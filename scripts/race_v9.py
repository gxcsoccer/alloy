"""V9 horse race: 3 approaches to balance irrelevance vs tool-calling.

Lane D: V8 data + dotted names only (no new irrelevance)
Lane E: V8 data + dotted names + reduced irrelevance (50 examples)
Lane F: Two-stage: V8+dotted (1200 steps) → add irrelevance (300 steps)

Each lane: 1500 steps, fp32 LoRA, all memory optimizations.
"""

import json
import os
import random
import subprocess
import sys
import time

random.seed(42)
DATA_DIR = os.path.expanduser("~/.cache/alloy/datasets")
XLAM_PATH = os.path.join(DATA_DIR, "xlam_clean_train.jsonl")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data, path):
    with open(path, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
    print(f"  Saved {len(data)} examples to {path}")


def has_tool_call(example):
    return any(
        '"tool_calls"' in m.get("content", "")
        for m in example["messages"]
        if m["role"] == "assistant"
    )


def combine(v2_examples, xlam_path, out_train, out_val, no_tool_ratio=0.25):
    """Combine v2 + xlam data with target no-tool ratio."""
    xlam = load_jsonl(xlam_path)
    xlam_short = [ex for ex in xlam if sum(len(m.get("content", "")) for m in ex["messages"]) < 1600]
    xlam_sample = random.sample(xlam_short, min(8000, len(xlam_short)))

    v2_tool = [ex for ex in v2_examples if has_tool_call(ex)]
    v2_no_tool = [ex for ex in v2_examples if not has_tool_call(ex)]

    total_tool = len(v2_tool) + len(xlam_sample)
    target_no_tool = int(total_tool * no_tool_ratio / (1 - no_tool_ratio))

    if len(v2_no_tool) < target_no_tool:
        oversampled = (v2_no_tool * (target_no_tool // len(v2_no_tool) + 1))[:target_no_tool]
    else:
        oversampled = v2_no_tool[:target_no_tool]

    all_data = v2_tool + xlam_sample + oversampled
    random.shuffle(all_data)

    # 90/10 split
    tool_ex = [ex for ex in all_data if has_tool_call(ex)]
    no_tool_ex = [ex for ex in all_data if not has_tool_call(ex)]
    n_tool = int(len(tool_ex) * 0.9)
    n_no_tool = int(len(no_tool_ex) * 0.9)

    train = tool_ex[:n_tool] + no_tool_ex[:n_no_tool]
    val = tool_ex[n_tool:] + no_tool_ex[n_no_tool:]
    random.shuffle(train)
    random.shuffle(val)

    save_jsonl(train, out_train)
    save_jsonl(val, out_val)
    print(f"  Train: {len(train)} ({n_tool} tool + {n_no_tool} no-tool)")
    print(f"  Val: {len(val)}")
    return train, val


def run_cmd(cmd, log_file=None):
    """Run a command and stream output."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    print(f"\n>>> {cmd[:100]}...")
    if log_file:
        with open(log_file, "w") as f:
            proc = subprocess.run(cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout[-500:])
    return proc.returncode


def train_and_eval(data_train, data_val, output_dir, log_prefix, steps=1500):
    """Train + eval, return eval results."""
    train_log = f"{log_prefix}_train.log"
    eval_log = f"{log_prefix}_eval.log"

    cmd = (
        f"python -u scripts/finetune_agent.py "
        f"--model ~/.cache/alloy/models/Nemotron-H-8B-Reasoning-128K "
        f"--data {data_train} --val-data {data_val} "
        f"--output {output_dir} "
        f"--steps {steps} --lr 1e-4 --lora-rank 64 "
        f"--max-seq-len 384 --quantize 4 "
        f"--val-interval 200 --warmup-steps 100 "
        f"--grad-checkpoint"
    )
    print(f"\n{'='*60}")
    print(f"TRAINING: {log_prefix}")
    print(f"{'='*60}")
    t0 = time.time()
    run_cmd(cmd, train_log)
    train_time = time.time() - t0
    print(f"Training done in {train_time/60:.0f} min. Log: {train_log}")

    # Extract best val loss
    best_val = None
    with open(train_log) as f:
        for line in f:
            if "Best val loss:" in line:
                best_val = float(line.split("Best val loss:")[1].split()[0])

    # Eval
    cmd = (
        f"python -u -m alloy.eval_agent "
        f"--model ~/.cache/alloy/models/Nemotron-H-8B-Reasoning-128K "
        f"--lora {output_dir}/best.npz --lora-rank 64 --quantize 4"
    )
    print(f"\nEVAL: {log_prefix}")
    run_cmd(cmd, eval_log)
    print(f"Eval done. Log: {eval_log}")

    # Parse eval results
    results = {"best_val_loss": best_val, "train_time_min": train_time / 60}
    with open(eval_log) as f:
        for line in f:
            if "simple_python" in line and "%" in line:
                results["simple_python"] = line.strip()
            elif "irrelevance" in line and "%" in line and "Running" not in line:
                results["irrelevance"] = line.strip()
            elif "OVERALL" in line:
                results["overall"] = line.strip()
            elif "Tool Selection" in line:
                results["tool_select"] = line.strip()

    return results


def main():
    print("=" * 60)
    print("V9 HORSE RACE: 3 approaches")
    print("=" * 60)

    # Generate base v2 data (will be generated fresh each time by gen_tool_data.py)
    # We need to control which generators are included

    # ================================================================
    # LANE D: V8 data + dotted names only (no new irrelevance)
    # ================================================================
    print("\n\n### LANE D: V8 + dotted names (no new irrelevance) ###")

    # Generate v2 data without irr_v9
    sys.path.insert(0, ".")
    exec(open("scripts/gen_tool_data.py").read(), globals())

    single = gen_single_tool_calls()
    parallel_ex = gen_parallel_calls()
    multiple_ex = gen_multiple_calls()
    multi_turn = gen_multi_turn()
    no_tool = gen_no_tool_examples()
    bfcl = convert_bfcl_to_training(max_examples=200)
    irrelevance_v8 = gen_bfcl_style_irrelevance()  # V8's coarse irrelevance
    exact_names = gen_exact_name_copy_examples()
    dotted = gen_dotted_namespace_calls()
    # NO irr_v9

    lane_d_v2 = single + parallel_ex + multiple_ex + multi_turn + no_tool + bfcl + irrelevance_v8 + exact_names + dotted
    random.shuffle(lane_d_v2)
    print(f"Lane D v2: {len(lane_d_v2)} examples")

    d_train_path = os.path.join(DATA_DIR, "race_d_train.jsonl")
    d_val_path = os.path.join(DATA_DIR, "race_d_val.jsonl")
    combine(lane_d_v2, XLAM_PATH, d_train_path, d_val_path)

    # ================================================================
    # LANE E: V8 + dotted + reduced irrelevance (limit to ~50)
    # ================================================================
    print("\n\n### LANE E: V8 + dotted + reduced irrelevance (50) ###")

    irr_v9_full = gen_bfcl_irrelevance_v9()
    # Take only 50 (mix of positive and negative)
    irr_v9_reduced = random.sample(irr_v9_full, min(50, len(irr_v9_full)))

    lane_e_v2 = single + parallel_ex + multiple_ex + multi_turn + no_tool + bfcl + irrelevance_v8 + exact_names + dotted + irr_v9_reduced
    random.shuffle(lane_e_v2)
    print(f"Lane E v2: {len(lane_e_v2)} examples")

    e_train_path = os.path.join(DATA_DIR, "race_e_train.jsonl")
    e_val_path = os.path.join(DATA_DIR, "race_e_val.jsonl")
    combine(lane_e_v2, XLAM_PATH, e_train_path, e_val_path)

    # ================================================================
    # LANE F: Two-stage (V8+dotted 1200 steps → add full irrelevance 300 steps)
    # ================================================================
    print("\n\n### LANE F: Two-stage training ###")

    # Stage 1 data = Lane D data (no irrelevance)
    f_s1_train = d_train_path  # reuse lane D data
    f_s1_val = d_val_path

    # Stage 2 data = full V9 data with irrelevance
    lane_f_v2 = single + parallel_ex + multiple_ex + multi_turn + no_tool + bfcl + irrelevance_v8 + exact_names + dotted + irr_v9_full
    random.shuffle(lane_f_v2)
    f_s2_train = os.path.join(DATA_DIR, "race_f_s2_train.jsonl")
    f_s2_val = os.path.join(DATA_DIR, "race_f_s2_val.jsonl")
    combine(lane_f_v2, XLAM_PATH, f_s2_train, f_s2_val)

    # ================================================================
    # RUN RACES
    # ================================================================
    results = {}

    # Lane D
    results["D"] = train_and_eval(
        d_train_path, d_val_path,
        "checkpoints/agent-lora-v9d", "race_d", steps=1500)

    # Lane E
    results["E"] = train_and_eval(
        e_train_path, e_val_path,
        "checkpoints/agent-lora-v9e", "race_e", steps=1500)

    # Lane F: Two-stage
    print(f"\n{'='*60}")
    print("LANE F: Stage 1 (V8+dotted, 1200 steps)")
    print(f"{'='*60}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    cmd_s1 = (
        f"python -u scripts/finetune_agent.py "
        f"--model ~/.cache/alloy/models/Nemotron-H-8B-Reasoning-128K "
        f"--data {f_s1_train} --val-data {f_s1_val} "
        f"--output checkpoints/agent-lora-v9f "
        f"--steps 1200 --lr 1e-4 --lora-rank 64 "
        f"--max-seq-len 384 --quantize 4 "
        f"--val-interval 300 --warmup-steps 100 "
        f"--grad-checkpoint"
    )
    t0 = time.time()
    run_cmd(cmd_s1, "race_f_s1_train.log")
    print(f"Stage 1 done in {(time.time()-t0)/60:.0f} min")

    print(f"\nLANE F: Stage 2 (+ irrelevance, 300 steps from stage 1 checkpoint)")
    # Stage 2: continue from stage 1's best checkpoint with irrelevance data
    # Lower LR for fine-tuning
    cmd_s2 = (
        f"python -u scripts/finetune_agent.py "
        f"--model ~/.cache/alloy/models/Nemotron-H-8B-Reasoning-128K "
        f"--data {f_s2_train} --val-data {f_s2_val} "
        f"--output checkpoints/agent-lora-v9f-s2 "
        f"--resume-lora checkpoints/agent-lora-v9f/best.npz "
        f"--steps 300 --lr 3e-5 --lora-rank 64 "
        f"--max-seq-len 384 --quantize 4 "
        f"--val-interval 100 --warmup-steps 30 "
        f"--grad-checkpoint"
    )
    run_cmd(cmd_s2, "race_f_s2_train.log")
    total_f_time = (time.time() - t0) / 60

    # Eval Lane F (use stage 2's best)
    cmd_eval = (
        f"python -u -m alloy.eval_agent "
        f"--model ~/.cache/alloy/models/Nemotron-H-8B-Reasoning-128K "
        f"--lora checkpoints/agent-lora-v9f-s2/best.npz --lora-rank 64 --quantize 4"
    )
    run_cmd(cmd_eval, "race_f_eval.log")

    results["F"] = {"train_time_min": total_f_time}
    with open("race_f_eval.log") as f:
        for line in f:
            if "simple_python" in line and "%" in line:
                results["F"]["simple_python"] = line.strip()
            elif "irrelevance" in line and "%" in line and "Running" not in line:
                results["F"]["irrelevance"] = line.strip()
            elif "OVERALL" in line:
                results["F"]["overall"] = line.strip()
            elif "Tool Selection" in line:
                results["F"]["tool_select"] = line.strip()

    # ================================================================
    # FINAL COMPARISON
    # ================================================================
    print("\n" + "=" * 60)
    print("HORSE RACE RESULTS")
    print("=" * 60)
    for lane, r in sorted(results.items()):
        print(f"\n--- Lane {lane} ---")
        for k, v in r.items():
            print(f"  {k}: {v}")

    # Save results
    with open("race_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to race_results.json")


if __name__ == "__main__":
    main()
