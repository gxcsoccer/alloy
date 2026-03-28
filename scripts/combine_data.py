"""Combine and split training data for agent fine-tuning V8.

V8 changes vs V7:
- Higher char limit (1600 vs 800) since system prompts are compressed
- Larger xlam sample (8000 vs 2000)
- No-tool ratio ~25% (v2 data now has more irrelevance examples)
- Output files: agent_v8_train.jsonl / agent_v8_val.jsonl

Combines:
- tool_calling_v2_train.jsonl (our generated data with no-tool + irrelevance + exact names)
- xlam_clean_train.jsonl (cleaned + compressed xlam)
"""

import json
import random
import os

random.seed(42)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def has_tool_call(example):
    return any(
        '"tool_calls"' in m.get("content", "")
        for m in example["messages"]
        if m["role"] == "assistant"
    )


def main():
    data_dir = os.path.expanduser("~/.cache/alloy/datasets")

    # Load datasets
    v2 = load_jsonl(os.path.join(data_dir, "tool_calling_v2_train.jsonl"))
    xlam = load_jsonl(os.path.join(data_dir, "xlam_clean_train.jsonl"))

    print(f"v2 data: {len(v2)} examples")
    print(f"xlam data: {len(xlam)} examples")

    # V8: higher char limit since system prompts are now compressed (~60% smaller)
    xlam_short = [ex for ex in xlam if sum(len(m.get("content", "")) for m in ex["messages"]) < 1600]
    print(f"xlam short (<1600 chars): {len(xlam_short)}/{len(xlam)}")

    # V8: much larger sample - we want 8k+ xlam examples
    xlam_sample = random.sample(xlam_short, min(8000, len(xlam_short)))
    print(f"xlam sampled: {len(xlam_sample)}")

    # Split v2 data into tool / no-tool
    v2_tool = [ex for ex in v2 if has_tool_call(ex)]
    v2_no_tool = [ex for ex in v2 if not has_tool_call(ex)]
    print(f"v2 tool: {len(v2_tool)}, v2 no-tool: {len(v2_no_tool)}")

    # Target ~25% no-tool in final dataset
    total_tool = len(v2_tool) + len(xlam_sample)
    target_no_tool = int(total_tool * 0.33)  # 25% of total = 33% of tool count
    if len(v2_no_tool) < target_no_tool:
        # Oversample with repetition
        oversampled = v2_no_tool * (target_no_tool // len(v2_no_tool) + 1)
        oversampled = oversampled[:target_no_tool]
        print(f"Oversampled no-tool: {len(v2_no_tool)} -> {len(oversampled)}")
    else:
        oversampled = v2_no_tool[:target_no_tool]
        print(f"Trimmed no-tool: {len(v2_no_tool)} -> {len(oversampled)}")

    # Combine
    all_data = v2_tool + xlam_sample + oversampled
    random.shuffle(all_data)

    # Split into tool / no-tool
    tool_examples = [ex for ex in all_data if has_tool_call(ex)]
    no_tool_examples = [ex for ex in all_data if not has_tool_call(ex)]

    print(f"\nCombined: {len(all_data)} total")
    print(f"  Tool calls: {len(tool_examples)} ({100*len(tool_examples)/len(all_data):.1f}%)")
    print(f"  No-tool:    {len(no_tool_examples)} ({100*len(no_tool_examples)/len(all_data):.1f}%)")

    # Split 90/10 maintaining ratio
    def split(examples, ratio=0.9):
        n = int(len(examples) * ratio)
        return examples[:n], examples[n:]

    tool_train, tool_val = split(tool_examples)
    no_tool_train, no_tool_val = split(no_tool_examples)

    train = tool_train + no_tool_train
    val = tool_val + no_tool_val
    random.shuffle(train)
    random.shuffle(val)

    print(f"\nTrain: {len(train)} ({len(tool_train)} tool + {len(no_tool_train)} no-tool)")
    print(f"Val:   {len(val)} ({len(tool_val)} tool + {len(no_tool_val)} no-tool)")
    print(f"No-tool ratio - train: {100*len(no_tool_train)/len(train):.1f}%, val: {100*len(no_tool_val)/len(val):.1f}%")

    # Save as V9
    train_path = os.path.join(data_dir, "agent_v9_train.jsonl")
    val_path = os.path.join(data_dir, "agent_v9_val.jsonl")
    for path, data in [(train_path, train), (val_path, val)]:
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
