"""Fine-tune Nemotron-H for tool calling via LoRA.

Usage:
    python scripts/finetune_agent.py \
        --model ~/.cache/alloy/models/Nemotron-H-8B-Reasoning-128K \
        --data ~/.cache/alloy/datasets/tool_calling_train.jsonl \
        --output checkpoints/agent-lora \
        --steps 500 --lr 1e-4 --lora-rank 64
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from alloy.convert import load_pretrained
from alloy.lora import linear_to_lora_layers


def load_data(path, tokenizer, max_seq_len=2048):
    """Load JSONL messages and convert to token sequences."""
    examples = []
    skipped = 0
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            messages = item["messages"]

            # Filter out 'tool' role messages (not supported by all tokenizers)
            messages = [m for m in messages if m["role"] in ("system", "user", "assistant")]

            # Format using chat template
            if hasattr(tokenizer, 'apply_chat_template'):
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False,
                    )
                except Exception:
                    skipped += 1
                    continue
            else:
                text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

            ids = tokenizer.encode(text)
            # Skip examples that are too long (prevents OOM spikes)
            if len(ids) > max_seq_len:
                skipped += 1
                continue
            if len(ids) < 10:
                skipped += 1
                continue
            examples.append(ids)

    if skipped:
        print(f"  Skipped {skipped} examples (too long/short/error)")
    return examples


def make_batches(examples, batch_size, sort_by_length=False):
    """Create batches with padding.

    Args:
        sort_by_length: If True, sort examples by length and batch similar
            lengths together. This stabilizes GPU memory usage and avoids
            cache thrashing from alternating short/long sequences.
            Within each bucket of 64 examples, order is shuffled to
            prevent overfitting to length-sorted order.
    """
    import random

    if sort_by_length:
        # Sort by length, then shuffle within buckets of 64
        indexed = sorted(enumerate(examples), key=lambda x: len(x[1]))
        bucket_size = 64
        bucketed = []
        for i in range(0, len(indexed), bucket_size):
            bucket = indexed[i:i + bucket_size]
            random.shuffle(bucket)
            bucketed.extend(bucket)
        ordered = [examples[idx] for idx, _ in bucketed]
    else:
        ordered = list(examples)
        random.shuffle(ordered)

    batches = []
    for i in range(0, len(ordered), batch_size):
        batch = ordered[i:i + batch_size]
        max_len = max(len(ex) for ex in batch)
        padded = [ex + [0] * (max_len - len(ex)) for ex in batch]
        batches.append(mx.array(padded))
    return batches


def loss_fn(model, batch):
    """Causal LM loss on the batch."""
    logits = model(batch[:, :-1])
    targets = batch[:, 1:]
    # Mask padding (token 0)
    mask = targets != 0
    ce = nn.losses.cross_entropy(logits, targets, reduction="none")
    return (ce * mask).sum() / mask.sum()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune for tool calling")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="checkpoints/agent-lora")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--quantize", type=int, choices=[4, 8], default=None, help="Quantize base model")
    parser.add_argument("--val-data", type=str, default=None, help="Validation data JSONL")
    parser.add_argument("--val-interval", type=int, default=100, help="Validate every N steps")
    parser.add_argument("--warmup-steps", type=int, default=50, help="LR warmup steps")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Use gradient checkpointing (save ~50%% memory, ~30%% slower)")
    parser.add_argument("--bf16", action="store_true", help="Train LoRA in bfloat16 (halves activation memory)")
    parser.add_argument("--resume-lora", type=str, default=None, help="Resume from existing LoRA weights")
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    from alloy.convert_cli import download_model
    model_dir = download_model(args.model)
    model = load_pretrained(model_dir)

    nparams_total = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Total params: {nparams_total / 1e9:.2f}B")

    # Apply LoRA BEFORE quantization (QLoRA pattern)
    print(f"Applying LoRA (rank={args.lora_rank})...")
    model.freeze()
    linear_to_lora_layers(model, lora_rank=args.lora_rank)

    if args.quantize:
        print(f"Quantizing base weights to {args.quantize}-bit (QLoRA)...")
        nn.quantize(model, bits=args.quantize, class_predicate=lambda path, m: isinstance(m, nn.Linear) and "lora" not in path)

    if args.resume_lora:
        from alloy.lora import load_lora_weights
        print(f"Resuming from LoRA weights: {args.resume_lora}")
        load_lora_weights(model, args.resume_lora)

    if args.bf16:
        # Convert LoRA params to bfloat16 for mixed-precision training.
        # Halves activation memory since the whole forward pass stays in bf16
        # (Mamba scan internally promotes to fp32 then casts back).
        bf16_params = [(n, p.astype(mx.bfloat16))
                       for n, p in tree_flatten(model.trainable_parameters())]
        model.load_weights(bf16_params, strict=False)
        print("Training in bfloat16 (activations ~50% smaller)")

    nparams_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"Trainable params: {nparams_trainable / 1e6:.1f}M ({100 * nparams_trainable / nparams_total:.2f}%)")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Load data
    print(f"Loading data: {args.data}")
    examples = load_data(args.data, tokenizer, args.max_seq_len)
    print(f"Loaded {len(examples)} examples, avg length: {sum(len(e) for e in examples) / len(examples):.0f} tokens")

    # Cosine LR schedule with warmup
    import math

    def cosine_lr(step):
        if step < args.warmup_steps:
            return args.lr * step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    optimizer = optim.AdamW(learning_rate=cosine_lr(0))
    if args.grad_checkpoint:
        n = model.enable_grad_checkpoint(attention_only=True)
        print(f"  Gradient checkpointing: {n} attention layers (O(L²) bottleneck only)")
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Load validation data
    val_examples = None
    if args.val_data:
        print(f"Loading validation data: {args.val_data}")
        val_examples = load_data(args.val_data, tokenizer, args.max_seq_len)
        print(f"  Loaded {len(val_examples)} val examples")

    def compute_val_loss():
        """Compute average loss on validation set."""
        if not val_examples:
            return None
        val_batches = make_batches(val_examples[:200], args.batch_size)
        total_loss = 0
        for batch in val_batches:
            loss = loss_fn(model, batch)
            mx.eval(loss)
            total_loss += loss.item()
        return total_loss / len(val_batches)

    # Training loop
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = args.batch_size * args.grad_accum
    print(f"\nTraining for {args.steps} steps, lr={args.lr}, batch_size={args.batch_size}")
    if args.grad_accum > 1:
        print(f"  Gradient accumulation: {args.grad_accum} steps (effective batch={effective_batch})")
    print(f"  Cosine schedule: warmup={args.warmup_steps} steps")
    if val_examples:
        print(f"  Validation every {args.val_interval} steps")

    step = 0
    best_val_loss = float("inf")
    tic = time.perf_counter()
    oom_skipped = 0

    # Set metal cache limit to prevent runaway memory growth
    if hasattr(mx, 'metal') and mx.metal.is_available():
        # Use ~80% of available memory as cache limit
        info = mx.metal.device_info()
        rec_size = info.get("recommended_max_working_set_size", 0)
        if rec_size > 0:
            mx.metal.set_cache_limit(int(rec_size * 0.8))
            print(f"  Metal cache limit: {int(rec_size * 0.8) / 1e9:.1f} GB")

    while step < args.steps:
        batches = make_batches(examples, args.batch_size, sort_by_length=True)
        accum_loss = 0.0
        accum_count = 0

        for batch in batches:
            # Update LR
            lr = cosine_lr(step)
            optimizer.learning_rate = lr

            # Pre-clear cache before long sequences to prevent fragmentation
            seq_len = batch.shape[1]
            if seq_len > 256:
                mx.metal.clear_cache()

            try:
                loss, grads = loss_and_grad(model, batch)

                if args.grad_accum > 1:
                    # Accumulate: scale grads by 1/accum, defer optimizer step
                    accum_loss += loss.item()
                    accum_count += 1

                    if accum_count < args.grad_accum:
                        # Just eval loss to free graph, don't update yet
                        mx.eval(loss)
                        mx.metal.clear_cache()
                        continue

                    # Accumulated enough — update
                    optimizer.update(model, grads)
                    mx.eval(model.parameters(), optimizer.state)
                    loss_val = accum_loss / accum_count
                    accum_loss = 0.0
                    accum_count = 0
                else:
                    optimizer.update(model, grads)
                    mx.eval(loss, model.parameters(), optimizer.state)
                    loss_val = loss.item()
            except Exception as e:
                # OOM or other error — skip this batch
                oom_skipped += 1
                mx.metal.clear_cache()
                accum_loss = 0.0
                accum_count = 0
                print(f"step {step + 1:>4d} | SKIPPED (OOM #{oom_skipped}: {e})")
                step += 1
                tic = time.perf_counter()
                continue

            step += 1

            # Clear metal cache periodically to prevent fragmentation
            if step % 10 == 0:
                mx.metal.clear_cache()

            toc = time.perf_counter()
            elapsed = toc - tic
            print(f"step {step:>4d} | loss {loss_val:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
            tic = time.perf_counter()

            if step % args.val_interval == 0:
                from alloy.lora import save_lora_weights
                save_lora_weights(model, str(output_dir / f"step_{step:04d}.npz"))
                print(f"  Saved checkpoint: step_{step:04d}.npz")

                if val_examples:
                    mx.metal.clear_cache()
                    vl = compute_val_loss()
                    mx.metal.clear_cache()
                    marker = ""
                    if vl < best_val_loss:
                        best_val_loss = vl
                        save_lora_weights(model, str(output_dir / "best.npz"))
                        marker = " (best)"
                    print(f"  Val loss: {vl:.4f}{marker}")

            if step >= args.steps:
                break

    # Final save
    from alloy.lora import save_lora_weights
    save_lora_weights(model, str(output_dir / "final.npz"))
    print(f"\nTraining complete. Final LoRA weights: {output_dir / 'final.npz'}")
    if val_examples:
        print(f"Best val loss: {best_val_loss:.4f} (saved as best.npz)")


if __name__ == "__main__":
    main()
