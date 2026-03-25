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
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            messages = item["messages"]

            # Format using chat template
            if hasattr(tokenizer, 'apply_chat_template'):
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            else:
                text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

            ids = tokenizer.encode(text)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            examples.append(ids)

    return examples


def make_batches(examples, batch_size):
    """Create batches with padding."""
    import random
    random.shuffle(examples)

    batches = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
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
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    from alloy.convert_cli import download_model
    model_dir = download_model(args.model)
    model = load_pretrained(model_dir)

    nparams_total = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Total params: {nparams_total / 1e9:.2f}B")

    # Apply LoRA
    print(f"Applying LoRA (rank={args.lora_rank})...")
    model.freeze()
    linear_to_lora_layers(model, lora_rank=args.lora_rank)

    nparams_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"Trainable params: {nparams_trainable / 1e6:.1f}M ({100 * nparams_trainable / nparams_total:.2f}%)")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Load data
    print(f"Loading data: {args.data}")
    examples = load_data(args.data, tokenizer, args.max_seq_len)
    print(f"Loaded {len(examples)} examples, avg length: {sum(len(e) for e in examples) / len(examples):.0f} tokens")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=args.lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {args.steps} steps, lr={args.lr}, batch_size={args.batch_size}")
    step = 0
    tic = time.perf_counter()

    while step < args.steps:
        batches = make_batches(examples, args.batch_size)
        for batch in batches:
            loss, grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)
            step += 1

            if step % 10 == 0:
                toc = time.perf_counter()
                elapsed = toc - tic
                print(f"step {step:>4d} | loss {loss.item():.4f} | {elapsed:.1f}s")
                tic = time.perf_counter()

            if step % 100 == 0:
                from alloy.lora import save_lora_weights
                save_lora_weights(model, str(output_dir / f"step_{step:04d}.npz"))
                print(f"  Saved checkpoint: step_{step:04d}.npz")

            if step >= args.steps:
                break

    # Final save
    from alloy.lora import save_lora_weights
    save_lora_weights(model, str(output_dir / "final.npz"))
    print(f"\nTraining complete. Final LoRA weights: {output_dir / 'final.npz'}")


if __name__ == "__main__":
    main()
