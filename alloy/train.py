"""Training script for HybridLM.

Usage:
    python -m alloy.train --config configs/toy.yaml --data data/train.jsonl
"""

import argparse
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml

from alloy.data.dataloader import Dataset
from alloy.models.hybrid_model import HybridConfig, HybridLM


def loss_fn(model: HybridLM, input_ids: mx.array) -> mx.array:
    """Compute cross-entropy loss for next-token prediction.

    Args:
        model: HybridLM instance.
        input_ids: Token IDs of shape [B, L]. The target is input_ids shifted
            by one position (standard causal LM objective).

    Returns:
        Scalar loss value.
    """
    # inputs: all tokens except the last; targets: all tokens except the first
    logits = model(input_ids[:, :-1])  # [B, L-1, vocab_size]
    targets = input_ids[:, 1:]  # [B, L-1]

    # Cross-entropy loss
    loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
    return loss


def train_step(model, optimizer, batch):
    """Execute a single training step: forward, backward, update.

    Args:
        model: HybridLM instance.
        optimizer: MLX optimizer.
        batch: Token IDs of shape [B, L].

    Returns:
        Scalar loss value for this step.
    """
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, batch)
    optimizer.update(model, grads)
    return loss


def load_config(config_path: str) -> HybridConfig:
    """Load HybridConfig from a YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        HybridConfig instance.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return HybridConfig(**raw)


def count_parameters(model: HybridLM) -> int:
    """Count total trainable parameters."""
    nparams = sum(x.size for k, x in nn.utils.tree_flatten(model.trainable_parameters()))
    return nparams


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train HybridLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--data", type=str, required=True, help="Path to training data (jsonl)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max training steps")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=100, help="LR warmup steps")
    args = parser.parse_args()

    # Load config and build model
    config = load_config(args.config)
    model = HybridLM(config)
    mx.eval(model.parameters())
    nparams = count_parameters(model)
    print(f"Model: {config.n_layers} layers, {nparams / 1e6:.1f}M parameters")

    # Optimizer with cosine schedule + warmup
    warmup = optim.linear_schedule(0.0, args.lr, args.warmup_steps)
    cosine = optim.cosine_decay(args.lr, args.max_steps - args.warmup_steps)
    lr_schedule = optim.join_schedules([warmup, cosine], [args.warmup_steps])
    optimizer = optim.AdamW(learning_rate=lr_schedule)

    # Try to use autoresearch tokenizer + parquet data if --data points to a directory
    # or "climbmix". Otherwise fall back to JSONL with a simple tokenizer.
    data_path = args.data
    use_parquet = False

    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from prepare import Tokenizer, make_dataloader, MAX_SEQ_LEN
        tokenizer = Tokenizer.from_directory()
        use_parquet = True
        print(f"Using climbmix data with BPE tokenizer (vocab={tokenizer.get_vocab_size()})")
        # Override vocab_size to match tokenizer
        if config.vocab_size != tokenizer.get_vocab_size():
            print(f"  Adjusting vocab_size: {config.vocab_size} → {tokenizer.get_vocab_size()}")
            config.vocab_size = tokenizer.get_vocab_size()
            model = HybridLM(config)
            mx.eval(model.parameters())
            nparams = count_parameters(model)
            print(f"  Rebuilt model: {nparams / 1e6:.1f}M parameters")
    except (ImportError, FileNotFoundError):
        use_parquet = False

    if use_parquet:
        train_loader = make_dataloader(tokenizer, args.batch_size, args.seq_len, "train")
    else:
        class CharTokenizer:
            """Minimal character-level tokenizer for testing."""
            def encode(self, text: str) -> list:
                return [ord(c) % config.vocab_size for c in text]
            def decode(self, ids: list) -> str:
                return "".join(chr(i) for i in ids)

        tokenizer = CharTokenizer()
        dataset = Dataset(
            args.data, tokenizer,
            seq_len=args.seq_len, batch_size=args.batch_size,
        )

    # Training loop
    print(f"Training for {args.max_steps} steps, batch_size={args.batch_size}, seq_len={args.seq_len}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    tic = time.perf_counter()

    def batch_iter():
        """Unified batch iterator for both data sources."""
        if use_parquet:
            while True:
                x, y, _ = next(train_loader)
                # Stack inputs and targets: [B, seq_len+1] -> input_ids [B, seq_len]
                yield mx.concatenate([x[:, :1], y], axis=1)  # reconstruct full sequence
        else:
            while True:
                for batch in dataset:
                    yield batch

    for batch in batch_iter():
        loss = train_step(model, optimizer, batch)
        mx.eval(loss, model.parameters(), optimizer.state)
        step += 1

        if step % 10 == 0:
            toc = time.perf_counter()
            elapsed = toc - tic
            tps = (10 * args.batch_size * args.seq_len) / elapsed
            print(
                f"step {step:>6d} | "
                f"loss {loss.item():.4f} | "
                f"lr {lr_schedule(step).item():.2e} | "
                f"{tps:.0f} tok/s"
            )
            tic = time.perf_counter()

        if step % args.save_every == 0:
            ckpt_path = output_dir / f"step_{step:06d}.safetensors"
            model.save_weights(str(ckpt_path))
            print(f"Saved checkpoint: {ckpt_path}")

        if step >= args.max_steps:
            break

    # Final save
    ckpt_path = output_dir / "final.safetensors"
    model.save_weights(str(ckpt_path))
    print(f"Training complete. Final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
