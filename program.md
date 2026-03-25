# autoresearch-mlx × Alloy

This is an autonomous research harness for the **Alloy Hybrid SSM-Attention** model,
adapted from [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx).

The goal: let an AI agent autonomously explore the hybrid architecture design space
(Mamba-2 + Attention interleaving) to find the optimal configuration on Apple Silicon.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `program.md` — this file (experiment protocol).
   - `prepare.py` — fixed: data prep, tokenizer, dataloader, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Hybrid model, optimizer, training loop.
4. **Verify data exists**: Check `~/.cache/autoresearch/` for data shards and tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with header row. Run `uv run train.py` once to establish YOUR baseline on this hardware.
6. **Confirm and go**.

## Experimentation

Each experiment runs on Apple Silicon via MLX with a **fixed 5-minute training budget**.
Launch: `uv run train.py`

**What you CAN modify** (all in `train.py`):
- Model architecture: depth, d_model, Mamba/Attention ratio, attn_layer_indices
- Mamba parameters: d_state, d_conv, expand, headdim, chunk_size
- Attention parameters: n_heads, n_kv_heads, window_size, full_attn_layers
- FFN: ffn_mult, activation function
- Optimizer: per-parameter LR groups, weight decay, betas, schedule
- Batch size, gradient accumulation
- Any architectural innovation (new block types, gating, normalization, etc.)

**What you CANNOT modify**:
- `prepare.py` — read-only (evaluation, data loading, tokenizer, constants).
- Dependencies in `pyproject.toml`.
- The evaluation harness (`evaluate_bpb`).

**The goal: lowest val_bpb.**

### Hybrid-Specific Search Dimensions

This is what makes this experiment unique vs vanilla autoresearch. Explore:

1. **Mamba:Attention ratio** — Current default is 3:1. Try 1:1, 7:1, all-Mamba, all-Attention.
2. **Attention layer placement** — Which layers get attention? First/last? Evenly spaced? Clustered?
3. **SSM hyperparameters** — d_state (16/32/64/128), expand factor (1/2/4), headdim.
4. **Sliding window** — Full attention on some layers, sliding window on others.
5. **Per-parameter LR** — Mamba params vs Attention params may need different learning rates.
6. **Architecture innovations** — Value embedding gates, residual lambdas, alternative FFN activations.

**Key insight from autoresearch**: With a fixed 5-minute budget, **smaller models that train faster often beat larger models**. The hybrid architecture's advantage is that Mamba layers are cheaper than Attention — exploit this.

### Simplicity Criterion

All else being equal, simpler is better. A 0.001 val_bpb improvement from 20 lines of hacky code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep.

## Output Format

The script prints a summary after training:

```
---
val_bpb:          2.534000
training_seconds: 312.4
total_seconds:    405.7
peak_vram_mb:     27528.9
total_tokens_M:   39.8
num_steps:        46
num_params_M:     50.3
depth:            8
architecture:     hybrid (mamba:6 attn:2)
```

Extract results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`

## Logging Results

Log to `results.tsv` (tab-separated):

```
commit	val_bpb	memory_gb	status	description
```

Example:
```
commit	val_bpb	memory_gb	status	description
abc1234	2.667000	26.9	keep	baseline hybrid 8L (6M+2A)
def5678	2.534000	26.9	keep	reduce depth to 4L (3M+1A)
ghi9012	2.700000	26.9	discard	all-mamba 8L (worse than hybrid)
```

## The Experiment Loop

LOOP FOREVER:

1. Check git state
2. Edit `train.py` with an experimental idea
3. `git add train.py && git commit -m "experiment: <description>"`
4. Run: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If grep empty → crash. `tail -n 50 run.log` to debug.
7. Record in `results.tsv`
8. If val_bpb improved: `git add results.tsv && git commit --amend --no-edit`
9. If equal or worse: record discard hash, then `git reset --hard <previous kept commit>`

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try radical architectural changes, read the Mamba-2 paper for ideas. The loop runs until manually interrupted.

**Timeout**: ~7 minutes per experiment (5 min train + ~2 min compile/eval on Apple Silicon). Kill runs exceeding 15 minutes.
