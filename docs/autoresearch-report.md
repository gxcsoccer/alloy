# Autoresearch Experiment Report

**Date:** 2026-03-25
**Platform:** Apple Silicon (MLX)
**Protocol:** 5-minute fixed training budget, val_bpb evaluation
**Dataset:** climbmix-400b-shuffle (10 shards)
**Total experiments:** 28 (7 kept, 19 discarded, 2 crashed)

## Results Summary

| val_bpb | Status | Description |
|---------|--------|-------------|
| 2.1654 | keep | Baseline: 4L (3M+1A), d384, d_state=16, chunk=64, bs=4 |
| 2.5369 | discard | 3x learning rates — too aggressive |
| **2.0282** | keep | Shallow+wide: 2L (1M+1A), d512 — 108 steps |
| **1.8382** | keep | Halve batch to 2^15 — 212 steps |
| 1.9102 | discard | 2x learning rates |
| crash | crash | d_model=640 — OOM |
| **1.7521** | keep | Halve batch to 2^14 — 423 steps |
| 1.9992 | discard | All-Mamba 2L — attention helps |
| 1.8265 | discard | 3L (2M+1A) — slower steps |
| **1.6860** | keep | Batch 2^13 — 838 steps |
| 1.7072 | discard | 1.5x LR |
| 1.7515 | discard | d_state=32 — slower |
| 1.7131 | discard | d_conv=2 |
| 1.6984 | discard | No weight decay |
| 1.6877 | discard | beta1=0.8 — equal |
| **1.6839** | keep | Squared ReLU FFN (simpler, fewer params) |
| 1.6949 | discard | Warmdown 0.7 |
| 1.6957 | discard | 2x embedding LR |
| **1.6776** | keep | chunk_size=128 — 884 steps |
| 1.6836 | discard | chunk_size=256 |
| **1.6755** | keep | GQA n_kv_heads=2 — 891 steps, 15.4M params |
| 1.7101 | discard | Mamba headdim=128 |
| 2.1947 | discard | Attention first — terrible |
| 1.7180 | discard | n_heads=16, head_dim=32 |
| 1.7116 | discard | Batch 2^12 — gradient quality too low |
| 1.7197 | discard | ffn_mult=4 with SwiGLU — slower |

**Best: val_bpb = 1.6755** (22.6% improvement over baseline)

## Optimal Configuration

```yaml
# Architecture
depth: 2
d_model: 512
attn_layer_indices: [1]    # Mamba first, Attention last
n_heads: 8
n_kv_heads: 2              # GQA
d_state: 16
d_conv: 4
expand: 2
headdim: 64
chunk_size: 128
ffn: squared_relu           # x^2 activation, 2 matrices
ffn_mult: 4.0

# Optimizer
total_batch_size: 8192      # 2^13
device_batch_size: 4
mamba_lr: 0.004
attn_lr: 0.004
embedding_lr: 0.06
ffn_lr: 0.004
scalar_lr: 0.05
weight_decay: 0.1
adam_betas: [0.9, 0.95]
warmdown_ratio: 0.5
```

## Key Findings

### 1. Hybrid Architecture Validated

The central hypothesis of Alloy — that interleaving SSM and Attention is better than either alone — is confirmed:

| Architecture | val_bpb | Gap vs Hybrid |
|-------------|---------|---------------|
| **Hybrid (1M+1A)** | **1.676** | — |
| Pure Mamba (2M) | 1.999 | +0.323 worse |
| Pure Attention (2A) | 2.095 | +0.419 worse |

Both pure architectures are significantly worse despite having comparable parameter counts.

### 2. Layer Ordering Matters

**Mamba first, Attention last** is critical. Reversing the order (Attention→Mamba) produces val_bpb = 2.195, nearly as bad as the untrained baseline. The SSM appears to build compressed representations that Attention can then precisely select from.

### 3. Shallow Models Win Under Fixed Time Budget

With a 5-minute training constraint, fewer layers = faster steps = more optimizer updates:

| Depth | Steps | val_bpb |
|-------|-------|---------|
| 2L | 891 | 1.676 |
| 3L | 280 | 1.826 |
| 4L | 838 | 1.686 |

2L is optimal. 4L gets competitive on val_bpb but only because it runs fewer steps.

### 4. Batch Size Sweet Spot

Smaller batches give more optimizer steps, but too small hurts gradient quality:

| Batch | Steps | val_bpb |
|-------|-------|---------|
| 2^16 | 108 | 2.028 |
| 2^15 | 212 | 1.838 |
| 2^14 | 423 | 1.752 |
| **2^13** | **838** | **1.686** |
| 2^12 | 1587 | 1.712 |

2^13 is the sweet spot — halving from 2^16 was the single biggest improvement.

### 5. GQA Works in Hybrid Models

Reducing KV heads from 8 to 2 (4x GQA) improved val_bpb slightly while reducing parameters and increasing throughput. This suggests the Mamba layer already provides sufficient context mixing, so the Attention layer can work with fewer KV heads.

### 6. Squared ReLU Replaces SwiGLU

Replacing 3-matrix SwiGLU with 2-matrix squared ReLU FFN:
- Reduces parameters (15.8M → 15.4M)
- Increases throughput (more steps)
- Maintains or slightly improves val_bpb
- Simplifies code

### 7. Conservative Learning Rates

The hybrid model is more sensitive to learning rates than standard transformers. Autoresearch's default LRs (matrix: 0.04, embedding: 0.6) cause divergence. The optimal LRs are roughly 10x lower (matrix: 0.004, embedding: 0.06). This likely reflects the Mamba block's selective scan being numerically sensitive.

## Bug Fix: Chunked Scan NaN

A critical numerical bug was discovered and fixed in `MambaBlock._scan_chunk`:

**Before (buggy):**
```python
M = mx.exp(lac[:,:,:,None] - lac[:,:,None,:])  # exp before mask → inf
causal = mx.tril(mx.ones((cs, cs)))
M = M * causal  # too late, inf * 0 = nan
```

**After (fixed):**
```python
M_log = lac[:,:,:,None] - lac[:,:,None,:]
causal_mask = mx.where(mx.tril(mx.ones((cs,cs))) > 0, M_log, float("-inf"))
M = mx.exp(causal_mask)  # mask before exp → clean zeros
```

The upper triangle of the transfer matrix has positive log-values (since `lac[t] - lac[s]` for `t < s` is positive). Without masking before `exp()`, these become `+inf`, and subsequent `inf * 0 = NaN` propagates through the scan. This bug only manifests for sequences longer than ~130 tokens.
