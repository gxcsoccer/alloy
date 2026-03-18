# Alloy

[中文版](README_zh.md)

Hybrid SSM-Attention language model built on [MLX](https://github.com/ml-explore/mlx) for Apple Silicon.

Alloy interleaves Mamba-2 (selective state-space) blocks with Attention blocks in a single model, combining the linear-time efficiency of SSMs with the precise recall of Attention.

## Features

- **Mamba-2 block** — selective scan with chunked parallel computation, pure MLX ops, autodiff-friendly
- **Attention block** — MHA / GQA / sliding-window, with RoPE
- **HybridLM** — configurable interleaved architecture (Jamba-style), RMSNorm + SwiGLU FFN, weight-tied LM head
- **Training** — `mx.value_and_grad` based loop, AdamW + cosine schedule, streaming JSONL dataloader with packing
- **LoRA** — freeze-and-inject adapter, save/load/merge
- **Generation** — autoregressive decoding with KV + SSM cache, top-p sampling, streaming output
- **Weight conversion** — load Jamba / Zamba2 weights from HuggingFace safetensors

## Quickstart

```bash
pip install -e ".[dev]"
```

### Train from scratch

```bash
python -m alloy.train \
  --config configs/toy.yaml \
  --data data/train.jsonl \
  --batch-size 4 \
  --seq-len 2048 \
  --lr 3e-4 \
  --max-steps 10000
```

Data format — one JSON object per line:

```jsonl
{"text": "..."}
{"text": "..."}
```

### Generate

```python
from alloy.models.hybrid_model import HybridConfig, HybridLM
from alloy.generate import generate
from alloy.train import load_config

config = load_config("configs/toy.yaml")
model = HybridLM(config)
model.load_weights("checkpoints/final.safetensors")

# Bring your own tokenizer
text = generate(model, tokenizer, "The quick brown", max_tokens=128)
```

### LoRA fine-tune

```python
from alloy.lora import linear_to_lora_layers, save_lora_weights, merge_lora_weights

model.freeze()
linear_to_lora_layers(model, lora_rank=16)
# ... train as usual, only LoRA params update ...
save_lora_weights(model, "adapter.npz")
merge_lora_weights(model)  # fold adapter into base weights
```

### Load HuggingFace weights

```python
from alloy.convert import load_pretrained

model = load_pretrained("path/to/jamba-or-zamba2")
```

## Model configurations

| Config | d_model | Layers | Params | Memory (bf16) | Use case |
|--------|---------|--------|--------|---------------|----------|
| `toy.yaml` | 512 | 12 | ~100M | ~0.2 GB | Architecture validation |
| `small.yaml` | 1024 | 24 | ~500M | ~1 GB | Quick experiments |
| `medium.yaml` | 2048 | 32 | ~1.5B | ~3 GB | Training starting point |

## Project structure

```
alloy/
├── alloy/
│   ├── models/
│   │   ├── mamba_block.py       # Mamba-2 selective scan
│   │   ├── attention_block.py   # MHA / GQA / sliding window
│   │   ├── hybrid_model.py      # HybridLM (config, block, full model)
│   │   └── cache.py             # MambaCache / AttentionCache / HybridCache
│   ├── data/
│   │   └── dataloader.py        # Streaming JSONL + packing
│   ├── kernels/                  # Metal kernels (future)
│   ├── generate.py              # Autoregressive generation
│   ├── train.py                 # Training loop + CLI
│   ├── lora.py                  # LoRA inject / save / merge
│   └── convert.py               # HuggingFace weight conversion
├── configs/                      # YAML model configs
├── tests/                        # 79 tests
├── docs/
│   └── spec.md                  # Full project spec
└── pyproject.toml
```

## Tests

```bash
python -m pytest tests/ -v
```

## Architecture

Each `HybridBlock` follows the pre-norm residual pattern:

```
x → RMSNorm → [MambaBlock or AttentionBlock] → + → RMSNorm → SwiGLU FFN → + → out
↑_______________________________________________↑    ↑________________________↑
```

Layer type (Mamba vs Attention) is determined by `attn_layer_indices` in config. For example, `[3, 7, 11]` in a 12-layer model gives a 3:1 SSM:Attention ratio.

The Mamba-2 selective scan uses a chunked parallel algorithm: each chunk of size C is computed via a C×C transfer-matrix matmul (fully parallel on Metal), with sequential state propagation between chunks. This gives ~3x speedup over naive sequential scan for long sequences.

## References

- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) — Gu & Dao
- [Transformers are SSMs](https://arxiv.org/abs/2405.21060) — Dao & Gu (Mamba-2)
- [Jamba: A Hybrid Transformer-Mamba LM](https://arxiv.org/abs/2403.19887)
- [Hymba: Hybrid-Head Architecture](https://developer.nvidia.com/blog/hymba-hybrid-head-architecture-boosts-small-language-model-performance/)
- [MLX](https://github.com/ml-explore/mlx)

## License

MIT
