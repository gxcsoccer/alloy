# Alloy

[中文版](README_zh.md)

Hybrid SSM-Attention language model built on [MLX](https://github.com/ml-explore/mlx) for Apple Silicon.

Alloy interleaves Mamba-2 (selective state-space) blocks with Attention blocks in a single model, combining the linear-time efficiency of SSMs with the precise recall of Attention.

## Features

- **Mamba-2 block** — selective scan with chunked parallel computation + fused Metal parallel scan kernel
- **Attention block** — MHA / GQA / sliding-window, with RoPE
- **HybridLM** — configurable interleaved architecture, supports Alloy-native, Zamba2, and Nemotron-H modes
- **Agent** — tool-calling execution loop with built-in tools, JSON tool call parsing, multi-turn conversations
- **Agent LoRA fine-tuning** — QLoRA pipeline with cosine LR, validation, OOM protection, and 4 memory optimizations (8x training speedup on Apple Silicon)
- **Agent evaluation** — 3-layer eval: JSON format validation, BFCL AST accuracy (5 categories), self-built tool tests
- **Training** — AdamW + cosine schedule, streaming JSONL dataloader with packing
- **LoRA** — freeze-and-inject adapter, save/load/merge
- **Generation** — autoregressive decoding with KV + SSM cache, top-p sampling, streaming output
- **Weight conversion** — load Zamba2 / Jamba / Nemotron-H weights from HuggingFace
- **Metal kernels** — fused conv1d+SiLU (8.3x), parallel associative scan (2.2x at cs=512)
- **bfloat16** — mixed precision support (2x memory reduction, scan internals auto-promote to fp32)
- **Autoresearch** — autonomous architecture search harness (28 experiments, 22.6% improvement)

## Quickstart

```bash
pip install -e ".[dev]"
pip install transformers huggingface_hub  # for pretrained models
```

### Chat with a model (easiest)

```bash
# Auto-downloads from HuggingFace, 4-bit quantized, only 1.3 GB memory
python -m alloy.chat --model Zyphra/Zamba2-1.2B-instruct --quantize 4
```

### OpenAI-compatible API server

```bash
# Start server
python -m alloy.serve --model Zyphra/Zamba2-1.2B-instruct --quantize 4 --port 8000

# Use with any OpenAI client
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

### Convert models

```bash
# Download + convert + quantize + save
python -m alloy.convert_cli --model Zyphra/Zamba2-1.2B --quantize 4 --output models/zamba2-4bit
```

### Evaluate

```bash
python -m alloy.eval --model Zyphra/Zamba2-1.2B --quantize 4
```

### Python API

```python
from alloy.convert import load_pretrained
from alloy.generate import generate
from transformers import AutoTokenizer

model = load_pretrained("path/to/Zamba2-1.2B")
model.to_bfloat16()  # optional: halve memory (6.9 GB → 3.5 GB)
tokenizer = AutoTokenizer.from_pretrained("path/to/Zamba2-1.2B")
text = generate(model, tokenizer, "The capital of France is", max_tokens=100)
```

### Train from scratch

```bash
# With climbmix data (auto-detected)
python prepare.py --num-shards 10
python -m alloy.train --config configs/toy.yaml --data climbmix --max-steps 2000

# With custom JSONL data
python -m alloy.train --config configs/toy.yaml --data data/train.jsonl
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

### Agent (tool-calling)

```bash
# Interactive agent with built-in tools (calculate, web_search, get_weather, get_time)
python -m alloy.agent --model nvidia/Nemotron-H-8B-Reasoning-128K \
    --lora checkpoints/agent-lora-v9d/best.npz --quantize 4
```

### Agent fine-tuning

```bash
# 1. Generate training data (tool calls, no-tool, dotted namespaces, BFCL conversion)
python scripts/gen_tool_data.py

# 2. Clean and compress xlam data (60% token savings)
python scripts/clean_xlam_data.py

# 3. Combine datasets (xlam + generated, 90/10 train/val split)
python scripts/combine_data.py

# 4. Fine-tune with QLoRA + memory optimizations
PYTHONPATH=. python scripts/finetune_agent.py \
    --model ~/.cache/alloy/models/Nemotron-H-8B-Reasoning-128K \
    --data ~/.cache/alloy/datasets/agent_v9_train.jsonl \
    --val-data ~/.cache/alloy/datasets/agent_v9_val.jsonl \
    --output checkpoints/agent-lora-v9d \
    --steps 5000 --lr 1e-4 --lora-rank 64 \
    --max-seq-len 384 --quantize 4 \
    --val-interval 500 --warmup-steps 200 \
    --grad-checkpoint
```

Memory optimizations (8x training speedup on Apple Silicon):
- `--grad-checkpoint` — attention-only gradient checkpointing (4 O(L^2) layers out of 52)
- Length-sorted batching — eliminates GPU cache thrashing (auto-enabled)
- Pre-long-sequence cache clearing — prevents Metal memory fragmentation (auto-enabled)
- `--bf16` — bfloat16 LoRA training (halves activation memory, but may reduce name accuracy)

### Agent evaluation

```bash
# Run BFCL (5 categories) + self-built tool eval
PYTHONPATH=. python -m alloy.eval_agent \
    --model ~/.cache/alloy/models/Nemotron-H-8B-Reasoning-128K \
    --lora checkpoints/agent-lora-v9d/best.npz \
    --lora-rank 64 --quantize 4
```

### Agent training results

| Version | Data | Steps | BFCL simple | Self tool_sel | Irrelevance | JSON format |
|---------|------|-------|------------|--------------|-------------|-------------|
| Base | — | — | 0% | 0% | — | N/A |
| V7 | 3.2k | 1000 | 35% | 27.3% | 10% | 100% |
| V8 | 10.5k | 2000 | **56%** | 45.5% | 8% | 100% |
| V9d | 10.7k | 1500 | 46% | **63.6%** | 32% | 100% |

Key findings:
- **Compact system prompt** saves 57% tokens, enabling 3x more training data
- **Dotted namespace training** (module.function) improves tool selection on real-world APIs
- **Irrelevance vs tool-calling trade-off**: improving irrelevance detection degrades tool-calling in supervised training — an open research problem
- See [docs/v10-plan.md](docs/v10-plan.md) for next steps

### Autoresearch (autonomous architecture search)

```bash
python prepare.py --num-shards 10
python train.py > run.log 2>&1  # 5-min budget per experiment
```

See [program.md](program.md) and [docs/autoresearch-report.md](docs/autoresearch-report.md) for details.

## Model configurations

| Config | d_model | Layers | Params | Use case |
|--------|---------|--------|--------|----------|
| `toy.yaml` | 512 | 12 | ~100M | Architecture validation |
| `small.yaml` | 1024 | 24 | ~500M | Quick experiments |
| `medium.yaml` | 2048 | 32 | ~1.5B | Full training |
| `autoresearch.yaml` | 512 | 2 | ~15M | Optimal for 5-min autoresearch budget |

## Key findings from autoresearch

28 autonomous experiments validated core architectural decisions:

| Architecture | val_bpb | Notes |
|-------------|---------|-------|
| **Hybrid (1M+1A)** | **1.676** | Best — SSM + Attention complement each other |
| Pure Mamba (2M) | 1.999 | +0.32 worse, lacks precise recall |
| Pure Attention (2A) | 2.095 | +0.42 worse, despite more steps |

**Key insights:**
- **Mamba first, Attention last** — reversed order catastrophic (2.195)
- **Shallow + wide wins** under fixed time budget (2L > 3L > 4L)
- **GQA effective** even in hybrid models (n_kv_heads=2 helps)
- Batch size 2^13 optimal (balance: gradient quality vs step count)

See [docs/autoresearch-report.md](docs/autoresearch-report.md) for all 28 experiments.

## Project structure

```
alloy/
├── alloy/
│   ├── models/
│   │   ├── mamba_block.py       # Mamba-2 (Alloy + Zamba2 modes)
│   │   ├── mamba_kernels.py     # Metal GPU kernels
│   │   ├── attention_block.py   # MHA / GQA / sliding window
│   │   ├── hybrid_model.py      # HybridLM + HybridBlock + grad checkpoint
│   │   └── cache.py             # MambaCache / AttentionCache / Zamba2HybridLayerCache
│   ├── data/
│   │   └── dataloader.py        # Streaming JSONL + packing
│   ├── agent.py                  # Agent execution loop with tool calling
│   ├── eval_agent.py             # BFCL + self-eval 3-layer evaluation
│   ├── chat.py                   # Interactive chat CLI
│   ├── serve.py                  # OpenAI-compatible HTTP API
│   ├── eval.py                   # Lightweight evaluation suite
│   ├── convert_cli.py            # Model conversion CLI
│   ├── generate.py               # Autoregressive generation
│   ├── train.py                  # Training loop + CLI
│   ├── lora.py                   # LoRA inject / save / merge
│   └── convert.py                # HuggingFace weight conversion
├── scripts/
│   ├── finetune_agent.py         # Agent LoRA fine-tuning (QLoRA + memory opts)
│   ├── gen_tool_data.py          # Training data generation (tool calls, irrelevance, dotted names)
│   ├── clean_xlam_data.py        # xlam data cleaning + prompt compression
│   ├── combine_data.py           # Dataset combination + train/val split
│   └── race_v9.py                # Multi-lane experiment runner
├── configs/                       # YAML model configs
├── tests/                         # 88 tests
├── docs/
│   ├── v10-plan.md               # Next training iteration plan
│   ├── roadmap.md                # Project roadmap
│   ├── autoresearch-report.md    # 28-experiment report
│   └── spec.md                   # Full project spec
├── prepare.py                     # Autoresearch data pipeline
├── train.py                       # Autoresearch training script
└── pyproject.toml
```

## Tests

```bash
python -m pytest tests/ -v   # 88 tests, ~0.5s
```

## Architecture

### Alloy mode (default)

Each `HybridBlock` follows the pre-norm residual pattern:

```
x → RMSNorm → [MambaBlock or AttentionBlock] → + → RMSNorm → FFN → + → out
↑_______________________________________________↑    ↑________________↑
```

### Zamba2 mode (for pretrained Zamba2 models)

Hybrid layers contain both mamba and attention:

```
                    ┌─ cat(x, emb) → Norm → Attention → Norm → FFN ─┐
x → shared_transformer ─────────────────────────────────────────────── linear
    └─ (x + linear_out) → Norm → MambaDecoder → + → out ───────────────────┘
```

## Performance

### Metal kernel acceleration

| Operation | Pure MLX | Metal Kernel | Speedup |
|-----------|----------|-------------|---------|
| Conv1d + SiLU | 3.6ms | 0.4ms | **8.3x** |
| Scan chunk (cs=512) | 11.6ms | 5.2ms | **2.2x** |
| MambaBlock forward | 36.7ms | 26.7ms | **1.38x** |

The parallel scan kernel auto-selects based on chunk size: matmul for cs<256 (hardware matrix engines optimal), Metal parallel scan for cs≥256 (O(cs·log cs) beats O(cs²)).

### Zamba2-1.2B inference

| Mode | Speed | Memory |
|------|-------|--------|
| No cache | 5.3 tok/s | 6.9 GB (fp32) |
| KV + SSM cache | 24.6 tok/s | 6.9 GB (fp32) |
| KV + SSM cache + bf16 | 24.6 tok/s | 3.5 GB |
| **KV + SSM cache + 4-bit** | **66.7 tok/s** | **1.3 GB** |

### Logit alignment with HuggingFace reference

| Configuration | Avg top-5 diff | Top-1 agreement |
|--------------|---------------|-----------------|
| Without LoRA adapters | 0.64 | — |
| **With LoRA adapters merged** | **0.23** | **80%** |

### Agent fine-tuning performance

| Metric | Before optimization | After optimization | Speedup |
|--------|--------------------|--------------------|---------|
| Training time (2000 steps) | 16.1h | 2.2h | **7.3x** |
| Avg step time | 29.1s | 2.5s | **11.6x** |
| p95 step time | 94.8s | 2.5s | **37.9x** |
| Max step time | 405s | 3.7s | **109x** |

Optimizations: length-sorted batching, pre-long-sequence cache clearing, attention-only gradient checkpointing (4/52 layers).

### Benchmark (quick eval, small sample)

| Benchmark | Score | Random baseline |
|-----------|-------|-----------------|
| MMLU (20 questions) | 80.0% | 25% |
| HellaSwag (5 questions) | 100.0% | 25% |

## References

- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) — Gu & Dao
- [Transformers are SSMs](https://arxiv.org/abs/2405.21060) — Dao & Gu (Mamba-2)
- [Jamba: A Hybrid Transformer-Mamba LM](https://arxiv.org/abs/2403.19887)
- [Zamba2](https://arxiv.org/abs/2411.15242) — Zyphra
- [Nemotron-H](https://arxiv.org/abs/2504.03624) — NVIDIA hybrid SSM-Transformer
- [BFCL](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html) — Berkeley Function Calling Leaderboard
- [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — Karpathy's autonomous research
- [MLX](https://github.com/ml-explore/mlx)

## License

MIT
