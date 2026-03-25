# Alloy Roadmap

**Date:** 2026-03-25
**Status:** Core framework complete. Choosing next direction.

## Completed

- Hybrid SSM-Attention architecture (Mamba-2 + Transformer)
- Metal kernels: conv1d+SiLU (8.3x), parallel scan (2.2x at cs=512)
- Zamba2-1.2B: weight conversion, LoRA adapter merging, generation (24.6 tok/s cached, 66.7 tok/s 4-bit)
- bfloat16 + 4-bit/8-bit quantization
- Autoresearch: 28 experiments, val_bpb 2.165→1.676
- Training pipeline: climbmix data, validated on toy model (69M, loss 8.7→4.4)
- Interactive chat CLI
- 88 tests

## Direction A: Inference Ecosystem (chosen)

Make Alloy the go-to hybrid SSM inference engine on Apple Silicon.

### Phase 1: Model Zoo
- [ ] Support more HF models: Jamba-1.5-Mini, Falcon Mamba, future Mamba-3
- [ ] `alloy convert` CLI for one-click HF→Alloy conversion
- [ ] Auto-download from HuggingFace Hub

### Phase 2: Serving
- [ ] `alloy serve` HTTP API (OpenAI-compatible format)
- [ ] Streaming SSE responses
- [ ] Concurrent request handling

### Phase 3: Distribution
- [ ] Publish to PyPI (`pip install alloy-mlx`)
- [ ] Pre-converted model weights on HuggingFace
- [ ] Documentation site

## Direction B: Train a Real Model (future)

Train a usable hybrid model from scratch on Apple Silicon.

- [ ] Download full climbmix dataset (6542 shards, ~600GB)
- [ ] Train `small.yaml` (500M params) for 10-20 hours
- [ ] Evaluate and publish to HuggingFace
- [ ] Compare hybrid vs pure transformer at same compute budget

**Prerequisite:** Needs dedicated multi-day training run.

## Direction C: Technical Blog / Paper (future)

Write up findings for the community.

### Topics
- Complete hybrid SSM-Attention implementation on Apple Silicon
- Autoresearch: 28 experiments, key architectural insights
  - Mamba first / Attention last ordering
  - Optimal Mamba:Attention ratio under fixed time budget
  - GQA effectiveness in hybrid models
- Metal kernel optimization: when matmul beats parallel scan (and vice versa)
- Zamba2 weight conversion: combined in_proj, D residual, gate-before-norm
- Practical tips: NaN in chunked scan, bf16 precision, dt clamping

**Prerequisite:** Can be done in parallel with A.
