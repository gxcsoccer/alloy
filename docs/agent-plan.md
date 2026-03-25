# Alloy Agent: Hybrid SSM Agent Model

## Base Model
**Nemotron-H-8B-Reasoning-128K** (NVIDIA)
- 8.1B params, 22 Mamba + 4 Attention layers
- 128K context, BF16, 16.2 GB
- 4-bit quantized: ~4 GB

## Architecture Match with Alloy
- Mamba-2 + Attention hybrid ✓
- GQA (n_kv_heads=8) ✓
- Squared ReLU FFN ✓
- Combined in_proj ✓
- d_state=128, n_groups=8, chunk_size=128 ✓

## Phases

### Phase 0: Model Loading
- [ ] Download Nemotron-H-8B-Reasoning-128K (16.2 GB)
- [ ] Add NemotronH weight conversion to Alloy convert.py
- [ ] Verify forward pass + generation quality
- [ ] Benchmark: tok/s, memory

### Phase 1: Agent Data Pipeline
- [ ] Design token format: `<think>`, `<tool_call>`, `<tool_result>`
- [ ] Collect/build training data:
  - Tool calling: Gorilla, ToolBench, API-Bank
  - Reasoning: GSM8K+CoT, MATH
  - Multi-turn tool interaction
  - Error recovery patterns
- [ ] Data preprocessing + validation

### Phase 2: Agent Fine-tuning
- [ ] Extend tokenizer with special tokens
- [ ] LoRA fine-tune (rank=64-128)
- [ ] Training: 3000-5000 steps on agent data
- [ ] Loss weighting: emphasize tool_call and think tokens

### Phase 3: Evaluation
- [ ] Tool call accuracy (JSON format, parameter extraction)
- [ ] Reasoning quality (GSM8K, multi-step logic)
- [ ] End-to-end task completion rate
- [ ] Compare with base model (before/after fine-tune)

### Phase 4: Production
- [ ] Agent serve API (tool definitions + execution loop)
- [ ] Tool execution sandbox
- [ ] Streaming output (show reasoning process)
- [ ] Publish as alloy-agent
