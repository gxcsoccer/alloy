# V10 Training Plan

## Background

### Current Best Results

| Version | BFCL simple | Self tool_sel | Irrelevance | Names | OVERALL | no_tool |
|---------|------------|--------------|-------------|-------|---------|---------|
| V8 | **56%** | 45.5% | 8% | **98%** | **21.2%** | 100% |
| V9d (Lane D) | 46% | **63.6%** | 32% | 90% | 16.8% | 80% |

### Core Findings from V8-V9 Experiments

1. **Severe underfitting**: V8 saw only 19.5% of training data (2000 steps / 10,241 examples). Lane D saw 14%. Models stopped before learning the full dataset.

2. **Irrelevance vs tool-calling is zero-sum in supervised training**: Any amount of "don't call this tool" data degrades tool-calling accuracy. Tested with full irrelevance (V9c: simple 30%, irr 72%), reduced (Lane E: simple 36%, irr 44%), and two-stage (Lane F: simple 30%, irr 72%). All hurt tool-calling.

3. **Dotted namespace data helps self-eval**: Lane D achieved 63.6% self tool_select (record), up from V8's 45.5%. Dotted names (module.function) improve the model's understanding of tool namespaces.

4. **xlam is undersampled**: Only using 8k of 57k available (14%). xlam provides the critical generalization signal for unseen BFCL functions.

5. **Memory optimizations validated** (7x faster):
   - Length-sorted batching: eliminates GPU cache thrashing
   - Pre-long-sequence cache clearing: prevents fragmentation
   - Attention-only gradient checkpointing: 4 attention layers (O(L^2)) out of 52
   - bf16 LoRA: works but degrades names accuracy — **do not use for production**

## V10 Changes

### Data Composition

| Component | V8/Lane D | V10 | Rationale |
|-----------|-----------|-----|-----------|
| Base (gen_tool_data) | V8 + dotted names | Same as Lane D | tool_select 63.6% validated |
| xlam sample | 8,000 | **15,000** | More diverse function signatures for BFCL generalization |
| no-tool ratio | 25% | **18%** | Reduce conservative bias, maximize tool-calling signal |
| V8 coarse irrelevance | 392 examples | **Remove** | Causes no_tool precision 80% (false positives on self-eval) |
| V9 BFCL-style irrelevance | 0 (Lane D) | **0** | Zero-sum trade-off confirmed; handle at inference time |

### Training Config

| Parameter | V8 | Lane D | V10 |
|-----------|-----|--------|-----|
| Steps | 2000 | 1500 | **5000** |
| Data coverage | 19.5% | 14.0% | **~35%** |
| Learning rate | 1e-4 | 1e-4 | 1e-4 |
| LoRA rank | 64 | 64 | 64 |
| max_seq_len | 384 | 384 | 384 |
| Quantize | 4-bit | 4-bit | 4-bit |
| val_interval | 200 | 200 | **500** |
| warmup_steps | 100 | 100 | **200** |
| grad_checkpoint | no | attention-only | **attention-only** |
| bf16 | no | no | **no** (degrades names) |
| Length-sorted batching | no | yes | **yes** |

### Expected Training Time

- 5000 steps x ~2.5s/step = ~3.5 hours (+ val overhead ~1h)
- Total: **~4.5 hours**

## Expected Outcomes

| Metric | V8 | Lane D | V10 Target |
|--------|-----|--------|------------|
| BFCL simple_python | 56% | 46% | **50-58%** |
| Self tool_select | 45.5% | 63.6% | **55-65%** |
| Names accuracy | 98% | 90% | **92-98%** |
| no_tool precision | 100% | 80% | **95-100%** |
| BFCL irrelevance | 8% | 32% | **5-15%** |
| BFCL OVERALL | 21.2% | 16.8% | **20-25%** |

Key hypothesis: more training steps + more xlam diversity will recover V8's simple_python accuracy while keeping Lane D's tool_select improvement.

## Implementation Steps

1. Modify `gen_tool_data.py`: remove `gen_bfcl_style_irrelevance()` and `gen_bfcl_irrelevance_v9()` from `generate_all()`
2. Modify `combine_data.py`: increase xlam sample to 15k, reduce no-tool ratio to 18%
3. Run data pipeline: gen_tool_data → clean_xlam → combine
4. Train with: `--steps 5000 --val-interval 500 --warmup-steps 200 --grad-checkpoint`
5. Eval with BFCL + self-eval

## Future Directions (Post-V10)

### If V10 succeeds (simple >= 55%, tool_select >= 55%):

1. **BFCL format alignment**: Analyze specific argument mismatches in BFCL AST checker. Fix parameter type/default handling.
2. **Inference-time irrelevance**: Add system prompt instruction "Only call a function if it can directly solve the user's request" instead of training data.
3. **Rank-128 LoRA**: More parameter capacity for complex patterns.
4. **Multi-epoch training**: 10k+ steps to see full dataset 1-2 times.

### If V10 doesn't meet targets:

1. **Diagnose**: Run verbose eval to identify specific failure patterns.
2. **BFCL-specific fine-tuning**: Convert more BFCL ground truth to training data (currently only 200 examples from second half).
3. **DPO for irrelevance**: Train preference model with (correct_call, wrong_call) pairs — avoids the supervised zero-sum problem.
4. **Separate LoRA adapters**: One for tool-calling, one for irrelevance, merge at inference.
