# Alloy × autoresearch-mlx 结合点分析

两个项目都基于 **MLX + Apple Silicon**，技术栈高度同源，结合潜力很大。

## 项目对比

| 维度 | Alloy | autoresearch-mlx |
|------|-------|------------------|
| **核心定位** | Hybrid SSM-Attention 模型框架 | 自主研究循环（AI agent 自动优化训练代码） |
| **模型架构** | Mamba-2 + Attention 交替堆叠 | 标准 GPT Transformer |
| **框架** | MLX ≥0.22 | MLX ≥0.30 |
| **评估指标** | Cross-entropy loss | val_bpb（bits-per-byte，词表无关） |
| **数据** | JSONL streaming + packing | HuggingFace parquet + BPE + best-fit packing |
| **优化循环** | 手动训练 | AI agent 自动编辑→训练→评估→git keep/revert |

## 五个关键结合点

### 1. 用 autoresearch 循环自动优化 Alloy 架构（最高价值）

autoresearch-mlx 的核心能力是让 AI agent 在固定时间预算内自动搜索最优训练配置。将 Alloy 的 hybrid model 替换进其 `train.py`，可以自动探索：

- **Mamba:Attention 层比例**（当前固定 4:1，可能不是最优）
- **哪些层放 Attention**（attn_layer_indices 的最优分布）
- **sliding window 大小** vs full attention 的混合策略
- **chunk_size 对 parallel scan 性能的影响**
- **d_state、d_conv、expand_factor 等 SSM 超参**

这比手动调参高效得多，而且 git 历史自动记录每次实验。

### 2. 借鉴 autoresearch 的评估体系

autoresearch-mlx 用 **val_bpb** 而非 cross-entropy loss，优势是词表大小无关，可以公平比较不同 tokenizer 的模型。Alloy 当前用 CE loss，可以：

- 引入 `evaluate_bpb()` 函数作为标准化评估指标
- 采用固定时间预算（5分钟）的评估协议，方便在 Apple Silicon 上做可复现的 benchmark
- 这对比较 Alloy 的 hybrid 架构 vs 纯 Transformer 特别有价值

### 3. 升级数据管道

autoresearch-mlx 的数据管道比 Alloy 更成熟：

| 特性 | Alloy 当前 | autoresearch-mlx |
|------|-----------|------------------|
| 数据源 | 本地 JSONL | HuggingFace parquet 自动下载 |
| Tokenizer | tiktoken/sentencepiece（外部） | rustbpe（自训练 BPE） |
| Packing | 简单拼接 + EOS | Best-fit bin packing（100% 利用率） |
| 批次 | Drop incomplete | BOS-aligned infinite iterator |

可以将 autoresearch 的 `prepare.py` 中的 `make_dataloader` 和 best-fit packing 移植到 Alloy，提升训练效率。

### 4. 优化器改进

autoresearch-mlx 经过多轮自动搜索，发现了 **per-parameter learning rate groups** 的最优配置：

```
embedding: 0.6, matrix: 0.04, unembedding: 0.004, scalar: 0.5
```

Alloy 当前用统一 learning rate 的 AdamW。可以：
- 为 Mamba 参数（A_log, dt, conv kernel）和 Attention 参数设不同学习率
- SSM 的 A_log 参数特别敏感，可能需要更小的 lr
- 这些都可以通过 autoresearch 循环自动搜索

### 5. 架构交叉验证（Hybrid vs Pure Transformer）

autoresearch-mlx 的 GPT 已经在 climbmix 上跑到 val_bpb=1.294。将 Alloy 的 hybrid model 接入同一评估框架，可以直接回答：

> **在相同时间预算和硬件上，Hybrid SSM-Attention 是否优于 Pure Transformer？**

这对 Alloy 项目的学术价值和实用性都是关键验证。

## 实施路径

```
Phase 1: 接入评估体系
  - 在 Alloy 中实现 val_bpb 评估
  - 接入 climbmix 数据集 + prepare.py 管道

Phase 2: 适配 autoresearch 协议
  - 编写 Alloy 版 train.py（符合 program.md 约定）
  - 输出格式对齐（val_bpb: / peak_vram_mb:）

Phase 3: 自动搜索
  - 用 Claude Code 作为 agent 跑 autoresearch 循环
  - 重点搜索 Mamba/Attention 比例和超参

Phase 4: 反哺
  - 将搜索到的最优配置固化为 Alloy 的新 config
  - 将改进的数据管道和优化器合入主线
```

最大的价值在于 **Phase 3**：autoresearch 的自动循环天然适合探索 Alloy 这种多维度超参空间（层比例 × SSM 参数 × Attention 参数），这是手动调参难以覆盖的。
