# Hybrid MLX Model — Project Spec

> 在 Apple Silicon 上基于 MLX 实现并训练 Hybrid SSM-Attention 架构的语言模型

---

## 背景与动机

Transformer 的 Attention 机制在长序列上计算复杂度为 O(n²)，而纯 SSM（如 Mamba）虽然高效但在精确回忆上不及 Attention。Hybrid 架构通过交错堆叠两种 block，取长补短：

- **SSM（Mamba-2）**：线性时间复杂度，擅长序列压缩和长程依赖的"摘要"
- **Attention**：精确，擅长精确 token 回忆，充当"快照记忆"

参考实现：NVIDIA Hymba（并行融合）、Jamba / Zamba（顺序交错）、Nemotron-3-Super（LatentMoE + Mamba-2）。

本项目目标：**在 macOS Apple Silicon 上，用 MLX 从零实现一个可训练的 Hybrid 模型框架**，分路线 B（架构实现）和路线 C（训练/微调）两阶段推进，代码完全共用，不需要切换框架。

---

## 路线 B — 架构实现

### 目标

实现可在 MLX 上前向推理的 Hybrid 模型，核心组件包括 Mamba block、Attention block 以及两者的交错组合层。

### 核心组件

#### 1. Mamba-2 Block

```
MambaBlock(d_model, d_state, d_conv, expand, headdim, chunk_size)
  ├── Input projection (expand)
  ├── Conv1D (causal, d_conv)
  ├── Selective scan (SSM core)
  │     ├── A, B, C 参数（input-dependent）
  │     └── Chunked parallel scan（chunk_size 分块计算）
  ├── Output gate (SiLU)
  └── Output projection
```

**关键挑战：selective scan 的 Metal 实现**

原版 Mamba 依赖 CUDA kernel（`selective_scan_cuda`），在 macOS 上不可用。两种替代方案：

| 方案 | 实现方式 | 性能 | 自动微分 |
|------|---------|------|---------|
| A | 纯 MLX ops 拼凑（`mx.cumsum` 等） | ★★☆ | ✅ 自动 |
| B | 自定义 Metal kernel（`mx.fast.metal_kernel`） | ★★★ | 需手写 `vjp` |

**建议**：路线 B 阶段用方案 A 跑通全流程，路线 C 训练稳定后再用方案 B 优化热点。

#### 2. Attention Block

支持以下变体（可配置）：

- `full`：标准 Multi-Head Attention（MHA）
- `gqa`：Grouped Query Attention（GQA），节省 KV cache
- `sliding_window`：滑动窗口 Attention，控制局部感受野

```python
AttentionBlock(d_model, n_heads, n_kv_heads, window_size=None)
```

Hymba 的策略可参考：在前、中、后三层使用 full attention，其余层用 sliding window，减少全局 attention 比例至约 10%。

#### 3. Hybrid Layer 组合策略

两种融合方式：

**顺序交错（Jamba 风格）**
```
[Mamba, Mamba, Mamba, Attention, Mamba, Mamba, Mamba, Attention, ...]
比例约 3:1 或 7:1（SSM:Attn）
```

**并行融合（Hymba 风格）**
```
HybridHead = SSM_head || Attention_head → concat → proj
参数比例建议 5:1（SSM:Attn）
```

**建议从顺序交错开始**，实现更简单，调试更直观。

#### 4. 完整模型结构

```
HybridLM
  ├── Embedding (vocab_size, d_model)
  ├── N × HybridBlock
  │     ├── RMSNorm (pre-norm)
  │     ├── MambaBlock 或 AttentionBlock（按配置交替）
  │     └── RMSNorm + FFN（可选，SwiGLU）
  ├── RMSNorm (final)
  └── LM Head (d_model → vocab_size, weight-tied)
```

#### 5. KV Cache / State Cache

- Attention 层：标准 KV cache（`[B, n_kv_heads, seq, head_dim]`）
- Mamba 层：SSM state cache（`[B, d_state, d_model]`），形态完全不同

需要实现统一的 `HybridCache` 管理类，在生成阶段分别更新两种 cache。

### 交付物（路线 B）

- [ ] `chimera/models/mamba_block.py` — Mamba-2 block（纯 MLX ops 实现）
- [ ] `chimera/models/attention_block.py` — Attention block（支持 GQA/sliding window）
- [ ] `chimera/models/hybrid_model.py` — HybridLM 主模型
- [ ] `chimera/models/cache.py` — HybridCache 统一管理
- [ ] `chimera/generate.py` — 推理 / 文本生成
- [ ] 配置文件（YAML / dataclass）— 控制 d_model、n_layers、交错比例等超参
- [ ] 单元测试：各 block 前向 pass、cache 更新正确性

---

## 路线 C — 训练 / 微调

### 前提

路线 C **直接复用路线 B 的全部代码**，唯一新增内容是：

1. 数据 pipeline（DataLoader）
2. 训练 loop（loss、backward、optimizer step）
3. 可选的 LoRA adapter 层

### 训练 Loop

```python
model = HybridLM(config)           # 路线 B 的代码，一字不改
optimizer = optim.AdamW(...)

loss_and_grad_fn = mx.value_and_grad(model, loss_fn)

for batch in dataloader:
    loss, grads = loss_and_grad_fn(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
```

MLX 的 `mx.value_and_grad` 对任意 `nn.Module` 生效，无需修改模型定义。

### 自动微分注意事项

- 纯 MLX ops 实现的 selective scan：**自动微分**，无需额外工作
- 自定义 Metal kernel 实现的 scan：**需手动实现 `vjp`**（vector-Jacobian product）
- 建议训练阶段始终使用纯 MLX ops 版本，推理阶段可切换到 Metal kernel 版本

### 规模建议（M4 Pro 64GB）

| 模型规模 | d_model | n_layers | 参数量 | 显存占用（bf16） | 建议用途 |
|---------|---------|----------|--------|----------------|---------|
| Toy | 512 | 12 | ~100M | ~0.2GB | 架构验证、单元测试 |
| Small | 1024 | 24 | ~500M | ~1GB | 快速实验 |
| Medium | 2048 | 32 | ~1.5B | ~3GB | 正式训练起点 |
| Large | 4096 | 32 | ~3B | ~6GB | 64GB 可跑，需量化推理 |

> 64GB 统一内存的最大优势：模型权重和 KV/SSM cache 共享内存，不会 OOM 触发 PCIe 瓶颈。

### LoRA 微调

对于从已有 Hybrid 模型（如 Jamba、Zamba）加载权重后做领域微调的场景：

```python
# 冻结主干，只训练 adapter
model.freeze()
linear_to_lora_layers(model, lora_rank=16)  # 参考 mlx-lm 的实现

# 训练 loop 完全相同，只是 grads 仅流向 LoRA 参数
```

支持目标层可配置：Mamba 的 input/output proj、Attention 的 q/k/v/o proj。

### 数据格式

支持标准 HuggingFace datasets 格式，tokenizer 对齐 `tiktoken` / `sentencepiece`：

```jsonl
{"text": "..."}
{"text": "..."}
```

### 交付物（路线 C）

- [ ] `chimera/data/dataloader.py` — 支持流式加载、packing
- [ ] `chimera/train.py` — 主训练脚本（支持 CLI 参数）
- [ ] `chimera/lora.py` — LoRA adapter 注入 / 保存 / 合并
- [ ] `chimera/convert.py` — 从 HuggingFace safetensors 加载 Jamba / Zamba 权重
- [ ] 训练配置 YAML 示例

---

## 项目结构（整体）

```
chimera/
├── chimera/
│   ├── models/
│   │   ├── mamba_block.py
│   │   ├── attention_block.py
│   │   ├── hybrid_model.py
│   │   └── cache.py
│   ├── data/
│   │   └── dataloader.py
│   ├── kernels/
│   │   └── selective_scan.metal    # 可选，路线 B 后期优化
│   ├── generate.py
│   ├── train.py
│   ├── lora.py
│   └── convert.py
├── configs/
│   ├── toy.yaml
│   ├── small.yaml
│   └── medium.yaml
├── tests/
├── README.md
└── pyproject.toml
```

---

## 阶段规划

| 阶段 | 内容 | 里程碑 |
|------|------|--------|
| B-1 | Mamba block（纯 MLX ops） + 单元测试 | selective scan 输出数值正确 |
| B-2 | Attention block + HybridCache | 单层 hybrid 前向 pass 通过 |
| B-3 | HybridLM 组装 + generate 脚本 | 可从随机权重生成文本 |
| B-4 | 加载 Jamba / Zamba 权重 + 对齐测试 | logits 与 HuggingFace 实现误差 < 1e-3 |
| C-1 | 训练 loop + DataLoader | toy 规模 loss 正常下降 |
| C-2 | LoRA 支持 | 微调 Jamba 在自定义数据集上收敛 |
| C-3 | Metal kernel selective scan + vjp | 推理吞吐量提升 ≥ 2× vs 纯 MLX ops |

---

## 参考资料

- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) — Gu & Dao
- [Transformers are SSMs](https://arxiv.org/abs/2405.21060) — Dao & Gu（Mamba-2）
- [Hymba: Hybrid-Head Architecture](https://developer.nvidia.com/blog/hymba-hybrid-head-architecture-boosts-small-language-model-performance/)
- [Jamba: A Hybrid Transformer-Mamba LM](https://arxiv.org/abs/2403.19887)
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — 参考 LoRA 实现和 generate 接口
- [mamba.py](https://github.com/alxndrTL/mamba.py) — MLX 版 Mamba 参考实现
