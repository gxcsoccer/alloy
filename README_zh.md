# Alloy

基于 [MLX](https://github.com/ml-explore/mlx) 在 Apple Silicon 上实现的 Hybrid SSM-Attention 语言模型。

Alloy 将 Mamba-2（选择性状态空间）块与 Attention 块交错堆叠在同一模型中，兼具 SSM 的线性时间效率和 Attention 的精确回忆能力。

## 特性

- **Mamba-2 块** — 选择性扫描 + 融合 Metal parallel scan 内核
- **Attention 块** — 支持 MHA / GQA / 滑动窗口，RoPE 位置编码
- **HybridLM** — 可配置的交错架构，支持 Alloy 原生模式和 Zamba2 模式
- **训练** — AdamW + 余弦退火，流式 JSONL 数据加载 + 序列拼接
- **LoRA** — 冻结主干 + 注入适配器，保存/加载/合并
- **生成** — 自回归解码，KV + SSM 缓存，top-p 采样，流式输出
- **权重转换** — 从 HuggingFace 加载 Zamba2 / Jamba 权重（Zamba2-1.2B 已验证，含 LoRA adapter 合并）
- **Metal 内核** — 融合 conv1d+SiLU (8.3x)、parallel associative scan (cs=512 时 2.2x)
- **bfloat16** — 混合精度支持（内存减半，scan 内部自动提升至 fp32）
- **Autoresearch** — 自主架构搜索（28 轮实验，val_bpb 提升 22.6%）

## 快速开始

```bash
pip install -e ".[dev]"
pip install transformers huggingface_hub  # 加载预训练模型需要
```

### 对话（最简单）

```bash
# 自动下载，4-bit 量化，仅需 1.3 GB 内存
python -m alloy.chat --model Zyphra/Zamba2-1.2B-instruct --quantize 4
```

### OpenAI 兼容 API 服务

```bash
# 启动服务
python -m alloy.serve --model Zyphra/Zamba2-1.2B-instruct --quantize 4 --port 8000

# 调用
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好！"}]}'
```

### 模型转换

```bash
python -m alloy.convert_cli --model Zyphra/Zamba2-1.2B --quantize 4 --output models/zamba2-4bit
```

### 评测

```bash
python -m alloy.eval --model Zyphra/Zamba2-1.2B --quantize 4
```

### Python API

```python
from alloy.convert import load_pretrained
from alloy.generate import generate
from transformers import AutoTokenizer

model = load_pretrained("path/to/Zamba2-1.2B")
model.to_bfloat16()  # 可选：内存减半 (6.9 GB → 3.5 GB)
tokenizer = AutoTokenizer.from_pretrained("path/to/Zamba2-1.2B")
text = generate(model, tokenizer, "The capital of France is", max_tokens=100)
```

### 从头训练

```bash
# 使用 climbmix 数据（自动检测）
python prepare.py --num-shards 10
python -m alloy.train --config configs/toy.yaml --data climbmix --max-steps 2000

# 使用自定义 JSONL 数据
python -m alloy.train --config configs/toy.yaml --data data/train.jsonl
```

### LoRA 微调

```python
from alloy.lora import linear_to_lora_layers, save_lora_weights, merge_lora_weights

model.freeze()
linear_to_lora_layers(model, lora_rank=16)
# ... 正常训练，只有 LoRA 参数会更新 ...
save_lora_weights(model, "adapter.npz")
merge_lora_weights(model)  # 将适配器合并回主干权重
```

### 自主架构搜索 (Autoresearch)

```bash
python prepare.py --num-shards 10
python train.py > run.log 2>&1  # 每轮 5 分钟预算
```

详见 [program.md](program.md) 和 [docs/autoresearch-report.md](docs/autoresearch-report.md)。

## 模型配置

| 配置 | d_model | 层数 | 参数量 | 用途 |
|------|---------|------|--------|------|
| `toy.yaml` | 512 | 12 | ~100M | 架构验证 |
| `small.yaml` | 1024 | 24 | ~500M | 快速实验 |
| `medium.yaml` | 2048 | 32 | ~1.5B | 正式训练 |
| `autoresearch.yaml` | 512 | 2 | ~15M | 5 分钟自主搜索最优 |

## Autoresearch 关键发现

28 轮自主实验验证了核心架构决策：

| 架构 | val_bpb | 说明 |
|------|---------|------|
| **Hybrid (1M+1A)** | **1.676** | 最优 — SSM 和 Attention 互补 |
| Pure Mamba (2M) | 1.999 | 差 0.32，缺乏精确回忆 |
| Pure Attention (2A) | 2.095 | 差 0.42，步数虽多但效果差 |

**关键洞察：**
- **Mamba 在前，Attention 在后** — 反序结果灾难性 (2.195)
- **浅层 + 宽模型** 在固定时间预算下更优 (2L > 3L > 4L)
- **GQA 有效** — n_kv_heads=2 在 hybrid 模型中同样有效
- Batch size 2^13 最优（梯度质量与步数的平衡）

## 项目结构

```
alloy/
├── alloy/
│   ├── models/
│   │   ├── mamba_block.py       # Mamba-2（Alloy + Zamba2 模式）
│   │   ├── mamba_kernels.py     # Metal GPU 内核
│   │   ├── attention_block.py   # MHA / GQA / 滑动窗口
│   │   ├── hybrid_model.py      # HybridLM + HybridBlock
│   │   └── cache.py             # MambaCache / AttentionCache / Zamba2HybridLayerCache
│   ├── data/
│   │   └── dataloader.py        # 流式 JSONL + 序列拼接
│   ├── generate.py              # 自回归生成
│   ├── train.py                 # 训练循环 + CLI
│   ├── lora.py                  # LoRA 注入/保存/合并
│   └── convert.py               # HuggingFace 权重转换
├── configs/                      # YAML 模型配置
├── tests/                        # 88 个测试
├── docs/
│   ├── spec.md                  # 完整项目规格说明
│   ├── autoresearch-report.md   # 28 轮实验报告
│   └── autoresearch-integration.md
├── prepare.py                    # Autoresearch 数据管道
├── train.py                      # Autoresearch 训练脚本
├── program.md                    # Autoresearch 实验协议
└── pyproject.toml
```

## 测试

```bash
python -m pytest tests/ -v   # 88 个测试, ~0.5s
```

## 架构

### Alloy 模式（默认）

每个 `HybridBlock` 采用 pre-norm 残差结构：

```
x → RMSNorm → [MambaBlock 或 AttentionBlock] → + → RMSNorm → FFN → + → out
↑________________________________________________↑    ↑________________↑
```

### Zamba2 模式（加载预训练 Zamba2 模型）

Hybrid 层同时包含 Mamba 和 Attention：

```
                    ┌─ cat(x, emb) → Norm → Attention → Norm → FFN ─┐
x → shared_transformer ─────────────────────────────────────────────── linear
    └─ (x + linear_out) → Norm → MambaDecoder → + → out ───────────────────┘
```

## 性能

### Metal 内核加速

| 操作 | 纯 MLX | Metal 内核 | 加速比 |
|------|--------|-----------|--------|
| Conv1d + SiLU | 3.6ms | 0.4ms | **8.3x** |
| Scan chunk (cs=512) | 11.6ms | 5.2ms | **2.2x** |
| MambaBlock 前向 | 36.7ms | 26.7ms | **1.38x** |

Parallel scan 内核根据 chunk size 自动选择：cs<256 用 matmul（硬件矩阵引擎最优），cs≥256 用 Metal parallel scan（O(cs·log cs) 优于 O(cs²)）。

### Zamba2-1.2B 推理

| 模式 | 速度 | 内存 |
|------|------|------|
| 无缓存 | 5.3 tok/s | 6.9 GB (fp32) |
| KV + SSM 缓存 | 24.6 tok/s | 6.9 GB (fp32) |
| KV + SSM 缓存 + bf16 | 24.6 tok/s | 3.5 GB |
| **KV + SSM 缓存 + 4-bit** | **66.7 tok/s** | **1.3 GB** |

### 与 HuggingFace 参考实现的 logit 对齐

| 配置 | 平均 top-5 差异 | Top-1 一致率 |
|------|----------------|-------------|
| 无 LoRA adapter | 0.64 | — |
| **合并 LoRA adapter** | **0.23** | **80%** |

### 评测（快速评测，小样本）

| 评测集 | 分数 | 随机基线 |
|--------|------|---------|
| MMLU (20 题) | 80.0% | 25% |
| HellaSwag (5 题) | 100.0% | 25% |

## 参考文献

- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) — Gu & Dao
- [Transformers are SSMs](https://arxiv.org/abs/2405.21060) — Dao & Gu (Mamba-2)
- [Jamba: A Hybrid Transformer-Mamba LM](https://arxiv.org/abs/2403.19887)
- [Zamba2](https://arxiv.org/abs/2411.15242) — Zyphra
- [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — Karpathy 自主研究
- [MLX](https://github.com/ml-explore/mlx)

## 许可证

MIT
