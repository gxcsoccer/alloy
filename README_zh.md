# Alloy

基于 [MLX](https://github.com/ml-explore/mlx) 在 Apple Silicon 上实现的 Hybrid SSM-Attention 语言模型。

Alloy 将 Mamba-2（选择性状态空间）块与 Attention 块交错堆叠在同一模型中，兼具 SSM 的线性时间效率和 Attention 的精确回忆能力。

## 特性

- **Mamba-2 块** — 选择性扫描，分块并行计算，纯 MLX 算子实现，支持自动微分
- **Attention 块** — 支持 MHA / GQA / 滑动窗口，使用 RoPE 位置编码
- **HybridLM** — 可配置的交错架构（Jamba 风格），RMSNorm + SwiGLU FFN，权重共享 LM Head
- **训练** — 基于 `mx.value_and_grad` 的训练循环，AdamW + 余弦退火，流式 JSONL 数据加载 + 序列拼接
- **LoRA** — 冻结主干 + 注入适配器，支持保存/加载/合并
- **生成** — 自回归解码，KV + SSM 缓存管理，top-p 采样，流式输出
- **权重转换** — 从 HuggingFace safetensors 加载 Jamba / Zamba2 权重

## 快速开始

```bash
pip install -e ".[dev]"
```

### 从头训练

```bash
python -m alloy.train \
  --config configs/toy.yaml \
  --data data/train.jsonl \
  --batch-size 4 \
  --seq-len 2048 \
  --lr 3e-4 \
  --max-steps 10000
```

数据格式 — 每行一个 JSON 对象：

```jsonl
{"text": "..."}
{"text": "..."}
```

### 文本生成

```python
from alloy.models.hybrid_model import HybridConfig, HybridLM
from alloy.generate import generate
from alloy.train import load_config

config = load_config("configs/toy.yaml")
model = HybridLM(config)
model.load_weights("checkpoints/final.safetensors")

# 需自备 tokenizer
text = generate(model, tokenizer, "从前有座山", max_tokens=128)
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

### 加载 HuggingFace 权重

```python
from alloy.convert import load_pretrained

model = load_pretrained("path/to/jamba-or-zamba2")
```

## 模型配置

| 配置 | d_model | 层数 | 参数量 | 显存占用 (bf16) | 用途 |
|------|---------|------|--------|----------------|------|
| `toy.yaml` | 512 | 12 | ~100M | ~0.2 GB | 架构验证、单元测试 |
| `small.yaml` | 1024 | 24 | ~500M | ~1 GB | 快速实验 |
| `medium.yaml` | 2048 | 32 | ~1.5B | ~3 GB | 正式训练起点 |

## 项目结构

```
alloy/
├── alloy/
│   ├── models/
│   │   ├── mamba_block.py       # Mamba-2 选择性扫描
│   │   ├── attention_block.py   # MHA / GQA / 滑动窗口
│   │   ├── hybrid_model.py      # HybridLM（配置、层、完整模型）
│   │   └── cache.py             # MambaCache / AttentionCache / HybridCache
│   ├── data/
│   │   └── dataloader.py        # 流式 JSONL + 序列拼接
│   ├── kernels/                  # Metal 内核（后续优化）
│   ├── generate.py              # 自回归生成
│   ├── train.py                 # 训练循环 + CLI
│   ├── lora.py                  # LoRA 注入/保存/合并
│   └── convert.py               # HuggingFace 权重转换
├── configs/                      # YAML 模型配置
├── tests/                        # 79 个测试
├── docs/
│   └── spec.md                  # 完整项目规格说明
└── pyproject.toml
```

## 测试

```bash
python -m pytest tests/ -v
```

## 架构

每个 `HybridBlock` 采用 pre-norm 残差结构：

```
x → RMSNorm → [MambaBlock 或 AttentionBlock] → + → RMSNorm → SwiGLU FFN → + → out
↑________________________________________________↑    ↑________________________↑
```

层类型（Mamba 或 Attention）由配置中的 `attn_layer_indices` 决定。例如，12 层模型中设 `[3, 7, 11]` 即 SSM:Attention = 3:1。

Mamba-2 的选择性扫描使用分块并行算法：每个大小为 C 的块内通过 C×C 传递矩阵乘法并行计算（在 Metal 上全并行），块间顺序传播状态。相比朴素顺序扫描，长序列上有约 3 倍加速。

## 参考文献

- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) — Gu & Dao
- [Transformers are SSMs](https://arxiv.org/abs/2405.21060) — Dao & Gu (Mamba-2)
- [Jamba: A Hybrid Transformer-Mamba LM](https://arxiv.org/abs/2403.19887)
- [Hymba: Hybrid-Head Architecture](https://developer.nvidia.com/blog/hymba-hybrid-head-architecture-boosts-small-language-model-performance/)
- [MLX](https://github.com/ml-explore/mlx)

## 许可证

MIT
