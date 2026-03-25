# 在 Apple Silicon 上从零构建 Hybrid SSM-Attention 语言模型

> 一个人 + Claude Code，一天内从零搭建框架、跑通 28 轮自主实验、实现 Metal GPU 加速、加载真实 1.2B 预训练模型并发布到 PyPI。

## 为什么做这个

Mamba（状态空间模型）和 Transformer 各有优劣：

- **Mamba**：线性时间复杂度，长序列高效，但"记忆模糊"
- **Transformer**：精确回忆能力强，但 O(n^2) 注意力是性能瓶颈

把两者交错堆叠——Mamba 负责压缩长程依赖，Attention 负责精确回忆——这就是 Hybrid SSM-Attention 架构的核心思想。Jamba、Zamba2 等模型已经验证了这条路线。

但现有实现都基于 PyTorch + CUDA。**Apple Silicon 用户被排除在外。**

Alloy 的目标：用 MLX 在 Mac 上原生实现整个 hybrid 架构，从训练到推理到 Metal GPU 加速。

## 核心架构

```
x → RMSNorm → [MambaBlock 或 AttentionBlock] → + → RMSNorm → FFN → + → out
```

每层要么是 Mamba（选择性状态空间），要么是 Attention（多头注意力）。层类型由 `attn_layer_indices` 配置决定，比如 12 层模型中设 `[3, 7, 11]` 就是 3:1 的 SSM:Attention 比例。

### Mamba-2 选择性扫描

Mamba 的核心是选择性扫描（selective scan）：

```
h[t] = A_disc[t] * h[t-1] + B[t] * x[t]
y[t] = C[t] · h[t]
```

这是一个线性递推，理论上 O(n) 就能处理整个序列。但朴素实现是顺序的，无法利用 GPU 并行性。

我们实现了**分块并行扫描**：将序列切成 chunk，每个 chunk 内通过 128×128 的传递矩阵做并行 matmul，chunk 间顺序传播状态。在 Apple Silicon 上比顺序扫描快 3x。

## 28 轮自主实验：Hybrid 架构到底有多强？

接入 [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx)（Karpathy 的自主研究框架的 MLX 版本）后，我们让 AI agent 在 5 分钟固定预算内自动搜索最优架构配置。28 轮实验，val_bpb 从 2.165 优化到 1.676（22.6% 提升）。

### 关键发现

**1. Hybrid 显著优于纯架构**

| 架构 | val_bpb |
|------|---------|
| Hybrid (1M+1A) | **1.676** |
| Pure Mamba (2M) | 1.999 (+0.32) |
| Pure Attention (2A) | 2.095 (+0.42) |

Mamba 和 Attention 确实是互补的——单独用任何一个都明显更差。

**2. 层顺序至关重要**

Mamba 在前、Attention 在后（val_bpb=1.676）远优于反序（val_bpb=2.195）。假说：SSM 先做长程压缩，Attention 再做精确选择。

**3. 固定时间预算下，浅模型胜**

| 深度 | 步数 | val_bpb |
|------|------|---------|
| 2 层 | 891 | 1.676 |
| 3 层 | 280 | 1.826 |
| 4 层 | 838 | 1.686 |

更浅的模型每步更快，在 5 分钟内能跑更多优化步。

## Metal 内核优化

### Conv1d + SiLU 融合内核：8.3x 加速

Mamba 的因果深度卷积原本需要 Python 循环 4 次（kernel_size=4），每次都是一次 MLX 操作调度。融合成一个 Metal kernel 后，3.6ms → 0.4ms。

### Parallel Associative Scan：大 chunk 时 2.2x

SSM 递推 `h[t] = a[t]*h[t-1] + b[t]` 是一个 monoid 上的 scan，可以用 Hillis-Steele 并行前缀和在 O(log n) 步完成。

但发现了一个有趣的 tradeoff：

| chunk_size | Matmul 方案 | Parallel Scan | 胜者 |
|------------|------------|---------------|------|
| 64 | 1.2ms | 1.8ms | Matmul |
| 128 | 2.2ms | 2.9ms | Matmul |
| 256 | 4.2ms | 4.0ms | 打平 |
| 512 | 11.6ms | 5.2ms | **Parallel Scan** |

**Apple Silicon 的矩阵引擎太强了**——对于小矩阵，O(n^2) 的 matmul 反而比 O(n·log n) 的 parallel scan 快，因为 matmul 用的是专用硬件。只有当 chunk_size≥256 时，matmul 的二次复杂度才开始吃亏。

我们实现了自动切换：小 chunk 用 matmul，大 chunk 用 parallel scan。

## 加载真实模型：Zamba2-1.2B 的坑

把 HuggingFace 上的 Zamba2-1.2B（1.2B 参数的 hybrid 模型）加载到 Alloy 是最耗时的部分。表面上是权重转换，实际遇到了一系列架构差异：

### 坑 1: in_proj 分割顺序

Zamba2 把 x, z(gate), B, C, dt 合并成一个 in_proj。HF 的分割顺序是 `[gate, conv_input, dt]`，我们最初实现成了 `[x, z, B+C, dt]`。结果：模型输出完全错误。

### 坑 2: Gate 在 Norm 之前

HuggingFace 的 `Zamba2RMSNormGated` 先做 `y * silu(gate)`，再做 RMSNorm。我们实现成了相反顺序。看似微小的差别，影响巨大。

### 坑 3: D residual 用未缩放的 x

SSM 的 D skip connection 应该用 dt 缩放**之前**的 x，不是之后的。这个 bug 导致 layer 0 就有 34% 的相对误差，38 层累积后模型完全不可用。

修复后："The capital of France is" → **Paris** 排名第一，生成流畅准确。

### 坑 4: 共享 Transformer + LoRA Adapter

Zamba2 的 6 个 hybrid 层共享同一个 transformer 权重，通过 LoRA adapter（rank=128）做 per-layer 差异化。我们在转换时直接将 adapter 合并进权重（W = W_base + B @ A），无运行时开销。

合并 adapter 后，logit 差异从 0.64 降到 0.23。

## 性能数据

最终 Zamba2-1.2B 在 Apple Silicon 上的推理性能：

| 配置 | 速度 | 内存 |
|------|------|------|
| FP32 无缓存 | 5.3 tok/s | 6.9 GB |
| FP32 + KV/SSM 缓存 | 24.6 tok/s | 6.9 GB |
| **4-bit 量化 + 缓存** | **66.7 tok/s** | **1.3 GB** |

4-bit 量化版本只需 1.3 GB 内存，8 GB 的 MacBook Air 都能跑。

## 一行命令即可使用

```bash
pip install alloy-mlx[serve]

# 对话
alloy-chat --model Zyphra/Zamba2-1.2B-instruct --quantize 4

# OpenAI 兼容 API
alloy-serve --model Zyphra/Zamba2-1.2B-instruct --quantize 4 --port 8000
```

## 开源

项目地址：[github.com/gxcsoccer/alloy](https://github.com/gxcsoccer/alloy)

PyPI：`pip install alloy-mlx`

88 个测试，MIT 许可证。欢迎 Star 和贡献。

---

*这个项目从零到发布用了一天，过程中大量使用 Claude Code 进行自主编码和调试。文中的 28 轮 autoresearch 实验、Metal kernel 编写、Zamba2 权重对齐的 debug 全程由 AI agent 驱动完成。*
