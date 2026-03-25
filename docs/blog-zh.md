# 我用一天时间，在 Mac 上造了一个语言模型推理引擎

> 从零开始，到 `pip install` 可用。中间经历了 28 次自动实验、4 个隐蔽 bug、和一个关于硬件的反直觉发现。

---

## 为什么要做这件事

去年开始，一种叫 **Mamba** 的新架构火了。

传统的 Transformer（就是 ChatGPT 背后的技术）有个致命问题：它处理文本的时间随长度**平方增长**。文本越长，越慢，越费内存。

Mamba 用了一种叫"状态空间模型"的方法，把复杂度降到了**线性**。简单说：文本长 10 倍，Mamba 只慢 10 倍，而 Transformer 会慢 100 倍。

但 Mamba 也有弱点——它的"记忆"是模糊的。让它回忆文章开头的一个细节，它经常答不上来。Transformer 的注意力机制恰好擅长这个。

于是有人想到：**把两者混着用。** 用 Mamba 处理大部分文本（快），在关键位置插几层 Attention（准）。这就是 Hybrid SSM-Attention 架构，Jamba、Zamba2 等模型都在用。

问题是：**这些实现全是 PyTorch + NVIDIA GPU 的。** 我手上只有一台 Mac。

所以我决定用 Apple 的 MLX 框架，从零实现一个完整的 hybrid 推理引擎。

> 📸 **【插图建议 1】** 一张架构对比图：左边是纯 Transformer（全部 Attention 层），右边是 Hybrid（Mamba 和 Attention 交替）。标注"O(n²)" vs "O(n)"。

---

## 怎么做的

### 第一步：搭积木

核心就两个"积木块"：

**Mamba 块**——一个递推公式：
```
h[t] = A * h[t-1] + B * x[t]    （状态更新）
y[t] = C · h[t]                  （输出）
```

看起来简单，但要在 GPU 上跑快，需要把序列切成小块，每块内用矩阵乘法并行计算。这个叫"分块并行扫描"。

**Attention 块**——标准的多头注意力，支持 GQA（分组查询注意力）和滑动窗口。

把这两种块按配置交替堆叠，就得到了一个 Hybrid 模型。比如 12 层模型，第 3、7、11 层放 Attention，其余放 Mamba，就是 3:1 的比例。

> 📸 **【插图建议 2】** 积木堆叠示意图：一列彩色方块，蓝色标 "Mamba"，橙色标 "Attention"，中间用箭头连接。底部写 "输入"，顶部写 "输出"。

### 第二步：让 AI 自己找最优配置

手动调参太慢了。我接入了 autoresearch 框架——让 AI 自己改代码、训练 5 分钟、看结果好不好、好就保留、不好就回滚。全自动，不需要人盯着。

**28 轮实验，跑出了三个关键发现：**

**发现 1：混合架构碾压纯架构**

| 架构 | 评分（越低越好） |
|------|:---:|
| **Hybrid（1 Mamba + 1 Attention）** | **1.676** |
| 纯 Mamba | 1.999 |
| 纯 Attention | 2.095 |

Mamba 和 Attention 真的是互补的。

**发现 2：顺序很重要**

Mamba 放前面、Attention 放后面：1.676。反过来？2.195——直接崩了。

我的理解是：Mamba 先做"粗加工"（压缩长程信息），Attention 再做"精加工"（精确定位）。就像先用粗砂纸打磨，再用细砂纸抛光。

**发现 3：小模型赢了**

在固定 5 分钟的训练预算下，2 层的小模型反而比 4 层的大模型效果好。原因很简单——小模型每步更快，5 分钟内能跑更多优化步。

> 📸 **【插图建议 3】** 28 轮实验的 val_bpb 折线图（从 2.165 下降到 1.676），标注关键转折点："浅层+宽模型"、"batch size 调优"、"GQA"。

### 第三步：Metal GPU 内核优化

MLX 在 Apple Silicon 上用 Metal GPU。但默认的算子组合有性能瓶颈。

**融合 Conv1d + SiLU 内核：8.3x 加速**

Mamba 里有一个 4-tap 的因果卷积，原来需要 Python 循环 4 次，每次都是一次 GPU 调度。我写了一个 Metal shader 把它融合成一次调度：3.6ms → 0.4ms。

**Parallel Scan 内核：一个反直觉的发现**

SSM 的递推理论上可以用"并行前缀和"算法加速，复杂度从 O(n²) 降到 O(n·log n)。我实现了一个融合的 Metal 内核。

结果：

| chunk 大小 | 矩阵乘法 | 并行扫描 | 谁赢了？ |
|-----------|---------|---------|---------|
| 64 | 1.2ms | 1.8ms | 矩阵乘法 |
| 128 | 2.2ms | 2.9ms | 矩阵乘法 |
| 256 | 4.2ms | 4.0ms | 打平 |
| 512 | 11.6ms | 5.2ms | **并行扫描** |

**Apple Silicon 的矩阵引擎太强了。** 对于小矩阵，专用硬件做 O(n²) 的乘法反而比通用计算做 O(n·log n) 更快。只有当矩阵大到一定程度，二次复杂度的代价才开始显现。

最终方案：自动切换。小块用矩阵乘法，大块用并行扫描。

> 📸 **【插图建议 4】** 柱状图对比：不同 chunk_size 下两种方案的耗时，交叉点标注 "256: breakeven"。

### 第四步：加载真实模型（踩坑记）

框架搭好后，我试着加载 Zamba2-1.2B——一个 12 亿参数的预训练模型。

**这是最痛苦的部分。** 表面上是权重格式转换，实际上踩了 4 个坑：

**坑 1：投影矩阵的分割顺序**

模型把多个投影合并成一个大矩阵。HuggingFace 的顺序是 `[gate, 卷积输入, 时间步]`，我实现成了 `[输入, gate, 参数]`。**结果：模型输出全是乱码。**

**坑 2：先乘门控还是先归一化？**

HuggingFace 先做 `y × gate`，再做 RMSNorm。我写反了。一个"先后顺序"的差别，排查了两小时。

**坑 3：D 残差用哪个 x？**

SSM 有一个 skip connection（D 参数）。它应该用时间步缩放**之前**的 x。我用了缩放**之后**的。这个 bug 导致第一层就有 34% 的误差，38 层累积后完全不可用。

**坑 4：共享权重 + LoRA 适配器**

Zamba2 的 6 个混合层共享同一套 Attention 权重，但每层有独立的 LoRA 适配器做差异化。我最初忽略了这些适配器，加载后 logit 差异降到了 0.23。

修完所有坑后，输入 "The capital of France is"，模型输出 **Paris**，排名第一。那一刻真的很开心。

> 📸 **【插图建议 5】** 修复前后对比截图：左边 "修复前 Top-5 预测" 全是乱码，右边 "修复后" Paris 排第一。

---

## 效果怎么样

### 性能

| 配置 | 速度 | 内存 |
|------|------|------|
| 原始模型 | 5.3 tok/s | 6.9 GB |
| + KV 缓存 | 24.6 tok/s | 6.9 GB |
| + bfloat16 | 24.6 tok/s | 3.5 GB |
| **+ 4-bit 量化** | **66.7 tok/s** | **1.3 GB** |

4-bit 量化后只要 **1.3 GB 内存**，8 GB 的 MacBook Air 都能跑。速度 66.7 tok/s，比人类阅读速度还快。

### 质量

和 HuggingFace 官方实现对比，Top-1 预测一致率 80%，平均 logit 差异仅 0.23。生成质量：

```
> The capital of France is
Paris. It is the largest city in France and the most populous city in
the European Union. It is located on the River Seine, in the
north-central part of the country.
```

事实准确，语句流畅。

### 一行命令可用

```bash
pip install alloy-mlx[serve]
alloy-chat --model Zyphra/Zamba2-1.2B-instruct --quantize 4
```

也可以启动一个 OpenAI 兼容的 API 服务：

```bash
alloy-serve --model Zyphra/Zamba2-1.2B-instruct --quantize 4 --port 8000
```

然后用任何支持 OpenAI API 的客户端连接。

> 📸 **【插图建议 6】** 终端截图：alloy-chat 的交互界面，用户输入 "Hello"，模型回复 "Hello! How can I assist you today?"。

---

## 长期规划

这个项目叫 **Alloy**（合金），名字来自它的核心理念——把 SSM 和 Attention 像合金一样熔合在一起。

### 已完成

- 完整的 Mamba-2 + Attention hybrid 架构
- Metal GPU 加速（conv1d 8.3x，parallel scan 2.2x）
- Zamba2-1.2B 加载 + 生成（含 LoRA 适配器合并）
- 4-bit / 8-bit 量化（1.3 GB 内存，66.7 tok/s）
- 对话 CLI + OpenAI 兼容 API 服务
- 28 轮自主架构搜索实验
- 88 个测试用例
- 已发布到 PyPI：`pip install alloy-mlx`

### 接下来

**短期：** 支持更多 hybrid 模型（Jamba、Falcon Mamba 等），让 Alloy 成为 Apple Silicon 上 hybrid SSM 模型的默认推理引擎。

**中期：** 从零训练一个 Alloy 原生的 hybrid 模型，发布到 HuggingFace。用实验数据回答一个核心问题：**在同等算力下，hybrid 架构到底比纯 Transformer 强多少？**

**长期：** Mamba 还在快速演进（Mamba-2 比 Mamba-1 快数倍），新的 SSM 变种不断涌现。Hybrid 架构可能是未来语言模型的主流形态。Alloy 要做的，就是让这些前沿架构在每个人的 Mac 上都能跑起来。

---

## 最后

这个项目从第一行代码到 PyPI 发布，用了一天时间。过程中大量使用 Claude Code 进行自主编码——28 轮自主实验、Metal 内核编写、Zamba2 权重对齐的 debug，全部由 AI 驱动。

我拥有的都是侥幸——每个 bug 恰好能找到原因，每次优化恰好能看到提升。我失去的都是人生——那些调试到凌晨的时间。

但这就是做技术的乐趣：**把不可能变成 `pip install`。**

项目地址：[github.com/gxcsoccer/alloy](https://github.com/gxcsoccer/alloy)

PyPI：`pip install alloy-mlx`

---

*首发于微信公众号「侥幸与人生」*
