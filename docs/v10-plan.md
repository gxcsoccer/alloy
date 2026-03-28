# V10 Plan: Usable Agent on Mac Mini 64GB

## Goal

在 Mac Mini 64GB 上跑起一个**实际可用**的 agent 模型（工具选择 >90%，参数准确 >80%，延迟 <5s）。

## 现状评估

### Nemotron-H-8B + LoRA 的天花板

| 指标 | 当前最好 | 可用标准 | 差距 |
|------|---------|---------|------|
| 工具选择 | 63.6% (V9d) | >90% | 远 |
| BFCL simple | 56% (V8) | >80% | 远 |
| 参数准确 | ~50% | >80% | 远 |
| Irrelevance | 32% (V9d) | >90% | 远 |

**结论：继续给 Nemotron-H 加训不是通往"可用 agent"的路。** 该模型预训练时没见过 tool calling，LoRA 容量不足以从零学会。

### 已建成的基础设施（可复用）

- Agent 执行框架：工具注册、多轮对话、流式输出
- 评估体系：BFCL 5 类 AST 评估 + self-built 工具测试
- 训练管线：数据生成、xlam 清洗压缩、QLoRA + 4 项内存优化（8x 加速）
- 模型层：bf16 支持、attention-only checkpoint、Mamba scan dtype 修复

---

## 新方向：接入已有 Tool Calling 能力的模型

### 2026 年 3 月开源模型调研

#### Qwen3.5 系列（2026 年 2 月发布）

| 模型 | 总参/激活参 | 架构 | 4-bit 内存 | Tool Calling | 64GB 可跑 |
|------|-----------|------|-----------|-------------|----------|
| Qwen3.5-0.8B | 0.8B | Dense | ~0.4GB | 原生 XML | YES |
| Qwen3.5-2B | 2B | Dense | ~1.1GB | 原生 XML | YES |
| Qwen3.5-4B | 4B | Dense | ~2.2GB | 原生 XML | YES |
| Qwen3.5-9B | 9B | Dense | ~5GB | 原生 XML | YES |
| Qwen3.5-27B | 27B | Dense | ~14GB | 原生 XML | YES |
| **Qwen3.5-35B-A3B** | **35B/3B** | **MoE** | **~20-24GB** | **原生 XML** | **YES** |
| Qwen3.5-122B-A10B | 122B/10B | MoE | ~67GB | 原生 XML | TIGHT |
| Qwen3.5-397B-A17B | 397B/17B | MoE | ~214GB | 原生 XML | NO |

特点：
- Gated Delta Networks（线性注意力）+ MoE 架构
- 原生多模态（视觉+语言）
- 原生 tool calling（XML 格式）
- 统一 instruct+base 模型
- MLX 已支持

#### Qwen3 Agent 专用

| 模型 | 总参/激活参 | 特点 |
|------|-----------|------|
| Qwen3-Coder-30B-A3B-Instruct | 30B/3B | 专为 agentic coding 设计，256K 上下文 |
| Qwen3-Next-80B-A3B-Instruct | 80B/3B | 长程推理 + 复杂工具使用 |

#### 其他竞品

| 模型 | 总参/激活参 | 架构 | Tool Calling |
|------|-----------|------|-------------|
| Llama 4 Scout | 109B/17B | MoE 16专家 | 原生 pythonic |
| Llama 4 Maverick | 400B/17B | MoE 128专家 | 原生 pythonic |
| Ministral-3B/8B/14B | Dense | Dense | 原生 |
| IBM Granite-20B-FC | 20B | Dense | 专用 FC 模型 |
| MiMo-V2-Flash | 309B/15B | MoE hybrid SWA/GA | 有（复杂 schema 会幻觉） |

#### BFCL 排行榜（2026, V4）

| 排名 | 模型 | 分数 |
|------|------|------|
| ~2 | Claude Opus 4.1 | 70.36% |
| ~3 | Claude Sonnet 4 | 70.29% |
| ~7 | GPT-5 | 59.22% |
| Top open-source | IBM Granite-20B-FC | — |

---

## 推荐方案

### 阶段一：立即可用（1-2 天）

**接入 Qwen3.5-35B-A3B（MoE，3B 激活）到 Alloy agent 框架**

- 用 `mlx-lm` 加载 4-bit 量化版（~20-24GB 内存）
- 改造 `alloy/agent.py` 支持 mlx-lm 后端
- 使用 Qwen3.5 原生 XML tool calling 格式
- 保留 Alloy 的工具注册、执行循环、流式输出

也可试 **Qwen3-Coder-30B-A3B-Instruct**（专为 agent 设计）。

预期效果：tool calling 开箱即用，质量远超我们训练的 LoRA。

### 阶段二：优化调优（1 周）

1. **Benchmark 对比**：Qwen3.5-9B vs 27B vs 35B-A3B vs Qwen3-Coder-30B-A3B
2. **LoRA 微调**（可选）：用我们的数据管线微调 Qwen3.5-9B，添加自定义工具
3. **Agent 框架增强**：多工具并行、错误重试、对话记忆

### 阶段三：Alloy 独特优势（2-4 周，研究方向）

1. **Nemotron-H 继续作为研究项目**：探索 hybrid SSM agent 的能力边界
2. **投机解码**：Nemotron-H（快速 Mamba）做 draft + Qwen 做 verifier
3. **长上下文 Agent**：SSM 的 O(L) 内存优势在长对话中不可替代

---

## 技术要点

### Qwen3.5 Tool Calling 格式

Qwen3.5 使用 XML 格式（非 JSON）：
```xml
<function=get_weather><parameter=city>Tokyo</parameter><parameter=unit>celsius</parameter></function>
```

需要适配 Alloy 的 `parse_tool_call()` 来支持这个格式。

### mlx-lm 集成

```bash
pip install mlx-lm
# 下载 + 量化
mlx_lm.convert --hf-path Qwen/Qwen3.5-35B-A3B -q --q-bits 4
# 或直接用社区量化版
mlx_lm.generate --model mlx-community/Qwen3.5-35B-A3B-4bit --prompt "..."
```

### MoE 在 MLX 上的性能

Qwen3.5-35B-A3B：35B 总参但只有 3B 激活 → 推理速度接近 3B dense 模型，但质量接近 35B dense。这是 64GB Mac Mini 的最佳选择。

---

## V8-V9 实验总结（归档）

### 训练结果

| Version | Data | Steps | BFCL simple | Self tool_sel | Irrelevance | Names | Train Time |
|---------|------|-------|------------|--------------|-------------|-------|------------|
| V8 | 10.5k | 2000 | **56%** | 45.5% | 8% | **98%** | 16.1h |
| V9 bf16 | 10.4k | 2000 | 32% | 18.2% | 66% | 74% | 2.2h |
| V9b fp32 | 10.4k | 2000 | 40% | 36.4% | 50% | 72% | 2.3h |
| V9c balanced | 10.4k | 2000 | 30% | 36.4% | **72%** | — | 2.3h |
| V9d Lane D | 10.7k | 1500 | 46% | **63.6%** | 32% | 90% | 1.1h |
| V9e Lane E | 10.7k | 1500 | 36% | 36.4% | 44% | 90% | 1.1h |
| V9f Lane F | 10.7k | 1200+300 | 30% | 27.3% | 72% | 72% | 1.5h |

### 关键发现

1. **Irrelevance vs tool-calling 是 zero-sum**：无法在 supervised training 中同时改善
2. **bf16 LoRA 降低 names accuracy**（98%→74%），不可用于生产
3. **内存优化 8x 加速**：length-sorted batching + 预清缓存 + attention checkpoint
4. **严重欠拟合**：模型只看了 14-20% 训练数据
5. **紧凑 prompt 省 57% tokens**：3x 更多数据进入训练
