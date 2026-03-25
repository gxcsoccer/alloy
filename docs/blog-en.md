# Can't Afford a GPU? Run LLMs on Your Mac — Building a Hybrid SSM Inference Engine from Scratch

> 1.2 billion parameters, 1.3 GB memory, 66.7 tok/s. From zero to `pip install` in one day — through 28 autonomous experiments, 4 sneaky bugs, and a counterintuitive discovery about hardware.

---

## Why Build This

If you've followed the AI space, you've heard of **Transformers**. GPT, Claude, Llama — they all use it.

At the core of the Transformer is the "attention mechanism" — every token in a text looks at every other token to figure out what's related to what.

It's powerful, but has a fatal flaw: **the longer the text, the slower it gets.** Every token attending to every other token means computation grows quadratically. 1,000 tokens looking at 1,000 tokens = 1 million operations. 10,000 tokens? 100 million.

Last year, a new architecture called **Mamba** took off. It uses a completely different approach — instead of every token looking at every other token, it maintains a "compressed memory" that each token updates and passes forward. **Computation becomes linear** — 10x longer text means only 10x slower, not 100x.

But Mamba has a weakness: its memory is "fuzzy." Ask it to recall a specific detail from the beginning of a long document, and it often can't. Transformer's attention mechanism excels at exactly this kind of precise recall.

So someone had an idea: **use both.** Most layers use Mamba (fast), with a few attention layers at key positions (precise). This is the **Hybrid architecture**.

![Architecture comparison](images/01-architecture.png)

*Three architectures from top to bottom: Transformer — every token attends to every other token, like a dense web of connections, slower as it grows. Mamba — information flows forward one step at a time, fast but lossy. Hybrid — the pipeline handles fast compression, the connections handle precise recall. Best of both worlds.*

The problem: all existing implementations are PyTorch + NVIDIA GPU. **Mac users are left out.**

So I decided to build one from scratch using Apple's MLX framework.

---

## How I Built It

### Step 1: Building Blocks

The entire model is just two types of "blocks" stacked alternately:

- **Mamba block** — fast processing, maintains "compressed memory" passed forward
- **Attention block** — precise recall, "looks back" at key positions

The configuration is flexible. For example, a 12-layer model with attention only at layers 3, 7, and 11 gives a 3:1 Mamba-to-Attention ratio. More Mamba = faster. More Attention = more precise. How to find the optimal ratio? That leads to the next step.

### Step 2: Let AI Find the Optimal Configuration

Manual tuning is too slow. Andrej Karpathy (former Tesla AI Director, OpenAI founding member) recently open-sourced a project called **autoresearch** — letting AI do its own research. It went viral across the AI community. I ported it to MLX and used it to automatically search for the optimal hybrid architecture. The idea is simple:

1. AI automatically modifies the model config (layers, size, learning rate, etc.)
2. Trains for 5 minutes
3. Checks if it improved
4. If yes, keep. If no, rollback.
5. Repeat. No human supervision needed.

**28 experiments, fully autonomous.**

![Experiment curve](images/02-experiments.png)

*Blue dots are kept improvements, gray are discarded attempts. The model score (val_bpb, lower is better) dropped from 2.165 to 1.676 — a 22.6% improvement.*

Three key findings:

**Finding 1: Hybrid crushes pure architectures**

| Architecture | Score (lower = better) |
|-------------|:---:|
| **Hybrid (1 Mamba + 1 Attention)** | **1.676** |
| Pure Mamba | 1.999 |
| Pure Attention | 2.095 |

The two building blocks are truly complementary — using either one alone is significantly worse.

**Finding 2: Order matters**

Mamba first, Attention last: 1.676. Reversed? 2.195 — it completely fell apart.

My interpretation: Mamba does "rough processing" first (compressing long-range information), then Attention does "fine processing" (precise selection). **Like sanding with coarse grit first, then fine grit. Reverse the order and you ruin everything.**

**Finding 3: Under a fixed time budget, smaller models win**

A 2-layer model outperformed a 4-layer one. Simple reason — smaller models run each step faster, fitting more optimization steps in 5 minutes. **More parameters isn't always better — what matters is how much you can learn per unit of time.**

### Step 3: GPU Acceleration — A Counterintuitive Discovery

Mac GPUs use Metal (Apple's graphics API, similar to NVIDIA's CUDA). The default compute patterns had performance bottlenecks, so I wrote custom GPU programs (Metal shaders) to speed things up.

One finding really surprised me.

Mamba's core computation can be implemented two ways:
- **Matrix multiplication**: classic approach, O(n²) compute, but can use the GPU's dedicated matrix hardware
- **Parallel scan**: smarter algorithm, only O(n·log n) compute

Intuitively, O(n·log n) should crush O(n²), right?

![Chunk size comparison](images/03-chunk-tradeoff.png)

*Orange: matrix multiplication (O(n²)). Blue: parallel scan (O(n·log n)). The red dashed line is the crossover — to the left, matmul wins thanks to hardware acceleration; to the right, quadratic complexity can't keep up and parallel scan takes over.*

**Not quite.** At small scales (chunk ≤ 128), matrix multiplication was actually faster. The reason: Apple Silicon has dedicated matrix multiplication hardware (AMX engine) that's blazingly fast. Parallel scan does less computation but uses general-purpose compute units with much lower throughput.

**Only when data gets large enough (chunk ≥ 256) does the quadratic cost finally exceed the hardware advantage.**

Final solution: automatic switching. Small chunks use matmul, large chunks use parallel scan. Best of both worlds.

### Step 4: Loading a Real Model — 4 Painful Bugs

With the framework ready, I tried loading a real pretrained model: **Zamba2-1.2B** (a 1.2 billion parameter hybrid model from Zyphra).

**This was the most painful part.** The model loaded, but the output was complete gibberish.

![Before and after fix](images/05-before-after.png)

*Asking "The capital of France is" — before the fix (top), the model outputs nonsense like "amo" and "imore". After the fix (bottom), "Paris" ranks #1, with the score jumping from 13 to 20.*

I tracked down 4 bugs:

**Bug 1: Sliced the cake in the wrong order.** The model packs multiple matrices into one large matrix. I split it in the wrong order — like opening someone's suitcase and putting the clothes in the laptop compartment.

**Bug 2: Paint first or sand first?** The model has a "gating" operation and a "normalization" operation. The official implementation does gating first, then normalization. I had it backwards. A tiny ordering difference, a massive impact.

**Bug 3: Use the raw material or the processed version?** The model has a skip connection that should use the data **before** processing. I used it **after**. This bug caused 34% error at the very first layer, compounding across 38 layers into total garbage. **Took two hours to find.**

**Bug 4: Shared template + personalized stickers.** Zamba2's 6 hybrid layers share one set of attention weights (the template), but each layer has its own fine-tuning parameters (stickers). I initially ignored these "stickers" — adding them improved accuracy by 64%.

---

## Results

### Speed and Memory

![Performance comparison](images/04-performance.png)

*Longer bars = faster. From top to bottom: the original model at 5.3 tok/s, adding cache brings it to 24.6, and 4-bit quantization reaches 66.7 tok/s while cutting memory from 6.9 GB to just 1.3 GB.*

| Configuration | Speed | Memory |
|------|------|------|
| Original | 5.3 tok/s | 6.9 GB |
| + KV Cache | 24.6 tok/s | 6.9 GB |
| **+ 4-bit Quantization** | **66.7 tok/s** | **1.3 GB** |

The last row is the highlight: **1.3 GB memory — even an 8 GB MacBook Air can run it.** At 66.7 tokens per second, that's faster than most people read.

### Generation Quality

```
> The capital of France is
Paris. It is the largest city in France and the most populous city in
the European Union. It is located on the River Seine, in the
north-central part of the country.
```

Factually accurate, fluent output.

### One Command to Use

Published to PyPI — anyone in the world can install it:

```bash
pip install alloy-mlx[serve]

# Chat (auto-downloads the model, ~4.5 GB)
alloy-chat --model Zyphra/Zamba2-1.2B-instruct --quantize 4

# Or start an OpenAI-compatible API server
alloy-serve --model Zyphra/Zamba2-1.2B-instruct --quantize 4 --port 8000
```

Once running, connect with any OpenAI API-compatible client — ChatGPT wrappers, LangChain, or even curl.

---

## What's Next

The project is called **Alloy** — named after its core idea of fusing different technologies together, like metals in an alloy, to get the best of each.

**Done:**
- Complete hybrid architecture (Mamba + Attention)
- Metal GPU acceleration
- Real pretrained model loading (Zamba2-1.2B)
- 4-bit quantization (1.3 GB memory)
- Chat CLI + API server
- 28 autonomous experiments
- Published to PyPI

**Next:**
- Support more hybrid models (making Alloy the default hybrid inference engine on Mac)
- Train an original model from scratch (answering with data: how much better is hybrid than pure Transformer?)
- Explore newer SSM variants (Mamba is evolving fast)

---

## Final Thoughts

This project went from the first line of code to `pip install` in one day. The entire process was heavily driven by Claude Code — 28 autonomous experiments, Metal kernel writing, weight alignment debugging, all AI-powered.

What I have is all luck — every bug happened to have a findable cause, every optimization happened to show improvement.

What I've lost is all life — those hours debugging into the night.

But that's the joy of building: **turning the impossible into `pip install`.**

---

**GitHub:** [github.com/gxcsoccer/alloy](https://github.com/gxcsoccer/alloy)

**Install:** `pip install alloy-mlx`
