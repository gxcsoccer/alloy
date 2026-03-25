"""
Autoresearch training script for Alloy Hybrid SSM-Attention model.
Single-device, single-file. Apple Silicon MLX.

This is the MUTABLE file in the autoresearch loop. The agent edits this file
to explore hybrid architecture configurations, optimizer settings, and
hyperparameters.

Usage: uv run train.py
"""

import gc
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, make_dataloader

# ===========================================================================
# Model Architecture — Hybrid SSM-Attention (Alloy)
# ===========================================================================


@dataclass
class HybridConfig:
    vocab_size: int = 8192
    d_model: int = 768
    n_layers: int = 8
    attn_layer_indices: List[int] = field(default_factory=lambda: [3, 7])
    n_heads: int = 6
    n_kv_heads: int = 6
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    chunk_size: int = 256
    ffn_mult: float = 2.667
    window_size: Optional[int] = None
    full_attn_layers: List[int] = field(default_factory=list)


class FFN(nn.Module):
    """Squared ReLU FFN (simpler, no gate — 2 matrices instead of 3)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x):
        return self.w2(mx.maximum(self.w1(x), 0) ** 2)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, headdim=64, chunk_size=256):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size

        d_inner = d_model * expand
        self.d_inner = d_inner
        self.n_heads = d_inner // headdim

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv_weight = mx.random.normal((d_inner, d_conv)) * 0.02
        self.conv_bias = mx.zeros((d_inner,))
        self.x_proj = nn.Linear(d_inner, self.n_heads * (d_state + d_state + 1), bias=False)
        self.A_log = mx.zeros([self.n_heads])
        self.dt_bias = mx.zeros([self.n_heads])
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def _depthwise_conv1d(self, x_padded, L):
        out = mx.zeros_like(x_padded[:, :L, :])
        for k in range(self.d_conv):
            out = out + x_padded[:, k : k + L, :] * self.conv_weight[:, k]
        return out + self.conv_bias

    def _scan_chunk(self, x_c, a_c, B_c, C_c, h, cs):
        B_size = x_c.shape[0]
        n_heads = self.n_heads
        d_state = self.d_state

        b_bar = B_c[..., None] * x_c[:, :, :, None, :]
        log_a = mx.log(mx.clip(a_c, a_min=1e-10, a_max=None))
        log_a_cum = mx.cumsum(log_a, axis=1)
        lac = log_a_cum.transpose(0, 2, 1)
        # Apply causal mask BEFORE exp to avoid inf: set upper-tri to -inf
        M_log = lac[:, :, :, None] - lac[:, :, None, :]
        causal_mask = mx.where(
            mx.tril(mx.ones((cs, cs))) > 0, M_log, mx.array(float("-inf"))
        )
        M = mx.exp(causal_mask)

        b_flat = b_bar.transpose(0, 2, 1, 3, 4).reshape(
            B_size, n_heads, cs, d_state * self.headdim
        )
        h_input = (M @ b_flat).reshape(B_size, n_heads, cs, d_state, self.headdim)

        decay = mx.exp(lac)
        h_from_init = decay[:, :, :, None, None] * h[:, :, None, :, :]
        h_all = h_from_init + h_input

        C_t = C_c.transpose(0, 2, 1, 3)
        y_chunk = (C_t[:, :, :, :, None] * h_all).sum(axis=3)
        y_chunk = y_chunk.transpose(0, 2, 1, 3)
        h_new = h_all[:, :, -1, :, :]
        return y_chunk, h_new

    def _selective_scan(self, x, A_disc, B, C):
        B_size, L, n_heads, headdim = x.shape
        h = mx.zeros((B_size, n_heads, self.d_state, headdim))
        cs = self.chunk_size
        outputs = []

        for start in range(0, L, cs):
            end = min(start + cs, L)
            chunk_len = end - start
            y_chunk, h = self._scan_chunk(
                x[:, start:end], A_disc[:, start:end],
                B[:, start:end], C[:, start:end], h, chunk_len,
            )
            outputs.append(y_chunk)

        return mx.concatenate(outputs, axis=1)

    def __call__(self, x):
        B_size, L, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz[..., : self.d_inner], xz[..., self.d_inner :]

        x_padded = mx.pad(x_branch, [(0, 0), (self.d_conv - 1, 0), (0, 0)])
        x_conv = self._depthwise_conv1d(x_padded, L)
        x_conv = nn.silu(x_conv)

        ssm_params = self.x_proj(x_conv)
        ssm_params = ssm_params.reshape(B_size, L, self.n_heads, 2 * self.d_state + 1)
        B_param = ssm_params[..., : self.d_state]
        C_param = ssm_params[..., self.d_state : 2 * self.d_state]
        dt = ssm_params[..., -1]

        dt = nn.softplus(dt + self.dt_bias)
        A = -mx.exp(self.A_log)
        A_disc = mx.exp(dt * A)

        x_heads = x_conv.reshape(B_size, L, self.n_heads, self.headdim)
        y = self._selective_scan(x_heads, A_disc, B_param, C_param)

        y = y.reshape(B_size, L, self.d_inner)
        y = y * nn.silu(z)
        return self.out_proj(y)


class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=8, n_kv_heads=None, window_size=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.window_size = window_size
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = nn.RoPE(self.head_dim)

    def __call__(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        # GQA: expand KV heads
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            Bs, n_kv, Ls, hd = k.shape
            k = mx.broadcast_to(k[:, :, None, :, :], (Bs, n_kv, n_rep, Ls, hd)).reshape(Bs, self.n_heads, Ls, hd)
            v = mx.broadcast_to(v[:, :, None, :, :], (Bs, n_kv, n_rep, Ls, hd)).reshape(Bs, self.n_heads, Ls, hd)

        # Causal mask
        mask = mx.full((L, L), float("-inf"))
        mask = mx.triu(mask, k=1)
        if self.window_size is not None:
            row_idx = mx.arange(L)[:, None]
            col_idx = mx.arange(L)[None, :]
            too_far = (row_idx - col_idx) >= self.window_size
            mask = mx.where(too_far, float("-inf"), mask)

        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale + mask
        weights = mx.softmax(scores, axis=-1)
        out = weights @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.o_proj(out)


class HybridBlock(nn.Module):
    def __init__(self, config: HybridConfig, layer_idx: int):
        super().__init__()
        self.is_attention = layer_idx in config.attn_layer_indices
        self.norm1 = nn.RMSNorm(config.d_model)

        if self.is_attention:
            use_full = config.window_size is None or layer_idx in config.full_attn_layers
            self.mixer = AttentionBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                window_size=None if use_full else config.window_size,
            )
        else:
            self.mixer = MambaBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                headdim=config.headdim,
                chunk_size=config.chunk_size,
            )

        self.norm2 = nn.RMSNorm(config.d_model)
        d_ff = int(config.d_model * config.ffn_mult)
        d_ff = ((d_ff + 255) // 256) * 256
        self.ffn = FFN(config.d_model, d_ff)

    def __call__(self, x):
        h = x + self.mixer(self.norm1(x))
        out = h + self.ffn(self.norm2(h))
        return out


class HybridLM(nn.Module):
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = [HybridBlock(config, i) for i in range(config.n_layers)]
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def init_weights(self):
        """Initialize weights for stable training."""
        scale = 3**0.5 * self.config.d_model**-0.5
        self.embedding.weight = (mx.random.normal(self.embedding.weight.shape) * 1.0).astype(mx.float32)
        self.lm_head.weight = (mx.random.normal(self.lm_head.weight.shape) * 0.001).astype(mx.float32)
        for layer in self.layers:
            if layer.is_attention:
                m = layer.mixer
                for proj in [m.q_proj, m.k_proj, m.v_proj]:
                    proj.weight = mx.random.uniform(-scale, scale, proj.weight.shape).astype(mx.float32)
                m.o_proj.weight = mx.zeros_like(m.o_proj.weight).astype(mx.float32)
            else:
                m = layer.mixer
                m.in_proj.weight = mx.random.uniform(-scale, scale, m.in_proj.weight.shape).astype(mx.float32)
                m.x_proj.weight = mx.random.uniform(-scale, scale, m.x_proj.weight.shape).astype(mx.float32)
                m.out_proj.weight = mx.zeros_like(m.out_proj.weight).astype(mx.float32)
                m.conv_weight = (mx.random.normal(m.conv_weight.shape) * 0.02).astype(mx.float32)
            # FFN: gate projects init, output project zero
            layer.ffn.w1.weight = mx.random.uniform(-scale, scale, layer.ffn.w1.weight.shape).astype(mx.float32)
            layer.ffn.w2.weight = mx.zeros_like(layer.ffn.w2.weight).astype(mx.float32)

    def __call__(self, input_ids, targets=None, reduction="mean"):
        """Forward pass compatible with autoresearch evaluate_bpb.

        Args:
            input_ids: Token IDs [B, L].
            targets: Target token IDs [B, L] (optional).
            reduction: "mean" or "none" for loss.

        Returns:
            If targets is None: logits [B, L, vocab_size].
            If targets is given: scalar loss (mean) or per-token loss (none).
        """
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x).astype(mx.float32)
        logits = 30.0 * mx.tanh(logits / 30.0)  # Clamp for stability

        if targets is None:
            return logits

        # Compute cross-entropy loss
        ce = nn.losses.cross_entropy(logits, targets, reduction="none")
        if reduction == "none":
            return ce
        return mx.mean(ce)


# ===========================================================================
# Custom AdamW with per-parameter learning rate groups
# ===========================================================================


class AdamW:
    def __init__(self, model, mamba_lr, attn_lr, embedding_lr, ffn_lr, scalar_lr,
                 weight_decay, adam_betas):
        self.param_config = {}
        self.adam_state = {}

        flat_params = tree_flatten(model.parameters())
        for path, param in flat_params:
            if "lm_head" in path:
                cfg = {"lr": embedding_lr * 0.01, "betas": adam_betas, "eps": 1e-10, "wd": 0.0}
            elif "embedding" in path and param.ndim == 2:
                cfg = {"lr": embedding_lr, "betas": adam_betas, "eps": 1e-10, "wd": 0.0}
            elif "mixer" in path and "A_log" in path:
                cfg = {"lr": scalar_lr * 0.1, "betas": adam_betas, "eps": 1e-10, "wd": 0.0}
            elif "mixer" in path and ("dt_bias" in path or "conv" in path):
                cfg = {"lr": scalar_lr, "betas": adam_betas, "eps": 1e-10, "wd": 0.0}
            elif "mixer" in path and any(k in path for k in ["in_proj", "x_proj", "out_proj"]):
                cfg = {"lr": mamba_lr, "betas": adam_betas, "eps": 1e-10, "wd": weight_decay}
            elif "mixer" in path and any(k in path for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                cfg = {"lr": attn_lr, "betas": adam_betas, "eps": 1e-10, "wd": weight_decay}
            elif "ffn" in path and param.ndim == 2:
                cfg = {"lr": ffn_lr, "betas": adam_betas, "eps": 1e-10, "wd": weight_decay}
            elif "norm" in path:
                cfg = {"lr": scalar_lr, "betas": adam_betas, "eps": 1e-10, "wd": 0.0}
            else:
                cfg = {"lr": ffn_lr, "betas": adam_betas, "eps": 1e-10, "wd": weight_decay}

            self.param_config[path] = cfg

        self.initial_lrs = {path: cfg["lr"] for path, cfg in self.param_config.items()}

    def _set_path_value(self, model, path, value):
        parts = path.split(".")
        obj = model
        for part in parts[:-1]:
            if isinstance(obj, list):
                obj = obj[int(part)]
            elif isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        last = parts[-1]
        if isinstance(obj, dict):
            obj[last] = value
        else:
            setattr(obj, last, value)

    def _step(self, path, grad, param, config):
        grad_f32 = grad.astype(mx.float32)
        param_f32 = param.astype(mx.float32)
        lr = config["lr"]
        beta1, beta2 = config["betas"]
        eps = config["eps"]
        wd = config["wd"]

        if path not in self.adam_state:
            self.adam_state[path] = {"m": mx.zeros_like(grad_f32), "v": mx.zeros_like(grad_f32), "t": 0}

        state = self.adam_state[path]
        state["t"] += 1
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad_f32
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad_f32 * grad_f32)

        bias1 = 1 - beta1 ** state["t"]
        bias2 = 1 - beta2 ** state["t"]
        denom = mx.sqrt(state["v"] / bias2) + eps
        step_size = lr / bias1

        param_f32 = param_f32 * (1 - lr * wd)
        param_f32 = param_f32 - step_size * (state["m"] / denom)
        return param_f32.astype(param.dtype)

    def update(self, model, grads):
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))
        for path, grad in flat_grads.items():
            if path not in self.param_config:
                continue
            config = self.param_config[path]
            param = flat_params[path]
            new_param = self._step(path, grad, param, config)
            self._set_path_value(model, path, new_param)

    def set_lr_multiplier(self, multiplier):
        for path, config in self.param_config.items():
            config["lr"] = self.initial_lrs[path] * multiplier

    @property
    def state(self):
        arrays = []
        for state in self.adam_state.values():
            arrays.extend([state["m"], state["v"]])
        return arrays


# ===========================================================================
# Hyperparameters (edit these to experiment)
# ===========================================================================

# Architecture — shallow+wide: fewer layers = faster steps = more updates
DEPTH = 2
D_MODEL = 512
ATTN_LAYER_INDICES = [1]  # 1 Mamba + 1 Attention
N_HEADS = 8
N_KV_HEADS = 8
D_STATE = 16
D_CONV = 4
EXPAND = 2
HEADDIM = 64
CHUNK_SIZE = 64
FFN_MULT = 4.0
WINDOW_SIZE = None
FULL_ATTN_LAYERS = []

# Optimizer — conservative LRs for hybrid model stability
TOTAL_BATCH_SIZE = 2**13
DEVICE_BATCH_SIZE = 4
MAMBA_LR = 0.004
ATTN_LR = 0.004
EMBEDDING_LR = 0.06
FFN_LR = 0.004
SCALAR_LR = 0.05
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

# Eval
FINAL_EVAL_BATCH_SIZE = 256
STARTUP_EXCLUDE_STEPS = 1


# ===========================================================================
# Training loop
# ===========================================================================


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def get_peak_memory_mb():
    return mx.get_peak_memory() / 1024 / 1024


t_start = time.time()
mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)
t_data = time.time()
print(f"Data/tokenizer loaded in {t_data - t_start:.1f}s")

# Build hybrid model
config = HybridConfig(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    n_layers=DEPTH,
    attn_layer_indices=ATTN_LAYER_INDICES,
    n_heads=N_HEADS,
    n_kv_heads=N_KV_HEADS,
    d_state=D_STATE,
    d_conv=D_CONV,
    expand=EXPAND,
    headdim=HEADDIM,
    chunk_size=CHUNK_SIZE,
    ffn_mult=FFN_MULT,
    window_size=WINDOW_SIZE,
    full_attn_layers=FULL_ATTN_LAYERS,
)

model = HybridLM(config)
model.init_weights()
mx.eval(model.parameters())
num_params = sum(param.size for _, param in tree_flatten(model.parameters()))

n_mamba = DEPTH - len(ATTN_LAYER_INDICES)
n_attn = len(ATTN_LAYER_INDICES)
print(f"Hybrid model: {DEPTH} layers ({n_mamba} Mamba + {n_attn} Attention), {num_params / 1e6:.1f}M params")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = AdamW(
    model,
    mamba_lr=MAMBA_LR,
    attn_lr=ATTN_LR,
    embedding_lr=EMBEDDING_LR,
    ffn_lr=FFN_LR,
    scalar_lr=SCALAR_LR,
    weight_decay=WEIGHT_DECAY,
    adam_betas=ADAM_BETAS,
)

loss_grad_fn = nn.value_and_grad(model, lambda model, inputs, targets: model(inputs, targets=targets))

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

smooth_train_loss = 0.0
total_training_time = 0.0
step = 0
t_compiled = None

while True:
    t0 = time.time()
    accum_grads = None
    train_loss = None

    for ga_i in range(grad_accum_steps):
        loss, grads = loss_grad_fn(model, x, y)
        mx.eval(loss, grads)
        if t_compiled is None:
            t_compiled = time.time()
            print(f"Model compiled in {t_compiled - t_data:.1f}s")
        loss_f = float(loss.item())
        if math.isnan(loss_f):
            print(f"FAIL: NaN in grad_accum iter {ga_i} at step {step}")
            raise SystemExit(1)
        train_loss = loss
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map(lambda lhs, rhs: lhs + rhs, accum_grads, grads)
        x, y, epoch = next(train_loader)

    if grad_accum_steps > 1:
        accum_grads = tree_map(lambda grad: grad * (1.0 / grad_accum_steps), accum_grads)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    optimizer.set_lr_multiplier(lrm)
    optimizer.update(model, accum_grads)
    mx.eval(model.parameters(), *optimizer.state)

    train_loss_f = float(train_loss.item())
    if train_loss_f > 100 or math.isnan(train_loss_f):
        print(f"FAIL (loss={train_loss_f}) at step {step} after optimizer update")
        raise SystemExit(1)

    dt = time.time() - t0
    if step >= STARTUP_EXCLUDE_STEPS:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
    remaining = max(0.0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
        f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
        f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    if step >= STARTUP_EXCLUDE_STEPS and total_training_time >= TIME_BUDGET:
        break

print()
t_train = time.time()
print(f"Training completed in {t_train - t_compiled:.1f}s")

total_tokens = step * TOTAL_BATCH_SIZE
print("Starting final eval...")
print(f"Final eval batch size: {FINAL_EVAL_BATCH_SIZE}")
val_bpb = evaluate_bpb(model, tokenizer, FINAL_EVAL_BATCH_SIZE)
t_eval = time.time()
print(f"Final eval completed in {t_eval - t_train:.1f}s")

peak_vram_mb = get_peak_memory_mb()

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_eval - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
print(f"architecture:     hybrid (mamba:{n_mamba} attn:{n_attn})")
