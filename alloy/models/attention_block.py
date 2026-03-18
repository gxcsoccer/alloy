"""Attention block supporting MHA, GQA, and sliding-window variants.

Follows the pre-norm residual pattern used in the HybridBlock wrapper.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class AttentionBlock(nn.Module):
    """Multi-head / grouped-query attention with optional sliding window.

    Args:
        d_model: Model dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads (< n_heads for GQA, = n_heads for MHA).
        window_size: If set, use sliding-window attention with this window size.
            None means full (causal) attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.window_size = window_size
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = nn.RoPE(self.head_dim)

    def _make_causal_mask(self, q_len: int, kv_len: int) -> mx.array:
        """Create a causal (+ optional sliding-window) attention mask.

        Args:
            q_len: Query sequence length.
            kv_len: Key/value sequence length (>= q_len when using cache).

        Returns:
            Mask of shape [1, 1, q_len, kv_len] with 0 for attend, -inf for block.
        """
        # Positions of queries and keys
        offset = kv_len - q_len
        q_idx = mx.arange(q_len)[:, None] + offset  # [q_len, 1]
        k_idx = mx.arange(kv_len)[None, :]  # [1, kv_len]

        # Causal: query can only attend to keys at positions <= its own
        mask = mx.where(k_idx <= q_idx, 0.0, -math.inf)

        # Sliding window: additionally block keys too far in the past
        if self.window_size is not None:
            mask = mx.where(q_idx - k_idx < self.window_size, mask, -math.inf)

        return mask[None, None, :, :]  # [1, 1, q_len, kv_len]

    def _repeat_kv(self, x: mx.array) -> mx.array:
        """Repeat KV heads to match the number of query heads (for GQA).

        Args:
            x: Tensor of shape [B, n_kv_heads, L, head_dim].

        Returns:
            Tensor of shape [B, n_heads, L, head_dim].
        """
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep == 1:
            return x
        # [B, n_kv_heads, L, head_dim] -> [B, n_kv_heads, n_rep, L, head_dim]
        B, n_kv, L, hd = x.shape
        x = mx.broadcast_to(x[:, :, None, :, :], (B, n_kv, n_rep, L, hd))
        return x.reshape(B, self.n_heads, L, hd)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, L, d_model].
            mask: Optional attention mask of shape [B, 1, L, S] or broadcastable.
                If None, a causal mask is generated automatically.
            cache: Optional AttentionCache for autoregressive generation.

        Returns:
            Output tensor of shape [B, L, d_model].
        """
        B, L, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        # q: [B, n_heads, L, head_dim], k/v: [B, n_kv_heads, L, head_dim]

        # Apply RoPE (with offset for cached positions)
        offset = 0 if cache is None else cache.seq_len
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # Update KV cache
        if cache is not None:
            cache.update(k, v)
            k, v = cache.keys, cache.values

        # GQA: expand KV heads to match query heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        kv_len = k.shape[2]

        # Build causal mask if not provided
        if mask is None:
            mask = self._make_causal_mask(L, kv_len)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale  # [B, n_heads, L, kv_len]
        scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        out = weights @ v  # [B, n_heads, L, head_dim]

        # Reshape and project output
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.o_proj(out)
