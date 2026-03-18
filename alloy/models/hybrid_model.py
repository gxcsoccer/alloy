"""Hybrid SSM-Attention language model.

Composes MambaBlock and AttentionBlock in an interleaved pattern
(Jamba-style sequential by default) with RMSNorm and SwiGLU FFN.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from alloy.models.attention_block import AttentionBlock
from alloy.models.cache import HybridCache
from alloy.models.mamba_block import MambaBlock


@dataclass
class HybridConfig:
    """Configuration for HybridLM.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model dimension.
        n_layers: Total number of layers.
        attn_layer_indices: Which layer indices use Attention (rest use Mamba).
            For example, [3, 7, 11] for a 12-layer model with 3:1 ratio.
        n_heads: Number of attention query heads.
        n_kv_heads: Number of KV heads (for GQA).
        d_state: Mamba SSM state dimension.
        d_conv: Mamba convolution width.
        expand: Mamba expansion factor.
        headdim: Mamba head dimension.
        chunk_size: Mamba chunk size for parallel scan.
        ffn_mult: FFN hidden dimension multiplier relative to d_model.
        window_size: Sliding-window size for attention layers (None = full).
        full_attn_layers: Layer indices that use full attention even when
            window_size is set (e.g., first and last attention layers).
        vocab_size_pad: If set, pad vocab_size to this multiple for efficiency.
    """

    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 12
    attn_layer_indices: List[int] = field(default_factory=lambda: [3, 7, 11])
    n_heads: int = 8
    n_kv_heads: int = 8
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    chunk_size: int = 256
    ffn_mult: float = 2.667
    window_size: Optional[int] = None
    full_attn_layers: List[int] = field(default_factory=list)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Args:
        d_model: Input/output dimension.
        d_ff: Hidden dimension.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class HybridBlock(nn.Module):
    """Single hybrid layer: RMSNorm + (Mamba or Attention) + RMSNorm + FFN.

    Args:
        config: Model configuration.
        layer_idx: Index of this layer (determines Mamba vs Attention).
    """

    def __init__(self, config: HybridConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
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
        # Round d_ff to nearest multiple of 256 for efficiency
        d_ff = ((d_ff + 255) // 256) * 256
        self.ffn = SwiGLU(config.d_model, d_ff)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        """Forward pass with residual connections.

        Args:
            x: Input tensor [B, L, d_model].
            mask: Attention mask (only used if this is an attention layer).
            cache: Layer-specific cache (MambaCache or AttentionCache).

        Returns:
            Output tensor [B, L, d_model].
        """
        if self.is_attention:
            h = x + self.mixer(self.norm1(x), mask=mask, cache=cache)
        else:
            h = x + self.mixer(self.norm1(x), cache=cache)
        out = h + self.ffn(self.norm2(h))
        return out


class HybridLM(nn.Module):
    """Hybrid SSM-Attention language model.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = [HybridBlock(config, i) for i in range(config.n_layers)]
        self.norm = nn.RMSNorm(config.d_model)
        # LM head is weight-tied with embedding (handled in __call__)

    def make_cache(self) -> HybridCache:
        """Create an empty HybridCache matching this model's architecture."""
        return HybridCache(self.config.n_layers, self.config.attn_layer_indices)

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[HybridCache] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape [B, L].
            cache: Optional HybridCache for autoregressive generation.

        Returns:
            Logits tensor of shape [B, L, vocab_size].
        """
        x = self.embedding(input_ids)  # [B, L, d_model]

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, cache=layer_cache)

        x = self.norm(x)

        # Weight-tied LM head: logits = x @ embedding.weight^T
        logits = x @ self.embedding.weight.T  # [B, L, vocab_size]
        return logits
