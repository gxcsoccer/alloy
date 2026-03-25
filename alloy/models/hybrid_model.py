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
    # Zamba2-specific options
    combined_proj: bool = False  # Mamba combined in_proj (x+z+B+C+dt)
    n_groups: int = 1  # B, C group count
    use_D: bool = False  # Mamba D skip parameter
    use_inner_norm: bool = False  # Mamba inner RMS norm
    attn_d_model: Optional[int] = None  # Attention hidden size (if != d_model)
    zamba2_hybrid: bool = False  # Hybrid layers have both mamba+attention
    # Nemotron-H: flat layer types (each layer is exactly one of mamba/attention/mlp)
    layer_types: Optional[List[str]] = None  # e.g. ["mamba","mlp","mamba","mlp",...,"attention","mlp"]
    ffn_hidden_size: Optional[int] = None  # explicit FFN hidden size (if not d_model * ffn_mult)


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


class SquaredReLUMLP(nn.Module):
    """MLP with squared ReLU activation (Nemotron-H style, 2 matrices)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(mx.maximum(self.up_proj(x), 0) ** 2)


class FlatLayer(nn.Module):
    """Single flat layer: RMSNorm + one operation (Mamba, Attention, or MLP).

    Used by Nemotron-H where each of the 52 layers does exactly one thing.
    """

    def __init__(self, config: HybridConfig, layer_idx: int, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.norm = nn.RMSNorm(config.d_model)

        if layer_type == "mamba":
            self.mixer = MambaBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                headdim=config.headdim,
                chunk_size=config.chunk_size,
                combined_proj=config.combined_proj,
                n_groups=config.n_groups,
                use_D=config.use_D,
                use_inner_norm=config.use_inner_norm,
            )
        elif layer_type == "attention":
            self.mixer = AttentionBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
            )
        elif layer_type == "mlp":
            d_ff = config.ffn_hidden_size or int(config.d_model * config.ffn_mult)
            self.mixer = SquaredReLUMLP(config.d_model, d_ff)

    def __call__(self, x: mx.array, mask=None, cache=None, **kwargs):
        if self.layer_type == "attention":
            return x + self.mixer(self.norm(x), mask=mask, cache=cache)
        elif self.layer_type == "mamba":
            return x + self.mixer(self.norm(x), cache=cache)
        else:
            return x + self.mixer(self.norm(x))


class HybridBlock(nn.Module):
    """Single hybrid layer: RMSNorm + (Mamba or Attention) + RMSNorm + FFN.

    Supports two modes:
    - **Alloy mode** (default): Each layer is EITHER Mamba OR Attention + FFN.
    - **Zamba2 mode** (zamba2_hybrid=True on attn layers): Hybrid layers contain
      Mamba → linear → Attention + FFN in sequence.

    Args:
        config: Model configuration.
        layer_idx: Index of this layer (determines Mamba vs Attention).
    """

    def __init__(self, config: HybridConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attention = layer_idx in config.attn_layer_indices
        self.zamba2_hybrid = config.zamba2_hybrid and self.is_attention

        mamba_kwargs = dict(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            headdim=config.headdim,
            chunk_size=config.chunk_size,
            combined_proj=config.combined_proj,
            n_groups=config.n_groups,
            use_D=config.use_D,
            use_inner_norm=config.use_inner_norm,
        )

        if self.zamba2_hybrid:
            # Zamba2: hybrid layer has mamba_decoder + shared_transformer
            self.mamba_decoder = MambaBlock(**mamba_kwargs)
            self.mamba_norm = nn.RMSNorm(config.d_model)  # mamba_decoder input norm
            self.linear = nn.Linear(config.d_model, config.d_model, bias=False)

            attn_dim = config.attn_d_model or config.d_model
            self.attn_norm = nn.RMSNorm(attn_dim)
            self.mixer = AttentionBlock(
                d_model=attn_dim,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
            )
            self.norm2 = nn.RMSNorm(config.d_model)
            d_ff = int(config.d_model * config.ffn_mult)
            d_ff = ((d_ff + 255) // 256) * 256
            self.ffn = SwiGLU(config.d_model, d_ff)
        elif self.is_attention:
            self.norm1 = nn.RMSNorm(config.d_model)
            use_full = config.window_size is None or layer_idx in config.full_attn_layers
            self.mixer = AttentionBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                window_size=None if use_full else config.window_size,
            )
            self.norm2 = nn.RMSNorm(config.d_model)
            d_ff = int(config.d_model * config.ffn_mult)
            d_ff = ((d_ff + 255) // 256) * 256
            self.ffn = SwiGLU(config.d_model, d_ff)
        else:
            self.norm1 = nn.RMSNorm(config.d_model)
            self.mixer = MambaBlock(**mamba_kwargs)
            # Zamba2 mamba-only layers have no FFN
            if config.zamba2_hybrid:
                self.norm2 = None
                self.ffn = None
            else:
                self.norm2 = nn.RMSNorm(config.d_model)
                d_ff = int(config.d_model * config.ffn_mult)
                d_ff = ((d_ff + 255) // 256) * 256
                self.ffn = SwiGLU(config.d_model, d_ff)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
        original_hidden: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass with residual connections.

        Args:
            x: Input tensor [B, L, d_model].
            mask: Attention mask (only used if this is an attention layer).
            cache: Layer-specific cache (MambaCache or AttentionCache).
            original_hidden: Original embedding [B, L, d_model] (Zamba2 only).

        Returns:
            Output tensor [B, L, d_model].
        """
        if self.zamba2_hybrid:
            # Zamba2 hybrid: shared transformer FIRST, then mamba
            from alloy.models.cache import Zamba2HybridLayerCache
            attn_cache = cache.attn_cache if isinstance(cache, Zamba2HybridLayerCache) else None
            mamba_cache = cache.mamba_cache if isinstance(cache, Zamba2HybridLayerCache) else None

            # 1. cat(hidden, original_emb) → norm → attention → norm → FFN
            attn_input = mx.concatenate([x, original_hidden], axis=-1)
            h = self.attn_norm(attn_input)
            h = self.mixer(h, mask=mask, cache=attn_cache)
            h = self.norm2(h)
            h = self.ffn(h)
            # 2. Linear projection
            transformer_hidden = self.linear(h)
            # 3. Mamba decoder: (hidden + transformer_hidden) → norm → mamba
            mamba_input = x + transformer_hidden
            out = x + self.mamba_decoder(self.mamba_norm(mamba_input), cache=mamba_cache)
            return out
        elif self.is_attention:
            h = x + self.mixer(self.norm1(x), mask=mask, cache=cache)
        else:
            h = x + self.mixer(self.norm1(x), cache=cache)
        if self.ffn is not None:
            out = h + self.ffn(self.norm2(h))
            return out
        return h


class HybridLM(nn.Module):
    """Hybrid SSM-Attention language model.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        if config.layer_types is not None:
            # Nemotron-H flat mode: each layer is exactly one type
            self.layers = [
                FlatLayer(config, i, lt)
                for i, lt in enumerate(config.layer_types)
            ]
        else:
            # Alloy/Zamba2 block mode: each block has mixer + FFN
            self.layers = [HybridBlock(config, i) for i in range(config.n_layers)]

        self.norm = nn.RMSNorm(config.d_model)
        # LM head: separate for Nemotron-H, weight-tied for others
        self.lm_head = None  # weight-tied by default; set during load if separate

    def to_bfloat16(self):
        """Convert all weights to bfloat16 for memory reduction.

        The scan computation internally promotes to float32 for precision,
        so this is safe for both training and inference.
        """
        from mlx.utils import tree_flatten, tree_unflatten
        params = [(k, v.astype(mx.bfloat16) if v.dtype == mx.float32 else v)
                  for k, v in tree_flatten(self.parameters())]
        self.load_weights(params, strict=False)
        return self

    def quantize(self, bits: int = 4, group_size: int = 64):
        """Quantize model weights for reduced memory and faster inference.

        Args:
            bits: Bits per weight (4 or 8). Default: 4.
            group_size: Quantization group size. Default: 64.

        Returns:
            self (quantized in-place).
        """
        # Quantize Linear layers only. Skip Embedding (weight-tied LM head
        # needs raw weight for x @ embedding.weight.T) and Mamba scalars.
        def predicate(path, module):
            if isinstance(module, nn.Linear):
                return True
            return False

        nn.quantize(self, group_size=group_size, bits=bits, class_predicate=predicate)
        return self

    def make_cache(self) -> HybridCache:
        """Create an empty HybridCache matching this model's architecture."""
        return HybridCache(
            self.config.n_layers,
            self.config.attn_layer_indices,
            zamba2_hybrid=self.config.zamba2_hybrid,
        )

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

        # Zamba2: original embedding for attention concatenation
        # During prefill: full embedding [B, L, d]. During decode: just new token [B, 1, d].
        # The attention KV cache handles the history; we only need current position's embedding.
        original_emb = x if self.config.zamba2_hybrid else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, cache=layer_cache, original_hidden=original_emb)

        x = self.norm(x)

        # LM head: separate or weight-tied
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = x @ self.embedding.weight.T  # [B, L, vocab_size]
        return logits
