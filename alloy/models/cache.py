"""Cache management for hybrid SSM-Attention models.

Mamba and Attention layers require fundamentally different cache formats.
HybridCache provides a unified interface for managing both during generation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

import mlx.core as mx


@dataclass
class MambaCache:
    """Cache for a single Mamba layer during autoregressive generation.

    Stores the SSM hidden state and the convolution buffer.

    Attributes:
        ssm_state: SSM recurrent state of shape [B, n_heads, d_state, headdim].
        conv_state: Convolution input buffer of shape [B, d_inner, d_conv].
    """

    ssm_state: Optional[mx.array] = None
    conv_state: Optional[mx.array] = None


@dataclass
class AttentionCache:
    """Cache for a single Attention layer during autoregressive generation.

    Stores key and value tensors for all past positions.

    Attributes:
        keys: Cached keys of shape [B, n_kv_heads, seq_len, head_dim].
        values: Cached values of shape [B, n_kv_heads, seq_len, head_dim].
    """

    keys: Optional[mx.array] = None
    values: Optional[mx.array] = None

    @property
    def seq_len(self) -> int:
        """Return the current cached sequence length."""
        if self.keys is None:
            return 0
        return self.keys.shape[2]

    def update(self, keys: mx.array, values: mx.array) -> "AttentionCache":
        """Append new keys and values to the cache.

        Args:
            keys: New keys of shape [B, n_kv_heads, new_len, head_dim].
            values: New values of shape [B, n_kv_heads, new_len, head_dim].

        Returns:
            Updated cache (self).
        """
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self


@dataclass
class Zamba2HybridLayerCache:
    """Cache for a Zamba2 hybrid layer that has both mamba and attention."""

    mamba_cache: MambaCache = None
    attn_cache: AttentionCache = None

    def __post_init__(self):
        if self.mamba_cache is None:
            self.mamba_cache = MambaCache()
        if self.attn_cache is None:
            self.attn_cache = AttentionCache()


class HybridCache:
    """Unified cache manager for a hybrid model with N layers.

    Each layer gets either a MambaCache or AttentionCache depending on
    its type. For Zamba2 hybrid layers, a composite cache holds both.

    Args:
        n_layers: Total number of layers.
        attn_layer_indices: Which layers are attention (rest are Mamba).
        zamba2_hybrid: If True, attention layers get Zamba2HybridLayerCache.
    """

    def __init__(self, n_layers: int, attn_layer_indices: List[int],
                 zamba2_hybrid: bool = False):
        self.n_layers = n_layers
        self.attn_layer_indices = set(attn_layer_indices)
        self.zamba2_hybrid = zamba2_hybrid
        self.caches: List[Union[MambaCache, AttentionCache, Zamba2HybridLayerCache]] = []
        for i in range(n_layers):
            if i in self.attn_layer_indices:
                if zamba2_hybrid:
                    self.caches.append(Zamba2HybridLayerCache())
                else:
                    self.caches.append(AttentionCache())
            else:
                self.caches.append(MambaCache())

    def __getitem__(self, layer_idx: int) -> Union[MambaCache, AttentionCache, Zamba2HybridLayerCache]:
        return self.caches[layer_idx]

    def reset(self) -> None:
        """Clear all cached states."""
        for i in range(self.n_layers):
            if i in self.attn_layer_indices:
                if self.zamba2_hybrid:
                    self.caches[i] = Zamba2HybridLayerCache()
                else:
                    self.caches[i] = AttentionCache()
            else:
                self.caches[i] = MambaCache()
