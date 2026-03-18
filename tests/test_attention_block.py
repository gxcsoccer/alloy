"""Tests for the Attention block."""

import mlx.core as mx
import pytest

from alloy.models.attention_block import AttentionBlock
from alloy.models.cache import AttentionCache


class TestAttentionBlock:
    """Tests for AttentionBlock."""

    def setup_method(self):
        self.block = AttentionBlock(d_model=64, n_heads=4, n_kv_heads=2)

    def test_output_shape(self):
        """Forward pass produces correct output shape [B, L, d_model]."""
        x = mx.random.normal((2, 32, 64))
        y = self.block(x)
        assert y.shape == (2, 32, 64)

    def test_single_token(self):
        """Single-token forward works."""
        x = mx.random.normal((1, 1, 64))
        y = self.block(x)
        assert y.shape == (1, 1, 64)

    def test_gqa_kv_heads(self):
        """GQA block correctly sets up fewer KV heads than query heads."""
        assert self.block.n_kv_heads == 2
        assert self.block.n_heads == 4

    def test_mha_default_kv_heads(self):
        """When n_kv_heads is None, defaults to n_heads (MHA)."""
        block = AttentionBlock(d_model=64, n_heads=4)
        assert block.n_kv_heads == 4

    def test_sliding_window(self):
        """Sliding-window attention block has window_size set."""
        sw_block = AttentionBlock(d_model=64, n_heads=4, window_size=8)
        assert sw_block.window_size == 8
        x = mx.random.normal((1, 16, 64))
        y = sw_block(x)
        assert y.shape == (1, 16, 64)

    def test_full_attention_default(self):
        """Default block uses full attention (no window)."""
        assert self.block.window_size is None

    def test_causal_mask(self):
        """Causal mask blocks future positions."""
        mask = self.block._make_causal_mask(4, 4)
        mx.eval(mask)
        # Lower-triangular: position 0 can only see position 0
        assert mask[0, 0, 0, 1].item() == float("-inf")
        # Position 3 can see all positions 0-3
        assert mask[0, 0, 3, 0].item() == 0.0
        assert mask[0, 0, 3, 3].item() == 0.0

    def test_cache_autoregressive(self):
        """Step-by-step generation with cache produces valid outputs."""
        mx.random.seed(42)
        block = AttentionBlock(d_model=64, n_heads=4, n_kv_heads=2)
        cache = AttentionCache()

        outputs = []
        for t in range(8):
            x_t = mx.random.normal((1, 1, 64))
            y_t = block(x_t, cache=cache)
            mx.eval(y_t, cache.keys, cache.values)
            outputs.append(y_t)
            assert cache.seq_len == t + 1

        y = mx.concatenate(outputs, axis=1)
        assert y.shape == (1, 8, 64)
