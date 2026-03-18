"""Tests for the Mamba-2 block."""

import mlx.core as mx
import pytest

from alloy.models.mamba_block import MambaBlock
from alloy.models.cache import MambaCache


class TestMambaBlock:
    """Tests for MambaBlock."""

    def setup_method(self):
        self.block = MambaBlock(d_model=64, d_state=16, d_conv=4, expand=2, headdim=32)

    def test_output_shape(self):
        """Forward pass produces correct output shape [B, L, d_model]."""
        x = mx.random.normal((2, 32, 64))
        y = self.block(x)
        assert y.shape == (2, 32, 64)

    def test_single_step(self):
        """Single-step forward (L=1) works for autoregressive generation."""
        x = mx.random.normal((1, 1, 64))
        y = self.block(x)
        assert y.shape == (1, 1, 64)

    def test_cache_shapes(self):
        """Cache states have the expected shapes after a forward pass."""
        cache = MambaCache()
        x = mx.random.normal((1, 8, 64))
        self.block(x, cache=cache)
        mx.eval(cache.ssm_state, cache.conv_state)
        # SSM state: [B, n_heads, d_state, headdim]
        assert cache.ssm_state.shape == (1, 4, 16, 32)
        # Conv state: [B, d_conv-1, d_inner]
        assert cache.conv_state.shape == (1, 3, 128)

    def test_autoregressive_matches_parallel(self):
        """Step-by-step generation with cache matches single-pass output."""
        mx.random.seed(42)
        block = MambaBlock(d_model=64, d_state=16, d_conv=4, expand=2, headdim=32)

        x = mx.random.normal((1, 8, 64))

        # Full parallel pass
        y_parallel = block(x)
        mx.eval(y_parallel)

        # Step-by-step with cache
        cache = MambaCache()
        outputs = []
        for t in range(8):
            y_t = block(x[:, t : t + 1, :], cache=cache)
            mx.eval(y_t, cache.ssm_state, cache.conv_state)
            outputs.append(y_t)
        y_sequential = mx.concatenate(outputs, axis=1)
        mx.eval(y_sequential)

        assert mx.allclose(y_parallel, y_sequential, atol=1e-4).item()

    def test_parameter_count(self):
        """Block has the expected number of sub-modules."""
        assert hasattr(self.block, "in_proj")
        assert hasattr(self.block, "conv_weight")
        assert hasattr(self.block, "out_proj")
