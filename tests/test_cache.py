"""Tests for the HybridCache system."""

import mlx.core as mx
import pytest

from alloy.models.cache import AttentionCache, HybridCache, MambaCache


class TestMambaCache:
    """Tests for MambaCache."""

    def test_initial_state_none(self):
        """New cache has None states."""
        cache = MambaCache()
        assert cache.ssm_state is None
        assert cache.conv_state is None


class TestAttentionCache:
    """Tests for AttentionCache."""

    def test_initial_seq_len_zero(self):
        """New cache reports seq_len of 0."""
        cache = AttentionCache()
        assert cache.seq_len == 0

    def test_update_from_empty(self):
        """First update sets keys and values."""
        cache = AttentionCache()
        k = mx.random.normal((1, 2, 4, 16))
        v = mx.random.normal((1, 2, 4, 16))
        cache.update(k, v)
        assert cache.seq_len == 4

    def test_update_append(self):
        """Subsequent updates append to existing cache."""
        cache = AttentionCache()
        k1 = mx.random.normal((1, 2, 4, 16))
        v1 = mx.random.normal((1, 2, 4, 16))
        cache.update(k1, v1)

        k2 = mx.random.normal((1, 2, 1, 16))
        v2 = mx.random.normal((1, 2, 1, 16))
        cache.update(k2, v2)
        assert cache.seq_len == 5


class TestHybridCache:
    """Tests for HybridCache."""

    def test_layer_types(self):
        """Cache assigns correct types per layer."""
        hc = HybridCache(n_layers=4, attn_layer_indices=[1, 3])
        assert isinstance(hc[0], MambaCache)
        assert isinstance(hc[1], AttentionCache)
        assert isinstance(hc[2], MambaCache)
        assert isinstance(hc[3], AttentionCache)

    def test_reset(self):
        """Reset clears all cached states."""
        hc = HybridCache(n_layers=4, attn_layer_indices=[1, 3])
        k = mx.random.normal((1, 2, 4, 16))
        v = mx.random.normal((1, 2, 4, 16))
        hc[1].update(k, v)
        assert hc[1].seq_len == 4

        hc.reset()
        assert hc[1].seq_len == 0
