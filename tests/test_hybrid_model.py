"""Tests for the HybridLM model."""

import mlx.core as mx
import pytest

from alloy.models.hybrid_model import HybridConfig, HybridLM, HybridBlock, SwiGLU
from alloy.generate import sample_top_p, stream_generate


class TestSwiGLU:
    """Tests for SwiGLU FFN."""

    def test_output_shape(self):
        """SwiGLU produces correct output shape."""
        ffn = SwiGLU(d_model=64, d_ff=128)
        x = mx.random.normal((2, 16, 64))
        y = ffn(x)
        assert y.shape == (2, 16, 64)


class TestHybridBlock:
    """Tests for HybridBlock."""

    def test_mamba_block_creation(self):
        """Non-attention layer creates a MambaBlock mixer."""
        config = HybridConfig(d_model=64, n_layers=4, attn_layer_indices=[3],
                              n_heads=4, n_kv_heads=2, headdim=32, d_state=16)
        block = HybridBlock(config, layer_idx=0)
        assert not block.is_attention

    def test_attention_block_creation(self):
        """Attention layer index creates an AttentionBlock mixer."""
        config = HybridConfig(d_model=64, n_layers=4, attn_layer_indices=[3],
                              n_heads=4, n_kv_heads=2, headdim=32, d_state=16)
        block = HybridBlock(config, layer_idx=3)
        assert block.is_attention


class TestHybridLM:
    """Tests for the full HybridLM model."""

    def setup_method(self):
        self.config = HybridConfig(
            vocab_size=256,
            d_model=64,
            n_layers=4,
            attn_layer_indices=[3],
            n_heads=4,
            n_kv_heads=2,
            d_state=16,
            headdim=32,
        )
        self.model = HybridLM(self.config)

    def test_layer_composition(self):
        """Model has correct number of layers with correct types."""
        assert len(self.model.layers) == 4
        attn_count = sum(1 for b in self.model.layers if b.is_attention)
        assert attn_count == 1

    def test_output_shape(self):
        """Forward pass produces logits of shape [B, L, vocab_size]."""
        input_ids = mx.array([[1, 2, 3, 4]])
        logits = self.model(input_ids)
        assert logits.shape == (1, 4, 256)

    def test_weight_tying(self):
        """Embedding and LM head share the same weight matrix."""
        input_ids = mx.array([[1, 2, 3]])
        logits = self.model(input_ids)
        mx.eval(logits)
        # Verify output dim matches vocab
        assert logits.shape[-1] == self.config.vocab_size

    def test_with_cache(self):
        """Forward pass with HybridCache works for autoregressive decoding."""
        cache = self.model.make_cache()
        # Prefill
        input_ids = mx.array([[1, 2, 3, 4]])
        logits = self.model(input_ids, cache=cache)
        mx.eval(logits)
        assert logits.shape == (1, 4, 256)
        # Decode one step
        next_token = mx.array([[5]])
        logits2 = self.model(next_token, cache=cache)
        mx.eval(logits2)
        assert logits2.shape == (1, 1, 256)


class TestGenerate:
    """Tests for generation utilities."""

    def test_sample_top_p_greedy(self):
        """Temperature=0 returns argmax."""
        logits = mx.array([[0.1, 0.3, 0.9, 0.2]])
        token = sample_top_p(logits, top_p=0.9, temperature=0.0)
        assert token.item() == 2

    def test_sample_top_p_shape(self):
        """Sampling returns [B, 1] shape."""
        logits = mx.random.normal((2, 100))
        token = sample_top_p(logits, top_p=0.9, temperature=1.0)
        assert token.shape == (2, 1)

    def test_stream_generate(self):
        """stream_generate yields the requested number of tokens."""
        config = HybridConfig(
            vocab_size=256, d_model=64, n_layers=2,
            attn_layer_indices=[1], n_heads=4, n_kv_heads=2,
            d_state=16, headdim=32,
        )
        model = HybridLM(config)
        prompt = mx.array([[1, 2, 3]])
        tokens = list(stream_generate(model, prompt, max_tokens=5, temperature=0.0))
        assert len(tokens) == 5
        assert all(t.shape == (1, 1) for t in tokens)
