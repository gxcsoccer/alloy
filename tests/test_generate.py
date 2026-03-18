"""Tests for the generation module."""

import mlx.core as mx
import pytest

from alloy.generate import sample_top_p, stream_generate, generate
from alloy.models.hybrid_model import HybridConfig, HybridLM


@pytest.fixture
def model():
    config = HybridConfig(
        vocab_size=256, d_model=64, n_layers=2,
        attn_layer_indices=[1], n_heads=4, n_kv_heads=2,
        d_state=16, headdim=32,
    )
    m = HybridLM(config)
    mx.eval(m.parameters())
    return m


class CharTokenizer:
    def encode(self, text):
        return [ord(c) % 256 for c in text]

    def decode(self, ids):
        return "".join(chr(max(32, min(i, 126))) for i in ids)


class TestSampleTopP:
    """Tests for top-p sampling."""

    def test_greedy(self):
        """Temperature=0 returns argmax."""
        logits = mx.array([[0.1, 0.3, 0.9, 0.2]])
        token = sample_top_p(logits, top_p=0.9, temperature=0.0)
        assert token.item() == 2

    def test_output_shape_batch(self):
        """Correct shape for batched input."""
        logits = mx.random.normal((4, 256))
        token = sample_top_p(logits, top_p=0.9, temperature=1.0)
        assert token.shape == (4, 1)

    def test_valid_token_range(self):
        """Sampled tokens are within vocab range."""
        logits = mx.random.normal((1, 100))
        for _ in range(10):
            token = sample_top_p(logits, top_p=0.9, temperature=1.0)
            assert 0 <= token.item() < 100

    def test_top_p_concentration(self):
        """With very low top_p, sampling concentrates on top token."""
        # Make one logit much larger
        logits = mx.array([[0.0, 0.0, 10.0, 0.0, 0.0]])
        tokens = [sample_top_p(logits, top_p=0.01, temperature=1.0).item() for _ in range(20)]
        # Most or all should be token 2
        assert tokens.count(2) >= 18


class TestStreamGenerate:
    """Tests for streaming generation."""

    def test_yields_correct_count(self, model):
        """Generates exactly max_tokens tokens."""
        prompt = mx.array([[1, 2, 3]])
        tokens = list(stream_generate(model, prompt, max_tokens=10, temperature=0.0))
        assert len(tokens) == 10

    def test_token_shapes(self, model):
        """Each yielded token has shape [1, 1]."""
        prompt = mx.array([[1, 2, 3]])
        for token in stream_generate(model, prompt, max_tokens=5, temperature=0.0):
            assert token.shape == (1, 1)

    def test_deterministic_with_seed(self, model):
        """Same seed produces same output."""
        prompt = mx.array([[1, 2, 3]])

        mx.random.seed(42)
        tokens1 = [t.item() for t in stream_generate(model, prompt, max_tokens=10, temperature=0.5)]

        mx.random.seed(42)
        tokens2 = [t.item() for t in stream_generate(model, prompt, max_tokens=10, temperature=0.5)]

        assert tokens1 == tokens2


class TestGenerate:
    """Tests for the high-level generate function."""

    def test_returns_string(self, model):
        """generate returns a string."""
        tokenizer = CharTokenizer()
        result = generate(model, tokenizer, "hello", max_tokens=10, temperature=0.0)
        assert isinstance(result, str)

    def test_starts_with_prompt(self, model):
        """Output starts with the encoded prompt tokens."""
        tokenizer = CharTokenizer()
        prompt = "abc"
        result = generate(model, tokenizer, prompt, max_tokens=5, temperature=0.0)
        # The first characters should decode from the prompt token IDs
        prompt_ids = tokenizer.encode(prompt)
        result_prefix = tokenizer.encode(result[:len(prompt)])
        assert result_prefix == prompt_ids

    def test_output_length(self, model):
        """Output has prompt + max_tokens characters (approx for char tokenizer)."""
        tokenizer = CharTokenizer()
        prompt = "hi"
        result = generate(model, tokenizer, prompt, max_tokens=20, temperature=0.0)
        result_tokens = tokenizer.encode(result)
        assert len(result_tokens) == len(tokenizer.encode(prompt)) + 20
