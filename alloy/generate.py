"""Autoregressive text generation for HybridLM."""

from typing import Iterator, Optional

import mlx.core as mx

from alloy.models.cache import HybridCache
from alloy.models.hybrid_model import HybridLM


def sample_top_p(logits: mx.array, top_p: float, temperature: float) -> mx.array:
    """Sample from logits with temperature and nucleus (top-p) filtering.

    Args:
        logits: Logits of shape [B, vocab_size].
        top_p: Cumulative probability threshold for nucleus sampling.
        temperature: Sampling temperature (1.0 = unchanged).

    Returns:
        Sampled token IDs of shape [B, 1].
    """
    if temperature <= 0:
        return mx.argmax(logits, axis=-1, keepdims=True)

    logits = logits / temperature

    # Sort descending
    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)

    # Cumulative probabilities
    cum_probs = mx.cumsum(sorted_probs, axis=-1)

    # Mask tokens beyond the top-p threshold (keep at least the top token)
    mask = cum_probs - sorted_probs > top_p
    sorted_logits = mx.where(mask, -float("inf"), sorted_logits)

    # Sample from filtered distribution
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    sampled_sorted_idx = mx.random.categorical(mx.log(sorted_probs + 1e-10))  # [B]

    # Map back to original vocab indices
    token = mx.take_along_axis(
        sorted_indices, sampled_sorted_idx[:, None], axis=-1
    )
    return token  # [B, 1]


def stream_generate(
    model: HybridLM,
    prompt_tokens: mx.array,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Iterator[mx.array]:
    """Yield token IDs one at a time (streaming generation).

    Args:
        model: A HybridLM instance.
        prompt_tokens: Prompt token IDs of shape [1, L].
        max_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Yields:
        Token ID arrays of shape [1, 1].
    """
    cache = model.make_cache()

    # Prefill: process the entire prompt at once
    logits = model(prompt_tokens, cache=cache)
    mx.eval(logits, *[c for layer_cache in cache.caches
                       for c in _eval_cache_arrays(layer_cache)])

    # Sample first new token from last position
    token = sample_top_p(logits[:, -1, :], top_p, temperature)
    yield token

    # Decode: generate one token at a time
    for _ in range(max_tokens - 1):
        logits = model(token, cache=cache)
        mx.eval(logits, *[c for layer_cache in cache.caches
                           for c in _eval_cache_arrays(layer_cache)])
        token = sample_top_p(logits[:, -1, :], top_p, temperature)
        yield token


def generate(
    model: HybridLM,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate text autoregressively from a prompt.

    Args:
        model: A HybridLM instance.
        tokenizer: Tokenizer with encode/decode methods.
        prompt: Input text prompt.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        Generated text string (including the prompt).
    """
    prompt_tokens = mx.array([tokenizer.encode(prompt)])  # [1, L]
    tokens = tokenizer.encode(prompt)

    for token in stream_generate(model, prompt_tokens, max_tokens, temperature, top_p):
        t = token.item()
        tokens.append(t)

    return tokenizer.decode(tokens)


def _eval_cache_arrays(layer_cache) -> list:
    """Collect non-None arrays from a cache object for mx.eval."""
    from alloy.models.cache import AttentionCache, MambaCache
    arrays = []
    if isinstance(layer_cache, AttentionCache):
        if layer_cache.keys is not None:
            arrays.extend([layer_cache.keys, layer_cache.values])
    elif isinstance(layer_cache, MambaCache):
        if layer_cache.ssm_state is not None:
            arrays.append(layer_cache.ssm_state)
        if layer_cache.conv_state is not None:
            arrays.append(layer_cache.conv_state)
    return arrays
