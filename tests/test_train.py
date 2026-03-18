"""Tests for the training module."""

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from alloy.models.hybrid_model import HybridConfig, HybridLM
from alloy.train import loss_fn, train_step, load_config, count_parameters


@pytest.fixture
def toy_config():
    return HybridConfig(
        vocab_size=256, d_model=64, n_layers=4,
        attn_layer_indices=[3], n_heads=4, n_kv_heads=2,
        d_state=16, headdim=32,
    )


@pytest.fixture
def toy_model(toy_config):
    model = HybridLM(toy_config)
    mx.eval(model.parameters())
    return model


class TestLossFn:
    """Tests for the loss function."""

    def test_loss_is_scalar(self, toy_model):
        """loss_fn returns a scalar."""
        batch = mx.random.randint(0, 256, (2, 32))
        loss = loss_fn(toy_model, batch)
        mx.eval(loss)
        assert loss.ndim == 0

    def test_loss_is_positive(self, toy_model):
        """Cross-entropy loss should be positive."""
        batch = mx.random.randint(0, 256, (2, 32))
        loss = loss_fn(toy_model, batch)
        mx.eval(loss)
        assert loss.item() > 0

    def test_loss_reasonable_range(self, toy_model, toy_config):
        """Initial loss should be near -ln(1/vocab_size)."""
        batch = mx.random.randint(0, 256, (2, 64))
        loss = loss_fn(toy_model, batch)
        mx.eval(loss)
        import math
        expected = math.log(toy_config.vocab_size)
        # Should be within 2x of random-init expected loss
        assert 0.1 < loss.item() < expected * 2


class TestTrainStep:
    """Tests for the training step."""

    def test_loss_decreases(self, toy_model):
        """Loss should decrease when overfitting a fixed batch."""
        optimizer = optim.AdamW(learning_rate=1e-3)
        batch = mx.random.randint(0, 256, (2, 32))

        losses = []
        for _ in range(10):
            loss = train_step(toy_model, optimizer, batch)
            mx.eval(loss, toy_model.parameters(), optimizer.state)
            losses.append(loss.item())

        assert losses[-1] < losses[0]

    def test_parameters_change(self, toy_model):
        """Parameters should be updated after a train step."""
        optimizer = optim.AdamW(learning_rate=1e-3)
        batch = mx.random.randint(0, 256, (2, 32))

        params_before = toy_model.embedding.weight.tolist()
        train_step(toy_model, optimizer, batch)
        mx.eval(toy_model.parameters(), optimizer.state)
        params_after = toy_model.embedding.weight.tolist()

        assert params_before != params_after


class TestLoadConfig:
    """Tests for config loading."""

    def test_load_toy_yaml(self):
        """Load the toy.yaml config."""
        config = load_config("configs/toy.yaml")
        assert config.d_model == 512
        assert config.n_layers == 12
        assert config.attn_layer_indices == [3, 7, 11]

    def test_load_small_yaml(self):
        """Load the small.yaml config."""
        config = load_config("configs/small.yaml")
        assert config.d_model == 1024
        assert config.n_layers == 24

    def test_load_medium_yaml(self):
        """Load the medium.yaml config."""
        config = load_config("configs/medium.yaml")
        assert config.d_model == 2048
        assert config.window_size == 4096

    def test_roundtrip(self, toy_config):
        """Write and reload a config."""
        import yaml
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            from dataclasses import asdict
            yaml.dump(asdict(toy_config), f)
            f.flush()
            loaded = load_config(f.name)
        assert loaded.d_model == toy_config.d_model
        assert loaded.attn_layer_indices == toy_config.attn_layer_indices


class TestCountParameters:
    """Tests for parameter counting."""

    def test_count(self, toy_model):
        """count_parameters returns a positive integer."""
        n = count_parameters(toy_model)
        assert n > 0
        assert isinstance(n, int)
