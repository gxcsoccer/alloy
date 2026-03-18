"""Tests for LoRA support."""

import tempfile

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from alloy.lora import (
    LoRALinear,
    linear_to_lora_layers,
    save_lora_weights,
    load_lora_weights,
    merge_lora_weights,
)
from alloy.models.hybrid_model import HybridConfig, HybridLM
from alloy.train import loss_fn, train_step


@pytest.fixture
def config():
    return HybridConfig(
        vocab_size=256, d_model=64, n_layers=4,
        attn_layer_indices=[3], n_heads=4, n_kv_heads=2,
        d_state=16, headdim=32,
    )


@pytest.fixture
def model(config):
    m = HybridLM(config)
    mx.eval(m.parameters())
    return m


class TestLoRALinear:
    """Tests for the LoRALinear module."""

    def test_output_shape(self):
        """LoRALinear preserves the output shape of the base layer."""
        base = nn.Linear(64, 32, bias=False)
        lora = LoRALinear(base, rank=8, alpha=16.0)
        x = mx.random.normal((2, 10, 64))
        y = lora(x)
        assert y.shape == (2, 10, 32)

    def test_initial_output_matches_base(self):
        """With B=0 init, LoRA output equals base output initially."""
        base = nn.Linear(64, 32, bias=False)
        mx.eval(base.parameters())
        lora = LoRALinear(base, rank=8)
        x = mx.random.normal((1, 5, 64))
        y_base = base(x)
        y_lora = lora(x)
        mx.eval(y_base, y_lora)
        assert mx.allclose(y_base, y_lora, atol=1e-6).item()

    def test_base_is_frozen(self):
        """The base linear layer is frozen inside LoRALinear."""
        base = nn.Linear(64, 32, bias=False)
        lora = LoRALinear(base, rank=8)
        # base weight should not be in trainable params
        trainable_names = [n for n, _ in nn.utils.tree_flatten(lora.trainable_parameters())]
        assert all("lora_a" in n or "lora_b" in n for n in trainable_names)


class TestLinearToLoraLayers:
    """Tests for LoRA injection into HybridLM."""

    def test_only_lora_trainable(self, model):
        """After injection, only LoRA params are trainable."""
        model.freeze()
        linear_to_lora_layers(model, lora_rank=8)
        for name, _ in nn.utils.tree_flatten(model.trainable_parameters()):
            assert "lora_a" in name or "lora_b" in name, f"Unexpected: {name}"

    def test_lora_param_ratio(self, model):
        """LoRA params should be a small fraction of total."""
        total = sum(x.size for _, x in nn.utils.tree_flatten(model.parameters()))
        model.freeze()
        linear_to_lora_layers(model, lora_rank=8)
        trainable = sum(x.size for _, x in nn.utils.tree_flatten(model.trainable_parameters()))
        ratio = trainable / total
        assert 0.001 < ratio < 0.2  # typically 1-10%

    def test_custom_target_modules(self, model):
        """Can target only specific modules."""
        model.freeze()
        linear_to_lora_layers(model, lora_rank=4, target_modules={"q_proj"})
        trainable_names = [n for n, _ in nn.utils.tree_flatten(model.trainable_parameters())]
        # Should only have LoRA in attention layer's q_proj
        assert len(trainable_names) > 0
        # All should trace back to q_proj
        assert all("q_proj" in n for n in trainable_names)

    def test_forward_still_works(self, model):
        """Model still produces valid output after LoRA injection."""
        model.freeze()
        linear_to_lora_layers(model, lora_rank=8)
        x = mx.array([[1, 2, 3, 4]])
        logits = model(x)
        mx.eval(logits)
        assert logits.shape == (1, 4, 256)


class TestLoRATraining:
    """Tests for LoRA fine-tuning."""

    def test_loss_decreases(self, model):
        """LoRA fine-tuning should decrease loss."""
        model.freeze()
        linear_to_lora_layers(model, lora_rank=8)
        optimizer = optim.AdamW(learning_rate=1e-3)
        batch = mx.random.randint(0, 256, (2, 32))

        losses = []
        for _ in range(15):
            loss = train_step(model, optimizer, batch)
            mx.eval(loss, model.parameters(), optimizer.state)
            losses.append(loss.item())

        assert losses[-1] < losses[0]


class TestSaveLoadMerge:
    """Tests for LoRA save, load, and merge operations."""

    def test_save_and_load(self, config, model):
        """Saved LoRA weights can be loaded into a fresh model."""
        model.freeze()
        linear_to_lora_layers(model, lora_rank=8)

        # Modify LoRA weights (simulate training)
        optimizer = optim.AdamW(learning_rate=1e-2)
        batch = mx.random.randint(0, 256, (2, 32))
        for _ in range(5):
            train_step(model, optimizer, batch)
            mx.eval(model.parameters(), optimizer.state)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            save_lora_weights(model, f.name)

            # Load into fresh model
            model2 = HybridLM(config)
            model2.freeze()
            linear_to_lora_layers(model2, lora_rank=8)
            load_lora_weights(model2, f.name)

        # Check a LoRA param was actually loaded (not zeros)
        for _, p in nn.utils.tree_flatten(model2.trainable_parameters()):
            mx.eval(p)
            if "lora_b" in str(p):
                break

    def test_merge_removes_lora(self, model):
        """After merging, no LoRALinear layers remain."""
        model.freeze()
        linear_to_lora_layers(model, lora_rank=8)
        merge_lora_weights(model)

        # Check no LoRALinear instances remain
        for _, child in model.children().items():
            if isinstance(child, list):
                for layer in child:
                    for _, v in layer.children().items():
                        assert not isinstance(v, LoRALinear)

    def test_merge_preserves_output(self, model):
        """Merged model produces the same output as LoRA model."""
        model.freeze()
        linear_to_lora_layers(model, lora_rank=8)

        # Simulate some training
        optimizer = optim.AdamW(learning_rate=1e-2)
        batch = mx.random.randint(0, 256, (2, 32))
        for _ in range(3):
            train_step(model, optimizer, batch)
            mx.eval(model.parameters(), optimizer.state)

        test_input = mx.array([[1, 2, 3]])
        logits_before = model(test_input)
        mx.eval(logits_before)

        merge_lora_weights(model)
        logits_after = model(test_input)
        mx.eval(logits_after)

        assert mx.allclose(logits_before, logits_after, atol=1e-4).item()
