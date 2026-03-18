"""Tests for the weight conversion module."""

import json
import os
import tempfile

import mlx.core as mx
import pytest

from alloy.convert import (
    convert_from_hf,
    convert_jamba,
    convert_zamba,
    load_hf_config,
    load_hf_weights,
)


def _make_jamba_dir(tmpdir, n_layers=4, d=64, di=128):
    """Create a mock Jamba model directory with config and weights."""
    cfg = {
        "model_type": "jamba",
        "vocab_size": 256,
        "hidden_size": d,
        "num_hidden_layers": n_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": di,
        "attn_layer_period": 4,
        "attn_layer_offset": 3,
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
    }
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(cfg, f)

    weights = {}
    weights["model.embed_tokens.weight"] = mx.random.normal((256, d))
    weights["model.final_layernorm.weight"] = mx.ones((d,))

    for i in range(n_layers):
        pre = f"model.layers.{i}"
        weights[f"{pre}.input_layernorm.weight"] = mx.ones((d,))
        if i == 3:
            for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                sz = d if p in ("q_proj", "o_proj") else 32
                weights[f"{pre}.attention.{p}.weight"] = mx.random.normal((sz, d))
        else:
            weights[f"{pre}.mamba.in_proj.weight"] = mx.random.normal((2 * di, d))
            weights[f"{pre}.mamba.conv1d.weight"] = mx.random.normal((di, 1, 4))
            weights[f"{pre}.mamba.conv1d.bias"] = mx.zeros((di,))
            weights[f"{pre}.mamba.x_proj.weight"] = mx.random.normal((132, di))
            weights[f"{pre}.mamba.out_proj.weight"] = mx.random.normal((d, di))
            weights[f"{pre}.mamba.A_log"] = mx.zeros((4,))
        for n in ("gate_proj", "up_proj", "down_proj"):
            s = (di, d) if n != "down_proj" else (d, di)
            weights[f"{pre}.feed_forward.{n}.weight"] = mx.random.normal(s)

    mx.savez(os.path.join(tmpdir, "model.npz"), **weights)
    return cfg


def _make_zamba_dir(tmpdir, n_layers=4, d=64, di=128):
    """Create a mock Zamba2 model directory with config and weights."""
    cfg = {
        "model_type": "zamba2",
        "vocab_size": 256,
        "hidden_size": d,
        "num_hidden_layers": n_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": di,
        "layers_block_type": ["mamba", "mamba", "mamba", "hybrid"],
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "n_mamba_heads": 4,
        "chunk_size": 64,
    }
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(cfg, f)

    weights = {}
    weights["model.embed_tokens.weight"] = mx.random.normal((256, d))
    weights["model.norm.weight"] = mx.ones((d,))

    for i in range(n_layers):
        pre = f"model.layers.{i}"
        weights[f"{pre}.input_layernorm.weight"] = mx.ones((d,))
        if i == 3:
            for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                sz = d if p in ("q_proj", "o_proj") else 32
                weights[f"{pre}.attention.{p}.weight"] = mx.random.normal((sz, d))
            for n in ("gate_proj", "up_proj", "down_proj"):
                s = (di, d) if n != "down_proj" else (d, di)
                weights[f"{pre}.mlp.{n}.weight"] = mx.random.normal(s)
        else:
            weights[f"{pre}.mamba_block.in_proj.weight"] = mx.random.normal((2 * di, d))
            weights[f"{pre}.mamba_block.conv1d.weight"] = mx.random.normal((di, 1, 4))
            weights[f"{pre}.mamba_block.conv1d.bias"] = mx.zeros((di,))
            weights[f"{pre}.mamba_block.x_proj.weight"] = mx.random.normal((132, di))
            weights[f"{pre}.mamba_block.out_proj.weight"] = mx.random.normal((d, di))
            weights[f"{pre}.mamba_block.A_log"] = mx.zeros((4,))
            for n in ("gate_proj", "up_proj", "down_proj"):
                s = (di, d) if n != "down_proj" else (d, di)
                weights[f"{pre}.feed_forward.{n}.weight"] = mx.random.normal(s)

    mx.savez(os.path.join(tmpdir, "model.npz"), **weights)
    return cfg


class TestLoadHfConfig:
    """Tests for config loading."""

    def test_load_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_jamba_dir(tmpdir)
            cfg = load_hf_config(tmpdir)
            assert cfg["model_type"] == "jamba"
            assert cfg["hidden_size"] == 64

    def test_missing_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_hf_config(tmpdir)


class TestLoadHfWeights:
    """Tests for weight loading."""

    def test_load_npz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_jamba_dir(tmpdir)
            weights = load_hf_weights(tmpdir)
            assert "model.embed_tokens.weight" in weights

    def test_no_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_hf_weights(tmpdir)


class TestConvertJamba:
    """Tests for Jamba conversion."""

    def test_config_extraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_jamba_dir(tmpdir)
            config, _ = convert_jamba(tmpdir)
            assert config.d_model == 64
            assert config.n_layers == 4
            assert config.attn_layer_indices == [3]

    def test_weight_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_jamba_dir(tmpdir)
            _, weights = convert_jamba(tmpdir)
            assert "embedding.weight" in weights
            assert "norm.weight" in weights
            assert "layers.0.mixer.in_proj.weight" in weights
            assert "layers.3.mixer.q_proj.weight" in weights

    def test_conv_weight_squeezed(self):
        """Conv1d weight is squeezed from [d, 1, k] to [d, k]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_jamba_dir(tmpdir)
            _, weights = convert_jamba(tmpdir)
            assert weights["layers.0.mixer.conv_weight"].ndim == 2

    def test_ffn_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_jamba_dir(tmpdir)
            _, weights = convert_jamba(tmpdir)
            assert "layers.0.ffn.w1.weight" in weights
            assert "layers.0.ffn.w2.weight" in weights
            assert "layers.0.ffn.w3.weight" in weights


class TestConvertZamba:
    """Tests for Zamba2 conversion."""

    def test_config_extraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_zamba_dir(tmpdir)
            config, _ = convert_zamba(tmpdir)
            assert config.d_model == 64
            assert config.attn_layer_indices == [3]
            assert config.chunk_size == 64

    def test_weight_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_zamba_dir(tmpdir)
            _, weights = convert_zamba(tmpdir)
            assert "embedding.weight" in weights
            assert "norm.weight" in weights
            assert "layers.1.mixer.in_proj.weight" in weights
            assert "layers.3.mixer.q_proj.weight" in weights


class TestConvertFromHf:
    """Tests for auto-detection."""

    def test_auto_detect_jamba(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_jamba_dir(tmpdir)
            config, _ = convert_from_hf(tmpdir)
            assert config.n_layers == 4

    def test_auto_detect_zamba(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_zamba_dir(tmpdir)
            config, _ = convert_from_hf(tmpdir)
            assert config.n_layers == 4

    def test_unknown_model_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump({"model_type": "unknown"}, f)
            with pytest.raises(ValueError, match="Unsupported"):
                convert_from_hf(tmpdir)
