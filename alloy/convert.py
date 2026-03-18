"""Weight conversion from HuggingFace models to Alloy format.

Supports loading weights from Jamba (AI21) and Zamba2 (Zyphra) model families
stored in safetensors format.

Note on architecture compatibility:
- Jamba uses Mamba-1 (with dt_rank projection), while Alloy uses Mamba-2.
  Jamba conversion extracts config and maps attention/FFN weights directly,
  but Mamba block weights require dt_proj folding.
- Zamba2 uses Mamba-2 (closer to our architecture) but has shared attention
  blocks with LoRA adapters. Shared weights are duplicated to per-layer.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from alloy.models.hybrid_model import HybridConfig, HybridLM


def load_hf_weights(model_path: str) -> Dict[str, mx.array]:
    """Load weights from HuggingFace safetensors files.

    Also supports .bin (PyTorch) files as fallback.

    Args:
        model_path: Path to HuggingFace model directory containing
            safetensors files and config.json.

    Returns:
        Dictionary mapping parameter names to MLX arrays.
    """
    model_dir = Path(model_path)
    weights = {}

    # Try safetensors first
    st_files = sorted(model_dir.glob("*.safetensors"))
    if st_files:
        for f in st_files:
            weights.update(mx.load(str(f)))
        return weights

    # Fallback to .bin files
    bin_files = sorted(model_dir.glob("*.bin"))
    if bin_files:
        for f in bin_files:
            weights.update(mx.load(str(f)))
        return weights

    # Fallback to .npz files
    npz_files = sorted(model_dir.glob("*.npz"))
    if npz_files:
        for f in npz_files:
            weights.update(mx.load(str(f)))
        return weights

    raise FileNotFoundError(
        f"No weight files (safetensors/bin/npz) found in {model_dir}"
    )


def load_hf_config(model_path: str) -> dict:
    """Load config.json from a HuggingFace model directory.

    Args:
        model_path: Path to HuggingFace model directory.

    Returns:
        Parsed config dictionary.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(config_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Jamba (AI21)
# ---------------------------------------------------------------------------
# HF Jamba layer keys:
#   model.embed_tokens.weight
#   model.layers.{i}.attention.{q,k,v,o}_proj.weight        (attention layers)
#   model.layers.{i}.mamba.{in_proj,conv1d,x_proj,out_proj}  (mamba layers)
#   model.layers.{i}.mamba.A_log
#   model.layers.{i}.mamba.D
#   model.layers.{i}.feed_forward.{gate,up,down}_proj.weight  (dense FFN)
#   model.layers.{i}.input_layernorm.weight
#   model.layers.{i}.post_attention_layernorm.weight           (not on all layers)
#   model.final_layernorm.weight
#   lm_head.weight


def _jamba_attn_layer_indices(n_layers: int, period: int, offset: int) -> List[int]:
    """Compute which layers are attention in Jamba's periodic pattern."""
    return [i for i in range(n_layers) if (i - offset) % period == 0 and i >= offset]


def convert_jamba(model_path: str) -> Tuple[HybridConfig, Dict[str, mx.array]]:
    """Convert Jamba model weights to Alloy format.

    Jamba uses Mamba-1 with dt_rank, while Alloy uses Mamba-2. The Mamba
    block weights (in_proj, x_proj, dt_proj) have different semantics.
    This conversion handles the structural mapping but callers should be
    aware of the Mamba-1 vs Mamba-2 differences.

    MoE (Mixture of Experts) layers are not supported — only the dense
    FFN weights are mapped. For MoE layers, only the first expert is used.

    Args:
        model_path: Path to HuggingFace Jamba model directory.

    Returns:
        Tuple of (config, state_dict) ready to load into HybridLM.
    """
    hf_cfg = load_hf_config(model_path)
    hf_weights = load_hf_weights(model_path)

    n_layers = hf_cfg["num_hidden_layers"]
    attn_period = hf_cfg.get("attn_layer_period", 8)
    attn_offset = hf_cfg.get("attn_layer_offset", 4)
    attn_indices = _jamba_attn_layer_indices(n_layers, attn_period, attn_offset)

    d_model = hf_cfg["hidden_size"]
    n_heads = hf_cfg["num_attention_heads"]
    n_kv_heads = hf_cfg.get("num_key_value_heads", n_heads)
    d_state = hf_cfg.get("mamba_d_state", 16)
    d_conv = hf_cfg.get("mamba_d_conv", 4)
    expand = hf_cfg.get("mamba_expand", 2)
    d_inner = d_model * expand
    headdim = max(64, d_inner // n_heads if n_heads > 0 else 64)

    config = HybridConfig(
        vocab_size=hf_cfg["vocab_size"],
        d_model=d_model,
        n_layers=n_layers,
        attn_layer_indices=attn_indices,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim,
        ffn_mult=hf_cfg.get("intermediate_size", int(d_model * 2.667)) / d_model,
    )

    alloy_weights = {}

    # Embedding
    _map(alloy_weights, hf_weights, "model.embed_tokens.weight", "embedding.weight")

    # Final norm
    _map(alloy_weights, hf_weights, "model.final_layernorm.weight", "norm.weight")

    # Per-layer
    for i in range(n_layers):
        hf_pre = f"model.layers.{i}"
        al_pre = f"layers.{i}"

        # Norms
        _map(alloy_weights, hf_weights,
             f"{hf_pre}.input_layernorm.weight", f"{al_pre}.norm1.weight")
        _map_optional(alloy_weights, hf_weights,
                      f"{hf_pre}.post_attention_layernorm.weight", f"{al_pre}.norm2.weight")

        if i in attn_indices:
            # Attention projections
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                _map(alloy_weights, hf_weights,
                     f"{hf_pre}.attention.{proj}.weight", f"{al_pre}.mixer.{proj}.weight")
        else:
            # Mamba block
            _map(alloy_weights, hf_weights,
                 f"{hf_pre}.mamba.in_proj.weight", f"{al_pre}.mixer.in_proj.weight")
            _map(alloy_weights, hf_weights,
                 f"{hf_pre}.mamba.x_proj.weight", f"{al_pre}.mixer.x_proj.weight")
            _map(alloy_weights, hf_weights,
                 f"{hf_pre}.mamba.out_proj.weight", f"{al_pre}.mixer.out_proj.weight")
            _map(alloy_weights, hf_weights,
                 f"{hf_pre}.mamba.A_log", f"{al_pre}.mixer.A_log")
            _map_optional(alloy_weights, hf_weights,
                          f"{hf_pre}.mamba.dt_bias", f"{al_pre}.mixer.dt_bias")

            # Conv1d: HF stores [d_inner, 1, d_conv], Alloy stores [d_inner, d_conv]
            conv_key = f"{hf_pre}.mamba.conv1d.weight"
            if conv_key in hf_weights:
                w = hf_weights[conv_key]
                if w.ndim == 3:
                    w = w.squeeze(1)  # [d_inner, 1, d_conv] -> [d_inner, d_conv]
                alloy_weights[f"{al_pre}.mixer.conv_weight"] = w
            _map_optional(alloy_weights, hf_weights,
                          f"{hf_pre}.mamba.conv1d.bias", f"{al_pre}.mixer.conv_bias")

        # FFN (SwiGLU): gate_proj -> w1, down_proj -> w2, up_proj -> w3
        # Handle both dense and MoE (use first expert for MoE)
        ffn_prefix = f"{hf_pre}.feed_forward"
        for hf_name, al_name in [("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")]:
            dense_key = f"{ffn_prefix}.{hf_name}.weight"
            expert_key = f"{ffn_prefix}.experts.0.{hf_name}.weight"
            if dense_key in hf_weights:
                alloy_weights[f"{al_pre}.ffn.{al_name}.weight"] = hf_weights[dense_key]
            elif expert_key in hf_weights:
                alloy_weights[f"{al_pre}.ffn.{al_name}.weight"] = hf_weights[expert_key]

    return config, alloy_weights


# ---------------------------------------------------------------------------
# Zamba2 (Zyphra)
# ---------------------------------------------------------------------------
# HF Zamba2 layer keys:
#   model.embed_tokens.weight
#   model.layers.{i}.mamba_block.{in_proj,conv1d,x_proj,dt_proj,out_proj}
#   model.layers.{i}.mamba_block.A_log
#   model.layers.{i}.attention.{q,k,v,o}_proj.weight   (shared blocks)
#   model.layers.{i}.mlp.{gate,up,down}_proj.weight     (shared MLP)
#   model.layers.{i}.input_layernorm.weight
#   model.layers.{i}.post_attention_layernorm.weight
#   model.norm.weight
#   lm_head.weight


def convert_zamba(model_path: str) -> Tuple[HybridConfig, Dict[str, mx.array]]:
    """Convert Zamba2 model weights to Alloy format.

    Zamba2 uses Mamba-2 (matching our architecture) with shared attention
    blocks. Shared attention weights are duplicated into each attention
    layer for simplicity.

    Args:
        model_path: Path to HuggingFace Zamba2 model directory.

    Returns:
        Tuple of (config, state_dict) ready to load into HybridLM.
    """
    hf_cfg = load_hf_config(model_path)
    hf_weights = load_hf_weights(model_path)

    n_layers = hf_cfg["num_hidden_layers"]
    d_model = hf_cfg["hidden_size"]
    n_heads = hf_cfg.get("num_attention_heads", 32)
    n_kv_heads = hf_cfg.get("num_key_value_heads", n_heads)

    # Zamba2 specifies layer types explicitly
    block_types = hf_cfg.get("layers_block_type", [])
    attn_indices = [i for i, t in enumerate(block_types) if t == "hybrid"]

    d_state = hf_cfg.get("mamba_d_state", 64)
    d_conv = hf_cfg.get("mamba_d_conv", 4)
    expand = hf_cfg.get("mamba_expand", 2)
    d_inner = d_model * expand
    n_mamba_heads = hf_cfg.get("n_mamba_heads", 8)
    headdim = d_inner // n_mamba_heads if n_mamba_heads else 64
    chunk_size = hf_cfg.get("chunk_size", 256)

    config = HybridConfig(
        vocab_size=hf_cfg["vocab_size"],
        d_model=d_model,
        n_layers=n_layers,
        attn_layer_indices=attn_indices,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim,
        chunk_size=chunk_size,
        ffn_mult=hf_cfg.get("intermediate_size", int(d_model * 2.667)) / d_model,
    )

    alloy_weights = {}

    # Embedding
    _map(alloy_weights, hf_weights, "model.embed_tokens.weight", "embedding.weight")

    # Final norm
    _map(alloy_weights, hf_weights, "model.norm.weight", "norm.weight")

    # Per-layer
    for i in range(n_layers):
        hf_pre = f"model.layers.{i}"
        al_pre = f"layers.{i}"

        # Norms
        _map(alloy_weights, hf_weights,
             f"{hf_pre}.input_layernorm.weight", f"{al_pre}.norm1.weight")
        _map_optional(alloy_weights, hf_weights,
                      f"{hf_pre}.post_attention_layernorm.weight", f"{al_pre}.norm2.weight")

        if i in attn_indices:
            # Attention (may be shared in HF, we store per-layer)
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                _map(alloy_weights, hf_weights,
                     f"{hf_pre}.attention.{proj}.weight", f"{al_pre}.mixer.{proj}.weight")

            # FFN for hybrid layers
            for hf_name, al_name in [("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")]:
                _map_optional(alloy_weights, hf_weights,
                              f"{hf_pre}.mlp.{hf_name}.weight", f"{al_pre}.ffn.{al_name}.weight")
        else:
            # Mamba2 block
            _map(alloy_weights, hf_weights,
                 f"{hf_pre}.mamba_block.in_proj.weight", f"{al_pre}.mixer.in_proj.weight")
            _map(alloy_weights, hf_weights,
                 f"{hf_pre}.mamba_block.x_proj.weight", f"{al_pre}.mixer.x_proj.weight")
            _map(alloy_weights, hf_weights,
                 f"{hf_pre}.mamba_block.out_proj.weight", f"{al_pre}.mixer.out_proj.weight")
            _map(alloy_weights, hf_weights,
                 f"{hf_pre}.mamba_block.A_log", f"{al_pre}.mixer.A_log")
            _map_optional(alloy_weights, hf_weights,
                          f"{hf_pre}.mamba_block.dt_bias", f"{al_pre}.mixer.dt_bias")

            # Conv1d
            conv_key = f"{hf_pre}.mamba_block.conv1d.weight"
            if conv_key in hf_weights:
                w = hf_weights[conv_key]
                if w.ndim == 3:
                    w = w.squeeze(1)
                alloy_weights[f"{al_pre}.mixer.conv_weight"] = w
            _map_optional(alloy_weights, hf_weights,
                          f"{hf_pre}.mamba_block.conv1d.bias", f"{al_pre}.mixer.conv_bias")

        # FFN for mamba layers (if present)
        if i not in attn_indices:
            for hf_name, al_name in [("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")]:
                _map_optional(alloy_weights, hf_weights,
                              f"{hf_pre}.feed_forward.{hf_name}.weight",
                              f"{al_pre}.ffn.{al_name}.weight")

    return config, alloy_weights


# ---------------------------------------------------------------------------
# Auto-detect and convert
# ---------------------------------------------------------------------------

def convert_from_hf(model_path: str) -> Tuple[HybridConfig, Dict[str, mx.array]]:
    """Auto-detect model type and convert from HuggingFace format.

    Args:
        model_path: Path to HuggingFace model directory.

    Returns:
        Tuple of (config, state_dict) ready to load into HybridLM.
    """
    hf_cfg = load_hf_config(model_path)
    model_type = hf_cfg.get("model_type", "").lower()

    if "jamba" in model_type:
        return convert_jamba(model_path)
    elif "zamba" in model_type:
        return convert_zamba(model_path)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            "Supported: jamba, zamba2"
        )


def load_pretrained(model_path: str) -> HybridLM:
    """Load a pretrained model from HuggingFace format.

    Convenience function that converts and loads weights in one step.

    Args:
        model_path: Path to HuggingFace model directory.

    Returns:
        HybridLM with loaded weights.
    """
    config, weights = convert_from_hf(model_path)
    model = HybridLM(config)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map(dst: dict, src: dict, src_key: str, dst_key: str) -> None:
    """Map a weight from src to dst, raising if missing."""
    if src_key not in src:
        raise KeyError(f"Expected weight key not found: {src_key}")
    dst[dst_key] = src[src_key]


def _map_optional(dst: dict, src: dict, src_key: str, dst_key: str) -> None:
    """Map a weight from src to dst if it exists."""
    if src_key in src:
        dst[dst_key] = src[src_key]
