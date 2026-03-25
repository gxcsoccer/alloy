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
import mlx.nn as nn

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
# Real Zamba2-1.2B HF weight structure:
#
# Global:
#   model.embed_tokens.weight
#   model.final_layernorm.weight
#
# Mamba-only layers (e.g., layer 0):
#   model.layers.{i}.input_layernorm.weight
#   model.layers.{i}.mamba.in_proj.weight       [d_inner*2 + d_bc + n_heads, d_model]
#   model.layers.{i}.mamba.conv1d.weight         [d_inner + d_bc, 1, d_conv]
#   model.layers.{i}.mamba.conv1d.bias           [d_inner + d_bc]
#   model.layers.{i}.mamba.A_log                 [n_mamba_heads]
#   model.layers.{i}.mamba.D                     [n_mamba_heads]
#   model.layers.{i}.mamba.dt_bias               [n_mamba_heads]
#   model.layers.{i}.mamba.norm.weight           [d_inner]
#   model.layers.{i}.mamba.out_proj.weight       [d_model, d_inner]
#   (no separate x_proj — B, C, dt are in in_proj)
#   (no FFN for mamba-only layers)
#
# Hybrid layers (e.g., layer 5):
#   model.layers.{i}.mamba_decoder.input_layernorm.weight
#   model.layers.{i}.mamba_decoder.mamba.{in_proj,conv1d,A_log,D,dt_bias,norm,out_proj}
#   model.layers.{i}.linear.weight               [d_model, d_model]
#   model.layers.{i}.shared_transformer.input_layernorm.weight   [attn_dim]
#   model.layers.{i}.shared_transformer.self_attn.{q,k,v,o}_proj.weight
#   model.layers.{i}.shared_transformer.pre_ff_layernorm.weight  [d_model]
#   model.layers.{i}.shared_transformer.feed_forward.gate_up_proj.weight  [2*ffn_dim, d_model]
#   model.layers.{i}.shared_transformer.feed_forward.down_proj.weight     [d_model, ffn_dim]
#   + adapter_list weights (LoRA, ignored for now)


def convert_zamba(model_path: str) -> Tuple[HybridConfig, Dict[str, mx.array]]:
    """Convert Zamba2 model weights to Alloy format.

    Handles the real Zamba2 weight structure including:
    - Combined in_proj (x + z + B + C + dt)
    - Conv1d on extended input (x + B + C)
    - D skip connection, inner RMS norm
    - Hybrid layers with mamba_decoder + shared_transformer
    - Fused gate_up_proj (split into w1 + w3)
    - Adapter weights (LoRA) are ignored

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

    block_types = hf_cfg.get("layers_block_type", [])
    attn_indices = [i for i, t in enumerate(block_types) if t == "hybrid"]

    d_state = hf_cfg.get("mamba_d_state", 64)
    d_conv = hf_cfg.get("mamba_d_conv", 4)
    expand = hf_cfg.get("mamba_expand", 2)
    n_groups = hf_cfg.get("mamba_ngroups", 1)
    d_inner = d_model * expand
    n_mamba_heads = hf_cfg.get("n_mamba_heads", d_inner // 64)
    headdim = d_inner // n_mamba_heads if n_mamba_heads else 64
    chunk_size = hf_cfg.get("chunk_size", 256)
    ffn_dim = hf_cfg.get("intermediate_size", int(d_model * 4))
    attn_d_model = hf_cfg.get("attention_hidden_size", d_model)

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
        ffn_mult=ffn_dim / d_model,
        combined_proj=True,
        n_groups=n_groups,
        use_D=True,
        use_inner_norm=True,
        attn_d_model=attn_d_model,
        zamba2_hybrid=True,
    )

    alloy_weights = {}

    # Embedding
    _map(alloy_weights, hf_weights, "model.embed_tokens.weight", "embedding.weight")

    # Final norm
    _map(alloy_weights, hf_weights, "model.final_layernorm.weight", "norm.weight")

    # Find the first hybrid layer that has shared_transformer weights
    shared_layer = None
    for i in attn_indices:
        if any(f"model.layers.{i}.shared_transformer" in k for k in hf_weights):
            shared_layer = i
            break

    for i in range(n_layers):
        hf_pre = f"model.layers.{i}"
        al_pre = f"layers.{i}"

        if i in attn_indices:
            # === Zamba2 hybrid layer: mamba_decoder + shared_transformer ===

            # Mamba decoder sub-block
            _convert_zamba_mamba_block(
                alloy_weights, hf_weights,
                hf_prefix=f"{hf_pre}.mamba_decoder.mamba",
                al_prefix=f"{al_pre}.mamba_decoder",
            )
            # Mamba decoder input norm
            _map(alloy_weights, hf_weights,
                 f"{hf_pre}.mamba_decoder.input_layernorm.weight",
                 f"{al_pre}.mamba_norm.weight")

            # Linear projection between mamba and attention
            _map(alloy_weights, hf_weights,
                 f"{hf_pre}.linear.weight",
                 f"{al_pre}.linear.weight")

            # Shared transformer: duplicate from the source layer + merge LoRA adapters
            src = f"model.layers.{shared_layer}" if shared_layer is not None else hf_pre
            adapter_idx = attn_indices.index(i)  # which adapter in the list

            # Attention projections (shared + per-layer LoRA adapter)
            attn_pre = f"{src}.shared_transformer.self_attn"
            adapter_map = {"q_proj": "linear_q", "k_proj": "linear_k", "v_proj": "linear_v"}
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                base_w = hf_weights[f"{attn_pre}.{proj}.weight"]
                # Merge LoRA adapter: W_eff = W_base + B @ A
                if proj in adapter_map:
                    a_key = f"{attn_pre}.{adapter_map[proj]}_adapter_list.{adapter_idx}.0.weight"
                    b_key = f"{attn_pre}.{adapter_map[proj]}_adapter_list.{adapter_idx}.1.weight"
                    if a_key in hf_weights and b_key in hf_weights:
                        A = hf_weights[a_key]  # [rank, in_dim]
                        B = hf_weights[b_key]  # [out_dim, rank]
                        base_w = base_w + B @ A
                alloy_weights[f"{al_pre}.mixer.{proj}.weight"] = base_w

            # Attention norm (shared)
            _map(alloy_weights, hf_weights,
                 f"{src}.shared_transformer.input_layernorm.weight",
                 f"{al_pre}.attn_norm.weight")

            # FFN: fused gate_up_proj → split into w1 (gate) + w3 (up)
            # Merge gate_up_proj adapter
            ffn_pre = f"{src}.shared_transformer.feed_forward"
            gate_up_key = f"{ffn_pre}.gate_up_proj.weight"
            if gate_up_key in hf_weights:
                gate_up = hf_weights[gate_up_key]  # [2 * ffn_dim, d_model]
                # Merge adapter
                a_key = f"{ffn_pre}.gate_up_proj_adapter_list.{adapter_idx}.0.weight"
                b_key = f"{ffn_pre}.gate_up_proj_adapter_list.{adapter_idx}.1.weight"
                if a_key in hf_weights and b_key in hf_weights:
                    A = hf_weights[a_key]  # [rank, d_model]
                    B = hf_weights[b_key]  # [2*ffn_dim, rank]
                    gate_up = gate_up + B @ A
                half = gate_up.shape[0] // 2
                alloy_weights[f"{al_pre}.ffn.w1.weight"] = gate_up[:half]  # gate
                alloy_weights[f"{al_pre}.ffn.w3.weight"] = gate_up[half:]  # up
            _map_optional(alloy_weights, hf_weights,
                          f"{ffn_pre}.down_proj.weight",
                          f"{al_pre}.ffn.w2.weight")

            # Pre-FFN norm (shared)
            _map_optional(alloy_weights, hf_weights,
                          f"{src}.shared_transformer.pre_ff_layernorm.weight",
                          f"{al_pre}.norm2.weight")

        else:
            # === Mamba-only layer ===
            _map(alloy_weights, hf_weights,
                 f"{hf_pre}.input_layernorm.weight",
                 f"{al_pre}.norm1.weight")

            _convert_zamba_mamba_block(
                alloy_weights, hf_weights,
                hf_prefix=f"{hf_pre}.mamba",
                al_prefix=f"{al_pre}.mixer",
            )

    return config, alloy_weights


def _convert_zamba_mamba_block(
    alloy_weights: dict,
    hf_weights: dict,
    hf_prefix: str,
    al_prefix: str,
) -> None:
    """Convert a single Zamba2 mamba block's weights.

    Handles in_proj, conv1d (squeeze), A_log, D, dt_bias, norm, out_proj.
    """
    _map(alloy_weights, hf_weights,
         f"{hf_prefix}.in_proj.weight", f"{al_prefix}.in_proj.weight")
    _map(alloy_weights, hf_weights,
         f"{hf_prefix}.out_proj.weight", f"{al_prefix}.out_proj.weight")
    _map(alloy_weights, hf_weights,
         f"{hf_prefix}.A_log", f"{al_prefix}.A_log")
    _map_optional(alloy_weights, hf_weights,
                  f"{hf_prefix}.D", f"{al_prefix}.D")
    _map_optional(alloy_weights, hf_weights,
                  f"{hf_prefix}.dt_bias", f"{al_prefix}.dt_bias")
    _map_optional(alloy_weights, hf_weights,
                  f"{hf_prefix}.norm.weight", f"{al_prefix}.norm.weight")

    # Conv1d: [conv_dim, 1, d_conv] → [conv_dim, d_conv]
    conv_key = f"{hf_prefix}.conv1d.weight"
    if conv_key in hf_weights:
        w = hf_weights[conv_key]
        if w.ndim == 3:
            w = w.squeeze(1)
        alloy_weights[f"{al_prefix}.conv_weight"] = w
    _map_optional(alloy_weights, hf_weights,
                  f"{hf_prefix}.conv1d.bias", f"{al_prefix}.conv_bias")


# ---------------------------------------------------------------------------
# Nemotron-H (NVIDIA)
# ---------------------------------------------------------------------------
# Flat 52-layer structure: alternating Mamba / MLP / Attention layers.
# Pattern: M-MLP-M-MLP-...-M-ATTN-MLP-M-MLP-...
#
# backbone.layers.{i}.norm.weight              — pre-norm for every layer
# backbone.layers.{i}.mixer.in_proj.weight     — Mamba: combined projection
# backbone.layers.{i}.mixer.conv1d.{weight,bias}
# backbone.layers.{i}.mixer.{A_log,D,dt_bias,norm.weight,out_proj.weight}
# backbone.layers.{i}.mixer.{q,k,v,o}_proj.weight  — Attention layers
# backbone.layers.{i}.mixer.{up,down}_proj.weight   — MLP layers
# backbone.embeddings.weight, backbone.norm_f.weight, lm_head.weight


def _parse_nemotron_pattern(pattern: str) -> List[str]:
    """Parse Nemotron-H hybrid_override_pattern into flat layer types.

    Pattern like 'M-M-M-M*-M-...' where M=Mamba block, M*=Attention block.
    M  → 2 layers: [mamba, mlp]
    M* → 3 layers: [mamba, attention, mlp]
    Total: 20×M + 4×M* = 20×2 + 4×3 = 52 layers.
    """
    entries = [e for e in pattern.rstrip("-").split("-") if e]
    layer_types = []
    for entry in entries:
        if entry == "M*":
            layer_types.extend(["mamba", "attention", "mlp"])
        else:
            layer_types.extend(["mamba", "mlp"])
    return layer_types


def convert_nemotron_h(model_path: str) -> Tuple[HybridConfig, Dict[str, mx.array]]:
    """Convert Nemotron-H model weights to Alloy format.

    Nemotron-H uses a flat layer structure where each layer is exactly one of:
    Mamba, Attention, or MLP. No nested blocks.
    """
    hf_cfg = load_hf_config(model_path)
    hf_weights = load_hf_weights(model_path)

    d_model = hf_cfg["hidden_size"]
    n_layers = hf_cfg["num_hidden_layers"]
    n_heads = hf_cfg.get("num_attention_heads", 32)
    n_kv_heads = hf_cfg.get("num_key_value_heads", 8)
    d_state = hf_cfg.get("ssm_state_size", 128)
    d_conv = hf_cfg.get("conv_kernel", 4)
    expand = hf_cfg.get("expand", 2)
    n_groups = hf_cfg.get("n_groups", 8)
    mamba_head_dim = hf_cfg.get("mamba_head_dim", 64)
    chunk_size = hf_cfg.get("chunk_size", 128)
    ffn_hidden = hf_cfg.get("intermediate_size", 21504)
    vocab_size = hf_cfg.get("vocab_size", 131072)

    # Parse layer pattern
    pattern = hf_cfg.get("hybrid_override_pattern", "")
    if pattern:
        layer_types = _parse_nemotron_pattern(pattern)
    else:
        # Infer from weights
        layer_types = []
        for i in range(n_layers):
            if f"backbone.layers.{i}.mixer.A_log" in hf_weights:
                layer_types.append("mamba")
            elif f"backbone.layers.{i}.mixer.q_proj.weight" in hf_weights:
                layer_types.append("attention")
            elif f"backbone.layers.{i}.mixer.down_proj.weight" in hf_weights:
                layer_types.append("mlp")

    attn_indices = [i for i, lt in enumerate(layer_types) if lt == "attention"]

    config = HybridConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=len(layer_types),
        attn_layer_indices=attn_indices,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=mamba_head_dim,
        chunk_size=chunk_size,
        ffn_hidden_size=ffn_hidden,
        combined_proj=True,
        n_groups=n_groups,
        use_D=True,
        use_inner_norm=True,
        layer_types=layer_types,
    )

    alloy_weights = {}

    # Embedding
    _map(alloy_weights, hf_weights, "backbone.embeddings.weight", "embedding.weight")

    # Final norm
    _map(alloy_weights, hf_weights, "backbone.norm_f.weight", "norm.weight")

    # Separate LM head
    _map(alloy_weights, hf_weights, "lm_head.weight", "lm_head.weight")

    # Per-layer
    for i, lt in enumerate(layer_types):
        hf_pre = f"backbone.layers.{i}"
        al_pre = f"layers.{i}"

        # Pre-norm (all layers have this)
        _map(alloy_weights, hf_weights,
             f"{hf_pre}.norm.weight", f"{al_pre}.norm.weight")

        if lt == "mamba":
            _convert_zamba_mamba_block(
                alloy_weights, hf_weights,
                hf_prefix=f"{hf_pre}.mixer",
                al_prefix=f"{al_pre}.mixer",
            )
        elif lt == "attention":
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                _map(alloy_weights, hf_weights,
                     f"{hf_pre}.mixer.{proj}.weight",
                     f"{al_pre}.mixer.{proj}.weight")
        elif lt == "mlp":
            _map(alloy_weights, hf_weights,
                 f"{hf_pre}.mixer.up_proj.weight",
                 f"{al_pre}.mixer.up_proj.weight")
            _map(alloy_weights, hf_weights,
                 f"{hf_pre}.mixer.down_proj.weight",
                 f"{al_pre}.mixer.down_proj.weight")

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
    elif "nemotron_h" in model_type:
        return convert_nemotron_h(model_path)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            "Supported: jamba, zamba2, nemotron_h"
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
    # Create separate lm_head if weights include it
    if "lm_head.weight" in weights:
        model.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
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
