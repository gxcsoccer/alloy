"""LoRA (Low-Rank Adaptation) support for HybridLM.

Reference: mlx-lm LoRA implementation.
"""

from pathlib import Path
from typing import Optional, Set

import mlx.core as mx
import mlx.nn as nn

from alloy.models.hybrid_model import HybridLM

# Default layers to apply LoRA to
DEFAULT_TARGET_MODULES: Set[str] = {
    "q_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "out_proj",
}


class LoRALinear(nn.Module):
    """Linear layer with a low-rank adapter.

    Computes: y = Linear(x) + scale * (x @ A^T @ B^T)
    where scale = lora_alpha / lora_rank.

    Only A and B are trainable; the original weight is frozen.

    Args:
        base: The original nn.Linear layer to wrap.
        rank: Rank of the low-rank matrices.
        alpha: Scaling factor.
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank

        in_dims = base.weight.shape[1]
        out_dims = base.weight.shape[0]

        # A: [rank, in_dims], initialized from N(0, 1/rank)
        self.lora_a = mx.random.normal((rank, in_dims)) * (1.0 / rank)
        # B: [out_dims, rank], initialized to zeros so LoRA starts as identity
        self.lora_b = mx.zeros((out_dims, rank))

        # Freeze the base linear layer
        self.base.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        # Base forward
        y = self.base(x)
        # LoRA delta: x @ A^T @ B^T * scale
        y = y + (x @ self.lora_a.T @ self.lora_b.T) * self.scale
        return y


def linear_to_lora_layers(
    model: HybridLM,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    target_modules: Optional[Set[str]] = None,
) -> None:
    """Replace target Linear layers in the model with LoRA layers in-place.

    The model should be frozen before calling this function. Only the
    LoRA parameters (A and B matrices) will be trainable.

    Args:
        model: HybridLM to modify.
        lora_rank: Rank of the low-rank matrices.
        lora_alpha: Scaling factor (effective scale = alpha / rank).
        target_modules: Set of module name suffixes to target.
            Defaults to q_proj, v_proj, o_proj, in_proj, out_proj.
    """
    targets = target_modules or DEFAULT_TARGET_MODULES

    def _replace_in_module(parent):
        """Recursively walk the module tree and replace matching Linear layers."""
        children = parent.children()
        for attr_name, child in children.items():
            if isinstance(child, nn.Linear) and attr_name in targets:
                setattr(parent, attr_name, LoRALinear(child, rank=lora_rank, alpha=lora_alpha))
            elif isinstance(child, nn.Module):
                _replace_in_module(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, nn.Module):
                        _replace_in_module(item)

    _replace_in_module(model)


def save_lora_weights(model: HybridLM, path: str) -> None:
    """Save only the LoRA adapter weights.

    Args:
        model: Model with LoRA layers.
        path: Output file path (.safetensors or .npz).
    """
    lora_weights = {}
    for name, param in nn.utils.tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            lora_weights[name] = param

    if not lora_weights:
        raise ValueError("No LoRA weights found. Did you call linear_to_lora_layers?")

    if path.endswith(".safetensors"):
        mx.save_safetensors(path, lora_weights)
    else:
        mx.savez(path, **lora_weights)


def load_lora_weights(model: HybridLM, path: str) -> None:
    """Load LoRA adapter weights into a model that already has LoRA layers.

    Args:
        model: Model with LoRA layers (from linear_to_lora_layers).
        path: Path to saved LoRA weights (.safetensors or .npz).
    """
    weights = dict(mx.load(path))

    model.load_weights(list(weights.items()), strict=False)


def merge_lora_weights(model: HybridLM) -> None:
    """Merge LoRA weights back into the base Linear layers.

    After merging, the model behaves as a standard model without LoRA overhead.
    LoRALinear layers are replaced with plain nn.Linear layers.

    Args:
        model: Model with LoRA layers to merge.
    """

    def _merge_in_module(parent):
        children = parent.children()
        for attr_name, child in children.items():
            if isinstance(child, LoRALinear):
                base = child.base
                delta = (child.lora_b @ child.lora_a) * child.scale
                base.weight = base.weight + delta
                base.unfreeze()
                setattr(parent, attr_name, base)
            elif isinstance(child, nn.Module):
                _merge_in_module(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, nn.Module):
                        _merge_in_module(item)

    _merge_in_module(model)
