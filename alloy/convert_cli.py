"""CLI for converting HuggingFace models to Alloy format.

Usage:
    python -m alloy.convert_cli --model Zyphra/Zamba2-1.2B
    python -m alloy.convert_cli --model Zyphra/Zamba2-1.2B --quantize 4
    python -m alloy.convert_cli --model ./local-model-dir --output models/my-model
"""

import argparse
import os
import sys
import time

import mlx.core as mx


def download_model(model_id: str, cache_dir: str = None) -> str:
    """Download a model from HuggingFace Hub.

    Args:
        model_id: HF model ID (e.g., 'Zyphra/Zamba2-1.2B') or local path.
        cache_dir: Cache directory. Default: ~/.cache/alloy/models/

    Returns:
        Local path to the model directory.
    """
    # If it's a local path, return as-is
    if os.path.isdir(model_id):
        return model_id

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "alloy", "models")

    model_name = model_id.replace("/", "--")
    local_dir = os.path.join(cache_dir, model_name)

    # Check if already downloaded
    if os.path.isdir(local_dir) and any(
        f.endswith(".safetensors") for f in os.listdir(local_dir)
    ):
        print(f"Model already cached: {local_dir}")
        return local_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub required for downloading.")
        print("  pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading {model_id} to {local_dir}...")
    snapshot_download(
        model_id,
        local_dir=local_dir,
        ignore_patterns=["*.bin", "*.onnx", "*.pt", "training_args*", "optimizer*"],
    )
    print(f"Downloaded: {local_dir}")
    return local_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to Alloy format"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HF model ID (e.g., Zyphra/Zamba2-1.2B) or local path",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for converted weights (default: prints info only)",
    )
    parser.add_argument(
        "--quantize", type=int, choices=[4, 8], default=None,
        help="Quantize to 4 or 8 bits after conversion",
    )
    parser.add_argument(
        "--bf16", action="store_true",
        help="Convert to bfloat16 after loading",
    )
    args = parser.parse_args()

    # Step 1: Download if needed
    t0 = time.time()
    model_dir = download_model(args.model)

    # Step 2: Convert
    from alloy.convert import convert_from_hf, load_hf_config
    from alloy.models.hybrid_model import HybridLM
    from mlx.utils import tree_flatten

    hf_cfg = load_hf_config(model_dir)
    print(f"Model type: {hf_cfg.get('model_type', 'unknown')}")

    print("Converting weights...")
    config, weights = convert_from_hf(model_dir)
    print(f"Config: {config.n_layers} layers, d_model={config.d_model}")
    print(f"  Attention layers: {config.attn_layer_indices}")
    print(f"  Weights: {len(weights)} tensors")

    # Step 3: Build model and load
    print("Building model...")
    model = HybridLM(config)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    nparams = sum(p.size for _, p in tree_flatten(model.parameters()))
    mem = sum(p.nbytes for _, p in tree_flatten(model.parameters()))
    print(f"Parameters: {nparams / 1e9:.2f}B ({mem / 1e9:.2f} GB)")

    # Step 4: Optional bf16
    if args.bf16:
        print("Converting to bfloat16...")
        model.to_bfloat16()
        mem = sum(p.nbytes for _, p in tree_flatten(model.parameters()))
        print(f"  Memory: {mem / 1e9:.2f} GB")

    # Step 5: Optional quantization
    if args.quantize:
        print(f"Quantizing to {args.quantize}-bit...")
        model.quantize(bits=args.quantize)
        mem = sum(p.nbytes for _, p in tree_flatten(model.parameters()))
        print(f"  Memory: {mem / 1e9:.2f} GB")

    # Step 6: Save if output specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        out_path = os.path.join(args.output, "model.safetensors")
        model.save_weights(out_path)

        # Save config
        import json
        config_dict = {k: v for k, v in config.__dict__.items()}
        with open(os.path.join(args.output, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        print(f"Saved to: {args.output}")
        print(f"  {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Quick test
    print("\nQuick generation test:")
    try:
        from transformers import AutoTokenizer
        from alloy.generate import stream_generate

        tok = AutoTokenizer.from_pretrained(model_dir)
        prompt = "The capital of France is"
        ids = tok.encode(prompt)
        tokens = list(ids)
        for token in stream_generate(model, mx.array([ids]), max_tokens=30, temperature=0.0):
            tokens.append(token.item())
            if tokens[-1] == tok.eos_token_id:
                break
        print(f"  {tok.decode(tokens)}")
    except Exception as e:
        print(f"  (skipped: {e})")


if __name__ == "__main__":
    main()
