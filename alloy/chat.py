"""Interactive chat CLI for Alloy models.

Usage:
    python -m alloy.chat --model path/to/Zamba2-1.2B
    python -m alloy.chat --model path/to/Zamba2-1.2B --quantize 4
"""

import argparse
import sys
import time

import mlx.core as mx

from alloy.convert import load_pretrained
from alloy.generate import stream_generate


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with Alloy models")
    parser.add_argument("--model", type=str, required=True, help="Path to HuggingFace model directory")
    parser.add_argument("--quantize", type=int, choices=[4, 8], default=None, help="Quantize to 4 or 8 bits")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    args = parser.parse_args()

    # Load model (supports HF model IDs with auto-download)
    from alloy.convert_cli import download_model
    print(f"Loading model: {args.model}...")
    t0 = time.time()
    model_dir = download_model(args.model)
    model = load_pretrained(model_dir)

    if args.quantize:
        print(f"Quantizing to {args.quantize}-bit...")
        model.quantize(bits=args.quantize)

    from mlx.utils import tree_flatten
    mem = sum(p.nbytes for _, p in tree_flatten(model.parameters()))
    print(f"Ready ({mem / 1e9:.1f} GB, {time.time() - t0:.1f}s)")

    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except ImportError:
        print("Error: `transformers` package required for tokenizer. Install with: pip install transformers")
        sys.exit(1)

    print(f"Model: {args.model}")
    if args.quantize:
        print(f"Quantization: {args.quantize}-bit")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}, Max tokens: {args.max_tokens}")
    print("Type your prompt and press Enter. Ctrl+C to exit.\n")

    while True:
        try:
            prompt = input("> ")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not prompt.strip():
            continue

        # Apply chat template if available (for instruct models)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            ids = tokenizer.encode(formatted)
        else:
            ids = tokenizer.encode(prompt)
        prompt_ids = mx.array([ids])
        tokens = list(ids)

        t0 = time.time()
        n_generated = 0

        for token in stream_generate(
            model, prompt_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ):
            t = token.item()
            if t == tokenizer.eos_token_id:
                break
            tokens.append(t)
            n_generated += 1
            # Stream: print only the new delta text
            new_text = tokenizer.decode(tokens[len(ids):])
            old_text = tokenizer.decode(tokens[len(ids):-1]) if n_generated > 1 else ""
            delta = new_text[len(old_text):]
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()

        elapsed = time.time() - t0
        speed = n_generated / elapsed if elapsed > 0 else 0
        print(f"\n[{n_generated} tokens, {speed:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
