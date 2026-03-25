"""Lightweight evaluation suite for Alloy models.

Runs standard benchmarks using log-likelihood scoring (no generation needed).

Usage:
    python -m alloy.eval --model Zyphra/Zamba2-1.2B
    python -m alloy.eval --model Zyphra/Zamba2-1.2B --quantize 4
"""

import argparse
import json
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn

# Built-in benchmark questions (subset for quick evaluation)
MMLU_SAMPLE = [
    # STEM
    {"q": "What is the SI unit of electric current?", "choices": ["Volt", "Ampere", "Ohm", "Watt"], "answer": 1},
    {"q": "Which planet is known as the Red Planet?", "choices": ["Venus", "Jupiter", "Mars", "Saturn"], "answer": 2},
    {"q": "What is the chemical symbol for gold?", "choices": ["Go", "Gd", "Au", "Ag"], "answer": 2},
    {"q": "DNA stands for:", "choices": ["Deoxyribonucleic acid", "Dinitrogen acid", "Deoxynitric acid", "Dinucleotide acid"], "answer": 0},
    {"q": "The speed of light is approximately:", "choices": ["300 km/s", "300,000 km/s", "3,000 km/s", "30,000 km/s"], "answer": 1},
    {"q": "What is the powerhouse of the cell?", "choices": ["Nucleus", "Ribosome", "Mitochondria", "Golgi body"], "answer": 2},
    {"q": "H2O is the chemical formula for:", "choices": ["Hydrogen peroxide", "Water", "Hydrochloric acid", "Helium oxide"], "answer": 1},
    {"q": "What force keeps planets in orbit around the sun?", "choices": ["Electromagnetic", "Gravity", "Nuclear", "Friction"], "answer": 1},
    # Humanities
    {"q": "Who wrote 'Romeo and Juliet'?", "choices": ["Dickens", "Shakespeare", "Austen", "Hemingway"], "answer": 1},
    {"q": "The Renaissance began in which country?", "choices": ["France", "England", "Italy", "Spain"], "answer": 2},
    {"q": "Who painted the Mona Lisa?", "choices": ["Michelangelo", "Raphael", "Da Vinci", "Botticelli"], "answer": 2},
    {"q": "The French Revolution began in which year?", "choices": ["1776", "1789", "1804", "1815"], "answer": 1},
    # Social science
    {"q": "What is the largest economy in the world by GDP?", "choices": ["China", "USA", "Japan", "Germany"], "answer": 1},
    {"q": "The United Nations was founded in:", "choices": ["1918", "1939", "1945", "1950"], "answer": 2},
    {"q": "Which continent has the most countries?", "choices": ["Asia", "Europe", "Africa", "South America"], "answer": 2},
    # Logic
    {"q": "If all dogs are animals, and all animals are living things, then:", "choices": ["All living things are dogs", "All dogs are living things", "Some animals are not dogs", "Dogs are not animals"], "answer": 1},
    {"q": "What comes next: 2, 4, 8, 16, ?", "choices": ["24", "30", "32", "36"], "answer": 2},
    {"q": "If it is raining, the ground is wet. The ground is not wet. Therefore:", "choices": ["It is raining", "It is not raining", "The ground is dry sometimes", "Cannot determine"], "answer": 1},
    {"q": "All squares are rectangles. All rectangles have four sides. Therefore:", "choices": ["All four-sided shapes are squares", "All squares have four sides", "Some rectangles are not squares", "Squares have five sides"], "answer": 1},
    {"q": "A train travels 60 km in 1 hour. How far does it travel in 2.5 hours?", "choices": ["120 km", "130 km", "150 km", "180 km"], "answer": 2},
]

HELLASWAG_SAMPLE = [
    {"ctx": "A man is standing in front of a grill. He", "choices": [
        "lights the grill and begins cooking food.",
        "starts playing the piano.",
        "begins swimming in a pool.",
        "reads a book about astronomy."
    ], "answer": 0},
    {"ctx": "A woman picks up a phone and dials a number. She", "choices": [
        "starts to fly through the air.",
        "waits for someone to answer.",
        "begins painting a picture.",
        "plants a tree in the ground."
    ], "answer": 1},
    {"ctx": "The chef adds salt and pepper to the dish. He then", "choices": [
        "puts the dish in the oven.",
        "throws the dish on the floor.",
        "starts mowing the lawn.",
        "goes skydiving."
    ], "answer": 0},
    {"ctx": "The students open their textbooks to chapter 5. The teacher", "choices": [
        "begins the football game.",
        "starts explaining the lesson.",
        "goes swimming.",
        "orders a pizza."
    ], "answer": 1},
    {"ctx": "It starts to rain heavily. People on the street", "choices": [
        "open their umbrellas and seek shelter.",
        "start sunbathing.",
        "begin building a snowman.",
        "plant flowers in the rain."
    ], "answer": 0},
]


def score_choice(model, tokenizer, prompt, choice):
    """Score a choice by computing average log-likelihood."""
    full = prompt + " " + choice
    ids = tokenizer.encode(full)
    prompt_len = len(tokenizer.encode(prompt))

    x = mx.array([ids])
    logits = model(x)
    mx.eval(logits)

    # Compute log-likelihood of the choice tokens
    log_probs = nn.losses.cross_entropy(
        logits[0, prompt_len - 1:-1],
        mx.array(ids[prompt_len:]),
        reduction="none",
    )
    mx.eval(log_probs)
    return -log_probs.mean().item()  # higher is better


def run_mmlu(model, tokenizer):
    """Run MMLU-style multiple choice evaluation."""
    correct = 0
    total = len(MMLU_SAMPLE)
    labels = ["A", "B", "C", "D"]

    for item in MMLU_SAMPLE:
        prompt = f"Question: {item['q']}\n"
        for i, c in enumerate(item["choices"]):
            prompt += f"{labels[i]}. {c}\n"
        prompt += "Answer:"

        scores = []
        for i, c in enumerate(item["choices"]):
            s = score_choice(model, tokenizer, prompt, f" {labels[i]}")
            scores.append(s)

        pred = scores.index(max(scores))
        if pred == item["answer"]:
            correct += 1

    return correct / total


def run_hellaswag(model, tokenizer):
    """Run HellaSwag-style commonsense evaluation."""
    correct = 0
    total = len(HELLASWAG_SAMPLE)

    for item in HELLASWAG_SAMPLE:
        scores = []
        for choice in item["choices"]:
            s = score_choice(model, tokenizer, item["ctx"], choice)
            scores.append(s)

        pred = scores.index(max(scores))
        if pred == item["answer"]:
            correct += 1

    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Evaluate Alloy models")
    parser.add_argument("--model", type=str, required=True, help="HF model ID or local path")
    parser.add_argument("--quantize", type=int, choices=[4, 8], default=None)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    from alloy.convert_cli import download_model
    from alloy.convert import load_pretrained

    print(f"Loading model: {args.model}")
    model_dir = download_model(args.model)
    model = load_pretrained(model_dir)

    if args.bf16:
        model.to_bfloat16()
    if args.quantize:
        model.quantize(bits=args.quantize)

    from mlx.utils import tree_flatten
    mem = sum(p.nbytes for _, p in tree_flatten(model.parameters()))
    print(f"Model ready: {mem / 1e9:.1f} GB")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except ImportError:
        print("Error: transformers required")
        sys.exit(1)

    print()

    # Run benchmarks
    print("Running MMLU (20 questions)...")
    t0 = time.time()
    mmlu = run_mmlu(model, tokenizer)
    print(f"  MMLU accuracy: {mmlu * 100:.1f}% ({time.time() - t0:.1f}s)")

    print("Running HellaSwag (5 questions)...")
    t0 = time.time()
    hellaswag = run_hellaswag(model, tokenizer)
    print(f"  HellaSwag accuracy: {hellaswag * 100:.1f}% ({time.time() - t0:.1f}s)")

    print()
    print("=" * 40)
    print(f"  MMLU:      {mmlu * 100:5.1f}%  (random=25%)")
    print(f"  HellaSwag: {hellaswag * 100:5.1f}%  (random=25%)")
    print("=" * 40)
    print()
    print("Note: This is a quick eval with small sample sizes.")
    print("For full benchmarks, use lm-evaluation-harness.")


if __name__ == "__main__":
    main()
