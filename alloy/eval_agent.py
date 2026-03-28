"""Agent / tool-calling evaluation suite for Alloy models.

Three evaluation layers:
  Layer 1 (Format): JSON parsability, schema compliance, tool name validity
  Layer 2 (Accuracy): BFCL AST matching (simple, multiple, parallel, irrelevance)
  Layer 3 (Self-built): End-to-end on Alloy's built-in tools

Usage:
    # Run all evals on base model
    python -m alloy.eval_agent --model ~/.cache/alloy/models/Nemotron-H-8B-Reasoning-128K --quantize 4

    # Run with LoRA adapter
    python -m alloy.eval_agent --model ~/.cache/alloy/models/Nemotron-H-8B-Reasoning-128K \
        --lora checkpoints/agent-lora-v4/final.npz --quantize 4

    # Only run BFCL subset
    python -m alloy.eval_agent --model ... --categories simple_python irrelevance --max-samples 50
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


# ============================================================
# BFCL data loading
# ============================================================

BFCL_DATA_DIR = Path(__file__).parent / ".." / "data" / "bfcl"

def _find_bfcl_data_dir():
    """Find BFCL data directory (bundled with bfcl_eval package)."""
    try:
        from bfcl_eval.eval_checker.eval_runner import PROMPT_PATH
        return Path(PROMPT_PATH)
    except ImportError:
        return None


def load_bfcl_category(category: str, max_samples: int = None):
    """Load BFCL test data and ground truth for a category.

    Returns list of (id, question_messages, functions, ground_truth) tuples.
    """
    data_dir = _find_bfcl_data_dir()
    if data_dir is None:
        print("  bfcl_eval not installed, skipping BFCL eval")
        return []

    from bfcl_eval.constants.category_mapping import VERSION_PREFIX
    prefix = VERSION_PREFIX

    prompt_file = data_dir / f"{prefix}_{category}.json"
    answer_file = data_dir / "possible_answer" / f"{prefix}_{category}.json"

    if not prompt_file.exists():
        print(f"  BFCL data not found: {prompt_file}")
        return []

    with open(prompt_file) as f:
        prompts = [json.loads(line) for line in f]

    answers = {}
    if answer_file.exists():
        with open(answer_file) as f:
            for line in f:
                item = json.loads(line)
                answers[item["id"]] = item["ground_truth"]

    examples = []
    for item in prompts:
        eid = item["id"]
        # question is a list of conversation turns, each a list of messages
        question = item["question"][0]  # first turn
        functions = item["function"]
        gt = answers.get(eid, None)
        examples.append((eid, question, functions, gt))

    if max_samples and len(examples) > max_samples:
        examples = examples[:max_samples]

    return examples


# ============================================================
# Model output generation
# ============================================================

def build_tool_prompt(user_messages, functions):
    """Build a prompt that asks the model to call functions.

    Adapts BFCL function descriptions to the format our model was trained on.
    """
    # Convert BFCL function format to our tool format
    tools = []
    for func in functions:
        tool = {
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": {},
        }
        params = func.get("parameters", {})
        if "properties" in params:
            for pname, pinfo in params["properties"].items():
                tool["parameters"][pname] = {
                    "type": pinfo.get("type", "string"),
                    "description": pinfo.get("description", ""),
                }
        tools.append(tool)

    tools_json = json.dumps(tools, separators=(',', ':'))
    system_prompt = f'Use EXACT function names. Respond with {{"tool_calls":[...]}} or plain text.\n{tools_json}'

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(user_messages)
    return messages


def generate_response(model, tokenizer, messages, max_tokens=200, temperature=0.0):
    """Generate model response given chat messages."""
    from alloy.generate import stream_generate

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer.encode(formatted)
    tokens = list(ids)

    for token in stream_generate(
        model, mx.array([ids]), max_tokens=max_tokens, temperature=temperature
    ):
        t = token.item()
        tokens.append(t)
        if t == tokenizer.eos_token_id:
            break

    output = tokenizer.decode(tokens[len(ids):]).strip()
    # Clean trailing noise
    for end in ["<SPECIAL", "\n\n\n", "<|", "</s>"]:
        if end in output:
            output = output[: output.index(end)]

    # Fix repetition artifacts: truncate at first complete JSON object
    if '{"tool_calls"' in output:
        try:
            for i in range(len(output), 0, -1):
                try:
                    json.loads(output[:i])
                    output = output[:i]
                    break
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

    return output.strip()


# ============================================================
# Output parsing
# ============================================================

def parse_model_output_to_bfcl(raw_output: str, functions: list):
    """Parse model's raw text output into BFCL format: [{"func_name": {"arg": val}}].

    Returns (parsed_list, parse_success, is_tool_call).
    """
    # Try to extract JSON from the output
    clean = raw_output.replace('\\"', '"').replace("\\'", "'")

    # Strategy 1: Find tool_calls JSON
    tc_match = re.search(r'"tool_calls"\s*:\s*\[(.+?)\]', clean, re.DOTALL)
    if tc_match:
        try:
            calls_str = "[" + tc_match.group(1) + "]"
            calls = json.loads(calls_str)
            result = []
            for call in calls:
                name = call.get("name", "")
                args = call.get("arguments", {})
                result.append({name: args})
            return result, True, True
        except json.JSONDecodeError:
            pass

    # Strategy 2: Find {"name": "...", "arguments": {...}} pattern(s)
    pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})'
    matches = re.findall(pattern, clean)
    if matches:
        result = []
        for name, args_str in matches:
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                try:
                    args = json.loads(args_str + "}")
                except json.JSONDecodeError:
                    args = {}
            result.append({name: args})
        return result, True, True

    # Strategy 3: Find func_name(arg=val, ...) Python-style calls
    func_names = {f["name"] for f in functions}
    py_pattern = r'(\w+)\(([^)]*)\)'
    py_matches = re.findall(py_pattern, clean)
    for fname, args_str in py_matches:
        if fname in func_names:
            # Parse keyword arguments
            args = {}
            for arg_match in re.finditer(r'(\w+)\s*=\s*(".*?"|\'.*?\'|[\d.]+|\w+)', args_str):
                key = arg_match.group(1)
                val = arg_match.group(2).strip("'\"")
                # Try to convert to number
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                args[key] = val
            return [{fname: args}], True, True

    # No tool call found — might be a direct text response (valid for irrelevance)
    return [], True, False


# ============================================================
# Layer 1: Format metrics
# ============================================================

def eval_format(raw_outputs: list, functions_list: list):
    """Evaluate Layer 1 format metrics.

    Returns dict with json_parse_rate, schema_compliance, tool_name_valid.
    """
    n = len(raw_outputs)
    json_parseable = 0
    schema_ok = 0
    tool_name_ok = 0

    for raw, funcs in zip(raw_outputs, functions_list):
        parsed, success, is_tool = parse_model_output_to_bfcl(raw, funcs)
        if success:
            json_parseable += 1

        if is_tool and parsed:
            # Check schema: each item should be {name: {args}}
            all_schema = True
            all_names = True
            valid_names = {f["name"] for f in funcs}
            for call in parsed:
                if not isinstance(call, dict) or len(call) != 1:
                    all_schema = False
                    break
                name = list(call.keys())[0]
                if not isinstance(call[name], dict):
                    all_schema = False
                if name not in valid_names:
                    all_names = False

            if all_schema:
                schema_ok += 1
            if all_names:
                tool_name_ok += 1
        elif not is_tool:
            # Direct response is valid format
            schema_ok += 1
            tool_name_ok += 1

    return {
        "json_parse_rate": json_parseable / n if n else 0,
        "schema_compliance": schema_ok / n if n else 0,
        "tool_name_valid": tool_name_ok / n if n else 0,
    }


# ============================================================
# Layer 2: BFCL AST accuracy
# ============================================================

def eval_bfcl_category(
    model, tokenizer, category: str, max_samples: int = None, verbose: bool = False
):
    """Run BFCL evaluation on a single category.

    Returns dict with accuracy, total, correct, errors, raw_outputs, and format_metrics.
    """
    examples = load_bfcl_category(category, max_samples)
    if not examples:
        return {"accuracy": 0, "total": 0, "correct": 0, "error": "no data"}

    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.constants.enums import Language

    is_irrelevance = "irrelevance" in category
    language = Language.PYTHON
    if "java" in category and "javascript" not in category:
        language = Language.JAVA
    elif "javascript" in category:
        language = Language.JAVASCRIPT

    correct = 0
    total = 0
    errors = []
    raw_outputs = []
    all_functions = []
    latencies = []

    for eid, question, functions, ground_truth in examples:
        messages = build_tool_prompt(question, functions)

        t0 = time.time()
        raw = generate_response(model, tokenizer, messages)
        latency = time.time() - t0
        latencies.append(latency)

        raw_outputs.append(raw)
        all_functions.append(functions)
        total += 1

        if is_irrelevance:
            # For irrelevance: model should NOT call any function
            _, _, is_tool = parse_model_output_to_bfcl(raw, functions)
            if not is_tool:
                correct += 1
            elif verbose:
                errors.append({"id": eid, "error": "called tool on irrelevant query", "output": raw[:200]})
        else:
            if ground_truth is None:
                continue

            parsed, _, is_tool = parse_model_output_to_bfcl(raw, functions)

            if not is_tool or not parsed:
                errors.append({"id": eid, "error": "no tool call generated", "output": raw[:200]})
                continue

            try:
                result = ast_checker(
                    functions, parsed, ground_truth, language, category, "alloy"
                )
                if result["valid"]:
                    correct += 1
                elif verbose:
                    errors.append({
                        "id": eid,
                        "error": result.get("error_type", "unknown"),
                        "detail": result.get("error", []),
                        "output": raw[:200],
                    })
            except Exception as e:
                errors.append({"id": eid, "error": f"checker exception: {e}", "output": raw[:200]})

        if total % 20 == 0:
            print(f"    [{category}] {total}/{len(examples)} done, "
                  f"acc={correct/total:.1%}, avg_latency={sum(latencies)/len(latencies):.1f}s")

    format_metrics = eval_format(raw_outputs, all_functions)

    return {
        "category": category,
        "accuracy": correct / total if total else 0,
        "total": total,
        "correct": correct,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "format": format_metrics,
        "errors": errors[:10],  # keep first 10 for debugging
    }


# ============================================================
# Layer 3: Self-built eval (Alloy built-in tools)
# ============================================================

SELF_EVAL_CASES = [
    # (query, expected_tool, expected_args_subset, description)
    # Tool selection tests
    ("What is 123 * 456?", "calculate", {"expression": "123 * 456"}, "basic multiplication"),
    ("Calculate 99 + 1", "calculate", {"expression": "99 + 1"}, "basic addition"),
    ("What's the square root of 144?", "calculate", {}, "sqrt - any calc call is ok"),
    ("What time is it?", "get_time", {}, "time query"),
    ("What's the current date?", "get_time", {}, "date query"),
    ("What's the weather in Tokyo?", "get_weather", {"city": "Tokyo"}, "weather query"),
    ("How's the weather in London today?", "get_weather", {"city": "London"}, "weather with city"),
    ("Search for machine learning tutorials", "web_search", {}, "search query"),
    ("Find information about quantum computing", "web_search", {}, "search query 2"),
    ("What's the temperature in Paris in Fahrenheit?", "get_weather", {"city": "Paris"}, "weather with unit"),
    # No-tool tests (should NOT call any tool)
    ("What is the capital of France?", None, {}, "general knowledge - no tool"),
    ("Explain what an API is", None, {}, "explanation - no tool"),
    ("Hello, how are you?", None, {}, "greeting - no tool"),
    ("What is Python used for?", None, {}, "general question - no tool"),
    ("Tell me a joke", None, {}, "creative - no tool"),
    # Edge cases
    ("What is 2+2 and what's the weather in NYC?", "calculate", {}, "multi-intent - at least one tool"),
]


def eval_self_built(model, tokenizer, verbose=False):
    """Run self-built eval on Alloy's 4 built-in tools."""
    from alloy.agent import BUILTIN_TOOLS, build_system_prompt, parse_tool_call

    system_prompt = build_system_prompt(BUILTIN_TOOLS)
    tool_names = set(BUILTIN_TOOLS.keys())

    correct_tool = 0
    correct_args = 0
    correct_no_tool = 0
    total_tool = 0
    total_no_tool = 0
    errors = []
    latencies = []

    for query, expected_tool, expected_args, desc in SELF_EVAL_CASES:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        t0 = time.time()
        raw = generate_response(model, tokenizer, messages)
        latency = time.time() - t0
        latencies.append(latency)

        tool_call = parse_tool_call(raw)

        if expected_tool is None:
            # Should NOT call a tool
            total_no_tool += 1
            if tool_call is None:
                correct_no_tool += 1
            elif verbose:
                errors.append({"query": query, "desc": desc, "error": "false positive tool call",
                              "got": tool_call, "output": raw[:200]})
        else:
            # Should call the right tool
            total_tool += 1
            if tool_call and tool_call.get("name") == expected_tool:
                correct_tool += 1
                # Check args if specified
                if expected_args:
                    args = tool_call.get("arguments", {})
                    args_match = all(
                        str(args.get(k, "")).lower() == str(v).lower()
                        for k, v in expected_args.items()
                    )
                    if args_match:
                        correct_args += 1
                    elif verbose:
                        errors.append({"query": query, "desc": desc, "error": "wrong args",
                                      "expected": expected_args, "got": args})
                else:
                    correct_args += 1  # no specific args required
            elif verbose:
                errors.append({"query": query, "desc": desc, "error": "wrong tool or no call",
                              "expected": expected_tool,
                              "got": tool_call.get("name") if tool_call else None,
                              "output": raw[:200]})

    return {
        "tool_select_acc": correct_tool / total_tool if total_tool else 0,
        "arg_match_acc": correct_args / total_tool if total_tool else 0,
        "no_tool_precision": correct_no_tool / total_no_tool if total_no_tool else 0,
        "total_tool_cases": total_tool,
        "total_no_tool_cases": total_no_tool,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "errors": errors[:10],
    }


# ============================================================
# Main
# ============================================================

BFCL_CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Alloy agent/tool-calling")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF ID")
    parser.add_argument("--lora", type=str, default=None, help="LoRA weights path")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--quantize", type=int, choices=[4, 8], default=None)
    parser.add_argument("--categories", nargs="+", default=BFCL_CATEGORIES,
                        help="BFCL categories to evaluate")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max samples per BFCL category (default 50 for speed)")
    parser.add_argument("--skip-bfcl", action="store_true", help="Skip BFCL eval")
    parser.add_argument("--skip-self", action="store_true", help="Skip self-built eval")
    parser.add_argument("--verbose", action="store_true", help="Show error details")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    # Load model
    from alloy.convert import load_pretrained
    from alloy.convert_cli import download_model

    print(f"Loading model: {args.model}")
    model_dir = download_model(args.model)
    model = load_pretrained(model_dir)

    model_label = "base"
    if args.lora:
        from alloy.lora import linear_to_lora_layers, load_lora_weights
        print(f"Applying LoRA (rank={args.lora_rank})...")
        model.freeze()
        linear_to_lora_layers(model, lora_rank=args.lora_rank)
        if args.quantize:
            nn.quantize(model, bits=args.quantize,
                        class_predicate=lambda p, m: isinstance(m, nn.Linear) and "lora" not in p)
        print(f"Loading LoRA weights: {args.lora}")
        load_lora_weights(model, args.lora)
        model_label = Path(args.lora).parent.name
    elif args.quantize:
        nn.quantize(model, bits=args.quantize)

    mx.eval(model.parameters())

    from mlx.utils import tree_flatten
    mem = sum(p.nbytes for _, p in tree_flatten(model.parameters()))
    print(f"Model ready: {mem / 1e9:.1f} GB\n")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    results = {"model": args.model, "label": model_label, "lora": args.lora}

    # Layer 2: BFCL eval
    if not args.skip_bfcl:
        print("=" * 60)
        print("Layer 2: BFCL AST Evaluation")
        print("=" * 60)

        bfcl_results = {}
        for cat in args.categories:
            print(f"\n  [{cat}] Running (max {args.max_samples} samples)...")
            r = eval_bfcl_category(model, tokenizer, cat,
                                   max_samples=args.max_samples, verbose=args.verbose)
            bfcl_results[cat] = r
            print(f"  [{cat}] Accuracy: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
            print(f"  [{cat}] Format: parse={r['format']['json_parse_rate']:.1%} "
                  f"schema={r['format']['schema_compliance']:.1%} "
                  f"names={r['format']['tool_name_valid']:.1%}")
            if args.verbose and r.get("errors"):
                for err in r["errors"][:3]:
                    print(f"    ERR: {err}")

        results["bfcl"] = bfcl_results

        # Summary
        print("\n" + "-" * 60)
        print(f"{'Category':<25} {'Accuracy':>10} {'Parse':>8} {'Schema':>8} {'Latency':>8}")
        print("-" * 60)
        total_correct = 0
        total_total = 0
        for cat in args.categories:
            r = bfcl_results.get(cat, {})
            if not r or r.get("error"):
                continue
            total_correct += r["correct"]
            total_total += r["total"]
            print(f"{cat:<25} {r['accuracy']:>9.1%} {r['format']['json_parse_rate']:>7.1%} "
                  f"{r['format']['schema_compliance']:>7.1%} {r['avg_latency']:>7.1f}s")
        if total_total:
            print("-" * 60)
            print(f"{'OVERALL':<25} {total_correct/total_total:>9.1%} {'':>8} {'':>8}")

    # Layer 3: Self-built eval
    if not args.skip_self:
        print("\n" + "=" * 60)
        print("Layer 3: Self-built Tool Eval (Alloy built-in tools)")
        print("=" * 60)

        self_results = eval_self_built(model, tokenizer, verbose=args.verbose)
        results["self_eval"] = self_results

        print(f"\n  Tool Selection Acc:  {self_results['tool_select_acc']:.1%}")
        print(f"  Arg Match Acc:      {self_results['arg_match_acc']:.1%}")
        print(f"  No-Tool Precision:  {self_results['no_tool_precision']:.1%}")
        print(f"  Avg Latency:        {self_results['avg_latency']:.1f}s")

        if args.verbose and self_results.get("errors"):
            print("\n  Errors:")
            for err in self_results["errors"]:
                print(f"    {err['desc']}: {err['error']}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Config: {'LoRA=' + args.lora if args.lora else 'base'}, "
          f"{'Q' + str(args.quantize) if args.quantize else 'fp32'}")
    if "bfcl" in results:
        bfcl = results["bfcl"]
        tc = sum(r.get("correct", 0) for r in bfcl.values())
        tt = sum(r.get("total", 0) for r in bfcl.values())
        print(f"BFCL Overall: {tc/tt:.1%} ({tc}/{tt})" if tt else "BFCL: no data")
    if "self_eval" in results:
        se = results["self_eval"]
        print(f"Self-eval: tool_select={se['tool_select_acc']:.1%}, "
              f"args={se['arg_match_acc']:.1%}, "
              f"no_tool={se['no_tool_precision']:.1%}")


if __name__ == "__main__":
    main()
