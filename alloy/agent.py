"""Agent execution loop: model generates tool calls, tools execute, results feed back.

Usage:
    python -m alloy.agent --model ~/.cache/alloy/models/Nemotron-H-8B-Reasoning-128K \
        --lora checkpoints/agent-lora-v3/final.npz --quantize 4
"""

import argparse
import json
import re
import sys
import time

import mlx.core as mx
import mlx.nn as nn

from alloy.convert import load_pretrained
from alloy.generate import stream_generate
from alloy.lora import linear_to_lora_layers, load_lora_weights


# ===========================================================================
# Built-in tools
# ===========================================================================

def tool_calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        # Only allow safe math operations
        allowed = set("0123456789+-*/.() %")
        expr = expression.replace("^", "**").replace("sqrt", "math.sqrt")
        if not all(c in allowed or c.isalpha() for c in expr):
            return json.dumps({"error": f"Unsafe expression: {expression}"})
        import math
        result = eval(expr, {"__builtins__": {}, "math": math})
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_get_time() -> str:
    """Get current date and time."""
    from datetime import datetime
    now = datetime.now()
    return json.dumps({"datetime": now.isoformat(), "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S")})


def tool_web_search(query: str, num_results: int = 3) -> str:
    """Simulated web search (placeholder)."""
    return json.dumps({"results": [f"[Simulated result for: {query}]"], "note": "Web search not connected. Connect a real search API."})


def tool_get_weather(city: str, unit: str = "celsius") -> str:
    """Simulated weather (placeholder)."""
    return json.dumps({"city": city, "temperature": 22, "unit": unit, "condition": "sunny", "note": "Simulated. Connect a real weather API."})


# Tool registry
BUILTIN_TOOLS = {
    "calculate": {
        "fn": tool_calculate,
        "description": "Evaluate a mathematical expression",
        "parameters": {"expression": {"type": "string", "description": "Math expression to evaluate"}},
    },
    "get_time": {
        "fn": tool_get_time,
        "description": "Get current date and time",
        "parameters": {},
    },
    "web_search": {
        "fn": tool_web_search,
        "description": "Search the web for information",
        "parameters": {"query": {"type": "string"}, "num_results": {"type": "integer"}},
    },
    "get_weather": {
        "fn": tool_get_weather,
        "description": "Get current weather for a city",
        "parameters": {"city": {"type": "string"}, "unit": {"type": "string"}},
    },
}


# ===========================================================================
# Tool call parsing
# ===========================================================================

def parse_tool_call(text: str):
    """Extract tool call from model output, handling escaped JSON and noise."""
    # Unescape the string first (model often outputs escaped JSON)
    clean = text.replace('\\"', '"').replace("\\'", "'")

    # Try to find [{"name": "tool", "arguments": {...}}]
    match = re.search(r'"name"\s*:\s*"([^"]+)".*?"arguments"\s*:\s*(\{[^}]*\})', clean)
    if match:
        name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            # Try adding closing brace
            try:
                args = json.loads(match.group(2) + "}")
            except:
                args = {}
        return {"name": name, "arguments": args}
    return None


def execute_tool(tool_call, tool_registry):
    """Execute a tool call and return the result."""
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name not in tool_registry:
        return json.dumps({"error": f"Unknown tool: {name}"})

    fn = tool_registry[name]["fn"]
    # Filter args to only those the function accepts
    import inspect
    sig = inspect.signature(fn)
    valid_args = {k: v for k, v in args.items() if k in sig.parameters}
    try:
        return fn(**valid_args)
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {e}"})


# ===========================================================================
# Agent loop
# ===========================================================================

def build_system_prompt(tool_registry):
    """Build system prompt with tool descriptions."""
    tools = []
    for name, info in tool_registry.items():
        tools.append({
            "name": name,
            "description": info["description"],
            "parameters": info["parameters"],
        })
    return f"""You are a helpful assistant with access to these tools:

{json.dumps(tools, indent=2)}

When you need to use a tool, respond with a JSON object:
{{"tool_calls": [{{"name": "tool_name", "arguments": {{...}}}}]}}

If no tool is needed, respond directly with text.
After receiving tool results, use them to answer the user's question."""


def agent_turn(model, tokenizer, messages, tool_registry, max_tokens=150):
    """Run one agent turn: generate → maybe call tool → return response."""
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(formatted)
    tokens = list(ids)

    # Generate with early stopping on tool call completion
    brace_count = 0
    started_json = False

    for token in stream_generate(model, mx.array([ids]), max_tokens=max_tokens, temperature=0.0):
        t = token.item()
        tokens.append(t)
        if t == tokenizer.eos_token_id:
            break

        # Track JSON braces to stop after first complete object
        char = tokenizer.decode([t])
        for c in char:
            if c == '{':
                started_json = True
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if started_json and brace_count <= 0:
                    # JSON object complete — stop generating
                    break
        if started_json and brace_count <= 0:
            break

    output = tokenizer.decode(tokens[len(ids):]).strip()
    # Clean trailing noise after first complete JSON
    for end in ['<SPECIAL', '\n\n\n']:
        if end in output:
            output = output[:output.index(end)]

    # Check if it's a tool call
    tool_call = parse_tool_call(output)
    if tool_call:
        return "tool_call", tool_call, output
    else:
        return "text", output, output


def run_agent_loop(model, tokenizer, user_input, tool_registry, max_turns=5):
    """Full agent loop: user → model → tool → model → ... → final answer."""
    system_prompt = build_system_prompt(tool_registry)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    for turn in range(max_turns):
        result_type, result, raw = agent_turn(model, tokenizer, messages, tool_registry)

        if result_type == "tool_call":
            tool_name = result["name"]
            tool_args = result["arguments"]
            print(f"  🔧 Calling {tool_name}({tool_args})")

            # Execute tool
            tool_result = execute_tool(result, tool_registry)
            print(f"  📋 Result: {tool_result}")

            # Add to conversation
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Tool result: {tool_result}\n\nPlease answer the original question based on this result."})
        else:
            # Direct text response
            return result

    return "I wasn't able to complete the task within the turn limit."


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Alloy Agent — AI with tool calling")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF ID")
    parser.add_argument("--lora", type=str, default=None, help="LoRA weights path")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--quantize", type=int, choices=[4, 8], default=None)
    args = parser.parse_args()

    # Load model
    from alloy.convert_cli import download_model
    print(f"Loading model: {args.model}")
    model_dir = download_model(args.model)
    model = load_pretrained(model_dir)

    if args.lora:
        print(f"Applying LoRA (rank={args.lora_rank})...")
        model.freeze()
        linear_to_lora_layers(model, lora_rank=args.lora_rank)
        if args.quantize:
            nn.quantize(model, bits=args.quantize,
                        class_predicate=lambda p, m: isinstance(m, nn.Linear) and "lora" not in p)
        print(f"Loading LoRA weights: {args.lora}")
        load_lora_weights(model, args.lora)
    elif args.quantize:
        model.quantize(bits=args.quantize)

    mx.eval(model.parameters())

    from mlx.utils import tree_flatten
    mem = sum(p.nbytes for _, p in tree_flatten(model.parameters()))
    print(f"Ready ({mem / 1e9:.1f} GB)")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    print(f"\nAlloy Agent")
    print(f"Tools: {', '.join(BUILTIN_TOOLS.keys())}")
    print(f"Type your question. Ctrl+C to exit.\n")

    while True:
        try:
            user_input = input("You> ")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input.strip():
            continue

        t0 = time.time()
        response = run_agent_loop(model, tokenizer, user_input, BUILTIN_TOOLS)
        elapsed = time.time() - t0

        print(f"\nAgent> {response}")
        print(f"  [{elapsed:.1f}s]\n")


if __name__ == "__main__":
    main()
