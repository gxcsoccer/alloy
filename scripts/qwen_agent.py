"""Lightweight agent powered by Qwen3-Coder via mlx-lm.

Usage:
    python scripts/qwen_agent.py --model /Users/lang/qwen3-coder-next-4bit
"""

import argparse
import json
import re
import sys
import time


# ===========================================================================
# Built-in tools
# ===========================================================================

def tool_calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        import math
        expr = expression.replace("^", "**")
        result = eval(expr, {"__builtins__": {}, "math": math})
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_get_time() -> str:
    """Get current date and time."""
    from datetime import datetime
    now = datetime.now()
    return json.dumps({"datetime": now.isoformat(), "date": now.strftime("%Y-%m-%d"),
                        "time": now.strftime("%H:%M:%S")})


def tool_web_search(query: str, num_results: int = 3) -> str:
    """Simulated web search (placeholder)."""
    return json.dumps({"results": [f"[Simulated result for: {query}]"],
                        "note": "Connect a real search API."})


def tool_get_weather(location: str, unit: str = "celsius") -> str:
    """Simulated weather (placeholder)."""
    return json.dumps({"location": location, "temperature": 22, "unit": unit,
                        "condition": "sunny", "note": "Connect a real weather API."})


# Tool registry: name → {fn, openai_schema}
TOOLS = {
    "calculate": {
        "fn": tool_calculate,
        "schema": {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression to evaluate"}
                    },
                    "required": ["expression"]
                }
            }
        }
    },
    "get_time": {
        "fn": tool_get_time,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get current date and time",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        }
    },
    "web_search": {
        "fn": tool_web_search,
        "schema": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "num_results": {"type": "integer", "description": "Number of results"}
                    },
                    "required": ["query"]
                }
            }
        }
    },
    "get_weather": {
        "fn": tool_get_weather,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    },
}


# ===========================================================================
# Tool call parsing (Qwen3 XML format)
# ===========================================================================

def parse_tool_calls(text):
    """Parse Qwen3's XML tool call format.

    Format:
        <tool_call>
        <function=name>
        <parameter=key>value</parameter>
        </function>
        </tool_call>

    Returns list of {"name": ..., "arguments": {...}} dicts, or empty list.
    """
    calls = []
    for block in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        content = block.group(1)
        func_match = re.search(r'<function=(\S+?)>(.*?)</function>', content, re.DOTALL)
        if not func_match:
            continue
        name = func_match.group(1)
        params_text = func_match.group(2)
        args = {}
        for param in re.finditer(r'<parameter=(\S+?)>\s*(.*?)\s*</parameter>', params_text, re.DOTALL):
            key = param.group(1)
            value = param.group(2).strip()
            # Try to parse as JSON value (number, bool, etc.)
            try:
                args[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                args[key] = value
        calls.append({"name": name, "arguments": args})
    return calls


def execute_tool(call):
    """Execute a tool call and return result string."""
    name = call["name"]
    args = call["arguments"]

    if name not in TOOLS:
        return json.dumps({"error": f"Unknown tool: {name}"})

    fn = TOOLS[name]["fn"]
    import inspect
    sig = inspect.signature(fn)
    valid_args = {k: v for k, v in args.items() if k in sig.parameters}
    try:
        return fn(**valid_args)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ===========================================================================
# Agent loop
# ===========================================================================

def agent_loop(model, tokenizer, user_input, max_turns=5):
    """Multi-turn agent loop: generate → call tools → feed results → repeat."""
    from mlx_lm import generate

    tool_schemas = [t["schema"] for t in TOOLS.values()]
    messages = [{"role": "user", "content": user_input}]

    for turn in range(max_turns):
        prompt = tokenizer.apply_chat_template(
            messages, tools=tool_schemas, tokenize=False, add_generation_prompt=True)

        response = generate(model, tokenizer, prompt=prompt, max_tokens=2048, verbose=False)

        # Check for tool calls
        tool_calls = parse_tool_calls(response)

        if not tool_calls:
            # Clean thinking tags if present
            clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            return clean

        # Execute tools and build result message
        messages.append({"role": "assistant", "content": response})

        for call in tool_calls:
            print(f"  🔧 {call['name']}({call['arguments']})", flush=True)
            result = execute_tool(call)
            print(f"  📋 {result}", flush=True)
            messages.append({
                "role": "tool",
                "name": call["name"],
                "content": result,
            })

    return "I wasn't able to complete the task within the turn limit."


# ===========================================================================
# Streaming agent loop
# ===========================================================================

def agent_loop_stream(model, tokenizer, user_input, max_turns=5):
    """Streaming version — prints text tokens as they arrive."""
    from mlx_lm import stream_generate as mlx_stream

    tool_schemas = [t["schema"] for t in TOOLS.values()]
    messages = [{"role": "user", "content": user_input}]

    for turn in range(max_turns):
        prompt = tokenizer.apply_chat_template(
            messages, tools=tool_schemas, tokenize=False, add_generation_prompt=True)

        # Stream generate
        full_response = ""
        in_think = False
        for chunk in mlx_stream(model, tokenizer, prompt=prompt, max_tokens=2048):
            token_text = chunk.text
            full_response += token_text

            # Don't print <think>...</think> content
            if "<think>" in full_response and not in_think:
                in_think = True
            if "</think>" in full_response and in_think:
                in_think = False
                continue
            if in_think:
                continue

            # Don't stream tool call XML
            if "<tool_call>" in full_response:
                continue

            print(token_text, end="", flush=True)

        # Check for tool calls
        tool_calls = parse_tool_calls(full_response)

        if not tool_calls:
            print(flush=True)
            clean = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
            return clean

        # Clear partial text, show tool calls
        print(flush=True)
        messages.append({"role": "assistant", "content": full_response})

        for call in tool_calls:
            print(f"  🔧 {call['name']}({call['arguments']})", flush=True)
            result = execute_tool(call)
            print(f"  📋 {result}", flush=True)
            messages.append({
                "role": "tool",
                "name": call["name"],
                "content": result,
            })

    return "Turn limit reached."


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Alloy Agent (Qwen3 backend)")
    parser.add_argument("--model", type=str, default="/Users/lang/qwen3-coder-next-4bit",
                        help="Model path (local or HuggingFace ID)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    from mlx_lm import load
    print(f"Loading: {args.model}")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"Ready ({time.time()-t0:.1f}s)")

    print(f"\nAlloy Agent (Qwen3)")
    print(f"Tools: {', '.join(TOOLS.keys())}")
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
        print(f"\nAgent> ", end="", flush=True)

        if args.no_stream:
            response = agent_loop(model, tokenizer, user_input)
            print(response)
        else:
            agent_loop_stream(model, tokenizer, user_input)

        elapsed = time.time() - t0
        print(f"  [{elapsed:.1f}s]\n")


if __name__ == "__main__":
    main()
