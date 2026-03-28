"""Clean xlam training data: fix double-escaped JSON + compress system prompts.

V8 changes:
- Fix double-escaped JSON to clean format
- Compress system prompts to compact one-line format (saves ~60% tokens)
- Strip verbose parameter descriptions to bare minimum

The original xlam data has assistant outputs like:
  {"tool_calls": "[{\"name\": \"func\", \"arguments\": {...}}]"}

This converts to clean format:
  {"tool_calls": [{"name": "func", "arguments": {...}}]}
"""

import json
import re
import random
import os

random.seed(42)


def compress_system_prompt(content):
    """Compress xlam system prompt to compact V8 format.

    Converts verbose tool descriptions to one-line JSON and removes
    filler text to save ~60% tokens.
    """
    tools = None

    # xlam format: 'You have access to these tools:\n"[{...}]"\nInstructions...'
    # The first line after prefix is a JSON string containing a JSON array
    prefix = "You have access to these tools:\n"
    if content.startswith(prefix):
        rest = content[len(prefix):]
        first_line = rest.split('\n')[0]
        try:
            tools_str = json.loads(first_line)  # Parse outer JSON string
            if isinstance(tools_str, str):
                tools = json.loads(tools_str)  # Parse inner JSON array
            elif isinstance(tools_str, list):
                tools = tools_str
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: try to find JSON array directly
    if tools is None:
        match = re.search(r'\[[\s\S]*?"name"[\s\S]*?\]', content)
        if match:
            try:
                tools = json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                pass

    if tools is None:
        # Can't parse tools, return a minimal version
        return re.sub(r'\s+', ' ', content).strip()

    # Compress each tool to minimal format
    compact_tools = []
    for t in tools:
        entry = {"name": t.get("name", "")}
        desc = t.get("description", "")
        # Truncate long descriptions to first sentence
        if len(desc) > 100:
            first_sentence = desc.split('.')[0] + '.'
            if len(first_sentence) < len(desc):
                desc = first_sentence
        entry["description"] = desc

        # Compress parameters: keep name, type, drop verbose descriptions
        params = t.get("parameters", {})
        if isinstance(params, dict) and params:
            compact_params = {}
            for pname, pinfo in params.items():
                if isinstance(pinfo, dict):
                    compact_params[pname] = {"type": pinfo.get("type", "string")}
                else:
                    compact_params[pname] = {"type": "string"}
            entry["parameters"] = compact_params
        compact_tools.append(entry)

    tools_json = json.dumps(compact_tools, separators=(',', ':'))
    return f'Use EXACT function names. Respond with {{"tool_calls":[...]}} or plain text.\n{tools_json}'


def clean_xlam_example(item):
    """Clean one xlam example's assistant output format + compress system prompt."""
    messages = item["messages"]
    cleaned = []

    for m in messages:
        if m["role"] == "assistant" and "tool_calls" in m.get("content", ""):
            try:
                parsed = json.loads(m["content"])
                tc = parsed.get("tool_calls", "")
                if isinstance(tc, str):
                    # Double-escaped: parse the inner string
                    inner = json.loads(tc)
                    cleaned_content = json.dumps({"tool_calls": inner}, separators=(',', ':'))
                    cleaned.append({"role": "assistant", "content": cleaned_content})
                else:
                    # Already clean, but re-serialize compact
                    cleaned_content = json.dumps(parsed, separators=(',', ':'))
                    cleaned.append({"role": "assistant", "content": cleaned_content})
            except (json.JSONDecodeError, TypeError):
                # Can't parse, skip this example
                return None
        elif m["role"] == "system":
            compressed = compress_system_prompt(m["content"])
            cleaned.append({"role": "system", "content": compressed})
        else:
            cleaned.append(m)

    return {"messages": cleaned}


def main():
    input_path = os.path.expanduser("~/.cache/alloy/datasets/xlam_train.jsonl")
    output_path = os.path.expanduser("~/.cache/alloy/datasets/xlam_clean_train.jsonl")

    examples = []
    skipped = 0

    with open(input_path) as f:
        for line in f:
            item = json.loads(line)
            cleaned = clean_xlam_example(item)
            if cleaned:
                examples.append(cleaned)
            else:
                skipped += 1

    print(f"Cleaned {len(examples)} examples, skipped {skipped}")

    # Verify format
    good = 0
    for ex in examples[:10]:
        for m in ex["messages"]:
            if m["role"] == "assistant" and "tool_calls" in m["content"]:
                parsed = json.loads(m["content"])
                if isinstance(parsed.get("tool_calls"), list):
                    good += 1
    print(f"Format check (first 10): {good}/10 have clean JSON lists")

    # Measure compression
    sys_lens = []
    for ex in examples[:100]:
        for m in ex["messages"]:
            if m["role"] == "system":
                sys_lens.append(len(m["content"]))
    if sys_lens:
        print(f"System prompt chars (first 100): mean={sum(sys_lens)/len(sys_lens):.0f}, min={min(sys_lens)}, max={max(sys_lens)}")

    random.shuffle(examples)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
