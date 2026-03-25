"""Generate tool calling training data for Agent fine-tuning.

Generates diverse tool calling examples in a format suitable for SFT:
- Single tool calls
- Multi-step tool calls
- No-tool responses
- Tool result handling

Output: JSONL with {messages: [...]} format.
"""

import json
import random
import os

random.seed(42)

# ============================================================
# Tool definitions
# ============================================================

TOOL_SETS = {
    "weather": [
        {"name": "get_weather", "description": "Get current weather for a city",
         "parameters": {"type": "object", "properties": {
             "city": {"type": "string", "description": "City name"},
             "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
         }, "required": ["city"]}},
    ],
    "search": [
        {"name": "web_search", "description": "Search the web for information",
         "parameters": {"type": "object", "properties": {
             "query": {"type": "string", "description": "Search query"},
             "num_results": {"type": "integer", "description": "Number of results"}
         }, "required": ["query"]}},
    ],
    "calculator": [
        {"name": "calculate", "description": "Evaluate a mathematical expression",
         "parameters": {"type": "object", "properties": {
             "expression": {"type": "string", "description": "Math expression to evaluate"}
         }, "required": ["expression"]}},
    ],
    "email": [
        {"name": "send_email", "description": "Send an email message",
         "parameters": {"type": "object", "properties": {
             "to": {"type": "string", "description": "Recipient email"},
             "subject": {"type": "string", "description": "Email subject"},
             "body": {"type": "string", "description": "Email body"}
         }, "required": ["to", "subject", "body"]}},
    ],
    "stock": [
        {"name": "get_stock_price", "description": "Get current stock price",
         "parameters": {"type": "object", "properties": {
             "symbol": {"type": "string", "description": "Stock ticker symbol"}
         }, "required": ["symbol"]}},
    ],
    "translation": [
        {"name": "translate", "description": "Translate text between languages",
         "parameters": {"type": "object", "properties": {
             "text": {"type": "string", "description": "Text to translate"},
             "from_lang": {"type": "string", "description": "Source language"},
             "to_lang": {"type": "string", "description": "Target language"}
         }, "required": ["text", "to_lang"]}},
    ],
    "file": [
        {"name": "read_file", "description": "Read contents of a file",
         "parameters": {"type": "object", "properties": {
             "path": {"type": "string", "description": "File path"}
         }, "required": ["path"]}},
        {"name": "write_file", "description": "Write content to a file",
         "parameters": {"type": "object", "properties": {
             "path": {"type": "string", "description": "File path"},
             "content": {"type": "string", "description": "Content to write"}
         }, "required": ["path", "content"]}},
    ],
    "database": [
        {"name": "query_db", "description": "Execute a SQL query",
         "parameters": {"type": "object", "properties": {
             "sql": {"type": "string", "description": "SQL query"},
             "database": {"type": "string", "description": "Database name"}
         }, "required": ["sql"]}},
    ],
}

# ============================================================
# Example templates
# ============================================================

WEATHER_EXAMPLES = [
    ("What's the weather in {city}?", {"city": "{city}", "unit": "celsius"}),
    ("How's the weather in {city} today?", {"city": "{city}", "unit": "celsius"}),
    ("Is it cold in {city}?", {"city": "{city}", "unit": "celsius"}),
    ("What's the temperature in {city} in Fahrenheit?", {"city": "{city}", "unit": "fahrenheit"}),
    ("Will I need an umbrella in {city}?", {"city": "{city}", "unit": "celsius"}),
]
CITIES = ["Tokyo", "London", "New York", "Paris", "Berlin", "Sydney", "Beijing", "Mumbai", "Cairo", "Toronto", "Seoul", "Bangkok", "Moscow", "Rome", "Dubai"]

SEARCH_EXAMPLES = [
    ("Search for {topic}", {"query": "{topic}", "num_results": 5}),
    ("Find information about {topic}", {"query": "{topic}", "num_results": 5}),
    ("What's the latest on {topic}?", {"query": "latest {topic} news", "num_results": 3}),
    ("Look up {topic} for me", {"query": "{topic}", "num_results": 5}),
]
TOPICS = ["quantum computing", "Mars exploration", "climate change solutions", "electric vehicles", "artificial intelligence ethics", "renewable energy", "space tourism", "gene editing CRISPR", "blockchain technology", "autonomous driving", "mRNA vaccines", "dark matter research", "ocean conservation", "nuclear fusion", "brain-computer interface"]

CALC_EXAMPLES = [
    ("What is {a} + {b}?", {"expression": "{a} + {b}"}),
    ("Calculate {a} * {b}", {"expression": "{a} * {b}"}),
    ("What's {a} divided by {b}?", {"expression": "{a} / {b}"}),
    ("Compute {a} to the power of {b}", {"expression": "{a} ** {b}"}),
    ("What is the square root of {a}?", {"expression": "sqrt({a})"}),
    ("How much is {a}% of {b}?", {"expression": "{a} / 100 * {b}"}),
]

STOCK_SYMBOLS = {"Apple": "AAPL", "Google": "GOOGL", "Microsoft": "MSFT", "Amazon": "AMZN", "Tesla": "TSLA", "Meta": "META", "Netflix": "NFLX", "NVIDIA": "NVDA"}

TRANSLATE_EXAMPLES = [
    ("Translate '{text}' to {lang}", {"text": "{text}", "to_lang": "{lang_code}"}),
    ("How do you say '{text}' in {lang}?", {"text": "{text}", "to_lang": "{lang_code}"}),
]
PHRASES = [("Hello, how are you?", "en"), ("Thank you very much", "en"), ("Good morning", "en"), ("I love programming", "en")]
LANGUAGES = [("Spanish", "es"), ("French", "fr"), ("German", "de"), ("Japanese", "ja"), ("Chinese", "zh"), ("Korean", "ko")]

NO_TOOL_EXAMPLES = [
    ("What is the capital of {country}?", "The capital of {country} is {capital}."),
    ("Explain what {concept} is in simple terms.", "{concept} is {explanation}."),
    ("Tell me a fun fact.", "{fact}"),
    ("What does {acronym} stand for?", "{acronym} stands for {full_name}."),
]

COUNTRIES = [("France", "Paris"), ("Japan", "Tokyo"), ("Brazil", "Brasília"), ("Australia", "Canberra"), ("Canada", "Ottawa"), ("Germany", "Berlin"), ("Italy", "Rome"), ("South Korea", "Seoul")]
CONCEPTS = [
    ("machine learning", "Machine learning is a type of artificial intelligence where computers learn patterns from data to make predictions, without being explicitly programmed for each task."),
    ("blockchain", "Blockchain is a distributed digital ledger that records transactions across many computers, making it very difficult to alter records retroactively."),
    ("API", "An API (Application Programming Interface) is a set of rules that allows different software programs to communicate with each other, like a menu in a restaurant that defines what you can order."),
]


def make_system_prompt(tools):
    """Format tools into a system prompt."""
    tool_desc = json.dumps(tools, indent=2)
    return f"You are a helpful assistant with access to the following tools:\n\n{tool_desc}\n\nWhen you need to use a tool, respond with a JSON object in the format: {{\"tool_call\": {{\"name\": \"tool_name\", \"arguments\": {{...}}}}}}\nIf no tool is needed, respond directly."


def make_example(tools_key, user_msg, tool_call=None, final_response=None, tool_result=None):
    """Create a training example in messages format."""
    tools = []
    for k in (tools_key if isinstance(tools_key, list) else [tools_key]):
        tools.extend(TOOL_SETS[k])

    messages = [
        {"role": "system", "content": make_system_prompt(tools)},
        {"role": "user", "content": user_msg},
    ]

    if tool_call:
        messages.append({"role": "assistant", "content": json.dumps({"tool_call": tool_call})})
        if tool_result:
            messages.append({"role": "tool", "content": json.dumps(tool_result)})
            if final_response:
                messages.append({"role": "assistant", "content": final_response})
    else:
        messages.append({"role": "assistant", "content": final_response or ""})

    return {"messages": messages}


def generate_all():
    examples = []

    # Weather examples
    for template, args_template in WEATHER_EXAMPLES:
        for city in CITIES:
            user = template.format(city=city)
            args = {k: v.format(city=city) for k, v in args_template.items()}
            examples.append(make_example("weather", user,
                tool_call={"name": "get_weather", "arguments": args}))

    # Search examples
    for template, args_template in SEARCH_EXAMPLES:
        for topic in TOPICS:
            user = template.format(topic=topic)
            args = {k: (v.format(topic=topic) if isinstance(v, str) else v) for k, v in args_template.items()}
            examples.append(make_example("search", user,
                tool_call={"name": "web_search", "arguments": args}))

    # Calculator
    for _ in range(60):
        a, b = random.randint(1, 999), random.randint(1, 999)
        template, args_template = random.choice(CALC_EXAMPLES)
        user = template.format(a=a, b=b)
        args = {k: v.format(a=a, b=b) for k, v in args_template.items()}
        examples.append(make_example("calculator", user,
            tool_call={"name": "calculate", "arguments": args}))

    # Stock
    for company, symbol in STOCK_SYMBOLS.items():
        for q in [f"What's {company}'s stock price?", f"How much is {symbol} trading at?", f"Check {company} stock"]:
            examples.append(make_example("stock", q,
                tool_call={"name": "get_stock_price", "arguments": {"symbol": symbol}}))

    # Translation
    for tmpl, _ in TRANSLATE_EXAMPLES:
        for phrase, src_lang in PHRASES:
            for lang_name, lang_code in LANGUAGES:
                user = tmpl.format(text=phrase, lang=lang_name)
                examples.append(make_example("translation", user,
                    tool_call={"name": "translate", "arguments": {"text": phrase, "from_lang": src_lang, "to_lang": lang_code}}))

    # No-tool examples (model should answer directly)
    for country, capital in COUNTRIES:
        examples.append(make_example(
            ["weather", "search", "calculator"], f"What is the capital of {country}?",
            final_response=f"The capital of {country} is {capital}."))

    for concept, explanation in CONCEPTS:
        examples.append(make_example(
            ["search", "calculator"], f"Explain what {concept} is.",
            final_response=explanation))

    # Multi-tool scenarios
    for city in CITIES[:5]:
        for company, symbol in list(STOCK_SYMBOLS.items())[:3]:
            examples.append(make_example(
                ["weather", "stock"],
                f"What's the weather in {city} and {company}'s stock price?",
                tool_call={"name": "get_weather", "arguments": {"city": city, "unit": "celsius"}}))

    # Multi-turn with tool results
    for city in CITIES[:8]:
        result = {"temperature": random.randint(-5, 35), "condition": random.choice(["sunny", "cloudy", "rainy", "snowy"]), "humidity": random.randint(30, 90)}
        examples.append(make_example(
            "weather",
            f"What's the weather in {city}?",
            tool_call={"name": "get_weather", "arguments": {"city": city, "unit": "celsius"}},
            tool_result=result,
            final_response=f"The weather in {city} is currently {result['condition']} with a temperature of {result['temperature']}°C and {result['humidity']}% humidity."))

    random.shuffle(examples)
    return examples


if __name__ == "__main__":
    examples = generate_all()
    out_path = os.path.expanduser("~/.cache/alloy/datasets/tool_calling_train.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Generated {len(examples)} training examples")
    print(f"Saved to: {out_path}")

    # Stats
    has_tool = sum(1 for ex in examples if any('"tool_call"' in m.get("content", "") for m in ex["messages"]))
    has_result = sum(1 for ex in examples if any(m["role"] == "tool" for m in ex["messages"]))
    no_tool = len(examples) - has_tool
    print(f"  Tool calls: {has_tool}")
    print(f"  With tool results: {has_result}")
    print(f"  No tool needed: {no_tool}")
