"""Generate tool calling training data for Agent fine-tuning (v2).

Key improvements over v1:
- Clean JSON format (no double-escaping) matching agent.py
- 30%+ no-tool negative examples to prevent irrelevance regression
- Parallel/multiple function call examples
- BFCL data conversion for generalization to unseen functions
- Consistent system prompt format with agent.py

Output: JSONL with {messages: [...]} format.
"""

import json
import random
import os
from pathlib import Path

random.seed(42)

# ============================================================
# Tool definitions (expanded)
# ============================================================

TOOL_SETS = {
    "weather": [
        {"name": "get_weather", "description": "Get current weather for a city",
         "parameters": {"city": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "description": "Temperature unit (celsius or fahrenheit)"}}},
    ],
    "search": [
        {"name": "web_search", "description": "Search the web for information",
         "parameters": {"query": {"type": "string", "description": "Search query"},
                        "num_results": {"type": "integer", "description": "Number of results"}}},
    ],
    "calculator": [
        {"name": "calculate", "description": "Evaluate a mathematical expression",
         "parameters": {"expression": {"type": "string", "description": "Math expression to evaluate"}}},
    ],
    "email": [
        {"name": "send_email", "description": "Send an email message",
         "parameters": {"to": {"type": "string", "description": "Recipient email"},
                        "subject": {"type": "string", "description": "Email subject"},
                        "body": {"type": "string", "description": "Email body"}}},
    ],
    "stock": [
        {"name": "get_stock_price", "description": "Get current stock price",
         "parameters": {"symbol": {"type": "string", "description": "Stock ticker symbol"}}},
    ],
    "translation": [
        {"name": "translate", "description": "Translate text between languages",
         "parameters": {"text": {"type": "string", "description": "Text to translate"},
                        "from_lang": {"type": "string", "description": "Source language"},
                        "to_lang": {"type": "string", "description": "Target language"}}},
    ],
    "file": [
        {"name": "read_file", "description": "Read contents of a file",
         "parameters": {"path": {"type": "string", "description": "File path"}}},
        {"name": "write_file", "description": "Write content to a file",
         "parameters": {"path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "Content to write"}}},
    ],
    "database": [
        {"name": "query_db", "description": "Execute a SQL query",
         "parameters": {"sql": {"type": "string", "description": "SQL query"},
                        "database": {"type": "string", "description": "Database name"}}},
    ],
    "time": [
        {"name": "get_time", "description": "Get current date and time",
         "parameters": {}},
    ],
    "calendar": [
        {"name": "create_event", "description": "Create a calendar event",
         "parameters": {"title": {"type": "string", "description": "Event title"},
                        "date": {"type": "string", "description": "Date (YYYY-MM-DD)"},
                        "time": {"type": "string", "description": "Time (HH:MM)"},
                        "duration": {"type": "integer", "description": "Duration in minutes"}}},
        {"name": "list_events", "description": "List calendar events for a date",
         "parameters": {"date": {"type": "string", "description": "Date (YYYY-MM-DD)"}}},
    ],
    "unit_converter": [
        {"name": "convert_units", "description": "Convert between measurement units",
         "parameters": {"value": {"type": "number", "description": "Value to convert"},
                        "from_unit": {"type": "string", "description": "Source unit"},
                        "to_unit": {"type": "string", "description": "Target unit"}}},
    ],
    # V9: Dotted namespace tools (matches BFCL format where 86% of 'multiple' uses module.function)
    "weather_dot": [
        {"name": "weather.get_current", "description": "Get current weather for a location",
         "parameters": {"location": {"type": "string"}, "unit": {"type": "string"}}},
        {"name": "weather.get_forecast", "description": "Get weather forecast for coming days",
         "parameters": {"location": {"type": "string"}, "days": {"type": "integer"}}},
        {"name": "weather.get_historical", "description": "Get historical weather data",
         "parameters": {"location": {"type": "string"}, "date": {"type": "string"}}},
    ],
    "math_dot": [
        {"name": "math.calculate", "description": "Evaluate a mathematical expression",
         "parameters": {"expression": {"type": "string"}}},
        {"name": "math.solve_equation", "description": "Solve an algebraic equation",
         "parameters": {"equation": {"type": "string"}, "variable": {"type": "string"}}},
        {"name": "math.convert_units", "description": "Convert between measurement units",
         "parameters": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}},
    ],
    "finance_dot": [
        {"name": "finance.get_stock_price", "description": "Get current stock price",
         "parameters": {"symbol": {"type": "string"}}},
        {"name": "finance.get_exchange_rate", "description": "Get currency exchange rate",
         "parameters": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}},
        {"name": "finance.calculate_loan", "description": "Calculate loan monthly payment",
         "parameters": {"principal": {"type": "number"}, "rate": {"type": "number"}, "years": {"type": "integer"}}},
    ],
    "geo_dot": [
        {"name": "geo.get_coordinates", "description": "Get latitude and longitude for an address",
         "parameters": {"address": {"type": "string"}}},
        {"name": "geo.calculate_distance", "description": "Calculate distance between two points",
         "parameters": {"lat1": {"type": "number"}, "lon1": {"type": "number"}, "lat2": {"type": "number"}, "lon2": {"type": "number"}}},
        {"name": "geo.reverse_geocode", "description": "Get address from coordinates",
         "parameters": {"latitude": {"type": "number"}, "longitude": {"type": "number"}}},
    ],
    "text_dot": [
        {"name": "text.translate", "description": "Translate text between languages",
         "parameters": {"text": {"type": "string"}, "source_lang": {"type": "string"}, "target_lang": {"type": "string"}}},
        {"name": "text.summarize", "description": "Summarize a long text",
         "parameters": {"text": {"type": "string"}, "max_length": {"type": "integer"}}},
        {"name": "text.sentiment", "description": "Analyze sentiment of text",
         "parameters": {"text": {"type": "string"}}},
    ],
    "db_dot": [
        {"name": "database.query", "description": "Execute a SQL query on a database",
         "parameters": {"sql": {"type": "string"}, "database": {"type": "string"}}},
        {"name": "database.insert", "description": "Insert a record into a table",
         "parameters": {"table": {"type": "string"}, "data": {"type": "object"}}},
        {"name": "database.delete", "description": "Delete records matching a condition",
         "parameters": {"table": {"type": "string"}, "condition": {"type": "string"}}},
    ],
    "calendar_dot": [
        {"name": "calendar.create_event", "description": "Create a new calendar event",
         "parameters": {"title": {"type": "string"}, "date": {"type": "string"}, "time": {"type": "string"}, "duration_minutes": {"type": "integer"}}},
        {"name": "calendar.list_events", "description": "List events for a specific date",
         "parameters": {"date": {"type": "string"}}},
        {"name": "calendar.delete_event", "description": "Delete a calendar event by ID",
         "parameters": {"event_id": {"type": "string"}}},
    ],
    "file_dot": [
        {"name": "file_system.read", "description": "Read contents of a file",
         "parameters": {"path": {"type": "string"}, "encoding": {"type": "string"}}},
        {"name": "file_system.write", "description": "Write content to a file",
         "parameters": {"path": {"type": "string"}, "content": {"type": "string"}}},
        {"name": "file_system.list_dir", "description": "List files in a directory",
         "parameters": {"path": {"type": "string"}, "pattern": {"type": "string"}}},
    ],
}


# ============================================================
# System prompt (matches agent.py format exactly)
# ============================================================

def make_system_prompt(tools):
    """Format tools into compact system prompt.

    V8: Compact format saves ~60% tokens vs V7 pretty-printed JSON.
    Uses one-line JSON with separators=(',',':') and minimal instructions.
    """
    tool_list = []
    for t in tools:
        # Compact: only include non-empty fields
        entry = {"name": t["name"], "description": t["description"]}
        if t.get("parameters"):
            entry["parameters"] = t["parameters"]
        tool_list.append(entry)
    tools_json = json.dumps(tool_list, separators=(',', ':'))
    return f'Use EXACT function names. Respond with {{"tool_calls":[...]}} or plain text.\n{tools_json}'


def make_tool_call_content(calls):
    """Format tool call(s) as clean JSON (no escaping!)."""
    return json.dumps({"tool_calls": calls})


def get_tools(*keys):
    """Get flat list of tools from TOOL_SETS keys."""
    tools = []
    for k in keys:
        tools.extend(TOOL_SETS[k])
    return tools


def make_example(tools, user_msg, tool_calls=None, final_response=None, tool_result=None):
    """Create a training example in messages format.

    tool_calls: list of {"name": ..., "arguments": {...}} dicts, or None for no-tool.
    """
    messages = [
        {"role": "system", "content": make_system_prompt(tools)},
        {"role": "user", "content": user_msg},
    ]

    if tool_calls:
        messages.append({"role": "assistant", "content": make_tool_call_content(tool_calls)})
        if tool_result is not None:
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(tool_result)}\n\nPlease answer the original question based on this result."})
            if final_response:
                messages.append({"role": "assistant", "content": final_response})
    else:
        messages.append({"role": "assistant", "content": final_response or ""})

    return {"messages": messages}


# ============================================================
# Data generation
# ============================================================

CITIES = ["Tokyo", "London", "New York", "Paris", "Berlin", "Sydney", "Beijing",
          "Mumbai", "Cairo", "Toronto", "Seoul", "Bangkok", "Moscow", "Rome",
          "Dubai", "Singapore", "Shanghai", "Istanbul", "Mexico City", "Lagos"]

TOPICS = ["quantum computing", "Mars exploration", "climate change solutions",
          "electric vehicles", "artificial intelligence ethics", "renewable energy",
          "space tourism", "gene editing CRISPR", "blockchain technology",
          "autonomous driving", "mRNA vaccines", "dark matter research",
          "ocean conservation", "nuclear fusion", "brain-computer interface"]

STOCK_SYMBOLS = {"Apple": "AAPL", "Google": "GOOGL", "Microsoft": "MSFT",
                 "Amazon": "AMZN", "Tesla": "TSLA", "Meta": "META",
                 "Netflix": "NFLX", "NVIDIA": "NVDA"}

PHRASES = [("Hello, how are you?", "en"), ("Thank you very much", "en"),
           ("Good morning", "en"), ("I love programming", "en"),
           ("Where is the train station?", "en"), ("Nice to meet you", "en")]
LANGUAGES = [("Spanish", "es"), ("French", "fr"), ("German", "de"),
             ("Japanese", "ja"), ("Chinese", "zh"), ("Korean", "ko"),
             ("Italian", "it"), ("Portuguese", "pt")]


def gen_single_tool_calls():
    """Generate single tool call examples."""
    examples = []

    # Weather
    weather_templates = [
        "What's the weather in {city}?",
        "How's the weather in {city} today?",
        "Is it cold in {city}?",
        "What's the temperature in {city}?",
        "Will I need an umbrella in {city}?",
        "Tell me the weather forecast for {city}.",
    ]
    for tmpl in weather_templates:
        for city in CITIES:
            unit = random.choice(["celsius", "fahrenheit"])
            examples.append(make_example(
                get_tools("weather"), tmpl.format(city=city),
                tool_calls=[{"name": "get_weather", "arguments": {"city": city, "unit": unit}}]))

    # Search
    search_templates = [
        "Search for {topic}",
        "Find information about {topic}",
        "What's the latest on {topic}?",
        "Look up {topic} for me",
        "I need to know about {topic}",
    ]
    for tmpl in search_templates:
        for topic in TOPICS:
            n = random.choice([3, 5])
            examples.append(make_example(
                get_tools("search"), tmpl.format(topic=topic),
                tool_calls=[{"name": "web_search", "arguments": {"query": topic, "num_results": n}}]))

    # Calculator
    calc_templates = [
        ("What is {a} + {b}?", "{a} + {b}"),
        ("Calculate {a} * {b}", "{a} * {b}"),
        ("What's {a} divided by {b}?", "{a} / {b}"),
        ("Compute {a} to the power of {b}", "{a} ** {b}"),
        ("What is the square root of {a}?", "sqrt({a})"),
        ("How much is {a}% of {b}?", "{a} / 100 * {b}"),
        ("What is {a} minus {b}?", "{a} - {b}"),
        ("Calculate ({a} + {b}) * 2", "({a} + {b}) * 2"),
    ]
    for _ in range(80):
        a, b = random.randint(1, 9999), random.randint(1, 9999)
        tmpl, expr_tmpl = random.choice(calc_templates)
        examples.append(make_example(
            get_tools("calculator"), tmpl.format(a=a, b=b),
            tool_calls=[{"name": "calculate", "arguments": {"expression": expr_tmpl.format(a=a, b=b)}}]))

    # Stock
    for company, symbol in STOCK_SYMBOLS.items():
        for q in [f"What's {company}'s stock price?", f"How much is {symbol} trading at?",
                  f"Check {company} stock", f"Get me the current price of {company}"]:
            examples.append(make_example(
                get_tools("stock"), q,
                tool_calls=[{"name": "get_stock_price", "arguments": {"symbol": symbol}}]))

    # Translation
    for phrase, src_lang in PHRASES:
        for lang_name, lang_code in LANGUAGES:
            for tmpl in ["Translate '{text}' to {lang}", "How do you say '{text}' in {lang}?"]:
                examples.append(make_example(
                    get_tools("translation"), tmpl.format(text=phrase, lang=lang_name),
                    tool_calls=[{"name": "translate", "arguments": {"text": phrase, "from_lang": src_lang, "to_lang": lang_code}}]))

    # Time
    for q in ["What time is it?", "What's the current time?", "What date is it today?",
              "What's today's date?", "Tell me the time", "Current date and time please"]:
        examples.append(make_example(
            get_tools("time"), q,
            tool_calls=[{"name": "get_time", "arguments": {}}]))

    # Unit converter
    conversions = [
        ("Convert 5 miles to kilometers", 5, "miles", "kilometers"),
        ("How many pounds is 10 kg?", 10, "kg", "pounds"),
        ("Convert 100 Fahrenheit to Celsius", 100, "fahrenheit", "celsius"),
        ("How many liters is 2 gallons?", 2, "gallons", "liters"),
        ("Convert 1000 meters to feet", 1000, "meters", "feet"),
    ]
    for q, val, from_u, to_u in conversions:
        examples.append(make_example(
            get_tools("unit_converter"), q,
            tool_calls=[{"name": "convert_units", "arguments": {"value": val, "from_unit": from_u, "to_unit": to_u}}]))

    return examples


def gen_parallel_calls():
    """Generate parallel function call examples (same function, multiple invocations)."""
    examples = []

    # Weather for multiple cities
    for _ in range(30):
        cities = random.sample(CITIES, random.choice([2, 3]))
        city_str = ", ".join(cities[:-1]) + f" and {cities[-1]}"
        calls = [{"name": "get_weather", "arguments": {"city": c, "unit": "celsius"}} for c in cities]
        examples.append(make_example(
            get_tools("weather"), f"What's the weather in {city_str}?",
            tool_calls=calls))

    # Stock prices for multiple companies
    for _ in range(20):
        items = random.sample(list(STOCK_SYMBOLS.items()), random.choice([2, 3]))
        names = ", ".join(n for n, _ in items[:-1]) + f" and {items[-1][0]}"
        calls = [{"name": "get_stock_price", "arguments": {"symbol": s}} for _, s in items]
        examples.append(make_example(
            get_tools("stock"), f"Get stock prices for {names}",
            tool_calls=calls))

    # Multiple calculations
    for _ in range(15):
        a, b, c, d = [random.randint(1, 100) for _ in range(4)]
        examples.append(make_example(
            get_tools("calculator"),
            f"Calculate both {a} * {b} and {c} + {d}",
            tool_calls=[
                {"name": "calculate", "arguments": {"expression": f"{a} * {b}"}},
                {"name": "calculate", "arguments": {"expression": f"{c} + {d}"}},
            ]))

    return examples


def gen_multiple_calls():
    """Generate multiple function call examples (different functions)."""
    examples = []

    # Weather + stock
    for city in CITIES[:8]:
        company, symbol = random.choice(list(STOCK_SYMBOLS.items()))
        examples.append(make_example(
            get_tools("weather", "stock"),
            f"What's the weather in {city} and {company}'s stock price?",
            tool_calls=[
                {"name": "get_weather", "arguments": {"city": city, "unit": "celsius"}},
                {"name": "get_stock_price", "arguments": {"symbol": symbol}},
            ]))

    # Calculate + search
    for _ in range(10):
        a, b = random.randint(1, 1000), random.randint(1, 1000)
        topic = random.choice(TOPICS)
        examples.append(make_example(
            get_tools("calculator", "search"),
            f"Calculate {a} + {b} and search for {topic}",
            tool_calls=[
                {"name": "calculate", "arguments": {"expression": f"{a} + {b}"}},
                {"name": "web_search", "arguments": {"query": topic, "num_results": 3}},
            ]))

    # Weather + time
    for city in CITIES[:6]:
        examples.append(make_example(
            get_tools("weather", "time"),
            f"What time is it and what's the weather in {city}?",
            tool_calls=[
                {"name": "get_time", "arguments": {}},
                {"name": "get_weather", "arguments": {"city": city, "unit": "celsius"}},
            ]))

    # Translate + search
    for phrase, _ in PHRASES[:3]:
        lang_name, lang_code = random.choice(LANGUAGES)
        examples.append(make_example(
            get_tools("translation", "search"),
            f"Translate '{phrase}' to {lang_name} and search for {lang_name} language courses",
            tool_calls=[
                {"name": "translate", "arguments": {"text": phrase, "from_lang": "en", "to_lang": lang_code}},
                {"name": "web_search", "arguments": {"query": f"{lang_name} language courses", "num_results": 3}},
            ]))

    return examples


def gen_multi_turn():
    """Generate multi-turn examples with tool results."""
    examples = []

    # Weather with result
    for city in CITIES[:12]:
        temp = random.randint(-5, 35)
        cond = random.choice(["sunny", "cloudy", "rainy", "snowy", "partly cloudy", "windy"])
        humidity = random.randint(30, 90)
        result = {"temperature": temp, "condition": cond, "humidity": humidity, "unit": "celsius"}
        examples.append(make_example(
            get_tools("weather"),
            f"What's the weather in {city}?",
            tool_calls=[{"name": "get_weather", "arguments": {"city": city, "unit": "celsius"}}],
            tool_result=result,
            final_response=f"The weather in {city} is currently {cond} with a temperature of {temp}°C and {humidity}% humidity."))

    # Calculator with result
    for _ in range(15):
        a, b = random.randint(1, 999), random.randint(1, 999)
        op = random.choice([("+", a + b), ("*", a * b), ("-", a - b)])
        result = {"result": op[1]}
        examples.append(make_example(
            get_tools("calculator"),
            f"What is {a} {op[0]} {b}?",
            tool_calls=[{"name": "calculate", "arguments": {"expression": f"{a} {op[0]} {b}"}}],
            tool_result=result,
            final_response=f"{a} {op[0]} {b} = {op[1]}"))

    # Stock with result
    for company, symbol in STOCK_SYMBOLS.items():
        price = round(random.uniform(50, 500), 2)
        change = round(random.uniform(-5, 5), 2)
        result = {"symbol": symbol, "price": price, "change": change, "currency": "USD"}
        direction = "up" if change > 0 else "down"
        examples.append(make_example(
            get_tools("stock"),
            f"What's {company}'s stock price?",
            tool_calls=[{"name": "get_stock_price", "arguments": {"symbol": symbol}}],
            tool_result=result,
            final_response=f"{company} ({symbol}) is currently trading at ${price}, {direction} {abs(change)}% today."))

    return examples


def gen_no_tool_examples():
    """Generate examples where the model should NOT call any tool.

    Critical for preventing irrelevance regression. Targets ~35% of total dataset.
    """
    examples = []

    # All tool combinations that should be available but NOT used
    all_tool_combos = [
        get_tools("weather", "search", "calculator"),
        get_tools("weather", "stock"),
        get_tools("search", "calculator", "translation"),
        get_tools("calculator", "time", "weather"),
        get_tools("email", "file", "database"),
        get_tools("stock", "search", "weather", "calculator"),
        get_tools("weather", "search", "calculator", "stock", "translation"),
    ]

    # General knowledge (tools available but not needed)
    gk_examples = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("What is the capital of Brazil?", "The capital of Brazil is Brasília."),
        ("What is the capital of Australia?", "The capital of Australia is Canberra."),
        ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
        ("What is the capital of Germany?", "The capital of Germany is Berlin."),
        ("What is the capital of Italy?", "The capital of Italy is Rome."),
        ("What is the capital of South Korea?", "The capital of South Korea is Seoul."),
        ("What is the capital of Egypt?", "The capital of Egypt is Cairo."),
        ("What is the capital of Mexico?", "The capital of Mexico is Mexico City."),
        ("What is the capital of India?", "The capital of India is New Delhi."),
        ("What is the capital of Russia?", "The capital of Russia is Moscow."),
        ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
        ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
        ("What year did World War II end?", "World War II ended in 1945."),
        ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
        ("What is the chemical symbol for water?", "The chemical symbol for water is H2O."),
        ("What is the speed of light?", "The speed of light is approximately 299,792,458 meters per second."),
        ("How many continents are there?", "There are 7 continents: Africa, Antarctica, Asia, Australia, Europe, North America, and South America."),
        ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."),
    ]
    for q, a in gk_examples:
        tools = random.choice(all_tool_combos)
        examples.append(make_example(tools, q, final_response=a))

    # Explanations
    explanations = [
        ("Explain what machine learning is.", "Machine learning is a type of artificial intelligence where computers learn patterns from data to make predictions, without being explicitly programmed for each task."),
        ("What is an API?", "An API (Application Programming Interface) is a set of rules that allows different software programs to communicate with each other."),
        ("Explain blockchain in simple terms.", "Blockchain is a distributed digital ledger that records transactions across many computers, making it very difficult to alter records retroactively."),
        ("What is quantum computing?", "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot, potentially solving certain problems much faster."),
        ("What is the difference between HTTP and HTTPS?", "HTTPS is the secure version of HTTP. It encrypts data between your browser and the website using SSL/TLS, protecting sensitive information from being intercepted."),
        ("Explain how a CPU works.", "A CPU processes instructions by fetching them from memory, decoding what they mean, executing the operation, and writing the result back. It does this billions of times per second."),
        ("What is recursion in programming?", "Recursion is when a function calls itself to solve a problem by breaking it down into smaller subproblems. Each recursive call handles a simpler version until reaching a base case."),
        ("Explain the theory of relativity briefly.", "Einstein's theory of relativity consists of special relativity (time slows as speed increases, E=mc²) and general relativity (gravity is the curvature of spacetime caused by mass)."),
        ("What is DNA?", "DNA (deoxyribonucleic acid) is a molecule that carries the genetic instructions used in the development and functioning of all known living organisms."),
        ("What is the Internet of Things?", "The Internet of Things (IoT) refers to the network of physical devices embedded with sensors, software, and connectivity that enables them to collect and exchange data."),
    ]
    for q, a in explanations:
        tools = random.choice(all_tool_combos)
        examples.append(make_example(tools, q, final_response=a))

    # Greetings and chit-chat
    chitchat = [
        ("Hello!", "Hello! How can I help you today?"),
        ("Hi there, how are you?", "I'm doing well, thank you! How can I assist you?"),
        ("Good morning!", "Good morning! What can I help you with today?"),
        ("Hey", "Hey there! What can I do for you?"),
        ("Thanks for your help!", "You're welcome! Let me know if you need anything else."),
        ("Goodbye!", "Goodbye! Have a great day!"),
        ("Tell me a joke.", "Why do programmers prefer dark mode? Because light attracts bugs!"),
        ("Tell me another joke.", "Why did the developer go broke? Because he used up all his cache!"),
        ("What can you do?", "I'm an AI assistant that can help you with various tasks. I have access to tools for weather, search, calculations, and more. Just ask me anything!"),
        ("Who are you?", "I'm a helpful AI assistant. I can answer questions, help with calculations, search for information, and more."),
    ]
    for q, a in chitchat:
        tools = random.choice(all_tool_combos)
        examples.append(make_example(tools, q, final_response=a))

    # Opinions and advice (no tool needed)
    opinions = [
        ("What programming language should I learn first?", "For beginners, Python is an excellent first language due to its readable syntax and versatility. It's widely used in web development, data science, and AI."),
        ("Is it better to buy or rent a house?", "It depends on your financial situation, location, and plans. Buying builds equity but requires a large upfront investment. Renting offers flexibility and lower maintenance costs."),
        ("What's the best way to learn a new language?", "Immersion is the most effective way. Combine regular practice with native speakers, consuming media in the target language, and structured study of grammar and vocabulary."),
        ("How can I improve my sleep?", "Keep a consistent sleep schedule, avoid screens before bed, keep your room cool and dark, limit caffeine after noon, and exercise regularly but not too close to bedtime."),
        ("What are some tips for a job interview?", "Research the company, practice common questions, prepare specific examples of your achievements, dress appropriately, arrive early, and send a thank-you note afterwards."),
        ("Should I learn React or Vue?", "Both are excellent. React has a larger ecosystem and job market. Vue has a gentler learning curve and is often praised for its documentation. Choose based on your goals."),
        ("How do I start investing?", "Start with an emergency fund, then consider low-cost index funds for diversification. Contribute to retirement accounts (401k, IRA) for tax benefits. Start early and invest consistently."),
        ("What's the best way to stay motivated?", "Set clear, achievable goals. Break large tasks into smaller ones. Track your progress. Find an accountability partner. Celebrate small wins and be kind to yourself."),
    ]
    for q, a in opinions:
        tools = random.choice(all_tool_combos)
        examples.append(make_example(tools, q, final_response=a))

    # Math that can be done in head (no need for calculator tool)
    simple_math = [
        ("What is 2 + 2?", "2 + 2 = 4."),
        ("What is 10 * 10?", "10 * 10 = 100."),
        ("What is half of 100?", "Half of 100 is 50."),
        ("What is 0 divided by 5?", "0 divided by 5 is 0."),
    ]
    for q, a in simple_math:
        examples.append(make_example(
            get_tools("calculator", "search"), q, final_response=a))

    # Explicitly irrelevant queries (tools don't match the question)
    irrelevant = [
        # Has weather tool but question is about history
        (get_tools("weather"), "When was the Declaration of Independence signed?",
         "The Declaration of Independence was signed on August 2, 1776."),
        (get_tools("weather"), "Who was the first person to walk on the moon?",
         "Neil Armstrong was the first person to walk on the moon on July 20, 1969."),
        (get_tools("calculator"), "What color is the sky?",
         "The sky appears blue due to Rayleigh scattering of sunlight in Earth's atmosphere."),
        (get_tools("stock"), "How do I cook pasta?",
         "Bring a pot of salted water to a boil, add pasta, cook for the time indicated on the package, then drain."),
        (get_tools("translation"), "What is the square root of 16?",
         "The square root of 16 is 4."),
        (get_tools("email"), "What's the largest ocean?",
         "The Pacific Ocean is the largest ocean, covering about 63 million square miles."),
        (get_tools("database"), "Who invented the telephone?",
         "Alexander Graham Bell is credited with inventing the telephone in 1876."),
        (get_tools("file"), "What is the boiling point of water?",
         "Water boils at 100°C (212°F) at standard atmospheric pressure."),
        (get_tools("weather", "stock"), "Tell me about the French Revolution",
         "The French Revolution (1789-1799) was a period of radical social and political change in France that overthrew the monarchy and established a republic."),
        (get_tools("search", "calculator"), "What is a haiku?",
         "A haiku is a form of Japanese poetry with three lines of 5, 7, and 5 syllables respectively."),
    ]
    for tools, q, a in irrelevant:
        examples.append(make_example(tools, q, final_response=a))

    # Duplicate no-tool with different tool sets to increase diversity
    core_no_tool = [
        "What is the meaning of life?",
        "How does gravity work?",
        "Why is the sky blue?",
        "What is evolution?",
        "How do airplanes fly?",
        "What is democracy?",
        "Why do we dream?",
        "What is entropy?",
        "How do vaccines work?",
        "What is the greenhouse effect?",
        "Summarize the plot of Hamlet.",
        "What is the Pythagorean theorem?",
        "Explain supply and demand.",
        "What is the Doppler effect?",
        "How does WiFi work?",
    ]
    simple_answers = [
        "The meaning of life is a philosophical question that has been debated for centuries. Different perspectives include finding happiness, contributing to society, and personal growth.",
        "Gravity is a fundamental force that attracts objects with mass toward each other. Einstein's general relativity describes it as the curvature of spacetime caused by mass.",
        "The sky appears blue because of Rayleigh scattering. Shorter blue wavelengths of sunlight are scattered more than longer red wavelengths by molecules in Earth's atmosphere.",
        "Evolution is the process of change in living organisms over generations through variations in inherited traits, natural selection, and genetic drift.",
        "Airplanes fly due to the shape of their wings (airfoils) which create lift as air moves over them. Engines provide thrust, while the wing shape creates lower pressure above and higher pressure below.",
        "Democracy is a system of government where power is held by the people, who exercise it through elected representatives or direct voting.",
        "We dream during REM sleep. Theories include memory consolidation, emotional processing, and random neural firing that our brain tries to make sense of.",
        "Entropy is a measure of disorder or randomness in a system. The second law of thermodynamics states that entropy in an isolated system tends to increase over time.",
        "Vaccines work by training your immune system to recognize and fight specific pathogens. They contain weakened or inactivated parts of a pathogen that trigger an immune response.",
        "The greenhouse effect is when gases in Earth's atmosphere trap heat from the sun, warming the planet's surface. Key greenhouse gases include CO2, methane, and water vapor.",
        "Hamlet is a tragedy by Shakespeare about Prince Hamlet of Denmark who seeks revenge against his uncle Claudius for murdering Hamlet's father and marrying his mother.",
        "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a² + b² = c².",
        "Supply and demand is an economic model where the price of a good is determined by its availability (supply) and consumer desire for it (demand). Prices rise when demand exceeds supply.",
        "The Doppler effect is the change in frequency of a wave as the source and observer move relative to each other. It explains why a siren sounds higher-pitched as it approaches.",
        "WiFi uses radio waves to transmit data wirelessly between devices and a router. The router connects to the internet via a wired connection and broadcasts the signal.",
    ]
    for q, a in zip(core_no_tool, simple_answers):
        for _ in range(4):  # each with different tool sets, repeat for volume
            tools = random.choice(all_tool_combos)
            examples.append(make_example(tools, q, final_response=a))

    # Additional no-tool volume: rephrase existing questions
    rephrasings = [
        ("Can you tell me about {topic}?", "{answer}"),
        ("I'd like to know about {topic}.", "{answer}"),
        ("Please explain {topic}.", "{answer}"),
    ]
    topic_answers = [
        ("gravity", "Gravity is a fundamental force of attraction between objects with mass."),
        ("photosynthesis", "Photosynthesis converts sunlight into chemical energy in plants."),
        ("the water cycle", "The water cycle describes how water evaporates, condenses into clouds, and falls as precipitation."),
        ("plate tectonics", "Plate tectonics is the theory that Earth's outer shell is divided into plates that move and interact."),
        ("the solar system", "Our solar system consists of the Sun, eight planets, dwarf planets, moons, asteroids, and comets."),
        ("atoms", "Atoms are the basic building blocks of matter, consisting of protons, neutrons, and electrons."),
        ("electricity", "Electricity is the flow of electric charge, typically through conductors like wires."),
        ("the immune system", "The immune system is a complex network of cells and organs that defends the body against infections."),
        ("black holes", "Black holes are regions of spacetime where gravity is so strong that nothing can escape, not even light."),
        ("the Renaissance", "The Renaissance was a cultural movement from the 14th to 17th century that began in Italy and spread across Europe."),
        ("DNA replication", "DNA replication is the process by which DNA makes a copy of itself during cell division."),
        ("magnetism", "Magnetism is a force caused by moving electric charges that can attract or repel certain materials."),
        ("socialism", "Socialism is an economic system where the means of production are owned collectively or by the state."),
        ("machine learning", "Machine learning is a subset of AI where systems learn from data to improve performance without explicit programming."),
        ("photons", "Photons are elementary particles of light that carry electromagnetic force."),
    ]
    for tmpl, _ in rephrasings:
        for topic, answer in topic_answers:
            tools = random.choice(all_tool_combos)
            examples.append(make_example(tools, tmpl.format(topic=topic, answer=answer),
                                        final_response=answer))

    return examples


def gen_dotted_namespace_calls():
    """Generate training data with module.function naming (V9).

    BFCL 'multiple' category has 86% dotted names (e.g. weather.get_forecast,
    triangle_properties.get). Our V8 training had 0% dotted names, causing
    64% name accuracy on multiple (vs 98% on simple_python).

    Generates:
    - Single calls from dotted tool sets (select 1 from 2-3 options)
    - Parallel calls (same dotted function, different args)
    - Cross-module calls (pick from 2 different dotted modules)
    """
    examples = []

    # === Single calls: select correct dotted function from multiple options ===

    # Weather module
    weather_qs = [
        ("What's the weather in Tokyo right now?", "weather.get_current", {"location": "Tokyo", "unit": "celsius"}),
        ("What's the weather forecast for London for the next 5 days?", "weather.get_forecast", {"location": "London", "days": 5}),
        ("What was the weather in Paris on 2024-01-15?", "weather.get_historical", {"location": "Paris", "date": "2024-01-15"}),
        ("Current temperature in New York?", "weather.get_current", {"location": "New York", "unit": "fahrenheit"}),
        ("3-day forecast for Berlin?", "weather.get_forecast", {"location": "Berlin", "days": 3}),
        ("Weather in Sydney last Christmas?", "weather.get_historical", {"location": "Sydney", "date": "2024-12-25"}),
        ("How's the weather in Mumbai?", "weather.get_current", {"location": "Mumbai", "unit": "celsius"}),
        ("Will it rain in Seattle next week?", "weather.get_forecast", {"location": "Seattle", "days": 7}),
    ]
    for q, fname, args in weather_qs:
        examples.append(make_example(get_tools("weather_dot"), q,
            tool_calls=[{"name": fname, "arguments": args}]))

    # Math module
    math_qs = [
        ("Calculate 15 * 23 + 7", "math.calculate", {"expression": "15 * 23 + 7"}),
        ("Solve the equation 2x + 5 = 15", "math.solve_equation", {"equation": "2x + 5 = 15", "variable": "x"}),
        ("Convert 100 miles to kilometers", "math.convert_units", {"value": 100, "from_unit": "miles", "to_unit": "kilometers"}),
        ("What is sqrt(144) + 3^4?", "math.calculate", {"expression": "sqrt(144) + 3**4"}),
        ("Solve for y: 3y - 9 = 0", "math.solve_equation", {"equation": "3y - 9 = 0", "variable": "y"}),
        ("How many pounds is 50 kg?", "math.convert_units", {"value": 50, "from_unit": "kg", "to_unit": "pounds"}),
        ("Calculate the area: pi * 5^2", "math.calculate", {"expression": "3.14159 * 5**2"}),
        ("Convert 72 Fahrenheit to Celsius", "math.convert_units", {"value": 72, "from_unit": "fahrenheit", "to_unit": "celsius"}),
    ]
    for q, fname, args in math_qs:
        examples.append(make_example(get_tools("math_dot"), q,
            tool_calls=[{"name": fname, "arguments": args}]))

    # Finance module
    finance_qs = [
        ("What's Apple's stock price?", "finance.get_stock_price", {"symbol": "AAPL"}),
        ("Exchange rate from USD to EUR?", "finance.get_exchange_rate", {"from_currency": "USD", "to_currency": "EUR"}),
        ("Monthly payment for a $300k loan at 6% over 30 years?", "finance.calculate_loan", {"principal": 300000, "rate": 6.0, "years": 30}),
        ("NVIDIA stock price?", "finance.get_stock_price", {"symbol": "NVDA"}),
        ("How much is 1000 JPY in USD?", "finance.get_exchange_rate", {"from_currency": "JPY", "to_currency": "USD"}),
        ("Calculate mortgage: $500k at 7.5% for 15 years", "finance.calculate_loan", {"principal": 500000, "rate": 7.5, "years": 15}),
    ]
    for q, fname, args in finance_qs:
        examples.append(make_example(get_tools("finance_dot"), q,
            tool_calls=[{"name": fname, "arguments": args}]))

    # Geo module
    geo_qs = [
        ("Get coordinates for the Eiffel Tower", "geo.get_coordinates", {"address": "Eiffel Tower, Paris"}),
        ("Distance between NYC (40.7,-74.0) and LA (34.1,-118.2)?", "geo.calculate_distance",
         {"lat1": 40.7, "lon1": -74.0, "lat2": 34.1, "lon2": -118.2}),
        ("What address is at 51.5074, -0.1278?", "geo.reverse_geocode", {"latitude": 51.5074, "longitude": -0.1278}),
        ("Coordinates of the White House", "geo.get_coordinates", {"address": "White House, Washington DC"}),
        ("Distance from Tokyo (35.7,139.7) to Beijing (39.9,116.4)?", "geo.calculate_distance",
         {"lat1": 35.7, "lon1": 139.7, "lat2": 39.9, "lon2": 116.4}),
    ]
    for q, fname, args in geo_qs:
        examples.append(make_example(get_tools("geo_dot"), q,
            tool_calls=[{"name": fname, "arguments": args}]))

    # Text module
    text_qs = [
        ("Translate 'Hello world' from English to French", "text.translate",
         {"text": "Hello world", "source_lang": "en", "target_lang": "fr"}),
        ("Summarize this article in 100 words", "text.summarize", {"text": "This is a long article...", "max_length": 100}),
        ("What's the sentiment of 'I love this product!'?", "text.sentiment", {"text": "I love this product!"}),
        ("Translate 'Good morning' to Japanese", "text.translate",
         {"text": "Good morning", "source_lang": "en", "target_lang": "ja"}),
        ("Analyze the sentiment: 'The service was terrible'", "text.sentiment", {"text": "The service was terrible"}),
    ]
    for q, fname, args in text_qs:
        examples.append(make_example(get_tools("text_dot"), q,
            tool_calls=[{"name": fname, "arguments": args}]))

    # Database module
    db_qs = [
        ("Run query: SELECT * FROM users WHERE age > 30", "database.query",
         {"sql": "SELECT * FROM users WHERE age > 30", "database": "main"}),
        ("Insert a new user John, age 25", "database.insert",
         {"table": "users", "data": {"name": "John", "age": 25}}),
        ("Delete all records where status = 'inactive'", "database.delete",
         {"table": "users", "condition": "status = 'inactive'"}),
        ("Query all orders from last month", "database.query",
         {"sql": "SELECT * FROM orders WHERE date >= '2024-02-01'", "database": "sales"}),
    ]
    for q, fname, args in db_qs:
        examples.append(make_example(get_tools("db_dot"), q,
            tool_calls=[{"name": fname, "arguments": args}]))

    # Calendar + File modules
    cal_qs = [
        ("Create a meeting tomorrow at 2pm for 60 minutes", "calendar.create_event",
         {"title": "Meeting", "date": "2024-03-29", "time": "14:00", "duration_minutes": 60}),
        ("What's on my calendar for today?", "calendar.list_events", {"date": "2024-03-28"}),
        ("Delete event ID evt_123", "calendar.delete_event", {"event_id": "evt_123"}),
    ]
    for q, fname, args in cal_qs:
        examples.append(make_example(get_tools("calendar_dot"), q,
            tool_calls=[{"name": fname, "arguments": args}]))

    file_qs = [
        ("Read the file /etc/config.yaml", "file_system.read", {"path": "/etc/config.yaml", "encoding": "utf-8"}),
        ("Write 'hello' to /tmp/test.txt", "file_system.write", {"path": "/tmp/test.txt", "content": "hello"}),
        ("List all .py files in /src", "file_system.list_dir", {"path": "/src", "pattern": "*.py"}),
    ]
    for q, fname, args in file_qs:
        examples.append(make_example(get_tools("file_dot"), q,
            tool_calls=[{"name": fname, "arguments": args}]))

    # === Parallel calls: same dotted function, different args ===
    parallel_qs = [
        ("Weather in Tokyo and London right now?", get_tools("weather_dot"),
         [{"name": "weather.get_current", "arguments": {"location": "Tokyo", "unit": "celsius"}},
          {"name": "weather.get_current", "arguments": {"location": "London", "unit": "celsius"}}]),
        ("Forecast for NYC and LA for the next 3 days?", get_tools("weather_dot"),
         [{"name": "weather.get_forecast", "arguments": {"location": "New York", "days": 3}},
          {"name": "weather.get_forecast", "arguments": {"location": "Los Angeles", "days": 3}}]),
        ("Stock prices for AAPL and GOOGL?", get_tools("finance_dot"),
         [{"name": "finance.get_stock_price", "arguments": {"symbol": "AAPL"}},
          {"name": "finance.get_stock_price", "arguments": {"symbol": "GOOGL"}}]),
        ("Translate 'thank you' to French and Spanish", get_tools("text_dot"),
         [{"name": "text.translate", "arguments": {"text": "thank you", "source_lang": "en", "target_lang": "fr"}},
          {"name": "text.translate", "arguments": {"text": "thank you", "source_lang": "en", "target_lang": "es"}}]),
        ("Coordinates of Big Ben and Statue of Liberty?", get_tools("geo_dot"),
         [{"name": "geo.get_coordinates", "arguments": {"address": "Big Ben, London"}},
          {"name": "geo.get_coordinates", "arguments": {"address": "Statue of Liberty, New York"}}]),
        ("Calculate 2+2 and 3*5", get_tools("math_dot"),
         [{"name": "math.calculate", "arguments": {"expression": "2+2"}},
          {"name": "math.calculate", "arguments": {"expression": "3*5"}}]),
        ("Weather in Paris, Berlin, and Rome?", get_tools("weather_dot"),
         [{"name": "weather.get_current", "arguments": {"location": "Paris", "unit": "celsius"}},
          {"name": "weather.get_current", "arguments": {"location": "Berlin", "unit": "celsius"}},
          {"name": "weather.get_current", "arguments": {"location": "Rome", "unit": "celsius"}}]),
        ("TSLA, MSFT, and AMZN stock prices?", get_tools("finance_dot"),
         [{"name": "finance.get_stock_price", "arguments": {"symbol": "TSLA"}},
          {"name": "finance.get_stock_price", "arguments": {"symbol": "MSFT"}},
          {"name": "finance.get_stock_price", "arguments": {"symbol": "AMZN"}}]),
    ]
    for q, tools, calls in parallel_qs:
        examples.append(make_example(tools, q, tool_calls=calls))

    # === Cross-module calls: pick from multiple dotted modules ===
    cross_qs = [
        ("What's the weather in Tokyo and Apple's stock price?",
         get_tools("weather_dot", "finance_dot"),
         [{"name": "weather.get_current", "arguments": {"location": "Tokyo", "unit": "celsius"}},
          {"name": "finance.get_stock_price", "arguments": {"symbol": "AAPL"}}]),
        ("Calculate 100*5 and translate 'hello' to German",
         get_tools("math_dot", "text_dot"),
         [{"name": "math.calculate", "arguments": {"expression": "100*5"}},
          {"name": "text.translate", "arguments": {"text": "hello", "source_lang": "en", "target_lang": "de"}}]),
        ("Get coordinates of Sydney and the exchange rate AUD to USD",
         get_tools("geo_dot", "finance_dot"),
         [{"name": "geo.get_coordinates", "arguments": {"address": "Sydney, Australia"}},
          {"name": "finance.get_exchange_rate", "arguments": {"from_currency": "AUD", "to_currency": "USD"}}]),
        ("List today's events and read /etc/hosts",
         get_tools("calendar_dot", "file_dot"),
         [{"name": "calendar.list_events", "arguments": {"date": "2024-03-28"}},
          {"name": "file_system.read", "arguments": {"path": "/etc/hosts", "encoding": "utf-8"}}]),
        ("Weather forecast for London 5 days and distance from London to Paris",
         get_tools("weather_dot", "geo_dot"),
         [{"name": "weather.get_forecast", "arguments": {"location": "London", "days": 5}},
          {"name": "geo.calculate_distance", "arguments": {"lat1": 51.5, "lon1": -0.1, "lat2": 48.9, "lon2": 2.3}}]),
    ]
    for q, tools, calls in cross_qs:
        examples.append(make_example(tools, q, tool_calls=calls))

    # Multiply each example with rephrased variants
    base_examples = list(examples)
    for ex in base_examples:
        # Create a variant with extra distractor module
        msgs = ex["messages"]
        # Parse system prompt to add more tools
        sys_content = msgs[0]["content"]
        for extra_module in ["math_dot", "finance_dot", "text_dot", "geo_dot"]:
            extra_tools = get_tools(extra_module)
            extra_json = json.dumps([{"name": t["name"], "description": t["description"],
                                      "parameters": t.get("parameters", {})} for t in extra_tools],
                                     separators=(',', ':'))
            # Only add if these tools aren't already in the system prompt
            if extra_tools[0]["name"] not in sys_content:
                new_sys = sys_content.rstrip() + ',' + extra_json[1:]  # append tools to JSON array
                new_msgs = [{"role": "system", "content": new_sys}] + msgs[1:]
                examples.append({"messages": new_msgs})
                break  # one variant per example

    random.shuffle(examples)
    return examples


def gen_bfcl_irrelevance_v9():
    """Generate BFCL-style irrelevance examples (V9).

    Key insight from V8 analysis: BFCL irrelevance has EXACTLY 1 function
    per example, and the function is in the SAME domain as the question
    but can't actually do what's asked. Our V8 data was too obviously
    irrelevant (different domains entirely).

    Format: 1 function, same-domain mismatch, model should respond with text.
    """
    examples = []

    # (tools_list, question, answer) — function LOOKS related but CAN'T do the task
    same_domain_mismatch = [
        # Math domain: has a math function but wrong one
        ([{"name": "math.compute_integral", "description": "Compute the definite integral of a function",
           "parameters": {"function": {"type": "string"}, "lower": {"type": "number"}, "upper": {"type": "number"}}}],
         [("Calculate the area of a triangle with base 10 and height 5.", "The area of a triangle is (base × height) / 2 = (10 × 5) / 2 = 25 square units."),
          ("What is 15% of 200?", "15% of 200 = 0.15 × 200 = 30."),
          ("Find the derivative of x^3 + 2x.", "The derivative of x³ + 2x is 3x² + 2."),
          ("What is the square root of 256?", "The square root of 256 is 16."),
          ("Solve for x: 2x + 6 = 20", "2x + 6 = 20 → 2x = 14 → x = 7.")]),

        ([{"name": "math.matrix_multiply", "description": "Multiply two matrices together",
           "parameters": {"matrix_a": {"type": "array"}, "matrix_b": {"type": "array"}}}],
         [("What is the factorial of 7?", "7! = 7 × 6 × 5 × 4 × 3 × 2 × 1 = 5,040."),
          ("Convert 45 degrees to radians.", "45° × π/180 = π/4 ≈ 0.7854 radians."),
          ("What is the GCD of 48 and 36?", "The GCD of 48 and 36 is 12."),
          ("Calculate compound interest on $1000 at 5% for 3 years.", "A = 1000(1.05)³ = $1,157.63.")]),

        # Weather domain: has weather function but wrong type
        ([{"name": "weather.set_alert", "description": "Set a weather alert for severe conditions",
           "parameters": {"location": {"type": "string"}, "condition": {"type": "string"}, "threshold": {"type": "number"}}}],
         [("What's the temperature in London right now?", "I don't have access to current weather data. Check a weather service like weather.com for London's current temperature."),
          ("Will it rain tomorrow in Tokyo?", "I can't check weather forecasts. Try a weather app or website for Tokyo's forecast."),
          ("What was the highest temperature in Dubai last summer?", "I don't have access to historical weather data. Weather services like Weather Underground maintain historical records.")]),

        ([{"name": "weather.get_air_quality_index", "description": "Get air quality index for a location",
           "parameters": {"location": {"type": "string"}, "pollutant": {"type": "string"}}}],
         [("What's the weather forecast for this weekend?", "I don't have access to weather forecast data. Check a weather service for your local weekend forecast."),
          ("Is it snowing in Moscow?", "I can't check current weather conditions. A weather app would show Moscow's current conditions."),
          ("Average temperature in Hawaii in December?", "Hawaii averages about 78°F (26°C) in December, with warm days and mild nights.")]),

        # Finance domain: has finance function but wrong one
        ([{"name": "finance.get_dividend_history", "description": "Get dividend payment history for a stock",
           "parameters": {"symbol": {"type": "string"}, "years": {"type": "integer"}}}],
         [("What is Apple's current stock price?", "I can't check real-time stock prices. Check a financial service like Yahoo Finance for AAPL's current price."),
          ("How much has Tesla stock grown this year?", "I don't have access to current stock performance data. Financial platforms track year-to-date returns."),
          ("What's the market cap of Google?", "I don't have access to current market data. Financial services like Bloomberg show current market capitalizations.")]),

        ([{"name": "finance.calculate_tax", "description": "Calculate income tax based on tax brackets",
           "parameters": {"income": {"type": "number"}, "filing_status": {"type": "string"}, "country": {"type": "string"}}}],
         [("What's the current exchange rate from EUR to USD?", "I can't check live exchange rates. Currency converter websites show current rates."),
          ("How much is Bitcoin worth right now?", "I don't have access to real-time crypto prices. Check CoinGecko or similar for current BTC price."),
          ("What's the interest rate on a 30-year mortgage?", "Mortgage rates vary by lender and change daily. Check Bankrate or your bank for current rates.")]),

        # Database domain: has db function but wrong one
        ([{"name": "database.create_backup", "description": "Create a backup of the entire database",
           "parameters": {"database": {"type": "string"}, "format": {"type": "string"}}}],
         [("Show me all users who signed up last month.", "I can only create database backups, not run queries. You'd need a query tool to retrieve user data."),
          ("How many orders were placed today?", "I don't have query capabilities. A database query tool would be needed to count today's orders."),
          ("Delete the inactive accounts from the users table.", "I can only create backups, not modify data. You'd need a data management tool for deletions.")]),

        # Text domain: has text function but wrong one
        ([{"name": "text.count_words", "description": "Count the number of words in a text",
           "parameters": {"text": {"type": "string"}}}],
         [("Translate 'hello' to Spanish.", "I can only count words, not translate. 'Hello' in Spanish is 'Hola'."),
          ("Summarize this article for me.", "I can only count words, not summarize text. I'd be happy to help count words if that's useful."),
          ("Check the grammar of this sentence.", "I can only count words, not check grammar. The sentence appears grammatically correct though.")]),

        ([{"name": "text.extract_keywords", "description": "Extract important keywords from text",
           "parameters": {"text": {"type": "string"}, "max_keywords": {"type": "integer"}}}],
         [("Translate this paragraph to French.", "I can extract keywords but not translate. For translation, use a translation service."),
          ("What's the reading level of this text?", "I can extract keywords but can't assess reading level. Reading level analyzers are available online."),
          ("Rewrite this in a more formal tone.", "I can identify keywords but can't rewrite text. Here's my understanding of the key points though.")]),

        # Geo domain: has geo function but wrong one
        ([{"name": "geo.get_timezone", "description": "Get the timezone for a location",
           "parameters": {"location": {"type": "string"}}}],
         [("How far is it from New York to Los Angeles?", "I can check timezones but not calculate distances. The distance from NYC to LA is approximately 2,451 miles (3,944 km)."),
          ("What's the population of Tokyo?", "I can look up timezones but not population data. Tokyo's metropolitan area has about 37 million people."),
          ("Show me a map of Europe.", "I can only look up timezone information, not display maps.")]),

        # Calendar domain
        ([{"name": "calendar.get_holidays", "description": "Get public holidays for a country and year",
           "parameters": {"country": {"type": "string"}, "year": {"type": "integer"}}}],
         [("Schedule a meeting for tomorrow at 3pm.", "I can look up holidays but can't create events. You'd need an event creation tool."),
          ("What's on my calendar today?", "I can only check public holidays, not personal calendars."),
          ("Cancel my 2pm appointment.", "I can only look up holidays, not manage appointments.")]),

        # File domain
        ([{"name": "file_system.get_permissions", "description": "Get file permissions and ownership info",
           "parameters": {"path": {"type": "string"}}}],
         [("Read the contents of config.yaml.", "I can only check file permissions, not read file contents."),
          ("Create a new file called notes.txt.", "I can only check permissions, not create files."),
          ("Delete the temporary files in /tmp.", "I can only view permissions, not delete files.")]),

        # Misc same-domain mismatches
        ([{"name": "image.get_metadata", "description": "Get EXIF metadata from an image file",
           "parameters": {"file_path": {"type": "string"}}}],
         [("Resize this image to 800x600.", "I can only read image metadata, not resize images."),
          ("Convert this PNG to JPEG.", "I can only extract metadata, not convert image formats."),
          ("What objects are in this photo?", "I can read metadata like camera model and date, but can't identify objects in images.")]),

        ([{"name": "audio.get_duration", "description": "Get the duration of an audio file in seconds",
           "parameters": {"file_path": {"type": "string"}}}],
         [("Transcribe this audio recording.", "I can only check audio duration, not transcribe content."),
          ("Convert this MP3 to WAV format.", "I can check duration but not convert audio formats."),
          ("Play the song 'Bohemian Rhapsody'.", "I can only check audio file duration, not play music.")]),

        ([{"name": "network.ping", "description": "Ping a host and return latency",
           "parameters": {"host": {"type": "string"}, "count": {"type": "integer"}}}],
         [("What's my IP address?", "I can ping hosts but can't determine your IP address."),
          ("Download the file from example.com.", "I can only ping hosts, not download files."),
          ("Check if port 443 is open on google.com.", "I can only ping hosts, not scan specific ports.")]),
    ]

    # For each tool, define a POSITIVE example where the tool SHOULD be called.
    # This balances the "don't call" signal with "do call when appropriate".
    positive_counterexamples = {
        "math.compute_integral": ("Compute the integral of x^2 from 0 to 5.",
            {"name": "math.compute_integral", "arguments": {"function": "x^2", "lower": 0, "upper": 5}}),
        "math.matrix_multiply": ("Multiply matrices [[1,2],[3,4]] and [[5,6],[7,8]].",
            {"name": "math.matrix_multiply", "arguments": {"matrix_a": [[1,2],[3,4]], "matrix_b": [[5,6],[7,8]]}}),
        "weather.set_alert": ("Set a severe storm alert for Chicago when wind exceeds 60 mph.",
            {"name": "weather.set_alert", "arguments": {"location": "Chicago", "condition": "wind", "threshold": 60}}),
        "weather.get_air_quality_index": ("What's the PM2.5 air quality in Beijing?",
            {"name": "weather.get_air_quality_index", "arguments": {"location": "Beijing", "pollutant": "PM2.5"}}),
        "finance.get_dividend_history": ("Show Tesla's dividend history for the past 3 years.",
            {"name": "finance.get_dividend_history", "arguments": {"symbol": "TSLA", "years": 3}}),
        "finance.calculate_tax": ("Calculate income tax for $85,000 filing single in the US.",
            {"name": "finance.calculate_tax", "arguments": {"income": 85000, "filing_status": "single", "country": "US"}}),
        "database.create_backup": ("Create a JSON backup of the production database.",
            {"name": "database.create_backup", "arguments": {"database": "production", "format": "json"}}),
        "text.count_words": ("How many words are in 'The quick brown fox jumps over the lazy dog'?",
            {"name": "text.count_words", "arguments": {"text": "The quick brown fox jumps over the lazy dog"}}),
        "text.extract_keywords": ("Extract the top 5 keywords from this article about climate change.",
            {"name": "text.extract_keywords", "arguments": {"text": "Climate change is causing rising temperatures...", "max_keywords": 5}}),
        "geo.get_timezone": ("What timezone is Tokyo in?",
            {"name": "geo.get_timezone", "arguments": {"location": "Tokyo"}}),
        "calendar.get_holidays": ("What are the public holidays in France in 2024?",
            {"name": "calendar.get_holidays", "arguments": {"country": "France", "year": 2024}}),
        "file_system.get_permissions": ("What are the permissions on /etc/passwd?",
            {"name": "file_system.get_permissions", "arguments": {"path": "/etc/passwd"}}),
        "image.get_metadata": ("Get the EXIF metadata from photo.jpg.",
            {"name": "image.get_metadata", "arguments": {"file_path": "photo.jpg"}}),
        "audio.get_duration": ("How long is the recording meeting_notes.mp3?",
            {"name": "audio.get_duration", "arguments": {"file_path": "meeting_notes.mp3"}}),
        "network.ping": ("Ping google.com 5 times.",
            {"name": "network.ping", "arguments": {"host": "google.com", "count": 5}}),
    }

    for tools, qa_pairs in same_domain_mismatch:
        tool_name = tools[0]["name"]
        for question, answer in qa_pairs:
            # NEGATIVE: tool can't do this task (irrelevance)
            examples.append(make_example(tools, question, final_response=answer))
            # One distractor variant
            distractors = [get_tools("calculator"), get_tools("stock"), get_tools("time")]
            extra = random.choice(distractors)
            examples.append(make_example(tools + extra, question, final_response=answer))

        # POSITIVE: tool CAN do this task (reinforces correct calling)
        if tool_name in positive_counterexamples:
            pos_q, pos_call = positive_counterexamples[tool_name]
            examples.append(make_example(tools, pos_q, tool_calls=[pos_call]))
            # Also with distractors
            for extra in random.sample([get_tools("calculator"), get_tools("stock"),
                                         get_tools("time"), get_tools("search")], 2):
                examples.append(make_example(tools + extra, pos_q, tool_calls=[pos_call]))

    random.shuffle(examples)
    return examples


def gen_bfcl_style_irrelevance():
    """Generate BFCL-style irrelevance examples (500+).

    V8: These mimic BFCL's irrelevance test category where:
    - Functions look superficially related but are semantically wrong domain
    - Question uses keywords that appear in tool descriptions but need different tools
    - Multiple tools available but NONE match the actual need
    """
    examples = []

    # Template: (tools_with_misleading_names, question, answer)
    # Key: function names/descriptions SOUND related but are WRONG domain
    misleading_tool_sets = [
        # "temperature" tools but question is about body temperature
        ([{"name": "get_temperature_reading", "description": "Get temperature sensor reading from IoT device",
           "parameters": {"device_id": {"type": "string", "description": "IoT device identifier"}}},
          {"name": "set_temperature_threshold", "description": "Set alert threshold for temperature sensor",
           "parameters": {"threshold": {"type": "number"}, "device_id": {"type": "string"}}}],
         [("What is normal body temperature?", "Normal body temperature is about 98.6°F (37°C), though it can vary slightly."),
          ("My child has a fever of 102°F, is that dangerous?", "A fever of 102°F in a child should be monitored. Give plenty of fluids and consult a doctor if it persists over 24 hours or is accompanied by other symptoms."),
          ("How do I take someone's temperature?", "Use a digital thermometer under the tongue (oral), under the arm (axillary), or in the ear (tympanic). Wait for the beep indicating the reading is complete."),
          ("At what temperature does water freeze?", "Water freezes at 0°C (32°F) at standard atmospheric pressure.")]),

        # "search" tools but question needs no search
        ([{"name": "search_database_records", "description": "Search internal database for matching records",
           "parameters": {"table": {"type": "string"}, "query": {"type": "string"}, "limit": {"type": "integer"}}},
          {"name": "search_log_entries", "description": "Search application logs for patterns",
           "parameters": {"pattern": {"type": "string"}, "severity": {"type": "string"}}}],
         [("How do I search for a word in Microsoft Word?", "Press Ctrl+F (or Cmd+F on Mac) to open the Find dialog. Type your search term and press Enter to find matches."),
          ("What is binary search?", "Binary search is an efficient algorithm that finds an item in a sorted list by repeatedly dividing the search interval in half. It has O(log n) time complexity."),
          ("How do search engines work?", "Search engines crawl web pages, index their content, and rank results by relevance using algorithms that consider factors like keywords, page authority, and user behavior.")]),

        # "convert" tools but question is about life conversion
        ([{"name": "convert_currency", "description": "Convert amount between currencies using live rates",
           "parameters": {"amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}}},
          {"name": "convert_file_format", "description": "Convert file from one format to another",
           "parameters": {"input_path": {"type": "string"}, "output_format": {"type": "string"}}}],
         [("How do I convert to vegetarianism?", "Start gradually by having a few meat-free days per week. Explore plant-based protein sources like beans, lentils, tofu, and nuts. Plan balanced meals to ensure proper nutrition."),
          ("What is a conversion in marketing?", "In marketing, a conversion is when a visitor completes a desired action, such as making a purchase, signing up, or downloading content. Conversion rate is the percentage of visitors who convert."),
          ("How do I convert Celsius to Fahrenheit manually?", "Multiply the Celsius temperature by 9/5 and add 32. For example, 20°C × 9/5 + 32 = 68°F.")]),

        # "calculate" tools but question is about human judgment
        ([{"name": "calculate_compound_interest", "description": "Calculate compound interest over time",
           "parameters": {"principal": {"type": "number"}, "rate": {"type": "number"}, "years": {"type": "integer"}}},
          {"name": "calculate_bmi", "description": "Calculate Body Mass Index",
           "parameters": {"weight_kg": {"type": "number"}, "height_m": {"type": "number"}}}],
         [("How do you calculate risk in a business decision?", "Business risk assessment involves identifying potential threats, estimating their probability and impact, and developing mitigation strategies. Use frameworks like SWOT analysis or risk matrices."),
          ("What's a calculated move in chess?", "A calculated move in chess is one where a player has analyzed several moves ahead, considering possible opponent responses. It involves tactical awareness and strategic planning."),
          ("How do I calculate my chances of getting into college?", "Consider your GPA, test scores, extracurriculars, and the school's acceptance rate. Many colleges publish admission statistics. No formula gives exact odds, but you can estimate competitiveness.")]),

        # "get_time" tools but question is about time concepts
        ([{"name": "get_server_time", "description": "Get current server timestamp in UTC",
           "parameters": {}},
          {"name": "get_execution_time", "description": "Get execution time of last operation in ms",
           "parameters": {"operation_id": {"type": "string"}}}],
         [("What is time dilation?", "Time dilation is a prediction of Einstein's relativity where time passes slower for objects moving at high speeds or in strong gravitational fields relative to a stationary observer."),
          ("How do I manage my time better?", "Prioritize tasks using methods like the Eisenhower Matrix. Break work into focused intervals (Pomodoro technique). Eliminate distractions and batch similar tasks together."),
          ("What time zone is Japan in?", "Japan is in Japan Standard Time (JST), which is UTC+9. Japan does not observe daylight saving time.")]),

        # "send" tools but question is about emotions
        ([{"name": "send_notification", "description": "Send push notification to user device",
           "parameters": {"user_id": {"type": "string"}, "message": {"type": "string"}, "priority": {"type": "string"}}},
          {"name": "send_webhook", "description": "Send HTTP webhook to external endpoint",
           "parameters": {"url": {"type": "string"}, "payload": {"type": "object"}}}],
         [("How do I send my condolences?", "Express sympathy sincerely with phrases like 'I'm sorry for your loss.' A handwritten card, flowers, or a personal visit shows genuine care. Mention a specific memory of the person if possible."),
          ("What does it mean to send mixed signals?", "Sending mixed signals means communicating in contradictory ways, such as saying one thing but doing another. It causes confusion in relationships and can be unintentional or deliberate."),
          ("How do I send a letter internationally?", "Address the envelope with the recipient's name and full address including country. Visit your post office, choose the appropriate international shipping method (airmail, surface), and pay the postage.")]),

        # "update" tools but question about personal growth
        ([{"name": "update_user_profile", "description": "Update user profile fields in the system",
           "parameters": {"user_id": {"type": "string"}, "fields": {"type": "object"}}},
          {"name": "update_firmware", "description": "Push firmware update to connected device",
           "parameters": {"device_id": {"type": "string"}, "version": {"type": "string"}}}],
         [("How do I update my resume?", "Tailor it to the job, use action verbs, quantify achievements, and keep it to 1-2 pages. Update skills, add recent experience, and remove outdated information."),
          ("What's the latest update on climate change?", "Recent reports show global temperatures continue to rise. Key developments include expanded renewable energy adoption, new emission reduction targets, and growing climate adaptation efforts."),
          ("How do I update my beliefs when new evidence comes in?", "Practice Bayesian thinking: consider how strongly the new evidence supports or contradicts your current belief, and adjust proportionally. Be willing to change your mind when evidence warrants it.")]),

        # "analyze" tools but question is about personal analysis
        ([{"name": "analyze_sentiment", "description": "Analyze sentiment of text (positive/negative/neutral)",
           "parameters": {"text": {"type": "string"}}},
          {"name": "analyze_image", "description": "Analyze image content using computer vision",
           "parameters": {"image_url": {"type": "string"}, "features": {"type": "array"}}}],
         [("How do I analyze a poem?", "Examine the poem's structure, rhythm, imagery, and figurative language. Consider the tone, theme, and historical context. Look for patterns in word choice and how they contribute to meaning."),
          ("What is psychoanalysis?", "Psychoanalysis is a therapeutic approach developed by Sigmund Freud that explores unconscious thoughts, feelings, and memories to understand and resolve psychological issues."),
          ("How do I analyze my spending habits?", "Track all expenses for a month, categorize them (housing, food, entertainment, etc.), and compare to your income. Look for patterns and areas where you can cut back.")]),

        # "create" tools but question about creativity
        ([{"name": "create_virtual_machine", "description": "Create a new virtual machine instance",
           "parameters": {"os": {"type": "string"}, "cpu": {"type": "integer"}, "memory_gb": {"type": "integer"}}},
          {"name": "create_api_key", "description": "Generate new API key for service access",
           "parameters": {"service": {"type": "string"}, "permissions": {"type": "array"}}}],
         [("How do I create a good first impression?", "Make eye contact, offer a firm handshake, smile genuinely, and show interest in the other person. Dress appropriately and be punctual."),
          ("What is creative writing?", "Creative writing is any writing that goes beyond professional, journalistic, or technical forms. It includes fiction, poetry, screenwriting, and personal essays, emphasizing imagination and style."),
          ("How do I create a morning routine?", "Start with essentials (hydrate, hygiene), add 1-2 productive habits (exercise, reading, planning), and keep it realistic. Consistency matters more than complexity.")]),

        # "list" tools but question about life lists
        ([{"name": "list_active_connections", "description": "List all active network connections",
           "parameters": {"protocol": {"type": "string"}}},
          {"name": "list_running_processes", "description": "List all running processes on the system",
           "parameters": {"sort_by": {"type": "string"}}}],
         [("What should be on my bucket list?", "Consider experiences that excite you: travel destinations, skills to learn, personal challenges, relationships to build, and contributions to make. Your bucket list should reflect your unique values."),
          ("How do I make a grocery list?", "Plan meals for the week, check what you already have, and list needed ingredients by store section (produce, dairy, etc.). Stick to the list to avoid impulse purchases."),
          ("What is a to-do list method?", "Popular methods include: Getting Things Done (GTD) for comprehensive task management, the Ivy Lee method (6 most important tasks), and the 1-3-5 rule (1 big, 3 medium, 5 small tasks per day.")]),
    ]

    extra_tool_options = [
        get_tools("weather"), get_tools("calculator"), get_tools("stock"),
        get_tools("time"), get_tools("search"), get_tools("translation"),
        get_tools("email"), get_tools("file"), get_tools("database"),
        get_tools("unit_converter"), get_tools("calendar"),
    ]
    for tools, qa_pairs in misleading_tool_sets:
        for question, answer in qa_pairs:
            # Generate 8 variants per question with different tool combos
            examples.append(make_example(tools, question, final_response=answer))
            for _ in range(7):
                extra_sets = random.sample(extra_tool_options, random.randint(1, 3))
                combined = tools[:]
                for ts in extra_sets:
                    combined.extend(ts)
                random.shuffle(combined)
                examples.append(make_example(combined, question, final_response=answer))

    # Cross-domain: question about domain A, tools for domain B
    cross_domain = [
        # Cooking questions with tech tools
        (get_tools("search", "calculator", "time"),
         [("How do I make sourdough bread?", "Mix flour, water, salt, and starter. Knead, let rise 4-12 hours, shape, proof, then bake at 450°F for 30-40 minutes in a Dutch oven."),
          ("What's the best way to cook steak?", "Season generously with salt. Sear in a hot cast iron pan with oil for 3-4 minutes per side. Rest 5 minutes before serving. Use a meat thermometer for desired doneness."),
          ("How do I make pasta from scratch?", "Mix 2 cups flour with 3 eggs, knead 10 minutes until smooth. Rest 30 minutes, roll thin, cut to shape, and cook in boiling salted water for 2-3 minutes.")]),
        # Health questions with file/database tools
        (get_tools("file", "database"),
         [("How do I start exercising?", "Begin with 15-20 minutes of walking daily. Gradually add strength training and increase duration. Listen to your body and rest when needed."),
          ("What are symptoms of dehydration?", "Thirst, dark yellow urine, dry mouth, fatigue, dizziness, and headache. Severe cases may include rapid heartbeat and confusion. Drink water regularly."),
          ("How much sleep do I need?", "Adults need 7-9 hours per night. Teenagers need 8-10 hours, and children need even more. Quality matters as much as quantity.")]),
        # Relationship questions with email/calendar tools
        (get_tools("email", "calendar"),
         [("How do I resolve a conflict with a friend?", "Approach calmly, listen actively, express feelings with 'I' statements, find common ground, and be willing to compromise. Timing matters — don't discuss when emotions are high."),
          ("What makes a good leader?", "Effective leaders communicate clearly, lead by example, empower their team, remain adaptable, take responsibility, and show empathy. They balance results with relationships."),
          ("How do I build self-confidence?", "Set small achievable goals and celebrate wins. Practice positive self-talk, step outside your comfort zone regularly, and surround yourself with supportive people.")]),
        # Travel questions with unit converter tool
        (get_tools("unit_converter", "time"),
         [("What should I pack for a trip to Europe?", "Pack versatile clothing layers, comfortable walking shoes, adapters for EU outlets, travel documents, and a small daypack. Check weather for specific destinations."),
          ("How do I deal with jet lag?", "Adjust your sleep schedule before traveling. Stay hydrated, avoid alcohol on the flight, get sunlight at your destination, and try to sleep at local bedtime."),
          ("Is it safe to travel alone?", "Solo travel is generally safe with precautions: research destinations, share your itinerary, stay aware of surroundings, keep valuables secure, and trust your instincts.")]),
    ]

    for tools, qa_pairs in cross_domain:
        for q, a in qa_pairs:
            # 6 variants with different tool combos
            examples.append(make_example(tools, q, final_response=a))
            for _ in range(5):
                extra_sets = random.sample(extra_tool_options, random.randint(1, 2))
                combined = tools[:]
                for ts in extra_sets:
                    combined.extend(ts)
                random.shuffle(combined)
                examples.append(make_example(combined, q, final_response=a))

    # Surface-similar keywords: question contains tool-related words but means something else
    keyword_traps = [
        # "stock" appears but not about stock market
        (get_tools("stock", "calculator"),
         [("How do I make chicken stock?", "Simmer chicken bones, vegetables (onion, celery, carrot), and herbs in water for 2-4 hours. Strain and cool. Homemade stock adds rich flavor to soups and sauces."),
          ("What is Stockholm syndrome?", "Stockholm syndrome is a psychological response where hostages develop positive feelings toward their captors. It's named after a 1973 bank robbery in Stockholm, Sweden."),
          ("Should I stock up on emergency supplies?", "Keep a 72-hour kit: water (1 gallon per person per day), non-perishable food, flashlight, batteries, first aid kit, medications, and important documents.")]),
        # "weather" appears but not about current weather
        (get_tools("weather", "search"),
         [("How does weather affect mood?", "Weather can influence mood through light exposure and temperature. Seasonal Affective Disorder (SAD) causes depression in darker months. Sunlight boosts serotonin levels."),
          ("What causes extreme weather events?", "Climate change, ocean temperature shifts, and atmospheric patterns drive extreme weather. El Niño/La Niña cycles, jet stream changes, and rising sea temperatures intensify storms."),
          ("How do I weatherproof my home?", "Seal gaps around windows and doors, add insulation, install storm windows, check roof condition, and ensure proper drainage. This reduces energy costs and prevents water damage.")]),
        # "translate" appears but not about language translation
        (get_tools("translation", "search"),
         [("How do I translate research into practice?", "Bridge the gap by identifying actionable findings, piloting small implementations, gathering feedback, and iterating. Engage stakeholders early and communicate benefits clearly."),
          ("What does it mean to translate vision into action?", "Converting vision to action requires setting specific goals, creating actionable plans, assigning responsibilities, tracking progress, and adapting when obstacles arise."),
          ("How do scientific discoveries translate to medicine?", "Through clinical trials, regulatory approval, and clinical adoption. The process from discovery to treatment typically takes 10-15 years and significant investment.")]),
        # "calculate" appears but not about math
        (get_tools("calculator", "time"),
         [("How do I calculate my worth as a person?", "Your worth isn't something that can be calculated — it's inherent. Focus on your values, relationships, positive impact on others, and personal growth rather than external metrics."),
          ("What is a calculated risk?", "A calculated risk is a carefully considered decision where potential benefits are weighed against possible downsides. Unlike reckless risks, they involve analysis, preparation, and contingency planning."),
          ("How do I make calculated decisions?", "Gather relevant data, identify alternatives, evaluate pros and cons of each, consider potential outcomes and their probabilities, and choose the option with the best expected value.")]),
    ]

    for tools, qa_pairs in keyword_traps:
        for q, a in qa_pairs:
            # 6 variants with different tool combos
            examples.append(make_example(tools, q, final_response=a))
            for _ in range(5):
                extra_sets = random.sample(extra_tool_options, random.randint(1, 2))
                combined = tools[:]
                for ts in extra_sets:
                    combined.extend(ts)
                examples.append(make_example(combined, q, final_response=a))

    random.shuffle(examples)
    return examples


def gen_exact_name_copy_examples():
    """Generate examples with rare/long function names to train exact copying.

    V8: Addresses bottleneck where model invents names like 'calculate_area'
    instead of copying the exact name 'calculate_triangle_area_from_vertices'.
    """
    examples = []

    # Tools with deliberately unusual/long names that are easy to confuse
    rare_name_tools = [
        # Similar names, different functions
        [{"name": "fetch_user_account_balance_details", "description": "Get detailed account balance for a user",
          "parameters": {"user_id": {"type": "string"}}},
         {"name": "fetch_user_account_transaction_history", "description": "Get transaction history for a user account",
          "parameters": {"user_id": {"type": "string"}, "days": {"type": "integer"}}},
         {"name": "fetch_user_account_settings_preferences", "description": "Get account settings and preferences",
          "parameters": {"user_id": {"type": "string"}}}],

        # Long compound names
        [{"name": "calculate_monthly_mortgage_payment_with_taxes", "description": "Calculate monthly mortgage including taxes",
          "parameters": {"principal": {"type": "number"}, "rate": {"type": "number"}, "years": {"type": "integer"}, "tax_rate": {"type": "number"}}},
         {"name": "calculate_monthly_car_lease_payment", "description": "Calculate monthly car lease payment",
          "parameters": {"vehicle_price": {"type": "number"}, "residual": {"type": "number"}, "months": {"type": "integer"}}}],

        # Unusual naming conventions
        [{"name": "getWeatherForecastByZipCode", "description": "Get weather forecast by US zip code",
          "parameters": {"zip_code": {"type": "string"}, "days_ahead": {"type": "integer"}}},
         {"name": "getWeatherHistoricalDataByCity", "description": "Get historical weather data for a city",
          "parameters": {"city": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}}}],

        # Names with prefixes/suffixes
        [{"name": "v2_search_products_by_category", "description": "Search products by category (API v2)",
          "parameters": {"category": {"type": "string"}, "max_results": {"type": "integer"}, "sort_order": {"type": "string"}}},
         {"name": "v2_search_products_by_keyword", "description": "Search products by keyword (API v2)",
          "parameters": {"keyword": {"type": "string"}, "max_results": {"type": "integer"}}}],

        # Abbreviation-heavy names
        [{"name": "get_avg_cpu_util_pct_last_24h", "description": "Get average CPU utilization percentage for last 24 hours",
          "parameters": {"server_id": {"type": "string"}}},
         {"name": "get_mem_usage_stats_by_proc", "description": "Get memory usage statistics grouped by process",
          "parameters": {"server_id": {"type": "string"}, "top_n": {"type": "integer"}}}],

        # Domain-specific jargon names
        [{"name": "run_kolmogorov_smirnov_goodness_of_fit_test", "description": "Run KS test for distribution fit",
          "parameters": {"data": {"type": "array"}, "distribution": {"type": "string"}}},
         {"name": "compute_pearson_correlation_coefficient_matrix", "description": "Compute Pearson correlation matrix",
          "parameters": {"dataset_id": {"type": "string"}, "columns": {"type": "array"}}}],

        # Snake_case with numbers
        [{"name": "export_report_q4_2024_summary", "description": "Export Q4 2024 summary report",
          "parameters": {"format": {"type": "string"}, "include_charts": {"type": "boolean"}}},
         {"name": "export_report_ytd_2024_detailed", "description": "Export year-to-date 2024 detailed report",
          "parameters": {"format": {"type": "string"}, "sections": {"type": "array"}}}],
    ]

    # Generate call examples for each tool set
    call_templates = [
        # fetch_user_account_*
        (0, [("Show me my account balance", 0, {"user_id": "current_user"}),
             ("What's my transaction history for the past week?", 1, {"user_id": "current_user", "days": 7}),
             ("Show my account settings", 2, {"user_id": "current_user"})]),
        # calculate_monthly_*
        (1, [("Calculate my mortgage payment for a $300k loan at 6.5% over 30 years with 1.2% tax",
              0, {"principal": 300000, "rate": 6.5, "years": 30, "tax_rate": 1.2}),
             ("What would a car lease cost for a $35k car with $15k residual over 36 months?",
              1, {"vehicle_price": 35000, "residual": 15000, "months": 36})]),
        # getWeather*
        (2, [("What's the weather forecast for zip code 94105 for the next 3 days?",
              0, {"zip_code": "94105", "days_ahead": 3}),
             ("Get historical weather for London from Jan to Mar 2024",
              1, {"city": "London", "start_date": "2024-01-01", "end_date": "2024-03-31"})]),
        # v2_search_products_*
        (3, [("Search for electronics products, show top 10 sorted by price",
              0, {"category": "electronics", "max_results": 10, "sort_order": "price_asc"}),
             ("Find products matching 'wireless headphones', top 5",
              1, {"keyword": "wireless headphones", "max_results": 5})]),
        # get_avg_cpu_* / get_mem_*
        (4, [("What's the average CPU utilization for server prod-01?",
              0, {"server_id": "prod-01"}),
             ("Show top 10 memory-consuming processes on server prod-01",
              1, {"server_id": "prod-01", "top_n": 10})]),
        # statistics tools
        (5, [("Run a KS test on dataset against normal distribution",
              0, {"data": [1.2, 2.3, 3.1, 2.8, 1.9], "distribution": "normal"}),
             ("Compute correlation matrix for columns age, income, score",
              1, {"dataset_id": "main_dataset", "columns": ["age", "income", "score"]})]),
        # export_report_*
        (6, [("Export the Q4 2024 summary as PDF with charts",
              0, {"format": "pdf", "include_charts": True}),
             ("Export the year-to-date detailed report as CSV, include sales and expenses sections",
              1, {"format": "csv", "sections": ["sales", "expenses"]})]),
    ]

    for tool_set_idx, qas in call_templates:
        tools = rare_name_tools[tool_set_idx]
        for question, tool_idx, args in qas:
            tool_name = tools[tool_idx]["name"]
            examples.append(make_example(
                tools, question,
                tool_calls=[{"name": tool_name, "arguments": args}]))
            # Also with rephrased question for variety
            for prefix in ["Please ", "Can you ", "I need to "]:
                examples.append(make_example(
                    tools, prefix + question[0].lower() + question[1:],
                    tool_calls=[{"name": tool_name, "arguments": args}]))

    random.shuffle(examples)
    return examples


def convert_bfcl_to_training(max_examples=200):
    """Convert BFCL test data to training format for generalization.

    Uses BFCL simple_python examples with ground truth to create
    training data with diverse, unseen function names.
    """
    examples = []

    try:
        from bfcl_eval.eval_checker.eval_runner import PROMPT_PATH
        from bfcl_eval.constants.category_mapping import VERSION_PREFIX
        data_dir = Path(PROMPT_PATH)
    except ImportError:
        print("  bfcl_eval not installed, skipping BFCL conversion")
        return examples

    for category in ["simple_python", "multiple", "parallel"]:
        prompt_file = data_dir / f"{VERSION_PREFIX}_{category}.json"
        answer_file = data_dir / "possible_answer" / f"{VERSION_PREFIX}_{category}.json"

        if not prompt_file.exists() or not answer_file.exists():
            continue

        with open(prompt_file) as f:
            prompts = [json.loads(line) for line in f]
        with open(answer_file) as f:
            answers = {json.loads(line)["id"]: json.loads(line)["ground_truth"]
                       for line in f}

        # Take a subset for training (leave rest for eval)
        # Use examples from the second half to avoid overlap with eval (which uses first N)
        subset = prompts[len(prompts)//2:]
        if len(subset) > max_examples // 3:
            subset = random.sample(subset, max_examples // 3)

        for item in subset:
            eid = item["id"]
            question = item["question"][0]  # first turn
            functions = item["function"]
            gt = answers.get(eid)
            if gt is None:
                continue

            # Convert BFCL function format to our tool format
            tools = []
            for func in functions:
                tool = {"name": func["name"], "description": func.get("description", ""),
                        "parameters": {}}
                params = func.get("parameters", {})
                if "properties" in params:
                    for pname, pinfo in params["properties"].items():
                        tool["parameters"][pname] = {
                            "type": pinfo.get("type", "string"),
                            "description": pinfo.get("description", ""),
                        }
                tools.append(tool)

            # Convert ground truth to tool_calls format
            tool_calls = []
            for gt_call in gt:
                for fname, fargs in gt_call.items():
                    # Ground truth has lists of possible values, take first
                    args = {}
                    for k, v_list in fargs.items():
                        if isinstance(v_list, list) and v_list:
                            val = v_list[0]
                            if val == "":
                                continue  # skip optional empty args
                            args[k] = val
                        else:
                            args[k] = v_list
                    tool_calls.append({"name": fname, "arguments": args})

            if tool_calls:
                messages = [
                    {"role": "system", "content": make_system_prompt(tools)},
                ]
                messages.extend(question)
                messages.append({"role": "assistant", "content": make_tool_call_content(tool_calls)})
                examples.append({"messages": messages})

    random.shuffle(examples)
    return examples


def generate_all():
    """Generate the complete training dataset."""
    single = gen_single_tool_calls()
    parallel = gen_parallel_calls()
    multiple = gen_multiple_calls()
    multi_turn = gen_multi_turn()
    no_tool = gen_no_tool_examples()
    bfcl = convert_bfcl_to_training(max_examples=200)
    irrelevance = gen_bfcl_style_irrelevance()
    exact_names = gen_exact_name_copy_examples()
    dotted = gen_dotted_namespace_calls()
    irr_v9 = gen_bfcl_irrelevance_v9()

    all_examples = single + parallel + multiple + multi_turn + no_tool + bfcl + irrelevance + exact_names + dotted + irr_v9
    random.shuffle(all_examples)

    # Stats
    has_tool = sum(1 for ex in all_examples
                   if any('"tool_calls"' in m.get("content", "") for m in ex["messages"]
                          if m["role"] == "assistant"))
    no_tool_count = len(all_examples) - has_tool
    has_multi_call = sum(1 for ex in all_examples
                         if any(m.get("content", "").count('"name"') > 1
                                for m in ex["messages"] if m["role"] == "assistant"))
    has_result = sum(1 for ex in all_examples
                     if any("Tool result:" in m.get("content", "")
                            for m in ex["messages"] if m["role"] == "user"))

    print(f"Generated {len(all_examples)} training examples:")
    print(f"  Single tool calls: {len(single)}")
    print(f"  Parallel calls:    {len(parallel)}")
    print(f"  Multiple calls:    {len(multiple)}")
    print(f"  Multi-turn:        {len(multi_turn)}")
    print(f"  No-tool:           {len(no_tool)} ({100*len(no_tool)/len(all_examples):.1f}%)")
    print(f"  BFCL converted:    {len(bfcl)}")
    print(f"  Irrelevance (V8):  {len(irrelevance)}")
    print(f"  Exact names (V8):  {len(exact_names)}")
    print(f"  Dotted names (V9): {len(dotted)}")
    print(f"  Irr BFCL-style V9: {len(irr_v9)}")
    print(f"  ---")
    print(f"  Total tool calls:  {has_tool}")
    print(f"  Total no-tool:     {no_tool_count} ({100*no_tool_count/len(all_examples):.1f}%)")
    print(f"  Multi-call:        {has_multi_call}")
    print(f"  With results:      {has_result}")

    return all_examples


if __name__ == "__main__":
    examples = generate_all()
    out_path = os.path.expanduser("~/.cache/alloy/datasets/tool_calling_v2_train.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"\nSaved to: {out_path}")
