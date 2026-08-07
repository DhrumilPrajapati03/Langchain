# -*- coding: utf-8 -*-
"""
Auto-generated Python script from: day9_test.ipynb
"""

# ### Day 9: Tools & Tool Calling

# Today you give your LLM superpowers — the ability to take real actions in the world. Tools transform a passive chatbot into an active agent that can search, calculate, fetch data, and call APIs.

# #### 1. What Is Tool Calling?

# Without tools:
#   User: "What's 2847 × 3921?"
#   LLM:  "11,161,287" ← might be wrong, LLMs hallucinate math
# 
# With tools:
#   User: "What's 2847 × 3921?"
#   LLM:  "I'll calculate that"
#     ↓
#   calls calculator_tool(2847 × 3921)
#     ↓
#   gets back: 11,157,087  ← always correct
#     ↓
#   LLM: "2847 × 3921 = 11,157,087"

# The LLM decides WHEN to use a tool and WHAT arguments to pass. You define WHAT tools exist.

# #### 2. How Tool Calling Works Internally

# User message
#      ↓
# LLM reads message + available tools
#      ↓
# LLM decides: "I need tool X with args {a: 1, b: 2}"
#      ↓
# LLM returns ToolCall object (not final answer yet)
#      ↓
# Your code executes the actual tool
#      ↓
# Tool result sent back to LLM
#      ↓
# LLM generates final answer using result

# ##### Setup

# Cell 1: Imports
from dotenv import load_dotenv
from langchain_groq import ChatGroq
#from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json, os, math, requests
from datetime import datetime

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
print("✅ Ready")

# #### 4. Creating Tools — Three Ways

# ##### Way 1: @tool decorator (simplest)

# Cell 2: Basic tools with @tool decorator
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use this for any arithmetic, algebra, or math calculations.
    Input should be a valid Python math expression like '2 + 2' or '100 * 0.18'.
    """
    try:
        # Safe eval - only math operations
        allowed = {
            'abs': abs, 'round': round,
            'min': min, 'max': max,
            'pow': pow, 'sqrt': math.sqrt,
            'pi': math.pi, 'e': math.e
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{result}"
    except Exception as ex:
        return f"Error: {ex}"


@tool
def get_current_datetime() -> str:
    """
    Get the current date and time.
    Use when user asks about current time, date, day of week, etc.
    Returns full datetime info including day name, date, and time.
    """
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y %H:%M:%S")


@tool
def word_counter(text: str) -> str:
    """
    Count words, characters, and sentences in a given text.
    Use when user asks about text statistics or length analysis.
    """
    words = len(text.split())
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))
    sentences = text.count('.') + text.count('!') + text.count('?')
    return json.dumps({
        "words": words,
        "characters": chars,
        "characters_no_spaces": chars_no_spaces,
        "sentences": max(sentences, 1)
    })


# Inspect tool metadata
print(f"Tool name: {calculator.name}")
print(f"Tool description: {calculator.description}")
print(f"Tool args schema: {calculator.args}")

# Cell 3: Test tools directly (before connecting to LLM)
print("Direct tool tests:")
print(f"  calculator('2847 * 3921') = {calculator.invoke('2847 * 3921')}")
print(f"  calculator('sqrt(144)') = {calculator.invoke('sqrt(144)')}")
print(f"  datetime = {get_current_datetime.invoke({})}")
print(f"  word_counter = {word_counter.invoke('Hello world. How are you?')}")

# ##### Way 2: Tools with complex inputs

# Cell 4: Tools that take structured inputs
from pydantic import BaseModel, Field
from typing import Optional

class WeatherInput(BaseModel):
    city: str = Field(description="City name to get weather for")
    unit: Optional[str] = Field(
        default="celsius",
        description="Temperature unit: 'celsius' or 'fahrenheit'"
    )

@tool(args_schema=WeatherInput)
def get_weather(city: str, unit: str = "celsius") -> str:
    """
    Get current weather for a city.
    Use when user asks about weather, temperature, or climate.
    """
    # Using Open-Meteo API - completely free, no key needed!
    try:
        # First get coordinates
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_resp = requests.get(geo_url, timeout=5).json()

        if not geo_resp.get("results"):
            return f"City '{city}' not found"

        lat = geo_resp["results"][0]["latitude"]
        lon = geo_resp["results"][0]["longitude"]
        name = geo_resp["results"][0]["name"]
        country = geo_resp["results"][0].get("country", "")

        # Get weather
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,weathercode,windspeed_10m"
            f"&temperature_unit={'fahrenheit' if unit=='fahrenheit' else 'celsius'}"
        )
        weather = requests.get(weather_url, timeout=5).json()
        current = weather["current"]

        temp = current["temperature_2m"]
        unit_symbol = "°F" if unit == "fahrenheit" else "°C"
        wind = current["windspeed_10m"]

        return (
            f"Weather in {name}, {country}: "
            f"{temp}{unit_symbol}, "
            f"Wind: {wind} km/h"
        )
    except Exception as e:
        return f"Weather fetch failed: {e}"


# Test it
print(get_weather.invoke({"city": "Ahmedabad", "unit": "celsius"}))
print(get_weather.invoke({"city": "Mumbai"}))

from ddgs import DDGS

@tool
def web_search(query: str) -> str:
    """
    Search the web for current information.
    Use for recent events, facts you're unsure about,
    prices, news, or anything requiring up-to-date info.
    Input should be a clear search query string.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found"

        formatted = []
        for r in results:
            formatted.append(
                f"Title: {r['title']}\n"
                f"URL: {r['href']}\n"
                f"Summary: {r['body'][:200]}"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search failed: {e}"


# Test it
result = web_search.invoke("Invincible latest season news")
print(result[:500])

# Cell 6: Bind tools to LLM
all_tools = [calculator, get_current_datetime, word_counter, get_weather, web_search]

# Bind - LLM now knows about these tools
llm_with_tools = llm.bind_tools(all_tools)

print(f"✅ LLM bound with {len(all_tools)} tools:")
for t in all_tools:
    print(f"   - {t.name}: {t.description[:60]}...")

# Cell 7: See tool calling in action - raw
message = HumanMessage(content="What is 15% of 8500?")
response = llm_with_tools.invoke([message])

print("Response type:", type(response))
print("\nContent:", response.content)
print("\nTool calls:", response.tool_calls)
# Notice: content might be empty - LLM wants to call a tool first!

# Cell 8: Understand the ToolCall object
response = llm_with_tools.invoke(
    [HumanMessage(content="What's the weather in Delhi?")]
)

if response.tool_calls:
    tool_call = response.tool_calls[0]
    print(tool_call)
    print("Tool Call Details:")
    print(f"  Name: {tool_call['name']}")
    print(f"  Args: {tool_call['args']}")
    print(f"  ID:   {tool_call['id']}")
else:
    print("No tool call made")
    print(response.content)

# Cell 9: Manual tool execution loop
# This is what happens inside every agent

def execute_tool_call(tool_call: dict) -> str:
    """Execute a single tool call and return result"""
    tool_map = {t.name: t for t in all_tools}
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name not in tool_map:
        return f"Unknown tool: {tool_name}"

    tool = tool_map[tool_name]
    result = tool.invoke(tool_args)
    return str(result)


def run_with_tools(user_message: str) -> str:
    """
    Complete tool-calling loop:
    1. Send message to LLM
    2. If LLM wants to use tools → execute them
    3. Send results back to LLM
    4. Get final answer
    """
    messages = [HumanMessage(content=user_message)]

    print(f"👤 User: {user_message}")

    # Step 1: First LLM call
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Step 2: Execute tools if requested
    while response.tool_calls:
        print(f"\n🔧 LLM wants to use tools:")

        tool_results = []
        for tool_call in response.tool_calls:
            print(f"   → {tool_call['name']}({tool_call['args']})")

            # Execute the tool
            result = execute_tool_call(tool_call)
            print(f"   ← Result: {result[:100]}")

            # Package result as ToolMessage
            tool_results.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                )
            )

        # Add tool results to message history
        messages.extend(tool_results)

        # Step 3: Call LLM again with tool results
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    # Step 4: Return final answer
    print(f"\n🤖 Final Answer: {response.content}")
    return response.content


# Test it!
run_with_tools("What is weather in Patdi?")

# Cell 10: Test multiple tool scenarios
print("\n" + "="*60)
run_with_tools("What's the weather in Mumbai right now?")

print("\n" + "="*60)
run_with_tools("What is current date?")

print("\n" + "="*60)
run_with_tools("Count the words in: 'The quick brown fox jumps over the lazy dog'")

# #### 7. Multi-Tool Calls in One Response

# Cell 11: LLM can call multiple tools in sequence
print("="*60)
run_with_tools(
    "What's the weather in Mumbai and delhi?"
)
# Watch: LLM calls get_weather TWICE - once per city!

# Cell 12: Complex multi-step tool use
print("="*60)
run_with_tools(
    "Search the web for the population of Tokyo, "
    "then calculate what 0.5% of that number is"
)
# Watch: LLM uses web_search first, then calculator with the result!

# Cell 13: Real-world style custom tools

# Simulated database
PRODUCT_DB = {
    "P001": {"name": "Laptop Pro", "price": 75000, "stock": 15},
    "P002": {"name": "Wireless Mouse", "price": 1500, "stock": 200},
    "P003": {"name": "USB Hub", "price": 2000, "stock": 50},
    "P004": {"name": "Monitor 27inch", "price": 35000, "stock": 8},
}

ORDER_DB = {
    "ORD-101": {"product": "P001", "qty": 2, "status": "shipped"},
    "ORD-102": {"product": "P002", "qty": 5, "status": "processing"},
    "ORD-103": {"product": "P003", "qty": 1, "status": "delivered"},
}


@tool
def check_product_stock(product_id: str) -> str:
    """
    Check stock and price for a product by its ID.
    Product IDs are in format P001, P002 etc.
    Use when user asks about product availability or pricing.
    """
    product = PRODUCT_DB.get(product_id.upper())
    if not product:
        return f"Product {product_id} not found"
    return json.dumps(product)


@tool
def check_order_status(order_id: str) -> str:
    """
    Check the status of a customer order.
    Order IDs are in format ORD-101, ORD-102 etc.
    Use when user asks about their order status or delivery.
    """
    order = ORDER_DB.get(order_id.upper())
    if not order:
        return f"Order {order_id} not found"

    product_info = PRODUCT_DB.get(order["product"], {})
    return json.dumps({
        "order_id": order_id,
        "product": product_info.get("name", "Unknown"),
        "quantity": order["qty"],
        "status": order["status"]
    })


@tool
def calculate_total(product_id: str, quantity: int) -> str:
    """
    Calculate total price for a product purchase.
    Use when user wants to know the total cost for buying multiple units.
    Input: product_id (e.g., P001) and quantity (number of units).
    """
    product = PRODUCT_DB.get(product_id.upper())
    if not product:
        return f"Product {product_id} not found"

    total = product["price"] * quantity
    return json.dumps({
        "product": product["name"],
        "unit_price": product["price"],
        "quantity": quantity,
        "total": total,
        "currency": "INR"
    })


# Create a shopping assistant
shopping_tools = [
    check_product_stock,
    check_order_status,
    calculate_total,
    calculator
]

shopping_llm = llm.bind_tools(shopping_tools)

def shopping_assistant(query: str) -> str:
    messages = [
        HumanMessage(content=f"""You are a helpful shopping assistant.
Answer the customer query using the available tools.
Always be helpful and specific.

Customer: {query}""")
    ]

    response = shopping_llm.invoke(messages)
    messages.append(response)

    while response.tool_calls:
        tool_map = {t.name: t for t in shopping_tools}
        tool_results = []

        for tc in response.tool_calls:
            result = tool_map[tc["name"]].invoke(tc["args"])
            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

        messages.extend(tool_results)
        response = shopping_llm.invoke(messages)
        messages.append(response)

    return response.content


# Test the shopping assistant
queries = [
    "Is the Laptop Pro in stock? How much does it cost?",
    "What's the status of order ORD-102?",
    "I want to buy 3 USB Hubs. What will be the total cost?",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"Customer: {q}")
    print(f"Assistant: {shopping_assistant(q)}")

# TOOL CREATION:
# - @tool                          → simple function → tool
# - @tool(args_schema=PydanticModel) → structured inputs
# - Tool.from_function()           → explicit control
# 
# TOOL EXECUTION:
# - tool.invoke(args)              → call tool directly
# - llm.bind_tools(tools)         → attach tools to LLM
# - response.tool_calls            → see what LLM wants to call
# - ToolMessage(content, id)       → send result back to LLM
# 
# DOCSTRING MATTERS:
# - The docstring IS the tool description the LLM reads.
# - Bad docstring → LLM won't know when to use the tool.
# - Be specific: "Use when user asks about X"
# 
# FREE APIS FOR TOOLS:
# - Open-Meteo    → weather (no key)
# - DuckDuckGo    → web search (no key)
# - REST Countries → country data (no key)
# - ExchangeRate  → currency rates (no key)
# - Open Library  → book data (no key)
