# -*- coding: utf-8 -*-
"""
Auto-generated Python script from: day11_test.ipynb
"""

# ### Day 11: ReAct Agents — Reason + Act Loops

# Today you learn the ReAct pattern — the foundation of all modern AI agents. Instead of one LLM call, agents loop: Think → Act → Observe → Think → Act... until the task is done.

# #### 1. What Makes an Agent Different?

# Chain (what you built so far):
#   Input → Step1 → Step2 → Step3 → Output
#   Fixed path, predetermined steps
# 
# Agent (what you build today):
#   Input → Think → "I need tool X" → Use X → Observe result
#          → Think → "I need tool Y" → Use Y → Observe result
#          → Think → "I have enough info" → Final Answer
#   Dynamic path, LLM decides next step every iteration

# #### 2. The ReAct Loop

# ┌─────────────────────────────────────────────┐
# │              ReAct Loop                     │
# │                                             │
# │  Question                                   │
# │     ↓                                       │
# │  REASON: "To answer this I need to..."      │
# │     ↓                                       │
# │  ACT: call_tool(name, args)                 │
# │     ↓                                       │
# │  OBSERVE: tool returned "..."               │
# │     ↓                                       │
# │  REASON: "Now I know X, but I still need Y" │
# │     ↓                                       │
# │  ACT: call_tool(name, args)                 │
# │     ↓                                       │
# │  OBSERVE: tool returned "..."               │
# │     ↓                                       │
# │  REASON: "I have all info needed"           │
# │     ↓                                       │
# │  Final Answer ✅                            │
# └─────────────────────────────────────────────┘

# Cell 1: Imports
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated, List
import operator, json, math, os, requests
from datetime import datetime
from ddgs import DDGS

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
print("✅ Ready")

response=llm.invoke("what is the meaning of dihh in slang word?")
print(response.content)

# Cell 2: All tools our agent will use

@tool
def calculator(expression: str) -> str:
    """
    Evaluate mathematical expressions.
    Use for any arithmetic, percentages, or numerical calculations.
    Input: valid Python math expression like '100 * 0.18' or 'sqrt(144)'.
    """
    try:
        allowed = {
            'abs': abs, 'round': round, 'min': min,
            'max': max, 'pow': pow, 'sqrt': math.sqrt,
            'pi': math.pi, 'e': math.e, 'log': math.log
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as ex:
        return f"Calculation error: {ex}"


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information.
    Use for recent facts, news, statistics, or anything
    that requires up-to-date knowledge.
    Input: a focused search query string.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"Title: {r['title']}\nInfo: {r['body'][:300]}"
            for r in results
        )
    except Exception as e:
        return f"Search error: {e}"


@tool
def get_weather(city: str) -> str:
    """
    Get current weather for any city.
    Use when user asks about temperature, weather conditions,
    or climate in a specific location.
    Input: city name as a string.
    """
    try:
        geo = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1",
            timeout=5
        ).json()
        if not geo.get("results"):
            return f"City '{city}' not found"

        r = geo["results"][0]
        lat, lon = r["latitude"], r["longitude"]
        name = r["name"]
        country = r.get("country", "")

        weather = requests.get(
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,weathercode,windspeed_10m,humidity",
            timeout=5
        ).json()

        c = weather["current"]
        return (
            f"{name}, {country}: "
            f"{c['temperature_2m']}°C, "
            f"Wind: {c['windspeed_10m']} km/h"
        )
    except Exception as e:
        return f"Weather error: {e}"


@tool
def get_current_time(timezone: str = "Asia/Kolkata") -> str:
    """
    Get current date and time.
    Use when user asks about current time, date, day, or year.
    Input: timezone string like 'Asia/Kolkata' or 'UTC'.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def wikipedia_search(topic: str) -> str:
    """
    Search Wikipedia for factual information about a topic.
    Use for historical facts, biographies, science concepts,
    or any topic needing encyclopedic knowledge.
    Input: topic or person to search for.
    """
    try:
        # Wikipedia API - free, no key needed
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + \
              topic.replace(" ", "_")
        resp = requests.get(url, timeout=5).json()

        if resp.get("type") == "disambiguation":
            return f"'{topic}' is ambiguous. Try a more specific term."

        title = resp.get("title", topic)
        extract = resp.get("extract", "No information found")
        return f"{title}: {extract[:500]}"
    except Exception as e:
        return f"Wikipedia error: {e}"


# All tools in one list
tools = [calculator, web_search, get_weather, get_current_time, wikipedia_search]
print(f"✅ {len(tools)} tools ready:")
for t in tools:
    print(f"   - {t.name}")

# Cell 3: Easiest way - LangGraph's prebuilt ReAct agent
from langgraph.prebuilt import create_react_agent
from click import prompt

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="""You are a helpful AI assistant with access to tools.

    For every task:
    1. Think about what information you need
    2. Use the most appropriate tool
    3. Analyze the result
    4. Use more tools if needed
    5. Give a clear, complete final answer

    Always show your reasoning. Be thorough but concise."""
)

print("✅ ReAct agent created")

# Cell 4: Run the agent and see the full ReAct loop
def run_agent(question: str, verbose: bool = True):
    """Run agent and display full reasoning trace"""
    print(f"\n{'='*60}")
    print(f"❓ Question: {question}")
    print('='*60)

    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    if verbose:
        print("\n📋 Full ReAct Trace:")
        for i, msg in enumerate(result["messages"]):
            if isinstance(msg, HumanMessage):
                print(f"\n[{i}] 👤 Human: {msg.content[:100]}")

            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"\n[{i}] 🧠 Reasoning → calling: {tc['name']}")
                        print(f"     Args: {tc['args']}")
                else:
                    print(f"\n[{i}] 🤖 Final Answer:")
                    print(f"     {msg.content}")

            elif isinstance(msg, ToolMessage):
                print(f"\n[{i}] 🔧 Tool Result ({msg.name}):")
                print(f"     {str(msg.content)[:200]}")

    # Return just the final answer
    final = result["messages"][-1].content
    return final


# Test simple queries
run_agent("What is 15 percantage of 85000?")

# Cell 5: Multi-step reasoning
run_agent(
    "Search for the population of Mumbai and calculate "
    "how many people that is per square kilometer "
    "given Mumbai is 603 square kilometers"
)

# Cell 6: Multi-tool chaining
run_agent(
    "What is the current weather in Ahmedabad? "
    "Also tell me what time it is and if I should carry an umbrella."
)

# Cell 7: Manual ReAct with LangGraph StateGraph
# This shows you EXACTLY what create_react_agent does internally

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

# ── State definition ──────────────────────────────────────────
class AgentState(TypedDict):
    """
    The state that flows through the graph.
    add_messages reducer appends new messages to history.
    """
    messages: Annotated[List, add_messages]


# ── Nodes ─────────────────────────────────────────────────────
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState) -> dict:
    """
    The thinking node.
    LLM reads message history and decides:
    - Call a tool, OR
    - Give final answer
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """
    The action node.
    Execute all tool calls from the last AI message.
    """
    tool_map = {t.name: t for t in tools}
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"   🔧 Executing: {tool_name}({tool_args})")

        if tool_name in tool_map:
            result = tool_map[tool_name].invoke(tool_args)
        else:
            result = f"Tool '{tool_name}' not found"

        tool_results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_name
            )
        )

    return {"messages": tool_results}


# ── Routing ───────────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    """
    Router: decide what happens after agent_node.
    - If LLM called tools → go to tool_node
    - If LLM gave answer → END
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tools"
    return "end"


# ── Build the graph ───────────────────────────────────────────
graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

# Set entry point
graph_builder.set_entry_point("agent")

# Add conditional routing after agent
graph_builder.add_conditional_edges(
    "agent",                    # from this node
    should_continue,            # use this function to decide
    {
        "use_tools": "tools",   # if tool calls → go to tools
        "end": END              # if done → end
    }
)

# After tools always go back to agent
graph_builder.add_edge("tools", "agent")

# Compile
manual_agent = graph_builder.compile()

print("✅ Manual ReAct agent built!")
print("\nGraph structure:")
print("  START → agent → (tool calls?) → tools → agent → ... → END")

# Cell 8: Visualize the graph structure
# This shows the exact flow as ASCII
print("""
Manual ReAct Graph:
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                    ┌────▼────┐
              ┌────►│  agent  │
              │     └────┬────┘
              │          │
              │    has tool calls?
              │     /          \\
              │   YES           NO
              │    │             │
              │ ┌──▼──┐       ┌──▼──┐
              │ │tools│       │ END │
              │ └──┬──┘       └─────┘
              │    │
              └────┘
""")

# Cell 9: Run the manual agent
def run_manual_agent(question: str):
    print(f"\n{'='*60}")
    print(f"❓ {question}")
    print('='*60)

    result = manual_agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # Show trace
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"\n👤 User: {msg.content[:100]}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n🧠 → Tool: {tc['name']}({tc['args']})")
            else:
                print(f"\n🤖 Answer: {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f"\n📥 Result: {msg.content[:150]}")


run_manual_agent("What is the capital of India? Tell me about it.")

# Cell 10: Add memory to the agent
from langgraph.checkpoint.memory import MemorySaver

# Memory saver persists state between invocations
memory = MemorySaver()

# Compile agent WITH memory
agent_with_memory = graph_builder.compile(checkpointer=memory)

# Thread ID groups messages into a conversation
config = {"configurable": {"thread_id": "conversation_1"}}

def chat_with_agent(question: str, config: dict):
    """Multi-turn agent conversation"""
    result = agent_with_memory.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config
    )
    answer = result["messages"][-1].content
    print(f"\n👤 You: {question}")
    print(f"🤖 Agent: {answer}")
    return answer


print("Multi-turn agent conversation:")
chat_with_agent("My name is Raj and I live in Ahmedabad", config)
chat_with_agent("What's the weather like where I live?", config)
# Agent remembers Ahmedabad from previous turn!
chat_with_agent("What is my name?", config)
# Agent remembers Raj!

# Cell 11: Prevent infinite loops
from langgraph.prebuilt import create_react_agent

safe_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a helpful assistant. Use tools wisely.",
)

def run_safe_agent(question: str, max_iterations: int = 10):
    """
    Run agent with iteration limit.
    Prevents runaway loops on complex/ambiguous queries.
    """
    result = safe_agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        {"recursion_limit": max_iterations}
    )

    # Count tool uses
    tool_uses = sum(
        1 for m in result["messages"]
        if isinstance(m, ToolMessage)
    )

    final = result["messages"][-1].content
    print(f"\n❓ {question}")
    print(f"🔧 Tool calls made: {tool_uses}")
    print(f"🤖 Answer: {final}")
    return final


run_safe_agent("Search for latest news about AI in India")

# Cell 12: Stream agent steps as they happen
print("Streaming ReAct agent:\n")

query = "What is the square root of the current year?"

for step in agent.stream(
    {"messages": [HumanMessage(content=query)]},
    stream_mode="values"
):
    last_msg = step["messages"][-1]

    if isinstance(last_msg, AIMessage):
        if last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                print(f"⚡ Calling: {tc['name']}({tc['args']})")
        elif last_msg.content:
            print(f"✅ Answer: {last_msg.content}")

    elif isinstance(last_msg, ToolMessage):
        print(f"📥 Got: {last_msg.content[:100]}")

# USE A CHAIN WHEN:              USE AN AGENT WHEN:
# ─────────────────────────────────────────────────────
# Steps are fixed & known        Steps are dynamic
# Simple linear flow             Needs decision making
# Fast execution needed          Multiple tool use
# Predictable outputs            Open-ended tasks
# Data transformation            Research & reasoning
# Summarization                  Planning & execution
# 
# EXAMPLES:
# Chain: PDF → extract text → summarize → translate
# Agent: "Research X, find recent news, compare with Y,
#         calculate Z, and write a report"
