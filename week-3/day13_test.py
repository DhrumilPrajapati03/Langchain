# -*- coding: utf-8 -*-
"""
Auto-generated Python script from: day13_test.ipynb
"""

# ### Day 13: LangGraph Patterns — Loops, Branches & Supervisor 🎯

# Today you master production-grade LangGraph patterns — retry logic, fallback branches, streaming state, and the supervisor pattern that orchestrates multiple specialized agents. This is the direct foundation of Project 3.

# ##### 1. What You're Learning Today
# - Pattern 1: Retry Loop        → retry failed operations automatically
# - Pattern 2: Fallback Branch   → graceful degradation on failure  
# - Pattern 3: Map-Reduce        → process many items in parallel
# - Pattern 4: Supervisor        → one agent coordinates many agents
# - Pattern 5: State Streaming   → watch graph execute in real time

# Cell 1: Imports
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List, Optional, Literal
import operator, json, time, os
from duckduckgo_search import DDGS

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
print("✅ Ready")

# ##### 2. Pattern 1 — Retry Loop with Error Handling

# Cell 2: Retry logic - critical for production agents

class RetryState(TypedDict):
    messages:     Annotated[List, add_messages]
    task:         str
    result:       Optional[str]
    attempts:     int
    max_attempts: int
    last_error:   Optional[str]
    success:      bool


def attempt_task_node(state: RetryState) -> dict:
    """
    Try to complete the task.
    Simulates occasional failures for demo.
    """
    attempt = state["attempts"] + 1
    print(f"attempt_task (try {attempt}/{state['max_attempts']})")

    try:
        response = llm.invoke([
            SystemMessage(content="""Complete the task.
Respond with valid JSON only:
{"result": "your answer", "confidence": 0.0-1.0}"""),
            HumanMessage(content=f"Task: {state['task']}")
        ])

        # Parse JSON response
        data = json.loads(response.content.strip())

        # Simulate failure if confidence is low
        if data.get("confidence", 0) < 0.6 and attempt < state["max_attempts"]:
            raise ValueError(
                f"Low confidence: {data.get('confidence')}"
            )

        return {
            "result":    data.get("result", response.content),
            "attempts":  attempt,
            "success":   True,
            "last_error": None
        }

    except json.JSONDecodeError as e:
        return {
            "attempts":   attempt,
            "success":    False,
            "last_error": f"JSON parse error: {e}"
        }
    except Exception as e:
        return {
            "attempts":   attempt,
            "success":    False,
            "last_error": str(e)
        }


def handle_failure_node(state: RetryState) -> dict:
    """Called when all retries exhausted"""
    print("handle_failure")
    return {
        "result": f"Task failed after {state['attempts']} attempts. "
                  f"Last error: {state['last_error']}"
    }


def should_retry(state: RetryState) -> str:
    """Router: retry, succeed, or give up"""
    if state["success"]:
        print(f"     → Success on attempt {state['attempts']}")
        return "success"

    if state["attempts"] >= state["max_attempts"]:
        print(f"     → Max attempts reached, giving up")
        return "give_up"

    wait = state["attempts"] * 1  # exponential-ish backoff
    print(f"     → Retrying in {wait}s (error: {state['last_error']})")
    time.sleep(wait)
    return "retry"


# Build retry graph
retry_graph = StateGraph(RetryState)
retry_graph.add_node("attempt_task",   attempt_task_node)
retry_graph.add_node("handle_failure", handle_failure_node)

retry_graph.add_edge(START, "attempt_task")
retry_graph.add_conditional_edges(
    "attempt_task",
    should_retry,
    {
        "retry":    "attempt_task",     # ← loop back
        "success":  END,
        "give_up":  "handle_failure"
    }
)
retry_graph.add_edge("handle_failure", END)

retry_runner = retry_graph.compile()

print("Testing retry pattern:\n")
result = retry_runner.invoke({
    "task":         "Explain what LangGraph is in one sentence",
    "attempts":     0,
    "max_attempts": 3,
    "success":      False,
    "messages":     []
})
print(f"\n✅ Result: {result['result']}")
print(f"   Attempts: {result['attempts']}")
print(f"   Success:  {result['success']}")

# ##### Pattern 2 — Fallback Branch

# Cell 3: Graceful degradation - try primary, fall back if needed

class FallbackState(TypedDict):
    messages:       Annotated[List, add_messages]
    query:          str
    primary_result: Optional[str]
    final_result:   Optional[str]
    used_fallback:  bool
    error:          Optional[str]


def primary_search_node(state: FallbackState) -> dict:
    """Primary: try web search first"""
    print("  📍 primary_search (web)")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(state["query"], max_results=2))
        if not results:
            raise ValueError("No search results")

        result = "\n".join(r["body"][:200] for r in results)
        return {
            "primary_result": result,
            "error": None
        }
    except Exception as e:
        return {
            "primary_result": None,
            "error": str(e)
        }


def fallback_llm_node(state: FallbackState) -> dict:
    """Fallback: use LLM knowledge directly"""
    print("  📍 fallback_llm (using internal knowledge)")
    response = llm.invoke([
        SystemMessage(content="Answer from your training knowledge. Be clear and helpful."),
        HumanMessage(content=state["query"])
    ])
    return {
        "final_result":  response.content,
        "used_fallback": True
    }


def process_primary_node(state: FallbackState) -> dict:
    """Process successful primary search result"""
    print("  📍 process_primary")
    response = llm.invoke([
        SystemMessage(content="Summarize these search results into a clear answer."),
        HumanMessage(content=f"Query: {state['query']}\n\nResults:\n{state['primary_result']}")
    ])
    return {
        "final_result":  response.content,
        "used_fallback": False
    }


def route_after_primary(state: FallbackState) -> str:
    """Check if primary succeeded"""
    if state.get("primary_result") and not state.get("error"):
        print("     → Primary succeeded")
        return "process"
    print(f"     → Primary failed ({state.get('error')}), using fallback")
    return "fallback"


# Build fallback graph
fallback_graph = StateGraph(FallbackState)
fallback_graph.add_node("primary_search",  primary_search_node)
fallback_graph.add_node("process_primary", process_primary_node)
fallback_graph.add_node("fallback_llm",    fallback_llm_node)

fallback_graph.add_edge(START, "primary_search")
fallback_graph.add_conditional_edges(
    "primary_search",
    route_after_primary,
    {
        "process":  "process_primary",
        "fallback": "fallback_llm"
    }
)
fallback_graph.add_edge("process_primary", END)
fallback_graph.add_edge("fallback_llm",    END)

fallback_runner = fallback_graph.compile()

# Test
queries = [
    "What is the capital of France?",
    "Latest Python version released in 2025",
]

for q in queries:
    print(f"\n{'='*50}")
    print(f"Query: {q}")
    res = fallback_runner.invoke({
        "query": q, "used_fallback": False, "messages": []
    })
    print(f"Used fallback: {res['used_fallback']}")
    print(f"Answer: {res['final_result'][:200]}")

# #### Pattern 3 — Map-Reduce

# Cell 4: Map-Reduce - process many items, combine results

class MapReduceState(TypedDict):
    messages:    Annotated[List, add_messages]
    documents:   List[str]
    summaries:   Annotated[List, operator.add]   # accumulate
    final_report: Optional[str]


def map_summarize_node(state: MapReduceState) -> dict:
    """
    MAP phase: summarize each document individually.
    In production this would fan out in parallel.
    """
    print(f"  📍 map_summarize ({len(state['documents'])} docs)")
    summaries = []

    for i, doc in enumerate(state["documents"]):
        print(f"     Summarizing doc {i+1}...")
        resp = llm.invoke([
            SystemMessage(content="Summarize in exactly 1 sentence."),
            HumanMessage(content=doc)
        ])
        summaries.append(f"Doc {i+1}: {resp.content}")

    return {"summaries": summaries}


def reduce_report_node(state: MapReduceState) -> dict:
    """
    REDUCE phase: combine all summaries into final report.
    """
    print("  📍 reduce_report")

    all_summaries = "\n".join(state["summaries"])
    resp = llm.invoke([
        SystemMessage(content="""Create a cohesive report from these summaries.
Structure: Executive Summary (2 sentences), Key Themes (3 bullets), Conclusion (1 sentence)."""),
        HumanMessage(content=all_summaries)
    ])

    return {"final_report": resp.content}


# Build map-reduce graph
mr_graph = StateGraph(MapReduceState)
mr_graph.add_node("map_summarize", map_summarize_node)
mr_graph.add_node("reduce_report", reduce_report_node)

mr_graph.add_edge(START,          "map_summarize")
mr_graph.add_edge("map_summarize","reduce_report")
mr_graph.add_edge("reduce_report", END)

mr_runner = mr_graph.compile()

# Test with multiple documents
documents = [
    """LangChain is a framework for building LLM applications.
    It provides tools for chaining prompts, managing memory,
    and integrating with external data sources.""",

    """LangGraph extends LangChain with graph-based workflows.
    It supports stateful agents, loops, and human-in-the-loop
    patterns for complex multi-step tasks.""",

    """LangSmith is the observability platform for LangChain apps.
    It provides tracing, debugging, and evaluation tools
    to monitor AI application performance in production.""",
]

print("Running Map-Reduce on 3 documents:\n")
result = mr_runner.invoke({
    "documents": documents,
    "summaries": [],
    "messages":  []
})

print(f"\n✅ Final Report:")
print(result["final_report"])

# #####  Pattern 4 — Supervisor Agent

# Cell 5: The Supervisor Pattern
# One LLM decides which specialist agent handles each subtask

class SupervisorState(TypedDict):
    messages:     Annotated[List, add_messages]
    task:         str
    subtasks:     List[str]
    results:      Annotated[List, operator.add]
    next_agent:   Optional[str]
    final_answer: Optional[str]
    iteration:    int


# ── Specialist agents ─────────────────────────────────────────

def researcher_node(state: SupervisorState) -> dict:
    """Specialist: web research"""
    print("  📍 Agent: RESEARCHER")
    current_task = state["subtasks"][state["iteration"]] \
        if state["subtasks"] else state["task"]

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(current_task, max_results=2))
        info = "\n".join(r["body"][:200] for r in results) \
            if results else "No results"
    except Exception:
        info = "Web search unavailable"

    resp = llm.invoke([
        SystemMessage(content="Summarize research findings concisely."),
        HumanMessage(content=f"Task: {current_task}\nFindings: {info}")
    ])

    return {
        "results":   [f"[RESEARCH] {resp.content}"],
        "iteration": state["iteration"] + 1
    }


def analyst_node(state: SupervisorState) -> dict:
    """Specialist: data analysis and reasoning"""
    print("  📍 Agent: ANALYST")
    current_task = state["subtasks"][state["iteration"]] \
        if state["subtasks"] else state["task"]

    resp = llm.invoke([
        SystemMessage(content="""You are an expert analyst.
Analyze the given information and provide insights.
Be specific with numbers and reasoning."""),
        HumanMessage(content=f"Analyze: {current_task}\n\nContext: {state['results']}")
    ])

    return {
        "results":   [f"[ANALYSIS] {resp.content}"],
        "iteration": state["iteration"] + 1
    }


def writer_node(state: SupervisorState) -> dict:
    """Specialist: writing and summarization"""
    print("  📍 Agent: WRITER")

    all_results = "\n\n".join(state["results"])
    resp = llm.invoke([
        SystemMessage(content="""You are an expert writer.
Create a clear, well-structured final answer from the provided information.
Use headers and bullet points where helpful."""),
        HumanMessage(content=f"""
Original task: {state['task']}

Research and analysis completed:
{all_results}

Write the final comprehensive answer.""")
    ])

    return {
        "results":      [f"[FINAL WRITE-UP] {resp.content}"],
        "final_answer": resp.content,
        "iteration":    state["iteration"] + 1
    }


# ── Supervisor node ───────────────────────────────────────────

SUPERVISOR_PROMPT = """You are a supervisor coordinating a team of specialist agents.

Available agents:
- RESEARCHER: searches web for current information and facts
- ANALYST: performs deep analysis, reasoning, and calculations
- WRITER: creates final polished answers and reports
- FINISH: call this when the task is fully complete

Work completed so far:
{results}

Current iteration: {iteration}
Original task: {task}

Decide which agent should act next.
Respond with JSON only: {{"next": "RESEARCHER|ANALYST|WRITER|FINISH", "reason": "why"}}"""


def supervisor_node(state: SupervisorState) -> dict:
    """The supervisor decides who acts next"""
    print("  📍 SUPERVISOR deciding...")

    results_text = "\n".join(state["results"]) if state["results"] else "None yet"

    resp = llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT.format(
            results=results_text,
            iteration=state["iteration"],
            task=state["task"]
        ))
    ])

    try:
        data = json.loads(resp.content.strip())
        next_agent = data.get("next", "FINISH")
        reason = data.get("reason", "")
    except Exception:
        next_agent = "FINISH"
        reason = "Parse error"

    # Safety: force FINISH after 5 iterations
    if state["iteration"] >= 5:
        next_agent = "FINISH"
        reason = "Max iterations reached"

    print(f"     → Next: {next_agent} ({reason})")
    return {"next_agent": next_agent}


def finalizer_node(state: SupervisorState) -> dict:
    """Create the final consolidated answer"""
    print("  📍 FINALIZER")

    if state.get("final_answer"):
        return {}   # writer already created final answer

    # If no writer was called, consolidate directly
    all_results = "\n\n".join(state["results"])
    resp = llm.invoke([
        SystemMessage(content="Create a clear final answer from all the work done."),
        HumanMessage(content=f"Task: {state['task']}\n\nWork done:\n{all_results}")
    ])
    return {"final_answer": resp.content}


def route_supervisor(state: SupervisorState) -> str:
    """Route based on supervisor's decision"""
    routes = {
        "RESEARCHER": "researcher",
        "ANALYST":    "analyst",
        "WRITER":     "writer",
        "FINISH":     "finalizer"
    }
    return routes.get(state.get("next_agent", "FINISH"), "finalizer")


# ── Build supervisor graph ────────────────────────────────────
sv_graph = StateGraph(SupervisorState)

sv_graph.add_node("supervisor", supervisor_node)
sv_graph.add_node("researcher", researcher_node)
sv_graph.add_node("analyst",    analyst_node)
sv_graph.add_node("writer",     writer_node)
sv_graph.add_node("finalizer",  finalizer_node)

sv_graph.add_edge(START, "supervisor")

sv_graph.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "analyst":    "analyst",
        "writer":     "writer",
        "finalizer":  "finalizer"
    }
)

# All specialists report back to supervisor
for agent in ["researcher", "analyst", "writer"]:
    sv_graph.add_edge(agent, "supervisor")

sv_graph.add_edge("finalizer", END)
supervisor_runner = sv_graph.compile()

print("✅ Supervisor graph ready\n")
print("Graph flow:")
print("  START → supervisor → (researcher|analyst|writer) → supervisor → ... → finalizer → END")

# Cell 6: Run the supervisor
print("="*60)
print("Task: Research Python vs JavaScript for AI development")
print("="*60 + "\n")

result = supervisor_runner.invoke({
    "task":      "Compare Python vs JavaScript for AI development. Which is better and why?",
    "subtasks":  [],
    "results":   [],
    "iteration": 0,
    "messages":  []
})

print(f"\n{'='*60}")
print("FINAL ANSWER:")
print('='*60)
print(result["final_answer"])
print(f"\nTotal iterations: {result['iteration']}")

# ##### Pattern 5 — Streaming Graph State

# Cell 7: Watch graph execute step by step in real time

print("Streaming supervisor execution:\n")

for step in supervisor_runner.stream(
    {
        "task":      "What are the top 3 use cases for LangGraph in production?",
        "subtasks":  [],
        "results":   [],
        "iteration": 0,
        "messages":  []
    },
    stream_mode="updates"   # get state updates after each node
):
    # step is a dict: {node_name: state_update}
    for node_name, state_update in step.items():
        print(f"\n⚡ Node '{node_name}' completed:")

        if "next_agent" in state_update:
            print(f"   Next agent: {state_update['next_agent']}")

        if "results" in state_update:
            for r in state_update["results"]:
                print(f"   Result: {r[:100]}...")

        if "final_answer" in state_update and state_update["final_answer"]:
            print(f"   ✅ Final answer ready!")

# ##### Combine Patterns — Production-Ready Agent

# Cell 8: A realistic agent combining multiple patterns

class ProductionState(TypedDict):
    messages:     Annotated[List, add_messages]
    query:        str
    intent:       Optional[str]
    search_result: Optional[str]
    analysis:     Optional[str]
    final:        Optional[str]
    attempts:     int
    error:        Optional[str]


def classify_node(state: ProductionState) -> dict:
    """Classify query intent"""
    resp = llm.invoke([
        SystemMessage(content="Classify as: RESEARCH, ANALYSIS, or SIMPLE. One word only."),
        HumanMessage(content=state["query"])
    ])
    intent = resp.content.strip().upper()
    return {"intent": intent}


def search_with_retry(state: ProductionState) -> dict:
    """Search with built-in retry"""
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(state["query"], max_results=3))
            if results:
                info = "\n".join(r["body"][:200] for r in results)
                return {"search_result": info, "attempts": attempt + 1, "error": None}
        except Exception as e:
            if attempt == 2:
                return {"search_result": None, "attempts": 3, "error": str(e)}
            time.sleep(attempt + 1)

    return {"search_result": None, "attempts": 3, "error": "All retries failed"}


def analyze_node(state: ProductionState) -> dict:
    """Deep analysis of search results"""
    context = state.get("search_result") or "No search results available"
    resp = llm.invoke([
        SystemMessage(content="Analyze and synthesize the information. Be thorough."),
        HumanMessage(content=f"Query: {state['query']}\nContext: {context}")
    ])
    return {"analysis": resp.content}


def simple_answer_node(state: ProductionState) -> dict:
    """Quick answer for simple queries"""
    resp = llm.invoke([
        SystemMessage(content="Answer concisely and directly."),
        HumanMessage(content=state["query"])
    ])
    return {"final": resp.content}


def compose_final_node(state: ProductionState) -> dict:
    """Compose final answer from analysis"""
    resp = llm.invoke([
        SystemMessage(content="Write a clear, complete answer using the analysis."),
        HumanMessage(content=f"Query: {state['query']}\nAnalysis: {state.get('analysis', '')}")
    ])
    return {"final": resp.content}


def route_by_intent(state: ProductionState) -> str:
    intent = state.get("intent", "SIMPLE")
    if intent == "RESEARCH":
        return "search"
    elif intent == "ANALYSIS":
        return "analyze"
    return "simple"


def route_after_search(state: ProductionState) -> str:
    if state.get("error"):
        return "analyze"   # fallback to analysis
    return "analyze"


# Build production graph
prod_graph = StateGraph(ProductionState)
prod_graph.add_node("classify",      classify_node)
prod_graph.add_node("search",        search_with_retry)
prod_graph.add_node("analyze",       analyze_node)
prod_graph.add_node("simple_answer", simple_answer_node)
prod_graph.add_node("compose_final", compose_final_node)

prod_graph.add_edge(START, "classify")
prod_graph.add_conditional_edges(
    "classify", route_by_intent,
    {"search": "search", "analyze": "analyze", "simple": "simple_answer"}
)
prod_graph.add_conditional_edges(
    "search", route_after_search,
    {"analyze": "analyze"}
)
prod_graph.add_edge("analyze",       "compose_final")
prod_graph.add_edge("simple_answer", END)
prod_graph.add_edge("compose_final", END)

prod_runner = prod_graph.compile()

# Test
test_queries = [
    "What is 2 + 2?",
    "Research the latest developments in LangGraph",
    "Analyze the pros and cons of using RAG vs fine-tuning",
]

for q in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {q}")
    res = prod_runner.invoke({
        "query":    q,
        "attempts": 0,
        "messages": []
    })
    print(f"Intent: {res.get('intent')}")
    print(f"Answer: {res.get('final', res.get('analysis', ''))[:200]}...")

# ##### Patterns Summary

# PATTERN         TRIGGER CONDITION          KEY NODES
# ─────────────────────────────────────────────────────────────────
# Retry Loop      Task may fail temporarily  attempt → check → retry/end
# Fallback Branch Primary source may fail    primary → check → fallback/process
# Map-Reduce      Many items to process      map(items) → reduce(results)
# Supervisor      Complex multi-step tasks   supervisor → specialist → supervisor
# Streaming       Need real-time visibility  .stream(mode="updates")
# 
# COMBINING PATTERNS (production checklist):
#   ✅ Retry on every external call (APIs, search, LLM)
#   ✅ Fallback for every critical path
#   ✅ Supervisor for tasks needing 3+ steps
#   ✅ Map-Reduce for batch document processing
#   ✅ Streaming for any user-facing interface
#   ✅ Checkpointing for multi-turn conversations
