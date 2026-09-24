# -*- coding: utf-8 -*-
"""
Auto-generated Python script from: day12_test.ipynb
"""

# ### Day 12: LangGraph Deep Dive

# Today you master LangGraph — the framework that turns simple agents into sophisticated AI systems. You'll learn state management, complex routing, parallel execution, and build graphs that can handle real-world complexity.

# #### 1. Why LangGraph Over Simple Agents?

# Simple ReAct Agent (Day 11):
#   Linear loop → tool → tool → answer
#   No complex branching, no parallel work,
#   no checkpointing, no human approval
# 
# LangGraph:
#   Full state machine with:
#   ✅ Custom state schemas
#   ✅ Parallel node execution
#   ✅ Conditional routing
#   ✅ Cycles and loops
#   ✅ Human-in-the-loop
#   ✅ Persistent checkpointing
#   ✅ Sub-graphs

# #### 2. Core LangGraph Concepts

# ┌─────────────────────────────────────────────────┐
# │  StateGraph                                     │
# │                                                 │
# │  State     → shared data flowing through graph  │
# │  Nodes     → functions that transform state     │
# │  Edges     → connections between nodes          │
# │  Routing   → conditional next-node decisions    │
# │  Reducers  → how state fields get updated       │
# └─────────────────────────────────────────────────┘

# #### 3. Setup

# Cell 1: Imports
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List, Literal, Optional
import operator, json, math, os
from datetime import datetime

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
print("✅ Ready")

# #### State — The Heart of LangGraph

# Cell 2: Understanding State and Reducers

# ── Basic state ───────────────────────────────────────────────
class BasicState(TypedDict):
    messages: Annotated[List, add_messages]  # reducer: append
    count: int                                # reducer: replace


# ── Custom reducer ────────────────────────────────────────────
def append_to_list(existing: List, new: List) -> List:
    """Custom reducer: append new items to existing list"""
    return existing + new


class RichState(TypedDict):
    messages:   Annotated[List, add_messages]   # append messages
    steps_taken: Annotated[List, append_to_list] # append steps
    final_answer: Optional[str]                  # replace each time
    iteration:  int                              # replace each time
    errors:     Annotated[List, append_to_list]  # append errors


# ── Show how reducers work ────────────────────────────────────
print("Reducer behavior:")
print("""
  add_messages:    [msg1] + [msg2] → [msg1, msg2]  (append)
  append_to_list:  [a, b] + [c]   → [a, b, c]     (append)
  plain field:     old_val → new_val               (replace)
""")

# #### Your First Custom Graph — Document Processor

# Cell 3: Multi-node document processing pipeline

class DocState(TypedDict):
    """State for document processing graph"""
    raw_text:   str
    language:   Optional[str]
    summary:    Optional[str]
    key_points: Optional[List[str]]
    translated: Optional[str]
    messages:   Annotated[List, add_messages]


# ── Node functions ─────────────────────────────────────────────

def detect_language_node(state: DocState) -> dict:
    """Node 1: Detect the language of the document"""
    print("  📍 Node: detect_language")

    response = llm.invoke([
        SystemMessage(content="Detect the language of this text. Reply with ONLY the language name. Example: 'English' or 'Hindi'"),
        HumanMessage(content=state["raw_text"][:500])
    ])

    language = response.content.strip()
    return {
        "language": language,
        "messages": [AIMessage(content=f"Detected language: {language}")]
    }


def summarize_node(state: DocState) -> dict:
    """Node 2: Summarize the document"""
    print("  📍 Node: summarize")

    response = llm.invoke([
        SystemMessage(content="Summarize this text in 2-3 sentences. Be concise."),
        HumanMessage(content=state["raw_text"])
    ])

    return {
        "summary": response.content,
        "messages": [AIMessage(content=f"Summary created")]
    }


def extract_points_node(state: DocState) -> dict:
    """Node 3: Extract key points as a list"""
    print("  📍 Node: extract_points")

    response = llm.invoke([
        SystemMessage(content="""Extract exactly 3 key points from this text.
Return as JSON array only: ["point 1", "point 2", "point 3"]
No other text."""),
        HumanMessage(content=state["raw_text"])
    ])

    try:
        points = json.loads(response.content.strip())
    except Exception:
        points = [response.content]

    return {
        "key_points": points,
        "messages": [AIMessage(content="Key points extracted")]
    }


def translate_node(state: DocState) -> dict:
    """Node 4: Translate summary to Hindi (only if not already Hindi)"""
    print("  📍 Node: translate")

    if state.get("language") == "Hindi":
        return {
            "translated": state["summary"],
            "messages": [AIMessage(content="Already in Hindi, skipping translation")]
        }

    response = llm.invoke([
        SystemMessage(content="Translate this text to Hindi. Return only the translation."),
        HumanMessage(content=state["summary"])
    ])

    return {
        "translated": response.content,
        "messages": [AIMessage(content="Translated to Hindi")]
    }


# ── Build the graph ───────────────────────────────────────────
doc_graph = StateGraph(DocState)

doc_graph.add_node("detect_language", detect_language_node)
doc_graph.add_node("summarize", summarize_node)
doc_graph.add_node("extract_points", extract_points_node)
doc_graph.add_node("translate", translate_node)

# Linear edges
doc_graph.add_edge(START, "detect_language")
doc_graph.add_edge("detect_language", "summarize")
doc_graph.add_edge("summarize", "extract_points")
doc_graph.add_edge("extract_points", "translate")
doc_graph.add_edge("translate", END)

doc_pipeline = doc_graph.compile()

# Run it
sample_text = """
LangGraph is a powerful library for building stateful multi-agent 
applications using LangChain. It was developed by the LangChain team 
and released in 2024. The library uses a graph-based approach where 
nodes represent processing steps and edges define the flow between them. 
LangGraph is particularly useful for complex workflows that require 
loops, branching, and human-in-the-loop interactions.
"""

print("Running document processing pipeline:\n")
result = doc_pipeline.invoke({"raw_text": sample_text, "messages": []})

print(f"\n✅ Results:")
print(f"   Language:   {result['language']}")
print(f"   Summary:    {result['summary'][:100]}...")
print(f"   Key Points: {result['key_points']}")
print(f"   Translated: {result['translated'][:100]}...")

# #### Conditional Routing — Smart Branching

# Cell 4: Conditional edges - graph decides its own path

class RoutingState(TypedDict):
    messages:   Annotated[List, add_messages]
    user_input: str
    intent:     Optional[str]
    response:   Optional[str]


def classify_intent_node(state: RoutingState) -> dict:
    """Classify user intent to route correctly"""
    print("  📍 Node: classify_intent")

    response = llm.invoke([
        SystemMessage(content="""Classify the user's intent into exactly one category:
MATH       - mathematical calculation needed
WEATHER    - weather information needed  
FACTUAL    - general knowledge question
CREATIVE   - creative writing or brainstorming
UNKNOWN    - doesn't fit other categories

Reply with ONLY the category word."""),
        HumanMessage(content=state["user_input"])
    ])

    intent = response.content.strip().upper()
    print(f"     Intent detected: {intent}")
    return {"intent": intent}


def math_node(state: RoutingState) -> dict:
    """Handle math queries"""
    print("  📍 Node: math_handler")
    response = llm.invoke([
        SystemMessage(content="You are a math expert. Solve step by step."),
        HumanMessage(content=state["user_input"])
    ])
    return {"response": response.content}


def weather_node(state: RoutingState) -> dict:
    """Handle weather queries"""
    print("  📍 Node: weather_handler")
    response = llm.invoke([
        SystemMessage(content="Help with weather queries. If you need real data, say so."),
        HumanMessage(content=state["user_input"])
    ])
    return {"response": response.content}


def factual_node(state: RoutingState) -> dict:
    """Handle factual queries"""
    print("  📍 Node: factual_handler")
    response = llm.invoke([
        SystemMessage(content="Answer factual questions accurately and concisely."),
        HumanMessage(content=state["user_input"])
    ])
    return {"response": response.content}


def creative_node(state: RoutingState) -> dict:
    """Handle creative queries"""
    print("  📍 Node: creative_handler")
    response = llm.invoke([
        SystemMessage(content="You are a creative assistant. Be imaginative and engaging."),
        HumanMessage(content=state["user_input"])
    ])
    return {"response": response.content}


def fallback_node(state: RoutingState) -> dict:
    """Handle unknown queries"""
    print("  📍 Node: fallback_handler")
    response = llm.invoke([
        HumanMessage(content=state["user_input"])
    ])
    return {"response": response.content}


# ── Router function ───────────────────────────────────────────
def route_by_intent(state: RoutingState) -> str:
    """Return the next node name based on intent"""
    intent_map = {
        "MATH":     "math_handler",
        "WEATHER":  "weather_handler",
        "FACTUAL":  "factual_handler",
        "CREATIVE": "creative_handler",
    }
    return intent_map.get(state["intent"], "fallback_handler")


# ── Build routing graph ───────────────────────────────────────
router_graph = StateGraph(RoutingState)

router_graph.add_node("classify_intent", classify_intent_node)
router_graph.add_node("math_handler",    math_node)
router_graph.add_node("weather_handler", weather_node)
router_graph.add_node("factual_handler", factual_node)
router_graph.add_node("creative_handler",creative_node)
router_graph.add_node("fallback_handler",fallback_node)

router_graph.add_edge(START, "classify_intent")

# Conditional edge - one source, many possible destinations
router_graph.add_conditional_edges(
    "classify_intent",   # from this node
    route_by_intent,     # call this function
    {                    # map return values to nodes
        "math_handler":    "math_handler",
        "weather_handler": "weather_handler",
        "factual_handler": "factual_handler",
        "creative_handler":"creative_handler",
        "fallback_handler":"fallback_handler",
    }
)

# All handlers go to END
for node in ["math_handler", "weather_handler",
             "factual_handler", "creative_handler", "fallback_handler"]:
    router_graph.add_edge(node, END)

router = router_graph.compile()

# Test routing
test_queries = [
    "What is 15% of 47,500?",
    "Will it rain in Mumbai today?",
    "Who invented the telephone?",
    "Write a haiku about Python programming",
]

for query in test_queries:
    print(f"\n{'='*50}")
    print(f"Query: {query}")
    result = router.invoke({
        "user_input": query,
        "messages": []
    })
    print(f"Response: {result['response'][:150]}...")

# #### Loops — Iterative Refinement

# Cell 5: Loops in LangGraph - refine until quality threshold

class RefinementState(TypedDict):
    messages:    Annotated[List, add_messages]
    topic:       str
    draft:       Optional[str]
    feedback:    Optional[str]
    quality:     Optional[int]   # 1-10 score
    iteration:   int
    final:       Optional[str]


def write_draft_node(state: RefinementState) -> dict:
    """Write or rewrite based on feedback"""
    print(f"  📍 Node: write_draft (iteration {state['iteration'] + 1})")

    if state.get("feedback") and state.get("draft"):
        # Rewrite based on feedback
        prompt = f"""Rewrite this draft based on the feedback.

Original draft: {state['draft']}

Feedback to address: {state['feedback']}

Write an improved version."""
    else:
        # First draft
        prompt = f"Write a short 3-sentence explanation of: {state['topic']}"

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "draft": response.content,
        "iteration": state["iteration"] + 1
    }


def evaluate_node(state: RefinementState) -> dict:
    """Evaluate draft quality and give feedback"""
    print(f"  📍 Node: evaluate")

    response = llm.invoke([
        SystemMessage(content="""Evaluate this text. Respond with JSON only:
{
  "score": <1-10>,
  "feedback": "<specific improvement needed or 'Looks great!' if score >= 8>"
}"""),
        HumanMessage(content=state["draft"])
    ])

    try:
        data = json.loads(response.content.strip())
        score = int(data.get("score", 5))
        feedback = data.get("feedback", "Needs improvement")
    except Exception:
        score = 5
        feedback = "Needs improvement"

    print(f"     Quality score: {score}/10")
    return {"quality": score, "feedback": feedback}


def finalize_node(state: RefinementState) -> dict:
    """Accept the draft as final"""
    print("  📍 Node: finalize ✅")
    return {"final": state["draft"]}


# ── Loop router ───────────────────────────────────────────────
def should_refine(state: RefinementState) -> str:
    """
    Continue refining if quality < 8 AND under iteration limit.
    Otherwise finalize.
    """
    quality = state.get("quality", 0)
    iteration = state.get("iteration", 0)

    if quality >= 8 or iteration >= 3:
        reason = "quality met" if quality >= 8 else "max iterations"
        print(f"     → Finalizing ({reason})")
        return "finalize"

    print(f"     → Refining (score {quality} < 8)")
    return "refine"


# ── Build refinement graph ────────────────────────────────────
refine_graph = StateGraph(RefinementState)

refine_graph.add_node("write_draft", write_draft_node)
refine_graph.add_node("evaluate",   evaluate_node)
refine_graph.add_node("finalize",   finalize_node)

refine_graph.add_edge(START, "write_draft")
refine_graph.add_edge("write_draft", "evaluate")

# Loop: evaluate → write_draft (refine) or finalize
refine_graph.add_conditional_edges(
    "evaluate",
    should_refine,
    {
        "refine":   "write_draft",  # ← loop back!
        "finalize": "finalize"
    }
)

refine_graph.add_edge("finalize", END)
refiner = refine_graph.compile()

# Test the refinement loop
print("Running iterative refinement:\n")
result = refiner.invoke({
    "topic": "How LangGraph enables multi-agent systems",
    "iteration": 0,
    "messages": []
})

print(f"\n✅ Final result after {result['iteration']} iteration(s):")
print(f"   Quality: {result['quality']}/10")
print(f"   Final: {result['final']}")

# #### Parallel Nodes — Speed Through Concurrency

# Cell 6: Run multiple nodes simultaneously

class ParallelState(TypedDict):
    messages:   Annotated[List, add_messages]
    topic:      str
    pros:       Optional[str]
    cons:       Optional[str]
    examples:   Optional[str]
    summary:    Optional[str]


def pros_node(state: ParallelState) -> dict:
    """Generate pros - runs in parallel"""
    print("  📍 Node: pros (parallel)")
    resp = llm.invoke([
        SystemMessage(content="List 3 pros only. Be brief."),
        HumanMessage(content=f"Pros of: {state['topic']}")
    ])
    return {"pros": resp.content}


def cons_node(state: ParallelState) -> dict:
    """Generate cons - runs in parallel"""
    print("  📍 Node: cons (parallel)")
    resp = llm.invoke([
        SystemMessage(content="List 3 cons only. Be brief."),
        HumanMessage(content=f"Cons of: {state['topic']}")
    ])
    return {"cons": resp.content}


def examples_node(state: ParallelState) -> dict:
    """Generate examples - runs in parallel"""
    print("  📍 Node: examples (parallel)")
    resp = llm.invoke([
        SystemMessage(content="Give 2 real-world examples only. Be brief."),
        HumanMessage(content=f"Examples of: {state['topic']}")
    ])
    return {"examples": resp.content}


def combine_node(state: ParallelState) -> dict:
    """Combine parallel results into final summary"""
    print("  📍 Node: combine (after parallel)")

    resp = llm.invoke([
        SystemMessage(content="Create a balanced summary from these inputs."),
        HumanMessage(content=f"""
Topic: {state['topic']}

Pros: {state['pros']}

Cons: {state['cons']}

Examples: {state['examples']}

Write a 2-sentence balanced summary.""")
    ])
    return {"summary": resp.content}


# ── Build parallel graph ──────────────────────────────────────
parallel_graph = StateGraph(ParallelState)

parallel_graph.add_node("pros",    pros_node)
parallel_graph.add_node("cons",    cons_node)
parallel_graph.add_node("examples",examples_node)
parallel_graph.add_node("combine", combine_node)

# START fans out to 3 parallel nodes
parallel_graph.add_edge(START,   "pros")
parallel_graph.add_edge(START,   "cons")
parallel_graph.add_edge(START,   "examples")

# All 3 parallel nodes fan in to combine
parallel_graph.add_edge("pros",    "combine")
parallel_graph.add_edge("cons",    "combine")
parallel_graph.add_edge("examples","combine")

parallel_graph.add_edge("combine", END)

parallel_runner = parallel_graph.compile()

import time
start = time.time()
result = parallel_runner.invoke({
    "topic": "Using AI agents in software development",
    "messages": []
})
elapsed = time.time() - start

print(f"\n✅ Parallel execution completed in {elapsed:.2f}s")
print(f"\nPros:\n{result['pros']}")
print(f"\nCons:\n{result['cons']}")
print(f"\nExamples:\n{result['examples']}")
print(f"\nSummary:\n{result['summary']}")

# #### Human-in-the-Loop

# Cell 7: Pause for human approval before continuing

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

class ApprovalState(TypedDict):
    messages:       Annotated[List, add_messages]
    task:           str
    plan:           Optional[str]
    approved:       Optional[bool]
    result:         Optional[str]


def plan_node(state: ApprovalState) -> dict:
    """Create an execution plan"""
    print("  📍 Node: plan")

    response = llm.invoke([
        SystemMessage(content="Create a brief 3-step action plan."),
        HumanMessage(content=f"Task: {state['task']}")
    ])

    plan = response.content
    print(f"\n📋 Generated Plan:\n{plan}\n")
    return {"plan": plan}


def human_approval_node(state: ApprovalState) -> dict:
    """
    Interrupt graph and wait for human input.
    Graph pauses here until resumed with approval decision.
    """
    print("  📍 Node: human_approval (PAUSED - waiting for human)")

    # interrupt() pauses the graph
    # value passed to interrupt is shown to the human
    user_decision = interrupt({
        "question": "Do you approve this plan?",
        "plan": state["plan"],
        "options": ["yes", "no"]
    })

    approved = str(user_decision).lower() in ["yes", "y", "approve", "ok"]
    return {"approved": approved}


def execute_node(state: ApprovalState) -> dict:
    """Execute the approved plan"""
    print("  📍 Node: execute")

    if not state.get("approved"):
        return {"result": "❌ Task cancelled - plan not approved"}

    response = llm.invoke([
        SystemMessage(content="Execute this plan concisely."),
        HumanMessage(content=f"Plan: {state['plan']}\nTask: {state['task']}")
    ])
    return {"result": f"✅ Executed:\n{response.content}"}


def route_approval(state: ApprovalState) -> str:
    """Route based on approval"""
    return "execute" if state.get("approved") else "rejected"


def rejected_node(state: ApprovalState) -> dict:
    print("  📍 Node: rejected")
    return {"result": "Task was rejected by human reviewer"}


# Build graph with human-in-the-loop
approval_graph = StateGraph(ApprovalState)

approval_graph.add_node("plan",           plan_node)
approval_graph.add_node("human_approval", human_approval_node)
approval_graph.add_node("execute",        execute_node)
approval_graph.add_node("rejected",       rejected_node)

approval_graph.add_edge(START, "plan")
approval_graph.add_edge("plan", "human_approval")
approval_graph.add_conditional_edges(
    "human_approval",
    route_approval,
    {"execute": "execute", "rejected": "rejected"}
)
approval_graph.add_edge("execute",  END)
approval_graph.add_edge("rejected", END)

# Memory required for human-in-the-loop
memory = MemorySaver()
approval_runner = approval_graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "approval_1"}}


# ── Step 1: Run until human approval needed ───────────────────
print("Step 1: Running until approval needed...\n")
initial_state = {
    "task": "Deploy a new Python microservice to production",
    "messages": []
}

result = approval_runner.invoke(initial_state, config=config)
print(f"\n⏸️  Graph paused. Current state: plan ready")
print(f"Plan: {result.get('plan', 'N/A')[:200]}")

# ── Step 2: Human provides decision ──────────────────────────
print("\nStep 2: Simulating human approval...\n")

# Resume with human decision
from langgraph.types import Command

final_result = approval_runner.invoke(
    Command(resume="yes"),   # human says yes
    config=config
)

print(f"\n✅ Final result:")
print(final_result.get("result", "No result"))

# #### Persistent Checkpointing

# Cell 8: Save and restore graph state across sessions

memory = MemorySaver()

# Agent with checkpointing
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def remember_fact(fact: str) -> str:
    """Store an important fact. Use when user shares important information."""
    return f"Fact stored: {fact}"

checkpointed_agent = create_react_agent(
    model=llm,
    tools=[remember_fact],
    checkpointer=memory
)

thread = {"configurable": {"thread_id": "session_abc"}}

# Session 1
print("=== Session 1 ===")
r1 = checkpointed_agent.invoke(
    {"messages": [HumanMessage(content="My name is Raj. I'm learning LangGraph.")]},
    config=thread
)
print(f"Agent: {r1['messages'][-1].content}")

# Session 2 - same thread_id = same memory!
print("\n=== Session 2 (same thread) ===")
r2 = checkpointed_agent.invoke(
    {"messages": [HumanMessage(content="What do you know about me?")]},
    config=thread
)
print(f"Agent: {r2['messages'][-1].content}")

# Session 3 - different thread_id = fresh memory
print("\n=== Session 3 (different thread) ===")
new_thread = {"configurable": {"thread_id": "session_xyz"}}
r3 = checkpointed_agent.invoke(
    {"messages": [HumanMessage(content="What do you know about me?")]},
    config=new_thread
)
print(f"Agent: {r3['messages'][-1].content}")
# Will say it doesn't know - different thread!

# #### LangGraph Patterns Cheat Sheet

# PATTERN              CODE                          USE WHEN
# ──────────────────────────────────────────────────────────────────
# Linear chain         A → B → C → END              Fixed steps
# Conditional branch   A → router → B or C          Intent-based routing
# Loop                 A → B → (back to A or END)   Iterative refinement
# Parallel fan-out     START → A, B, C              Independent parallel tasks
# Parallel fan-in      A, B, C → D                  Combine parallel results
# Human-in-loop        A → interrupt() → B          Approval needed
# Checkpointing        compile(checkpointer=memory) Persist across sessions
# Sub-graphs           node = other_graph.compile() Modular complex graphs
# 
# STATE REDUCERS:
#   add_messages     → append to message list
#   operator.add     → add numbers or concatenate lists
#   custom function  → any merge logic you define
#   no annotation    → replace value each time
