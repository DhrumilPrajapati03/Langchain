# -*- coding: utf-8 -*-
"""
Auto-generated Python script from: day14_test.ipynb
"""

# ### Day 14: Multi-Agent Orchestration

# Today you build real multi-agent systems — multiple specialized AI agents working together, sharing memory, communicating results, and coordinated by a supervisor. This is the direct foundation of Project 3 tomorrow.

# ##### 1. What Is Multi-Agent Orchestration?

# Single Agent (Days 11-13):
#   One LLM + tools → handles everything itself
#   Good for: simple tasks, linear workflows
# 
# Multi-Agent System (Today):
#   Agent A (Researcher) ──┐
#   Agent B (Analyst)    ──┤→ Supervisor → Coordinated Output
#   Agent C (Writer)     ──┘
#   Agent D (Critic)     ──┘
# 
#   Each agent is specialized, has its own:
#   - System prompt (persona + expertise)
#   - Tool set (only what it needs)
#   - Memory (conversation within its scope)

# ##### 2. Three Orchestration Architectures

# 1. SUPERVISOR (hierarchical)
#    Supervisor decides who acts next
#    Best for: structured workflows
# 
# 2. SWARM (peer-to-peer)  
#    Agents hand off directly to each other
#    Best for: pipeline-style workflows
# 
# 3. MIXTURE OF EXPERTS
#    Router picks the best specialist
#    Best for: domain-specific routing

# Cell 1: Imports
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, BaseMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List, Optional, Dict, Any
import operator, json, math, os, time, requests
from duckduckgo_search import DDGS

load_dotenv()

# Different temperature for different agent roles
llm_precise = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0
)    # for analysis, facts
llm_creative = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0.7
)   # for writing
llm_fast = ChatGroq(
    model="llama-3.1-8b-instant", temperature=0
)        # for routing

print("✅ Three LLM instances ready")
print("   llm_precise   → analysis & facts (temp=0)")
print("   llm_creative  → writing (temp=0.7)")
print("   llm_fast      → routing decisions (temp=0)")

# ##### Build Specialized Agent Tools

# Cell 2: Each agent gets its own specialized tools

# ── Researcher tools ──────────────────────────────────────────
@tool
def search_web(query: str) -> str:
    """Search web for current information and recent facts."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No results found"
        return "\n\n".join(
            f"[{r['title']}]\n{r['body'][:300]}"
            for r in results
        )
    except Exception as e:
        return f"Search error: {e}"


@tool
def search_wikipedia(topic: str) -> str:
    """Search Wikipedia for encyclopedic knowledge."""
    try:
        url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + topic.replace(" ", "_")
        )
        resp = requests.get(url, timeout=5).json()
        return f"{resp.get('title', topic)}:\n{resp.get('extract', 'Not found')[:600]}"
    except Exception as e:
        return f"Wikipedia error: {e}"


@tool
def fetch_url_content(url: str) -> str:
    """Fetch and extract text content from a URL."""
    try:
        resp = requests.get(url, timeout=8)
        # Simple text extraction
        text = resp.text[:3000]
        return f"Content from {url}:\n{text}"
    except Exception as e:
        return f"Fetch error: {e}"


# ── Analyst tools ─────────────────────────────────────────────
@tool
def calculate(expression: str) -> str:
    """Evaluate mathematical expressions accurately."""
    try:
        allowed = {
            'abs': abs, 'round': round, 'min': min,
            'max': max, 'pow': pow, 'sqrt': math.sqrt,
            'pi': math.pi, 'e': math.e, 'log': math.log,
            'sum': sum, 'len': len
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as ex:
        return f"Calculation error: {ex}"


@tool
def compare_data(items: str) -> str:
    """
    Compare multiple items or data points.
    Input: comma-separated list of items to compare.
    Returns structured comparison analysis.
    """
    item_list = [i.strip() for i in items.split(",")]
    return json.dumps({
        "items_compared": item_list,
        "count": len(item_list),
        "analysis_ready": True
    })


@tool
def extract_key_metrics(text: str) -> str:
    """
    Extract numbers, percentages, and key metrics from text.
    Use to find quantitative data in research results.
    """
    import re
    numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|B|M|K|bn|mn)?\b', text)
    return json.dumps({
        "metrics_found": numbers[:20],
        "count": len(numbers)
    })


# ── Writer tools ──────────────────────────────────────────────
@tool
def check_grammar(text: str) -> str:
    """
    Check text for common grammar issues.
    Returns suggestions for improvement.
    """
    issues = []
    if text[0].islower():
        issues.append("Start with capital letter")
    if not text.endswith(('.', '!', '?')):
        issues.append("Add ending punctuation")
    sentences = text.split('.')
    long_sentences = [s for s in sentences if len(s.split()) > 30]
    if long_sentences:
        issues.append(f"{len(long_sentences)} sentence(s) over 30 words — consider splitting")
    return json.dumps({
        "issues": issues,
        "issue_count": len(issues),
        "suggestion": "Looks good!" if not issues else "Review noted issues"
    })


@tool
def format_as_markdown(content: str, doc_type: str = "report") -> str:
    """
    Format content as structured markdown.
    doc_type options: report, blog, summary, analysis
    """
    headers = {
        "report":   ["Executive Summary", "Key Findings", "Analysis", "Conclusion"],
        "blog":     ["Introduction", "Main Content", "Key Takeaways", "Conclusion"],
        "summary":  ["Overview", "Details", "Next Steps"],
        "analysis": ["Background", "Analysis", "Insights", "Recommendations"]
    }
    structure = headers.get(doc_type, headers["report"])
    template = f"# Document\n\n"
    for section in structure:
        template += f"## {section}\n[content here]\n\n"
    return f"Suggested structure for {doc_type}:\n{template}"


# ── Critic tools ──────────────────────────────────────────────
@tool
def fact_check_claims(claims: str) -> str:
    """
    Identify claims that need verification.
    Input: text containing claims to evaluate.
    """
    claim_list = [c.strip() for c in claims.split('.') if len(c.strip()) > 20]
    return json.dumps({
        "claims_identified": len(claim_list),
        "needs_verification": claim_list[:5],
        "recommendation": "Verify statistical claims and recent data"
    })


@tool
def score_content(content: str, criteria: str = "accuracy,clarity,depth") -> str:
    """
    Score content quality on given criteria.
    criteria: comma-separated list of aspects to evaluate.
    """
    criteria_list = [c.strip() for c in criteria.split(",")]
    scores = {c: min(10, max(1, len(content) // 100 + 5)) for c in criteria_list}
    avg = sum(scores.values()) / len(scores)
    return json.dumps({
        "scores": scores,
        "average": round(avg, 1),
        "verdict": "Good" if avg >= 7 else "Needs improvement"
    })


print("✅ Specialized tools created:")
print("   Researcher: search_web, search_wikipedia, fetch_url_content")
print("   Analyst:    calculate, compare_data, extract_key_metrics")
print("   Writer:     check_grammar, format_as_markdown")
print("   Critic:     fact_check_claims, score_content")

# ##### Define Shared State

# Cell 3: Shared state all agents read and write

class MultiAgentState(TypedDict):
    # ── Core ──────────────────────────────────────────────────
    messages:       Annotated[List, add_messages]
    task:           str
    current_agent:  Optional[str]

    # ── Agent outputs ──────────────────────────────────────────
    research:       Optional[str]   # Researcher output
    analysis:       Optional[str]   # Analyst output
    draft:          Optional[str]   # Writer output
    critique:       Optional[str]   # Critic output
    final_report:   Optional[str]   # Finished product

    # ── Workflow control ───────────────────────────────────────
    next_agent:     Optional[str]
    completed_steps: Annotated[List[str], operator.add]
    iteration:      int
    quality_score:  Optional[float]
    approved:       bool

    # ── Shared memory (all agents can read) ───────────────────
    shared_context: Optional[str]
    key_facts:      Annotated[List[str], operator.add]


print("✅ Shared MultiAgentState defined")
print("""
   Fields:
   ├── task, messages       (input)
   ├── research             (Researcher writes here)
   ├── analysis             (Analyst writes here)
   ├── draft                (Writer writes here)
   ├── critique             (Critic writes here)
   ├── final_report         (final output)
   ├── shared_context       (all agents can read)
   └── key_facts            (accumulated by all agents)
""")

# ##### Build Each Specialized Agent

# Cell 4: Researcher Agent

RESEARCHER_PROMPT = """You are an elite research agent with access to web search tools.

Your responsibilities:
- Find accurate, up-to-date information
- Gather data from multiple sources
- Extract key facts and statistics
- Summarize findings clearly
- Flag uncertain or conflicting information

Current task: {task}
Existing context: {context}

Use your tools to research thoroughly, then provide a comprehensive research report."""

def researcher_agent(state: MultiAgentState) -> dict:
    """
    Researcher: gathers information from web and Wikipedia.
    Has access to: search_web, search_wikipedia, fetch_url_content
    """
    print("\n🔍 RESEARCHER AGENT activated")

    researcher_llm = llm_precise.bind_tools(
        [search_web, search_wikipedia, fetch_url_content]
    )

    messages = [
        SystemMessage(content=RESEARCHER_PROMPT.format(
            task=state["task"],
            context=state.get("shared_context") or "None yet"
        )),
        HumanMessage(content=f"Research this thoroughly: {state['task']}")
    ]

    # ReAct loop for researcher
    max_steps = 5
    for step in range(max_steps):
        response = researcher_llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break   # Done researching

        # Execute tool calls
        tool_map = {
            "search_web": search_web,
            "search_wikipedia": search_wikipedia,
            "fetch_url_content": fetch_url_content
        }

        for tc in response.tool_calls:
            print(f"   🔧 {tc['name']}({list(tc['args'].values())[0][:50]}...)")
            result = tool_map.get(tc["name"], search_web).invoke(tc["args"])
            messages.append(
                HumanMessage(
                    content=f"Tool result for {tc['name']}:\n{result}",
                    name="tool_result"
                )
            )

    research_output = messages[-1].content
    print(f"   ✅ Research complete ({len(research_output)} chars)")

    # Extract key facts
    fact_resp = llm_precise.invoke([
        SystemMessage(content="Extract 3 key facts as JSON array: [\"fact1\", \"fact2\", \"fact3\"]"),
        HumanMessage(content=research_output)
    ])

    try:
        facts = json.loads(fact_resp.content.strip())
    except Exception:
        facts = [research_output[:100]]

    return {
        "research": research_output,
        "key_facts": facts,
        "completed_steps": ["research"],
        "current_agent": "researcher",
        "messages": [AIMessage(content=f"Research completed: {research_output[:200]}...")]
    }

# Cell 5: Analyst Agent

ANALYST_PROMPT = """You are a senior data analyst and strategic thinker.

Your responsibilities:
- Analyze research findings critically
- Identify patterns, trends, and insights
- Perform calculations when needed
- Compare and contrast information
- Draw evidence-based conclusions

Task: {task}
Research provided: {research}
Key facts: {facts}

Analyze deeply and provide actionable insights."""

def analyst_agent(state: MultiAgentState) -> dict:
    """
    Analyst: performs deep analysis on research output.
    Has access to: calculate, compare_data, extract_key_metrics
    """
    print("\n📊 ANALYST AGENT activated")

    analyst_llm = llm_precise.bind_tools(
        [calculate, compare_data, extract_key_metrics]
    )

    research = state.get("research") or "No research available"
    facts = "\n".join(state.get("key_facts", []))

    messages = [
        SystemMessage(content=ANALYST_PROMPT.format(
            task=state["task"],
            research=research[:1000],
            facts=facts
        )),
        HumanMessage(content="Analyze the research and provide insights.")
    ]

    # ReAct loop for analyst
    for _ in range(4):
        response = analyst_llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        tool_map = {
            "calculate": calculate,
            "compare_data": compare_data,
            "extract_key_metrics": extract_key_metrics
        }

        for tc in response.tool_calls:
            print(f"   🔧 {tc['name']}(...)")
            result = tool_map.get(tc["name"], calculate).invoke(tc["args"])
            messages.append(
                HumanMessage(
                    content=f"Tool result:\n{result}",
                    name="tool_result"
                )
            )

    analysis_output = messages[-1].content
    print(f"   ✅ Analysis complete ({len(analysis_output)} chars)")

    return {
        "analysis": analysis_output,
        "completed_steps": ["analysis"],
        "current_agent": "analyst",
        "shared_context": f"Research:\n{research[:500]}\n\nAnalysis:\n{analysis_output[:500]}",
        "messages": [AIMessage(content=f"Analysis completed: {analysis_output[:200]}...")]
    }

# Cell 6: Writer Agent

WRITER_PROMPT = """You are an expert technical writer who creates clear, engaging content.

Your responsibilities:
- Transform research and analysis into polished writing
- Structure content logically with headers and sections
- Use clear, professional language
- Make complex topics accessible
- Create content the reader will find valuable

Task: {task}
Research: {research}
Analysis: {analysis}

Write a comprehensive, well-structured report."""

def writer_agent(state: MultiAgentState) -> dict:
    """
    Writer: creates polished final content.
    Has access to: check_grammar, format_as_markdown
    """
    print("\n✍️  WRITER AGENT activated")

    writer_llm = llm_creative.bind_tools(
        [check_grammar, format_as_markdown]
    )

    research = state.get("research") or ""
    analysis = state.get("analysis") or ""

    messages = [
        SystemMessage(content=WRITER_PROMPT.format(
            task=state["task"],
            research=research[:800],
            analysis=analysis[:800]
        )),
        HumanMessage(content="Write the comprehensive report now.")
    ]

    for _ in range(3):
        response = writer_llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        tool_map = {
            "check_grammar": check_grammar,
            "format_as_markdown": format_as_markdown
        }

        for tc in response.tool_calls:
            print(f"   🔧 {tc['name']}(...)")
            result = tool_map.get(tc["name"], check_grammar).invoke(tc["args"])
            messages.append(
                HumanMessage(
                    content=f"Tool result:\n{result}",
                    name="tool_result"
                )
            )

    draft = messages[-1].content
    print(f"   ✅ Draft complete ({len(draft)} chars)")

    return {
        "draft": draft,
        "completed_steps": ["writing"],
        "current_agent": "writer",
        "messages": [AIMessage(content=f"Draft created: {draft[:200]}...")]
    }

# Cell 7: Critic Agent

CRITIC_PROMPT = """You are a rigorous editor and quality assurance expert.

Your responsibilities:
- Evaluate content quality objectively
- Check for factual accuracy and logic
- Identify gaps, weaknesses, or unclear sections
- Score content on key dimensions
- Decide if content meets quality bar or needs revision

Task: {task}
Draft to review: {draft}

Be specific in your critique. Score 1-10 on: accuracy, clarity, depth, structure."""

def critic_agent(state: MultiAgentState) -> dict:
    """
    Critic: reviews and scores draft quality.
    Has access to: fact_check_claims, score_content
    """
    print("\n🔎 CRITIC AGENT activated")

    critic_llm = llm_precise.bind_tools(
        [fact_check_claims, score_content]
    )

    draft = state.get("draft") or "No draft available"

    messages = [
        SystemMessage(content=CRITIC_PROMPT.format(
            task=state["task"],
            draft=draft[:1500]
        )),
        HumanMessage(content="Review this draft critically.")
    ]

    for _ in range(3):
        response = critic_llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        tool_map = {
            "fact_check_claims": fact_check_claims,
            "score_content": score_content
        }

        for tc in response.tool_calls:
            print(f"   🔧 {tc['name']}(...)")
            result = tool_map.get(tc["name"], score_content).invoke(tc["args"])
            messages.append(
                HumanMessage(
                    content=f"Tool result:\n{result}",
                    name="tool_result"
                )
            )

    critique = messages[-1].content

    # Extract quality score from critique
    score_resp = llm_fast.invoke([
        SystemMessage(content="Extract the overall quality score from this critique. Reply with ONLY a number 1-10."),
        HumanMessage(content=critique)
    ])

    try:
        quality = float(score_resp.content.strip())
    except Exception:
        quality = 6.0

    print(f"   ✅ Critique complete | Quality score: {quality}/10")

    return {
        "critique": critique,
        "quality_score": quality,
        "completed_steps": ["critique"],
        "current_agent": "critic",
        "messages": [AIMessage(content=f"Critique: Score {quality}/10")]
    }

# Cell 8: Supervisor Agent

SUPERVISOR_PROMPT = """You are the project supervisor coordinating a team of specialist agents.

Team:
- RESEARCHER  → gathers facts and information from web
- ANALYST     → analyzes data and extracts insights
- WRITER      → creates polished written content
- CRITIC      → reviews quality and provides feedback
- FINISH      → call when final report is ready and approved

Current status:
  Task: {task}
  Completed steps: {completed}
  Quality score: {quality}
  Iteration: {iteration}

Rules:
1. Always start with RESEARCHER
2. RESEARCHER → ANALYST → WRITER → CRITIC is the standard flow
3. If quality < 7.5 and iteration < 3 → send back to WRITER
4. If quality >= 7.5 → FINISH
5. Never exceed 4 iterations

Respond with JSON only: {{"next": "AGENT_NAME", "reason": "brief explanation"}}"""


def supervisor_agent(state: MultiAgentState) -> dict:
    """
    Supervisor: orchestrates all specialist agents.
    Decides who acts next based on current state.
    """
    print("\n👔 SUPERVISOR deciding next agent...")

    completed = state.get("completed_steps", [])
    quality = state.get("quality_score")
    iteration = state.get("iteration", 0)

    response = llm_fast.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT.format(
            task=state["task"],
            completed=completed,
            quality=quality or "Not yet scored",
            iteration=iteration
        ))
    ])

    try:
        data = json.loads(response.content.strip())
        next_agent = data.get("next", "FINISH")
        reason = data.get("reason", "")
    except Exception:
        # Fallback logic if JSON parse fails
        if "research" not in completed:
            next_agent = "RESEARCHER"
        elif "analysis" not in completed:
            next_agent = "ANALYST"
        elif "writing" not in completed:
            next_agent = "WRITER"
        elif "critique" not in completed:
            next_agent = "CRITIC"
        else:
            next_agent = "FINISH"
        reason = "Fallback routing"

    # Safety override
    if iteration >= 4:
        next_agent = "FINISH"
        reason = "Max iterations reached"

    print(f"   → Next: {next_agent} | Reason: {reason}")

    return {
        "next_agent": next_agent,
        "iteration": iteration + 1
    }


def finalizer_agent(state: MultiAgentState) -> dict:
    """Compile everything into the final approved report"""
    print("\n🏁 FINALIZER compiling report...")

    draft = state.get("draft") or ""
    critique = state.get("critique") or ""

    if state.get("quality_score", 0) >= 7.5:
        # High quality - use draft with minor polish
        final = llm_creative.invoke([
            SystemMessage(content="Polish this draft slightly. Keep structure. Improve flow."),
            HumanMessage(content=f"Draft:\n{draft}\n\nCritique notes:\n{critique[:300]}")
        ]).content
    else:
        # Lower quality - rebuild from scratch
        final = llm_creative.invoke([
            SystemMessage(content="Write a comprehensive final report from scratch."),
            HumanMessage(content=f"""
Task: {state['task']}
Research: {state.get('research', '')[:600]}
Analysis: {state.get('analysis', '')[:600]}
""")
        ]).content

    print(f"   ✅ Final report ready ({len(final)} chars)")

    return {
        "final_report": final,
        "approved": True,
        "completed_steps": ["finalized"]
    }

# Cell 9: Build the full multi-agent graph

def route_supervisor(state: MultiAgentState) -> str:
    """Map supervisor decision to graph node"""
    routes = {
        "RESEARCHER": "researcher",
        "ANALYST":    "analyst",
        "WRITER":     "writer",
        "CRITIC":     "critic",
        "FINISH":     "finalizer"
    }
    return routes.get(
        state.get("next_agent", "FINISH"),
        "finalizer"
    )


# ── Build graph ───────────────────────────────────────────────
mas_graph = StateGraph(MultiAgentState)

# Add all nodes
mas_graph.add_node("supervisor", supervisor_agent)
mas_graph.add_node("researcher", researcher_agent)
mas_graph.add_node("analyst",    analyst_agent)
mas_graph.add_node("writer",     writer_agent)
mas_graph.add_node("critic",     critic_agent)
mas_graph.add_node("finalizer",  finalizer_agent)

# Entry point
mas_graph.add_edge(START, "supervisor")

# Supervisor routes to any agent
mas_graph.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "analyst":    "analyst",
        "writer":     "writer",
        "critic":     "critic",
        "finalizer":  "finalizer"
    }
)

# All agents report back to supervisor
for agent in ["researcher", "analyst", "writer", "critic"]:
    mas_graph.add_edge(agent, "supervisor")

# Finalizer ends the graph
mas_graph.add_edge("finalizer", END)

# Compile with memory
memory = MemorySaver()
multi_agent_system = mas_graph.compile(checkpointer=memory)

print("✅ Multi-Agent System compiled!")
print("""
Graph:
  START
    ↓
  SUPERVISOR
  ↙  ↓  ↓  ↘  ↘
 R   A   W   C   F
  ↘  ↓  ↓  ↗
  SUPERVISOR (loop)
    ↓ (FINISH)
  FINALIZER
    ↓
   END
""")

# Cell 10: Full multi-agent run with streaming

task = "What are the most impactful applications of LangGraph in production AI systems in 2025?"

print("="*60)
print(f"TASK: {task}")
print("="*60)

config = {"configurable": {"thread_id": "mas_run_1"}}

# Stream execution
for step in multi_agent_system.stream(
    {
        "task":            task,
        "messages":        [],
        "completed_steps": [],
        "key_facts":       [],
        "iteration":       0,
        "approved":        False
    },
    config=config,
    stream_mode="updates"
):
    for node, update in step.items():
        print(f"\n{'─'*40}")
        print(f"⚡ {node.upper()} completed")

        if update.get("research"):
            print(f"   Research: {update['research'][:120]}...")
        if update.get("analysis"):
            print(f"   Analysis: {update['analysis'][:120]}...")
        if update.get("draft"):
            print(f"   Draft: {update['draft'][:120]}...")
        if update.get("critique"):
            print(f"   Critique: {update['critique'][:100]}...")
        if update.get("quality_score"):
            print(f"   Quality Score: {update['quality_score']}/10")
        if update.get("next_agent"):
            print(f"   Next Agent: {update['next_agent']}")
        if update.get("final_report"):
            print(f"   ✅ FINAL REPORT READY!")

# Cell 11: Get and display final report
final_state = multi_agent_system.get_state(config)

print("\n" + "="*60)
print("FINAL REPORT")
print("="*60)
print(final_state.values.get("final_report", "Report not generated"))
print("\n" + "="*60)
print(f"Steps completed: {final_state.values.get('completed_steps')}")
print(f"Quality score:   {final_state.values.get('quality_score')}/10")
print(f"Iterations:      {final_state.values.get('iteration')}")

# ##### Inter-Agent Communication Patterns

# Cell 12: Agents communicating via shared state

class CommState(TypedDict):
    messages:       Annotated[List, add_messages]
    inbox:          Dict[str, str]      # agent → message
    outbox:         Dict[str, str]      # agent → message
    shared_memory:  Dict[str, Any]      # key-value shared store


def send_message(state: CommState, from_agent: str,
                 to_agent: str, message: str) -> dict:
    """Helper: add message to inbox"""
    inbox = dict(state.get("inbox", {}))
    inbox[to_agent] = f"From {from_agent}: {message}"
    return {"inbox": inbox}


def read_message(state: CommState, agent: str) -> Optional[str]:
    """Helper: read message from inbox"""
    return state.get("inbox", {}).get(agent)


def write_to_shared(state: CommState, key: str,
                    value: Any) -> dict:
    """Helper: write to shared memory"""
    mem = dict(state.get("shared_memory", {}))
    mem[key] = value
    return {"shared_memory": mem}


# Example: Researcher writes facts, Analyst reads them
def researcher_comm_node(state: CommState) -> dict:
    print("  📍 Researcher writing to shared memory")
    return {
        **write_to_shared(state, "key_findings",
                          ["LangGraph released 2024",
                           "Used in production at Uber",
                           "Supports human-in-loop"]),
        **send_message(state, "researcher", "analyst",
                       "Research complete. Check shared_memory.key_findings")
    }


def analyst_comm_node(state: CommState) -> dict:
    print("  📍 Analyst reading shared memory")
    findings = state.get("shared_memory", {}).get("key_findings", [])
    msg = read_message(state, "analyst")
    print(f"     Message received: {msg}")
    print(f"     Findings found: {findings}")
    return {
        **write_to_shared(state, "analysis_result",
                          f"Analyzed {len(findings)} findings")
    }


# Quick test
comm_graph = StateGraph(CommState)
comm_graph.add_node("researcher", researcher_comm_node)
comm_graph.add_node("analyst",    analyst_comm_node)
comm_graph.add_edge(START,        "researcher")
comm_graph.add_edge("researcher", "analyst")
comm_graph.add_edge("analyst",    END)

comm_runner = comm_graph.compile()
result = comm_runner.invoke({
    "messages": [], "inbox": {}, "outbox": {}, "shared_memory": {}
})
print(f"\nShared memory: {result['shared_memory']}")

# ##### Architecture Comparison

# ARCHITECTURE    CONTROL      SPEED      COMPLEXITY   BEST FOR
# ──────────────────────────────────────────────────────────────────
# Supervisor      Centralized  Medium     Medium       Structured tasks
#                                                      Quality control
#                                                      Approval flows
# 
# Swarm           Distributed  Fast       Low          Pipeline tasks
#                                                      Linear workflows
#                                                      Data processing
# 
# Mixture of      Routing      Very fast  Low          Classification
# Experts         based                               Domain routing
#                                                      Simple delegation
# 
# Hierarchical    Multi-level  Slow       High         Enterprise workflows
# Supervisor                                          Complex decisions
#                                                      Large agent teams
# 
# CHOOSING:
#   < 3 agents,  linear flow  → Swarm
#   3-5 agents,  need quality → Supervisor
#   5+ agents,  complex logic → Hierarchical Supervisor
#   Domain routing needed    → Mixture of Experts
