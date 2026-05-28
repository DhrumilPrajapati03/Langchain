# core/graph.py
# Wires all agents into the final graph

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from core.llm import get_llm
from core.state import ResearchState
from agents.researcher import researcher_node
from agents.analyst    import analyst_node
from agents.writer     import writer_node
from agents.critic     import critic_node
from agents.supervisor import supervisor_node


def finalizer_node(state: ResearchState) -> dict:
    """Polish and finalize the report"""
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    print("\n🏁 FINALIZER polishing report...")

    draft  = state.get("draft_report") or ""
    score  = state.get("quality_score") or 0

    if score >= 7.5:
        # Minor polish only
        llm = get_llm(
            model="llama-3.3-70b-versatile", temperature=0.3
        )
        final = llm.invoke([
            SystemMessage(content="Lightly polish this report. Fix flow, keep all content."),
            HumanMessage(content=draft)
        ]).content
    else:
        # Full rewrite with all context
        llm = get_llm(
            model="llama-3.3-70b-versatile", temperature=0.5
        )
        final = llm.invoke([
            SystemMessage(content="""Write a high-quality research report.
Use all available research and analysis.
Structure: Executive Summary, Background, Key Findings,
Analysis, Implications, Conclusion, Sources."""),
            HumanMessage(content=f"""
Topic: {state['topic']}
Research: {state.get('research_notes', '')[:1000]}
Analysis: {state.get('analysis', '')[:600]}
Key Insights: {state.get('key_insights', [])}
""")
        ]).content

    word_count = len(final.split())
    print(f"   ✅ Final report ready ({word_count} words)")

    return {
        "final_report": final,
        "word_count":   word_count,
        "status":       "complete"
    }


def route_supervisor(state: ResearchState) -> str:
    """Map supervisor decision to graph node"""
    return {
        "RESEARCHER": "researcher",
        "ANALYST":    "analyst",
        "WRITER":     "writer",
        "CRITIC":     "critic",
        "FINISH":     "finalizer"
    }.get(state.get("next_agent", "FINISH"), "finalizer")


def build_graph():
    """Build and compile the research agent graph"""
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst",    analyst_node)
    graph.add_node("writer",     writer_node)
    graph.add_node("critic",     critic_node)
    graph.add_node("finalizer",  finalizer_node)

    # Entry
    graph.add_edge(START, "supervisor")

    # Supervisor routes
    graph.add_conditional_edges(
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
        graph.add_edge(agent, "supervisor")

    graph.add_edge("finalizer", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)