# agents/analyst.py

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.llm import get_llm
from core.state import ResearchState
from core.tools import ANALYST_TOOLS

llm = get_llm(
    model="llama-3.3-70b-versatile", temperature=0
).bind_tools(ANALYST_TOOLS)

SYSTEM = """You are a senior research analyst and strategic thinker.

Your job: transform raw research into actionable insights.

Analysis framework:
1. Extract key statistics and data points
2. Identify major trends and patterns
3. Compare and contrast different viewpoints
4. Assess implications and significance
5. Highlight gaps or areas of uncertainty

Be specific, evidence-based, and insightful."""


def analyst_node(state: ResearchState) -> dict:
    """Analyst agent — extracts insights from research"""
    print("\n📊 ANALYST starting...")

    research  = state.get("research_notes") or "No research available"
    topic     = state["topic"]
    tool_map  = {t.name: t for t in ANALYST_TOOLS}

    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"""Analyze this research on: {topic}

RESEARCH NOTES:
{research[:2000]}

Extract insights, patterns, and key takeaways.
Use your tools for any calculations needed.""")
    ]

    for _ in range(4):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            print(f"   🔧 {tc['name']}(...)")
            result = tool_map.get(tc["name"], ANALYST_TOOLS[0]).invoke(tc["args"])
            messages.append(
                HumanMessage(
                    content=f"[{tc['name']} result]:\n{result}",
                    name="tool_result"
                )
            )

    analysis = messages[-1].content

    # Extract key insights as list
    insight_resp = get_llm(
        model="llama-3.1-8b-instant", temperature=0
    ).invoke([
        SystemMessage(content="Extract 5 key insights as JSON array: [\"insight1\", ...]"),
        HumanMessage(content=analysis)
    ])

    try:
        import json
        insights = json.loads(insight_resp.content.strip())
    except Exception:
        insights = [analysis[:100]]

    print(f"   ✅ Analysis complete ({len(insights)} insights)")

    return {
        "analysis":     analysis,
        "key_insights": insights,
        "completed":    ["analysis"],
        "messages":     [AIMessage(content=f"Analysis complete: {len(insights)} insights extracted")],
        "status":       "analysis_done"
    }