# agents/writer.py

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.llm import get_llm
from core.state import ResearchState
from core.tools import WRITER_TOOLS

llm = get_llm(
    model="llama-3.3-70b-versatile", temperature=0.7
).bind_tools(WRITER_TOOLS)

SYSTEM = """You are an expert research writer who creates
clear, engaging, and comprehensive reports.

Report structure:
# [Title]

## Executive Summary
[2-3 sentences covering the main finding]

## Background
[Context and why this topic matters]

## Key Findings
[Main research results with evidence]

## Analysis & Insights
[Deep dive into what the findings mean]

## Implications
[Real-world impact and applications]

## Conclusion
[Summary and forward-looking statement]

## Sources
[List of references]

Writing standards:
✅ Professional but accessible tone
✅ Evidence-based claims only
✅ Clear structure with headers
✅ Concrete examples where possible
✅ 600-900 words for standard depth"""


def writer_node(state: ResearchState) -> dict:
    """Writer agent — creates polished report"""
    print("\n✍️  WRITER starting...")

    topic     = state["topic"]
    research  = state.get("research_notes") or ""
    analysis  = state.get("analysis") or ""
    insights  = state.get("key_insights") or []
    sources   = state.get("raw_sources") or []
    revision  = state.get("revision_notes") or ""
    tool_map  = {t.name: t for t in WRITER_TOOLS}

    # Build context
    context = f"""Topic: {topic}

Research Notes:
{research[:1200]}

Analysis:
{analysis[:800]}

Key Insights:
{chr(10).join(f'- {i}' for i in insights[:5])}

Sources:
{chr(10).join(sources[:5])}"""

    if revision:
        context += f"\n\nRevision Notes from Critic:\n{revision}"
        instruction = "Rewrite the report addressing ALL critic feedback."
    else:
        instruction = "Write the comprehensive research report now."

    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"{context}\n\n{instruction}")
    ]

    for _ in range(3):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            print(f"   🔧 {tc['name']}(...)")
            result = tool_map.get(tc["name"], WRITER_TOOLS[0]).invoke(tc["args"])
            messages.append(
                HumanMessage(
                    content=f"[{tc['name']} result]:\n{result}",
                    name="tool_result"
                )
            )

    draft = messages[-1].content
    word_count = len(draft.split())
    print(f"   ✅ Draft written ({word_count} words)")

    return {
        "draft_report": draft,
        "word_count":   word_count,
        "completed":    ["writing"],
        "messages":     [AIMessage(content=f"Draft written: {word_count} words")],
        "status":       "draft_done"
    }