# agents/researcher.py

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.llm import get_llm
from core.state import ResearchState
from core.tools import RESEARCHER_TOOLS

llm = get_llm(
    model="llama-3.3-70b-versatile", temperature=0
).bind_tools(RESEARCHER_TOOLS)

SYSTEM = """You are an elite research agent.

Your mission: gather comprehensive, accurate information on the given topic.

Research approach by depth:
- quick:    2-3 searches, key facts only
- standard: 4-5 searches, multiple angles
- deep:     6-8 searches, exhaustive coverage

Always:
✅ Use multiple search queries for different angles
✅ Check Wikipedia for background context
✅ Search for recent news and developments
✅ Note your sources explicitly
✅ Flag any conflicting information found"""


def researcher_node(state: ResearchState) -> dict:
    """Research agent — gathers raw information"""
    print("\n🔍 RESEARCHER starting...")

    depth      = state.get("depth", "standard")
    topic      = state["topic"]
    max_steps  = {"quick": 3, "standard": 5, "deep": 8}.get(depth, 5)

    tool_map   = {t.name: t for t in RESEARCHER_TOOLS}
    messages   = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"""Research this topic at '{depth}' depth:

TOPIC: {topic}

Use your tools to gather comprehensive information.
Start broad, then get specific.""")
    ]

    sources = []

    for step in range(max_steps):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            print(f"   🔧 {tc['name']}({list(tc['args'].values())[0][:60]}...)")
            result = tool_map.get(tc["name"], RESEARCHER_TOOLS[0]).invoke(tc["args"])

            # Track sources
            if "URL:" in result:
                import re
                urls = re.findall(r'URL:\s*(https?://\S+)', result)
                sources.extend(urls)

            messages.append(
                HumanMessage(
                    content=f"[{tc['name']} result]:\n{result}",
                    name="tool_result"
                )
            )

    # Compile final research notes
    compile_resp = get_llm(
        model="llama-3.3-70b-versatile", temperature=0
    ).invoke([
        SystemMessage(content="""Compile all research into structured notes.
Format:
## Key Facts
[bullet points]

## Background
[2-3 paragraphs]

## Recent Developments
[bullet points]

## Sources
[list URLs if any]"""),
        HumanMessage(content=f"Research conversation:\n{str([m.content[:200] for m in messages])}")
    ])

    notes = compile_resp.content
    print(f"   ✅ Research complete ({len(notes)} chars, {len(sources)} sources)")

    return {
        "research_notes": notes,
        "raw_sources":    sources[:10],
        "completed":      ["research"],
        "messages":       [AIMessage(content=f"Research completed on: {topic}")],
        "status":         "research_done"
    }