# agents/critic.py

import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.llm import get_llm
from core.state import ResearchState
from core.tools import CRITIC_TOOLS

llm = get_llm(
    model="llama-3.3-70b-versatile", temperature=0
).bind_tools(CRITIC_TOOLS)

SYSTEM = """You are a rigorous research editor and quality reviewer.

Evaluation criteria:
1. ACCURACY    — Are claims supported by evidence?
2. COMPLETENESS — Does it cover the topic thoroughly?
3. CLARITY     — Is it easy to understand?
4. STRUCTURE   — Is it well-organized?
5. DEPTH       — Does it provide genuine insight?

Score each 1-10. Overall score = average.

Output format:
SCORES:
- Accuracy: X/10
- Completeness: X/10
- Clarity: X/10
- Structure: X/10
- Depth: X/10
- Overall: X/10

STRENGTHS:
[what works well]

WEAKNESSES:
[specific issues]

REVISION NOTES:
[exact instructions for improvement]

VERDICT: APPROVE | REVISE"""


def critic_node(state: ResearchState) -> dict:
    """Critic agent — reviews and scores the draft"""
    print("\n🔎 CRITIC starting...")

    topic     = state["topic"]
    draft     = state.get("draft_report") or "No draft available"
    tool_map  = {t.name: t for t in CRITIC_TOOLS}

    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"""Review this research report:

TOPIC: {topic}
EXPECTED DEPTH: {state.get('depth', 'standard')}

DRAFT:
{draft[:3000]}

Evaluate thoroughly and provide specific feedback.""")
    ]

    for _ in range(3):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            print(f"   🔧 {tc['name']}(...)")
            result = tool_map.get(tc["name"], CRITIC_TOOLS[0]).invoke(tc["args"])
            messages.append(
                HumanMessage(
                    content=f"[{tc['name']} result]:\n{result}",
                    name="tool_result"
                )
            )

    critique = messages[-1].content

    # Extract score
    score_resp = get_llm(
        model="llama-3.1-8b-instant", temperature=0
    ).invoke([
        SystemMessage(content="Extract the Overall score number from this critique. Reply with ONLY the number."),
        HumanMessage(content=critique)
    ])

    try:
        score = float(score_resp.content.strip().replace("/10", ""))
        score = max(1.0, min(10.0, score))
    except Exception:
        score = 6.5

    # Extract revision notes
    revision = ""
    if "REVISION NOTES:" in critique:
        revision = critique.split("REVISION NOTES:")[-1].split("VERDICT:")[0].strip()

    verdict = "APPROVE" if score >= 7.5 else "REVISE"
    print(f"   ✅ Critique done | Score: {score}/10 | Verdict: {verdict}")

    return {
        "critique":       critique,
        "quality_score":  score,
        "revision_notes": revision if verdict == "REVISE" else None,
        "completed":      ["critique"],
        "messages":       [AIMessage(content=f"Quality score: {score}/10 — {verdict}")],
        "status":         f"critique_{verdict.lower()}"
    }