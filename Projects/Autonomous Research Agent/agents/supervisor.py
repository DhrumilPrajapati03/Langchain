# agents/supervisor.py

import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.llm import get_llm
from core.state import ResearchState

llm = get_llm(model="llama-3.1-8b-instant", temperature=0)

SYSTEM = """You are the research project supervisor.

Workflow rules:
1. Always start: RESEARCHER (gather info)
2. Then: ANALYST (extract insights)
3. Then: WRITER (create report)
4. Then: CRITIC (review quality)
5. If critic score < 7.5 AND iteration < 3: WRITER (revise)
6. If critic score >= 7.5 OR iteration >= 3: FINISH

Current state:
  Topic:     {topic}
  Completed: {completed}
  Score:     {score}
  Iteration: {iteration}
  Status:    {status}

Respond with JSON only:
{{"next": "RESEARCHER|ANALYST|WRITER|CRITIC|FINISH", "reason": "brief reason"}}"""


def supervisor_node(state: ResearchState) -> dict:
    """Supervisor — orchestrates all agents"""
    print("\n👔 SUPERVISOR deciding...")

    completed = state.get("completed", [])
    score     = state.get("quality_score")
    iteration = state.get("iteration", 0)
    status    = state.get("status", "starting")

    response = llm.invoke([
        SystemMessage(content=SYSTEM.format(
            topic=state["topic"],
            completed=completed,
            score=score or "Not yet scored",
            iteration=iteration,
            status=status
        ))
    ])

    try:
        data = json.loads(response.content.strip())
        next_agent = data.get("next", "FINISH")
        reason = data.get("reason", "")
    except Exception:
        # Deterministic fallback
        if "research" not in completed:
            next_agent, reason = "RESEARCHER", "Start with research"
        elif "analysis" not in completed:
            next_agent, reason = "ANALYST", "Analyze research"
        elif "writing" not in completed:
            next_agent, reason = "WRITER", "Write draft"
        elif "critique" not in completed:
            next_agent, reason = "CRITIC", "Review quality"
        elif score and score < 7.5 and iteration < 3:
            next_agent, reason = "WRITER", "Revise based on feedback"
        else:
            next_agent, reason = "FINISH", "Task complete"

    # Hard safety limit
    if iteration >= 5:
        next_agent, reason = "FINISH", "Max iterations"

    print(f"   → {next_agent} | {reason}")

    return {
        "next_agent": next_agent,
        "iteration":  iteration + 1,
        "messages":   [AIMessage(content=f"Supervisor: → {next_agent}")]
    }