# core/state.py
# Shared state for all agents

from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langgraph.graph.message import add_messages
import operator


class ResearchState(TypedDict):
    # ── Input ─────────────────────────────────────────────────
    topic:          str
    depth:          str          # "quick" | "standard" | "deep"
    messages:       Annotated[List, add_messages]

    # ── Agent outputs ──────────────────────────────────────────
    research_notes: Optional[str]
    raw_sources:    Annotated[List[str], operator.add]
    analysis:       Optional[str]
    key_insights:   Annotated[List[str], operator.add]
    draft_report:   Optional[str]
    critique:       Optional[str]
    final_report:   Optional[str]

    # ── Workflow control ───────────────────────────────────────
    next_agent:     Optional[str]
    completed:      Annotated[List[str], operator.add]
    iteration:      int
    quality_score:  Optional[float]
    revision_notes: Optional[str]

    # ── Metadata ───────────────────────────────────────────────
    start_time:     Optional[float]
    word_count:     Optional[int]
    status:         Optional[str]