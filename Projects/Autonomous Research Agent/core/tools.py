# core/tools.py
# All tools available to agents

import math
import json
import requests
from datetime import datetime
from langchain_core.tools import tool
from duckduckgo_search import DDGS


# ── Research Tools ────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """
    Search web for current information and facts.
    Use for recent news, statistics, and up-to-date data.
    Input: focused search query string.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No results found"
        return "\n\n".join(
            f"SOURCE: {r['title']}\nURL: {r['href']}\n{r['body'][:350]}"
            for r in results
        )
    except Exception as e:
        return f"Search error: {e}"


@tool
def wikipedia_lookup(topic: str) -> str:
    """
    Get encyclopedic information from Wikipedia.
    Use for background knowledge, history, and established facts.
    Input: topic or person name.
    """
    try:
        url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + topic.replace(" ", "_")
        )
        resp = requests.get(url, timeout=6).json()
        if resp.get("type") == "disambiguation":
            return f"'{topic}' is ambiguous. Try more specific term."
        title = resp.get("title", topic)
        extract = resp.get("extract", "Not found")
        wiki_url = resp.get("content_urls", {}).get("desktop", {}).get("page", "")
        return f"Wikipedia - {title}:\n{extract[:800]}\nURL: {wiki_url}"
    except Exception as e:
        return f"Wikipedia error: {e}"


@tool
def search_news(query: str) -> str:
    """
    Search for recent news articles on a topic.
    Use for current events and latest developments.
    Input: news search query.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=4))
        if not results:
            return "No news found"
        return "\n\n".join(
            f"[{r.get('date', 'N/A')}] {r['title']}\n{r.get('body', '')[:250]}"
            for r in results
        )
    except Exception as e:
        return f"News search error: {e}"


# ── Analysis Tools ────────────────────────────────────────────

@tool
def calculate(expression: str) -> str:
    """
    Evaluate mathematical expressions.
    Use for statistics, percentages, and numerical analysis.
    Input: Python math expression like '(100 * 0.15)' or 'sqrt(144)'.
    """
    try:
        allowed = {
            'abs': abs, 'round': round, 'min': min,
            'max': max, 'pow': pow, 'sqrt': math.sqrt,
            'pi': math.pi, 'e': math.e, 'log': math.log,
            'sum': sum
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as ex:
        return f"Calculation error: {ex}"


@tool
def extract_statistics(text: str) -> str:
    """
    Extract numerical data and statistics from text.
    Use to identify key metrics in research findings.
    Input: text containing numbers and statistics.
    """
    import re
    patterns = {
        "percentages": r'\d+(?:\.\d+)?%',
        "large_numbers": r'\$?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|trillion|B|M|T))?',
        "years": r'\b20\d{2}\b|\b19\d{2}\b',
    }
    found = {}
    for label, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            found[label] = list(set(matches))[:8]

    return json.dumps(found, indent=2) if found else "No statistics found"


@tool
def get_current_date() -> str:
    """Get current date and time for timestamping reports."""
    return datetime.now().strftime("%B %d, %Y at %H:%M")


# ── Quality Tools ─────────────────────────────────────────────

@tool
def count_words(text: str) -> str:
    """Count words and estimate reading time."""
    words = len(text.split())
    reading_time = max(1, words // 200)
    return json.dumps({
        "word_count": words,
        "reading_time_minutes": reading_time
    })


@tool
def check_coverage(topic: str, content: str) -> str:
    """
    Check if content covers the key aspects of a topic.
    Returns what's covered and what might be missing.
    Input: topic string and content to evaluate.
    """
    # Simple heuristic check
    content_lower = content.lower()
    topic_words = topic.lower().split()
    coverage = sum(1 for w in topic_words if w in content_lower)
    coverage_pct = (coverage / max(len(topic_words), 1)) * 100

    return json.dumps({
        "topic_coverage": f"{coverage_pct:.0f}%",
        "covered_keywords": coverage,
        "total_keywords": len(topic_words),
        "assessment": "Good coverage" if coverage_pct > 60 else "May need more depth"
    })


# ── Tool sets per agent ───────────────────────────────────────
RESEARCHER_TOOLS = [web_search, wikipedia_lookup, search_news]
ANALYST_TOOLS    = [calculate, extract_statistics, get_current_date]
WRITER_TOOLS     = [count_words, get_current_date]
CRITIC_TOOLS     = [check_coverage, count_words]