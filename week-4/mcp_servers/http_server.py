
# mcp_servers/http_server.py
# Production-style MCP server over HTTP

from mcp.server.fastmcp import FastMCP
from datetime import datetime
import json, math

mcp = FastMCP(
    "Production Tools Server",
    host="127.0.0.1",
    port=8765
)

# ── Tools ─────────────────────────────────────────────────────

@mcp.tool()
def get_timestamp(format: str = "iso") -> str:
    """
    Get current timestamp.
    format: "iso" | "human" | "unix"
    """
    now = datetime.now()
    formats = {
        "iso":   now.isoformat(),
        "human": now.strftime("%B %d, %Y at %I:%M %p"),
        "unix":  str(int(now.timestamp()))
    }
    return formats.get(format, now.isoformat())


@mcp.tool()
def unit_converter(
    value: float,
    from_unit: str,
    to_unit: str
) -> str:
    """
    Convert between common units.
    Supports: km/miles, kg/lbs, celsius/fahrenheit, liters/gallons
    """
    conversions = {
        ("km",         "miles"):      lambda x: x * 0.621371,
        ("miles",      "km"):         lambda x: x * 1.60934,
        ("kg",         "lbs"):        lambda x: x * 2.20462,
        ("lbs",        "kg"):         lambda x: x * 0.453592,
        ("celsius",    "fahrenheit"): lambda x: (x * 9/5) + 32,
        ("fahrenheit", "celsius"):    lambda x: (x - 32) * 5/9,
        ("liters",     "gallons"):    lambda x: x * 0.264172,
        ("gallons",    "liters"):     lambda x: x * 3.78541,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key not in conversions:
        return f"Conversion {from_unit}→{to_unit} not supported"

    result = conversions[key](value)
    return f"{value} {from_unit} = {result:.4f} {to_unit}"


@mcp.tool()
def text_analyzer(text: str) -> str:
    """
    Analyze text statistics.
    Returns word count, char count, sentence count, reading time.
    """
    words     = text.split()
    sentences = text.count(".") + text.count("!") + text.count("?")
    return json.dumps({
        "characters":          len(text),
        "characters_no_space": len(text.replace(" ", "")),
        "words":               len(words),
        "sentences":           max(sentences, 1),
        "paragraphs":          text.count("\n\n") + 1,
        "reading_time_mins":   max(1, len(words) // 200),
        "avg_word_length":     round(
            sum(len(w) for w in words) / max(len(words), 1), 2
        )
    }, indent=2)


@mcp.tool()
def generate_report_template(
    report_type: str,
    title: str
) -> str:
    """
    Generate a markdown report template.
    report_type: "technical" | "executive" | "research"
    """
    templates = {
        "technical": f"""# {title}

## Overview
[Brief technical overview]

## Architecture
[System design and components]

## Implementation Details
[Code and technical specifics]

## Performance Metrics
[Benchmarks and measurements]

## Known Issues
[Current limitations and bugs]

## Next Steps
[Future improvements]
""",
        "executive": f"""# {title}

## Executive Summary
[2-3 sentence summary for leadership]

## Business Impact
[ROI and business value]

## Key Metrics
[Critical KPIs]

## Risks & Mitigations
[Top risks and how they are addressed]

## Recommendation
[Clear recommended action]
""",
        "research": f"""# {title}

## Abstract
[150-word summary]

## Introduction
[Background and motivation]

## Methodology
[Research approach]

## Results
[Findings with data]

## Discussion
[Interpretation and implications]

## Conclusion
[Summary and future work]

## References
[Citations]
"""
    }

    template = templates.get(
        report_type.lower(),
        templates["technical"]
    )
    return template


if __name__ == "__main__":
    print("Starting HTTP MCP server on http://127.0.0.1:8765")
    mcp.run(transport="streamable-http")
