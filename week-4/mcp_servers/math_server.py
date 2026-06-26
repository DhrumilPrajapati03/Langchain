
# mcp_servers/math_server.py
# A simple MCP server exposing math tools

from mcp.server.fastmcp import FastMCP
import math

# Create MCP server
mcp = FastMCP("Math Tools Server")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together"""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b


@mcp.tool()
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Supports: +, -, *, /, **, sqrt, sin, cos, log, pi, e
    Example: "2 ** 10" or "sqrt(144)"
    """
    allowed = {
        "sqrt":  math.sqrt,
        "sin":   math.sin,
        "cos":   math.cos,
        "log":   math.log,
        "abs":   abs,
        "round": round,
        "pi":    math.pi,
        "e":     math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as ex:
        return f"Error: {ex}"


@mcp.tool()
def statistics(numbers: list) -> dict:
    """
    Calculate statistics for a list of numbers.
    Returns mean, median, min, max, std deviation.
    """
    if not numbers:
        return {"error": "Empty list"}

    n    = len(numbers)
    mean = sum(numbers) / n
    sorted_nums = sorted(numbers)
    mid  = n // 2
    median = (
        sorted_nums[mid]
        if n % 2 != 0
        else (sorted_nums[mid-1] + sorted_nums[mid]) / 2
    )
    variance = sum((x - mean) ** 2 for x in numbers) / n
    std_dev  = math.sqrt(variance)

    return {
        "count":   n,
        "mean":    round(mean, 4),
        "median":  median,
        "min":     min(numbers),
        "max":     max(numbers),
        "std_dev": round(std_dev, 4),
        "sum":     sum(numbers)
    }


@mcp.resource("math://constants")
def get_constants() -> str:
    """Expose math constants as a resource"""
    return json.dumps({
        "pi":      math.pi,
        "e":       math.e,
        "tau":     math.tau,
        "inf":     "infinity",
        "phi":     (1 + math.sqrt(5)) / 2
    }, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
