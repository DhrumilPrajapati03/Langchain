
# mcp_servers/knowledge_server.py
# MCP server that exposes a knowledge base

from mcp.server.fastmcp import FastMCP
from datetime import datetime
import json

mcp = FastMCP("Knowledge Base Server")

# In-memory knowledge base (in production: use a database)
KNOWLEDGE_BASE = {
    "langchain": {
        "description": "Open-source framework for building LLM applications",
        "created_by":  "Harrison Chase",
        "year":        2022,
        "language":    ["Python", "JavaScript"],
        "key_features": ["Chains", "Agents", "Memory", "RAG", "Tools"]
    },
    "langgraph": {
        "description": "Library for stateful multi-agent applications",
        "created_by":  "LangChain Team",
        "year":        2024,
        "language":    ["Python"],
        "key_features": ["StateGraph", "Nodes", "Edges", "Human-in-loop"]
    },
    "langsmith": {
        "description": "Observability platform for LangChain applications",
        "created_by":  "LangChain Team",
        "year":        2023,
        "language":    ["Python"],
        "key_features": ["Tracing", "Datasets", "Evaluation", "Monitoring"]
    },
    "chromadb": {
        "description": "Open-source vector database for AI applications",
        "created_by":  "Chroma Team",
        "year":        2023,
        "language":    ["Python"],
        "key_features": ["Embeddings", "Similarity Search", "Persistence"]
    }
}

NOTES = {}   # user notes storage


@mcp.tool()
def lookup(topic: str) -> str:
    """
    Look up information about a technology or tool.
    Available topics: langchain, langgraph, langsmith, chromadb
    """
    topic_lower = topic.lower().replace("-", "").replace(" ", "")
    info = KNOWLEDGE_BASE.get(topic_lower)
    if not info:
        available = list(KNOWLEDGE_BASE.keys())
        return f"Topic not found. Available: {available}"
    return json.dumps(info, indent=2)


@mcp.tool()
def search_knowledge(query: str) -> str:
    """
    Search knowledge base by keyword.
    Returns all matching entries.
    """
    query_lower = query.lower()
    results = {}

    for topic, info in KNOWLEDGE_BASE.items():
        # Search in description and features
        searchable = (
            info["description"].lower() +
            " ".join(info["key_features"]).lower()
        )
        if query_lower in searchable or query_lower in topic:
            results[topic] = info

    if not results:
        return f"No results for: {query}"
    return json.dumps(results, indent=2)


@mcp.tool()
def save_note(key: str, content: str) -> str:
    """
    Save a note to the knowledge base.
    key:     unique identifier for this note
    content: the note content to save
    """
    NOTES[key] = {
        "content":   content,
        "saved_at":  datetime.now().isoformat()
    }
    return f"Note saved: {key}"


@mcp.tool()
def get_note(key: str) -> str:
    """
    Retrieve a saved note by key.
    Returns the note content and when it was saved.
    """
    note = NOTES.get(key)
    if not note:
        available = list(NOTES.keys())
        return f"Note not found: {key}. Available: {available}"
    return json.dumps(note, indent=2)


@mcp.tool()
def list_all_topics() -> str:
    """List all topics available in the knowledge base"""
    topics = []
    for topic, info in KNOWLEDGE_BASE.items():
        topics.append({
            "topic":       topic,
            "description": info["description"],
            "year":        info["year"]
        })
    return json.dumps(topics, indent=2)


@mcp.resource("knowledge://index")
def get_index() -> str:
    """Resource: full knowledge base index"""
    return json.dumps(
        {k: v["description"] for k, v in KNOWLEDGE_BASE.items()},
        indent=2
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
