# Cell 6: Connect to HTTP MCP server
# (Start the HTTP server first in a terminal!)
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import os
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
env_path = os.path.join(repo_root, ".env")
load_dotenv(dotenv_path=env_path)

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(f"GROQ_API_KEY is not set. Check {env_path}")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
print("✅ Ready")


async def run_with_http_server(question: str):
    """Connect to HTTP MCP server"""

    server_config = {
        "production_tools": {
            "url": "http://127.0.0.1:8765/mcp",
            "transport": "streamable_http",
            "headers": {
                "Accept": "text/event-stream"
            },
        }
    }

    try:
        client = MultiServerMCPClient(server_config)
        tools = await client.get_tools()
        print(f"HTTP server tools: {[t.name for t in tools]}")

        agent = create_react_agent(model=llm, tools=tools)
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=question)]
        })
        return result["messages"][-1].content

    except Exception as e:
        return f"Server not running: {e}\nStart it with: python mcp_servers/http_server.py"


async def main():
    result = await run_with_http_server(
        "Convert 100 km to miles and tell me the current timestamp"
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())