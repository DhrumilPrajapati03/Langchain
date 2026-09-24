# -*- coding: utf-8 -*-
"""
Auto-generated Python script from: day1_test.ipynb
"""

# Cell 1: Imports and env check
from dotenv import load_dotenv
import os

load_dotenv()
print("✅ Environment loaded")
print(f"API Key present: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")

# Best models available on Groq free tier (as of 2025)
groq_models = {
    "best_overall":   "openai/gpt-oss-20b",   # use this most often
    "fastest":        "llama-3.1-8b-instant",       # when speed matters
    "coding":         "qwen-2.5-coder-32b",         # for code generation
    "reasoning":      "deepseek-r1-distill-llama-70b", # for complex tasks
}

from langchain_groq import ChatGroq

llm_groq = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature=0.7
)

response = llm_groq.invoke("What is the capital of Gujarat?")
print(response)
print(response.content)
print(response.usage_metadata)
print(response.response_metadata)

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Initialize the Groq model
llm = ChatGroq(model="llama-3.1-8b-instant")

# Create the agent
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

# pip install -qU langchain langchain-groq
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import os

# Make sure your API key is set in your environment variables
# os.environ["GROQ_API_KEY"] = "your-groq-api-key"

# 1. We wrap your function in the @tool decorator
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

tools = [get_weather]

# 2. Initialize the Groq model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# 3. Create a strict prompt template for the Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{input}"),
    # The agent_scratchpad is fundamentally required by Langchain AgentExecutors 
    # to store intermediate responses / tool thought processes:
    ("placeholder", "{agent_scratchpad}"),
])

# 4. Create the core agent that binds the LLM and the tools
agent = create_tool_calling_agent(
    llm=llm, 
    tools=tools, 
    prompt=prompt
)

# 5. Wrap it in an AgentExecutor which loops through the tools automatically
# Tip: set verbose=False if you don't want to see the terminal printout of its thoughts
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. Run the agent (AgentExecutor expects {"input": "..."} instead of a raw message list)
result = agent_executor.invoke({
    "input": "What is the weather in Patdi"
})

# Print just the final answer
print("\nFinal Output:", result["output"])
