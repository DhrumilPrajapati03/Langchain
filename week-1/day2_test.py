# -*- coding: utf-8 -*-
"""
Auto-generated Python script from: day2_test.ipynb
"""

# Day 2: LLM Basics — Tokens, Completions & Cost

# Today you'll understand how LLMs actually work under the hood — tokens, temperature, streaming, batching, and system prompts. This is the foundation everything else builds on.

# 1. What Are Tokens?
# LLMs don't read words — they read tokens. A token is roughly:
# - 1 word ≈ 1.3 tokens
# - 1 token is generally equivalent to about 4 characters or roughly 0.75 words
# - meaning 100 tokens are approximately 75 words. A rough rule of thumb is 1 word equals 1.33 tokens
# - "LangChain" = 2 tokens (Lang + Chain)
# - "hello" = 1 token
# - Code and special characters tokenize differently

import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

texts = [
    "Hello world",
    "LangChain is amazing",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
]

for text in texts:
    tokens = encoder.encode(text)
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Count: {len(tokens)}\n")

# Cell 2: Understanding temperature
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

question = "Give me a one-word color"

for temp in [0.0, 0.7, 1.5]:
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=temp)
    responses = [llm.invoke(question).content for _ in range(3)]
    print(f"Temperature {temp}: {responses}")

# Cell 3: max_tokens controls response length
llm_short = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=20      # cuts off after 20 tokens
)

llm_long = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=500
)

q = "Explain how the internet works"
print("SHORT:", llm_short.invoke(q).content)
print("\nLONG:", llm_long.invoke(q).content)

# Cell 4: System prompts in action
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

# Without system prompt
response1 = llm.invoke("What is recursion?")
print("WITHOUT SYSTEM PROMPT:")
print(response1.content)
print("\n" + "="*60 + "\n")

# With system prompt - acts like a strict teacher
messages_teacher = [
    SystemMessage(content="You are a strict computer science professor. Always use technical terminology. Keep answers under 3 sentences."),
    HumanMessage(content="What is recursion?")
]
response2 = llm.invoke(messages_teacher)
print("AS STRICT PROFESSOR:")
print(response2.content)
print("\n" + "="*60 + "\n")

# Same question, different persona
messages_child = [
    SystemMessage(content="You are explaining concepts to a 10-year-old. Use simple words and fun analogies only."),
    HumanMessage(content="What is recursion?")
]
response3 = llm.invoke(messages_child)
print("FOR A 10-YEAR-OLD:")
print(response3.content)

# Cell 5: invoke - waits for full response
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

response = llm.invoke("Count from 1 to 5 slowly")
print("INVOKE (all at once):")
print(response.content)

# Cell 6: stream - token by token (like ChatGPT typing effect)
print("STREAM (token by token):")
for chunk in llm.stream("Count from 1 to 5 slowly"):
    print(chunk.content, end="", flush=True)
print()  # newline at end

# Cell 7: batch - multiple prompts in one call (efficient)
questions = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?"
]

print("BATCH (parallel):")
responses = llm.batch(questions)
for q, r in zip(questions, responses):
    print(f"Q: {q}")
    print(f"A: {r.content[:100]}...\n")  # first 100 chars

# ![image.png](attachment:image.png)

# Cell 8: PromptTemplate basics
from langchain_core.prompts import ChatPromptTemplate

# Define a reusable template
template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Always answer in {language}."),
    ("human", "{question}")
])

# Fill in variables
chain = template | llm  # pipe operator - we'll deep dive this tomorrow

# Use it
response = chain.invoke({
    "domain": "machine learning",
    "language": "simple English",
    "question": "What is overfitting?"
})
print(response.content)

# Cell 9: Reuse same template with different inputs
inputs = [
    {"domain": "cooking", "language": "English", "question": "What is blanching?"},
    {"domain": "finance", "language": "English", "question": "What is compound interest?"},
    {"domain": "physics",  "language": "English", "question": "What is entropy?"},
]

for inp in inputs:
    resp = chain.invoke(inp)
    print(f"Domain: {inp['domain']}")
    print(f"Answer: {resp.content[:150]}...\n")

# Cell 10: Track your token usage smartly
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

response = llm.invoke("Explain decorators in Python in 2 sentences")

metadata = response.response_metadata
print("=== TOKEN USAGE ===")
print(f"Model: {metadata.get('model', 'N/A')}")

# Handle different metadata structures
usage = metadata.get('usage', metadata.get('token_usage', {}))
if hasattr(usage, '__dict__'):
    usage = usage.__dict__   # some versions return an object
    
print(f"Input tokens:  {usage.get('prompt_tokens', 'N/A')}")
print(f"Output tokens: {usage.get('completion_tokens', 'N/A')}")
print(f"Total tokens:  {usage.get('total_tokens', 'N/A')}")

# - openai/gpt-oss-20b → 14,400 requests/day, 131K tokens/min
# - llama-3.1-8b-instant → 14,400 requests/day, 131K tokens/min
# - When you hit limits → switch to Ollama locally, zero limits
