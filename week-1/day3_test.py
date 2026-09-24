# -*- coding: utf-8 -*-
"""
Auto-generated Python script from: day3_test.ipynb
"""

# ### LangChain Chains & LCEL 🔗
# - Today you'll learn the most important concept in LangChain — the pipe | operator and LCEL (LangChain Expression Language). Everything in this course — RAG, agents, multi-agent systems — is built on this foundation.

# #### 1. What is LCEL?
# - LCEL is LangChain's way of composing components together using the | pipe operator — same concept as Unix pipes.
# - `input → prompt | llm | output_parser → final output`
# - Each component takes input, transforms it, passes to next. Clean, readable, and powerful.

# #### 2. Without LCEL vs With LCEL

from pydantic.v1.parse import load_str_bytes
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model= "llama-3.1-8b-instant", temperature=0.7)

# Step 1: format prompt manually
prompt_template = ChatPromptTemplate.from_messages([
    ("system","You are a helpfull assistant."),
    ("human","{question}")
])

formatted_prompt = prompt_template.invoke({"question": "What is python?"})

# Step 2: pass to LLM manually  
response = llm.invoke(formatted_prompt)

# Step 3: extract content manually
output = response.content
print(output)

# Cell 2: The LCEL way - clean and composable
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model= "llama-3.1-8b-instant", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

output = chain.invoke({"question": "what is python?"})
print(output)
print(type(output))

# Cell 3: StrOutputParser - plain text
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "Name 3 programming languages"})
print(type(result))   # str
print(result)

# Cell 4: JsonOutputParser - structured data
from langchain_core.output_parsers import JsonOutputParser

json_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data extractor. 
    Always respond with valid JSON only. No extra text.
    Format: {{"name": "", "founded": 0, "language": ""}}"""),
    ("human", "Tell me about the {framework} framework")
])

json_chain = json_prompt | llm | JsonOutputParser()

result = json_chain.invoke({"framework": "LangChain"})
print(type(result))        # dict!
print(result)
print(result["name"])      # access like normal dict
print(result["founded"])

# Cell 5: CommaSeparatedListOutputParser
from langchain_core.output_parsers import CommaSeparatedListOutputParser

list_parser = CommaSeparatedListOutputParser()

list_prompt = ChatPromptTemplate.from_messages([
    ("system", "Return only a comma-separated list. No other text."),
    ("human", "List 5 {category}")
])

list_chain = list_prompt | llm | list_parser

result = list_chain.invoke({"category": "Python libraries for AI"})
print(type(result))   # list!
print(result)         # ['LangChain', 'PyTorch', ...]
for item in result:
    print(f"  - {item}")

# #### RunnableSequence — Chaining Multiple Steps

# This is the most common one.
# 
# - It runs tasks one after another, like a pipeline.
# 
#  💡 Analogy:
# Making tea:
# - Boil water
# - Add tea leaves
# - Pour into cup
# - Each step depends on the previous one.

# Key Idea:
# - Output of step 1 → goes into step 2
# - Linear flow

# Cell 6: Multi-step chain
# Step 1: Generate a topic summary
# Step 2: Then translate it

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

# Chain 1: summarize
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following topic in exactly 2 sentences."),
    ("human", "{topic}")
])

# Chain 2: translate (takes output of chain 1)
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following text to {language}. Return only the translation."),
    ("human", "{text}")
])

# Build individual chains
summarize_chain = summarize_prompt | llm | StrOutputParser()
translate_chain  = translate_prompt | llm | StrOutputParser()

# Combine them - output of first feeds into second
from langchain_core.runnables import RunnablePassthrough

full_chain = (
    summarize_chain
    | (lambda summary: {"text": summary, "language": "Hindi"})
    | translate_chain
)

result = full_chain.invoke({"topic": "How neural networks learn"})
print("Hindi Summary:")
print(result)

# #### RunnableParallel — Run Multiple Chains Simultaneously

# Runs multiple tasks simultaneously on the same input.
# 
# 💡 Analogy:
# 
# You ask 3 friends:
# 
# - One translates
# - One summarizes
# - One explains
# 
# All work at the same time.

#  Key Idea:
# - Same input → multiple outputs
# - Faster (parallel execution)

# Cell 7: Run chains in parallel - much faster than sequential
from langchain_core.runnables import RunnableParallel

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

# Three different analysis chains
pros_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "List only the PROS. Be concise, max 3 points."),
        ("human", "{technology}")
    ]) | llm | StrOutputParser()
)

cons_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "List only the CONS. Be concise, max 3 points."),
        ("human", "{technology}")
    ]) | llm | StrOutputParser()
)

usecase_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "List only 3 best use cases. Be concise."),
        ("human", "{technology}")
    ]) | llm | StrOutputParser()
)

# Run ALL THREE in parallel
analysis_chain = RunnableParallel(
    pros=pros_chain,
    cons=cons_chain,
    use_cases=usecase_chain
)

import time
start = time.time()
result = analysis_chain.invoke({"technology": "Python"})
elapsed = time.time() - start

print(f"All 3 responses in {elapsed:.2f}s\n")
print("PROS:")
print(result["pros"])
print("\nCONS:")
print(result["cons"])
print("\nUSE CASES:")
print(result["use_cases"])

# #### RunnablePassthrough — Pass Input Along

# This is used when you want to:
# - 👉 Keep the original input AND add new data

# 💡 Analogy:
# 
# You submit a form:
# - Keep original data
# - Add extra computed fields

# 🔥 Key Idea:
# - Doesn’t replace input
# - Adds extra info

# Cell 8: RunnablePassthrough - keep original input alongside output
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "Explain this concept in one sentence."),
    ("human", "{concept}")
])

chain = RunnableParallel(
    concept=RunnablePassthrough(),     # passes original input through unchanged
    explanation=(explain_prompt | llm | StrOutputParser())
)

result = chain.invoke({"concept": "recursion"})
print(f"Concept: {result['concept']}")
print(f"Explanation: {result['explanation']}")

# ##### Simple mental model
# - RunnableSequence = Step-by-step workflows
# - RunnableParallel = Do multiple things at once
# - RunnablePassthrough = Keep original data + add info

# #### Streaming with LCEL Chains

# Cell 9: Stream entire chain output
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a storyteller."),
        ("human", "Tell a 3-sentence story about {topic}")
    ])
    | llm
    | StrOutputParser()
)

print("Streaming story:\n")
for chunk in chain.stream({"topic": "a programmer who discovers AI"}):
    print(chunk, end="", flush=True)
print("\n")

# #### Code Review Assistant

# Cell 10: Code Review Assistant
# Input: code snippet
# Output: bugs, improvements, and rating — all in parallel

from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)

bugs_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a code reviewer. Find bugs only. Be specific."),
        ("human", "Review this code:\n{code}")
    ]) | llm | StrOutputParser()
)

improve_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a senior engineer. Suggest improvements only. Max 3 points."),
        ("human", "Review this code:\n{code}")
    ]) | llm | StrOutputParser()
)

rating_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Rate code quality 1-10. Reply with just: 'Rating: X/10 - one reason'"),
        ("human", "Rate this code:\n{code}")
    ]) | llm | StrOutputParser()
)

code_review_chain = RunnableParallel(
    bugs=bugs_chain,
    improvements=improve_chain,
    rating=rating_chain
)

# Test it with buggy code
test_code = """
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    average = total / len(numbers)
    return average

result = calculate_average([1, 2, 3, 4, 5])
print(result)
"""

print("🔍 Analyzing code...\n")
review = code_review_chain.invoke({"code": test_code})

print("🐛 BUGS FOUND:")
print(review["bugs"])
print("\n💡 IMPROVEMENTS:")
print(review["improvements"])
print("\n⭐ RATING:")
print(review["rating"])

# ![image.png](attachment:image.png)
