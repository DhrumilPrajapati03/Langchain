# -*- coding: utf-8 -*-
"""
Auto-generated Python script from: day4_test.ipynb
"""

# ### Prompt Engineering

# Today you'll learn how to dramatically improve LLM outputs without changing the model — just by changing how you talk to it. This is one of the highest-leverage skills in AI development.

# #### 1. Why Prompt Engineering Matters
# Same model, same question, wildly different quality:

# Cell 1: Bad prompt vs Good prompt
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

# BAD prompt
bad = llm.invoke("fix my code: def add(a,b) return a+b").content
print("BAD PROMPT OUTPUT:")
print(bad)

print("\n" + "="*60 + "\n")

# GOOD prompt
good_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior Python developer doing code review.
When given broken code:
1. Identify the exact bug with line reference
2. Show the fixed code in a code block
3. Explain WHY it was broken in one sentence
Always be precise and technical."""),
    ("human", "Fix this code:\n```python\n{code}\n```")
])

good_chain = good_prompt | llm | StrOutputParser()
good = good_chain.invoke({"code": "def add(a,b) return a+b"})
print("GOOD PROMPT OUTPUT:")
print(good)

# #### 2. Zero-Shot vs Few-Shot Prompting
# - Zero-shot = just ask, no examples
# - Few-shot = show examples before asking — teaches the model the exact format you want

# Cell 2: Zero-shot (no examples)
zero_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the sentiment of the given text."),
    ("human", "{text}")
])

chain = zero_shot_prompt | llm | StrOutputParser()
result = chain.invoke({"text": "This product is okay I guess"})
print("ZERO-SHOT:")
print(result)  # verbose, inconsistent format

# #### Few-shot

# Cell 3: Few-shot (with examples) - consistent output
from langchain_core.prompts import FewShotChatMessagePromptTemplate

# Define your examples
examples = [
    {
        "input": "I absolutely love this product!",
        "output": "POSITIVE | Confidence: High"
    },
    {
        "input": "This is the worst purchase I've ever made.",
        "output": "NEGATIVE | Confidence: High"
    },
    {
        "input": "It arrived on time I suppose.",
        "output": "NEUTRAL | Confidence: Medium"
    },
    {
        "input": "Not bad, not great either.",
        "output": "NEUTRAL | Confidence: Low"
    }
]

# Example prompt template
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# Few-shot prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# Full prompt with few-shot examples
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify sentiment. Format: SENTIMENT | Confidence: Level"),
    few_shot_prompt,
    ("human", "{text}")
])

chain = final_prompt | llm | StrOutputParser()

# Test with ambiguous inputs
test_cases = [
    "This product changed my life honestly",
    "Meh, it works I guess",
    "Total waste of money, avoid at all costs",
    "Surprisingly decent for the price"
]

print("FEW-SHOT RESULTS:")
for text in test_cases:
    result = chain.invoke({"text": text})
    print(f"  Input: '{text}'")
    print(f"  Output: {result}\n")

# #### Chain of Thought (CoT) Prompting
# - CoT forces the model to think step by step before answering — dramatically improves accuracy on reasoning tasks.

# Cell 4: Without CoT - often wrong on math/logic
no_cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question directly."),
    ("human", "{question}")
])

chain = no_cot_prompt | llm | StrOutputParser()
result = chain.invoke({
    "question": "If a train travels 120km in 1.5 hours, then stops for 20 minutes, then travels 80km in 1 hour, what is the average speed for the entire journey?"
})
print("WITHOUT CoT:")
print(result)

# Cell 5: With CoT - accurate reasoning
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a precise problem solver.
ALWAYS follow this exact process:
Step 1: Identify what is being asked
Step 2: List all given information
Step 3: Show your calculations step by step
Step 4: State the final answer clearly

Never skip steps. Show all working."""),
    ("human", "{question}")
])

chain = cot_prompt | llm | StrOutputParser()
result = chain.invoke({
    "question": "If a train travels 120km in 1.5 hours, then stops for 20 minutes, then travels 80km in 1 hour, what is the average speed for the entire journey?"
})
print("WITH CoT:")
print(result)

# Cell 6: CoT with few-shot examples (most powerful combo)
cot_examples = [
    {
        "input": "A shop buys items for ₹200 and sells for ₹250. If they sell 40 items, what is the total profit?",
        "output": """Step 1: What is asked? → Total profit from selling 40 items
Step 2: Given info:
  - Cost price per item: ₹200
  - Selling price per item: ₹250
  - Number of items: 40
Step 3: Calculations:
  - Profit per item = ₹250 - ₹200 = ₹50
  - Total profit = ₹50 × 40 = ₹2,000
Step 4: Final Answer → Total profit = ₹2,000"""
    }
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=cot_examples
)

cot_few_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "Solve problems step by step. Always show your working."),
    few_shot,
    ("human", "{question}")
])

chain = cot_few_shot_prompt | llm | StrOutputParser()
result = chain.invoke({
    "question": "A developer earns ₹80,000/month. After 30% tax, they save 25% of remaining. How much do they save per year?"
})
print("CoT + FEW-SHOT:")
print(result)

# Cell 7: Dynamic system prompts based on user context
from langchain_core.prompts import ChatPromptTemplate

adaptive_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a coding tutor for {skill_level} developers.

Adjust your explanation based on skill level:
- beginner: Use analogies, avoid jargon, explain every term
- intermediate: Use proper terms, show code examples
- expert: Be concise, focus on edge cases and best practices

Topic domain: {domain}
Response format: {format}"""),
    ("human", "{question}")
])

chain = adaptive_prompt | llm | StrOutputParser()

# Same question, three different audiences
question = "Explain decorators"

for level in ["beginner", "intermediate", "expert"]:
    print(f"\n{'='*60}")
    print(f"SKILL LEVEL: {level.upper()}")
    print('='*60)
    result = chain.invoke({
        "skill_level": level,
        "domain": "Python",
        "format": "concise explanation with one code example",
        "question": question
    })
    print(result)

# Cell 8: Structured output with Pydantic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define exact output structure
class CodeReview(BaseModel):
    language: str = Field(description="Programming language detected")
    bugs: List[str] = Field(description="List of bugs found")
    improvements: List[str] = Field(description="List of suggested improvements")
    rating: int = Field(description="Code quality rating from 1-10")
    summary: str = Field(description="One sentence overall assessment")

parser = JsonOutputParser(pydantic_object=CodeReview)

structured_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a code reviewer. 
Analyze the code and respond ONLY with a JSON object.
{format_instructions}"""),
    ("human", "Review this code:\n```python\n{code}\n```")
]).partial(format_instructions=parser.get_format_instructions())

chain = structured_prompt | llm | parser

test_code = """
def get_user(users, id):
    for u in users:
        if u['id'] == id:
            return u
    return None

users = [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]
print(get_user(users, 1))
"""

result = chain.invoke({"code": test_code})
print(f"Language: {result['language']}")
print(f"Rating: {result['rating']}/10")
print(f"Summary: {result['summary']}")
print(f"\nBugs ({len(result['bugs'])}):")
for bug in result['bugs']:
    print(f"  - {bug}")
print(f"\nImprovements ({len(result['improvements'])}):")
for imp in result['improvements']:
    print(f"  - {imp}")

# Cell 9: Generate → Critique → Improve pattern
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

# Step 1: Generate first draft
generate_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Write a concise technical blog introduction."),
        ("human", "Topic: {topic}")
    ]) | llm | StrOutputParser()
)

# Step 2: Critique the draft
critique_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """Critique this blog introduction.
Identify exactly 3 specific weaknesses.
Format as numbered list only."""),
        ("human", "{draft}")
    ]) | llm | StrOutputParser()
)

# Step 3: Improve based on critique
improve_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Rewrite the blog introduction fixing all the mentioned weaknesses."),
        ("human", "Original:\n{draft}\n\nWeaknesses to fix:\n{critique}")
    ]) | llm | StrOutputParser()
)

# Run the full pipeline
topic = "Why every developer should learn LangChain in 2025"

print("STEP 1: Generating draft...")
draft = generate_chain.invoke({"topic": topic})
print(draft)

print("\nSTEP 2: Critiquing...")
critique = critique_chain.invoke({"draft": draft})
print(critique)

print("\nSTEP 3: Improved version...")
final = improve_chain.invoke({"draft": draft, "critique": critique})
print(final)

# TECHNIQUE          WHEN TO USE                    EXAMPLE TRIGGER
# ─────────────────────────────────────────────────────────────────
# Zero-shot         Simple, clear tasks             "Translate this to French"
# Few-shot          Consistent format needed        Sentiment, classification
# Chain-of-Thought  Math, logic, multi-step tasks   "Solve step by step"
# CoT + Few-shot    Complex reasoning + format      Best of both worlds
# Pydantic output   Need structured data in code    JSON with typed fields
# Generate→Critique Quality matters, iterative      Blog posts, code, plans
# Dynamic prompts   Different users/contexts        Skill-level adaptation
# 
# GOLDEN RULES:
#   ✅ Be specific about format ("Reply as JSON", "Max 3 bullet points")
#   ✅ Give the model a role ("You are a senior engineer...")
#   ✅ Show don't tell (examples > instructions)
#   ✅ Break complex tasks into steps
#   ❌ Never say "don't do X" — say "always do Y instead"
#   ❌ Never leave output format ambiguous
