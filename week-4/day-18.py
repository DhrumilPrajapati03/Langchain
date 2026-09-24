# -*- coding: utf-8 -*-
"""
Auto-generated Python script from: day-18.ipynb
"""

# ### Day 18: Structured Outputs

# Today you master structured outputs — making LLMs return exactly the data structure your code needs, every time. No more parsing failures, no more inconsistent formats, no more KeyError in production.

# #### 1. The Problem with Unstructured Output

# What you've been doing (fragile):
response = llm.invoke("Extract name and age from: John is 25 years old")
print(response.content)
# Output: "The name is John and the age is 25"
# Now how do you get just the age? String parsing = fragile ❌

# What you'll do after today (robust):
result = structured_llm.invoke("Extract name and age from: John is 25 years old")
print(result.name)   # "John"  ✅
print(result.age)    # 25      ✅  (already an int, not a string!)

# Cell 1: Imports
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser
)
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Optional, Literal, Any
import json, os, time
from dotenv import load_dotenv

load_dotenv()

llm  = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
fast = ChatGroq(model="llama-3.1-8b-instant",    temperature=0)

print("✅ Ready")

# #### Method 1 — Pydantic with with_structured_output

# Cell 2: Basic Pydantic structured output
# The cleanest, most reliable method

class PersonInfo(BaseModel):
    """Extract person information from text"""
    name:        str   = Field(description="Full name of the person")
    age:         int   = Field(description="Age as integer")
    city:        str   = Field(description="City where person lives")
    occupation:  str   = Field(description="Job or occupation")


# Bind schema to LLM
structured_llm = llm.with_structured_output(PersonInfo)

# Test it
result = structured_llm.invoke(
    "Raj Patel is a 27-year-old ML Engineer living in Ahmedabad"
)

print(f"Type:       {type(result)}")          # <class 'PersonInfo'>
print(f"Name:       {result.name}")           # Raj Patel
print(f"Age:        {result.age}")            # 27  (int, not string!)
print(f"City:       {result.city}")           # Ahmedabad
print(f"Occupation: {result.occupation}")     # ML Engineer

# Access as dict
print(f"\nAs dict: {result.model_dump()}")

# Cell 3: Nested Pydantic models
class Address(BaseModel):
    street:  Optional[str] = Field(default=None, description="Street address")
    city:    str           = Field(description="City name")
    state:   str           = Field(description="State or province")
    country: str           = Field(description="Country name")


class Company(BaseModel):
    name:     str           = Field(description="Company name")
    industry: str           = Field(description="Industry sector")
    size:     Optional[str] = Field(
        default=None,
        description="Company size: startup/small/medium/large/enterprise"
    )


class ProfessionalProfile(BaseModel):
    """Extract complete professional profile from text"""
    full_name:   str            = Field(description="Person's full name")
    age:         Optional[int]  = Field(default=None, description="Age if mentioned")
    title:       str            = Field(description="Job title")
    company:     Company        = Field(description="Company details")
    address:     Address        = Field(description="Location details")
    skills:      List[str]      = Field(description="List of technical skills")
    years_exp:   Optional[int]  = Field(
        default=None,
        description="Years of experience if mentioned"
    )


structured_llm = llm.with_structured_output(ProfessionalProfile)

text = """
Priya Sharma is a Senior Data Scientist at TechCorp India,
a large enterprise in the fintech industry based in Bangalore, Karnataka.
She has 6 years of experience and specializes in PyTorch,
transformers, LangChain, and MLOps. She's 29 years old.
"""

result = structured_llm.invoke(text)

print("Extracted Professional Profile:")
print(f"  Name:     {result.full_name}")
print(f"  Title:    {result.title}")
print(f"  Company:  {result.company.name} ({result.company.industry})")
print(f"  Location: {result.address.city}, {result.address.state}")
print(f"  Skills:   {result.skills}")
print(f"  Exp:      {result.years_exp} years")
print(f"\nFull JSON:\n{json.dumps(result.model_dump(), indent=2)}")

# #### Field Validation — Guarantee Data Quality

# Cell 4: Validators ensure data is always valid

class ProductReview(BaseModel):
    """Structured product review extraction"""
    product_name: str = Field(description="Name of the product")

    rating: float = Field(
        description="Rating from 1.0 to 5.0",
        ge=1.0,   # greater than or equal
        le=5.0    # less than or equal
    )

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment"
    )

    pros:  List[str] = Field(
        description="List of positive points",
        min_length=0,
        max_length=10
    )
    cons:  List[str] = Field(
        description="List of negative points",
        min_length=0,
        max_length=10
    )

    summary:      str  = Field(description="One sentence summary")
    would_recommend: bool = Field(description="Would reviewer recommend this?")

    # Custom validator
    @field_validator("rating")
    @classmethod
    def round_rating(cls, v):
        """Round to nearest 0.5"""
        return round(v * 2) / 2

    @field_validator("pros", "cons")
    @classmethod
    def clean_list(cls, v):
        """Remove empty strings from lists"""
        return [item.strip() for item in v if item.strip()]


structured_llm = llm.with_structured_output(ProductReview)

review_text = """
I bought the Sony WH-1000XM5 headphones last month.
The noise cancellation is absolutely incredible — best I've ever used.
Sound quality is fantastic and battery life lasts forever.
However, the ear cups get uncomfortable after 3 hours
and the price is quite steep at $350.
Overall I'd give it 4.5 out of 5 and yes, I'd recommend it to anyone
who travels frequently or works in a noisy environment.
"""

result = structured_llm.invoke(review_text)
print(f"Product:     {result.product_name}")
print(f"Rating:      {result.rating}/5.0")
print(f"Sentiment:   {result.sentiment}")
print(f"Recommend:   {result.would_recommend}")
print(f"Pros:        {result.pros}")
print(f"Cons:        {result.cons}")
print(f"Summary:     {result.summary}")

# #### Method 2 — JSON Mode

# Cell 5: JSON output parser - flexible, no schema required

from langchain_core.output_parsers import JsonOutputParser

# Simple JSON extraction
json_parser = JsonOutputParser()

json_prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract information and return as JSON only.
No markdown, no explanation, just valid JSON.
Schema: {{
  "title": "string",
  "technologies": ["array", "of", "strings"],
  "difficulty": "beginner|intermediate|advanced",
  "estimated_hours": number,
  "prerequisites": ["array"]
}}"""),
    ("human", "{text}")
])

json_chain = json_prompt | llm | json_parser

result = json_chain.invoke({
    "text": """Building a RAG chatbot with LangChain requires knowledge
of Python, vector databases, and LLM APIs. It's an intermediate
project that typically takes 20-30 hours for someone with basic
Python experience. You'll use ChromaDB, LangChain, and Groq API."""
})

print(f"Type: {type(result)}")   # dict - already parsed!
print(json.dumps(result, indent=2))
print(f"\nDifficulty: {result['difficulty']}")
print(f"Hours:      {result['estimated_hours']}")
print(f"Tech stack: {result['technologies']}")

# Cell 6: JSON with dynamic schemas
def extract_structured(text: str, schema: Dict) -> Dict:
    """
    Universal extractor - pass any schema, get structured data back.
    No need to define a Pydantic model for every use case.
    """
    schema_str = json.dumps(schema, indent=2)

    response = llm.invoke([
        SystemMessage(content=f"""Extract information from the text.
Return ONLY valid JSON matching this exact schema:
{schema_str}

Rules:
- Return null for fields not mentioned in text
- Arrays should be empty [] if no items found
- Booleans must be true/false (not yes/no)
- Numbers must be numeric (not strings)"""),
        HumanMessage(content=text)
    ])

    # Clean and parse
    content = response.content.strip()
    # Remove markdown code blocks if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    return json.loads(content)


# Test with different schemas
job_schema = {
    "job_title":    "string",
    "company":      "string",
    "salary_min":   "number or null",
    "salary_max":   "number or null",
    "remote":       "boolean",
    "skills":       ["array of strings"],
    "experience_years": "number or null"
}

job_text = """
Senior ML Engineer at DataCorp. Remote friendly position.
Salary: 18-25 LPA. Must have 4+ years with PyTorch,
LangChain, and MLOps experience. FastAPI knowledge preferred.
"""

result = extract_structured(job_text, job_schema)
print("Job Extraction:")
print(json.dumps(result, indent=2))

# #### Method 3 — Pydantic with Retry

# Cell 7: Auto-retry on validation failure

from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser

class StrictCodeReview(BaseModel):
    """Strict code review with validation"""
    language:   str       = Field(description="Programming language")
    bugs:       List[str] = Field(description="List of bugs found")
    rating:     int       = Field(description="Code quality 1-10", ge=1, le=10)
    complexity: Literal["low", "medium", "high"] = Field(
        description="Code complexity level"
    )
    is_production_ready: bool = Field(
        description="Is this code ready for production?"
    )
    refactored_code: Optional[str] = Field(
        default=None,
        description="Improved version of the code if needed"
    )

    @model_validator(mode="after")
    def check_consistency(self):
        """If production ready, rating must be >= 7"""
        if self.is_production_ready and self.rating < 7:
            raise ValueError(
                "Production ready code must have rating >= 7"
            )
        return self


# Parser with auto-fix on failure
base_parser   = PydanticOutputParser(pydantic_object=StrictCodeReview)
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=llm
)

review_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior code reviewer.
Analyze the code and provide structured feedback.
{format_instructions}"""),
    ("human", "Review this code:\n```python\n{code}\n```")
]).partial(format_instructions=base_parser.get_format_instructions())

# Test with problematic code
test_code = """
def divide(a, b):
    return a / b

result = divide(10, 0)
print(result)
"""

chain = review_prompt | llm | fixing_parser

result = chain.invoke({"code": test_code})
print(f"Language:    {result.language}")
print(f"Bugs:        {result.bugs}")
print(f"Rating:      {result.rating}/10")
print(f"Complexity:  {result.complexity}")
print(f"Prod Ready:  {result.is_production_ready}")
if result.refactored_code:
    print(f"Fixed Code:\n{result.refactored_code}")

# #### Streaming Structured Outputs

# Cell 8: Stream JSON as it generates

from langchain_core.output_parsers import JsonOutputParser

stream_prompt = ChatPromptTemplate.from_messages([
    ("system", """Analyze this topic and return JSON with:
{{
  "topic": "string",
  "summary": "string (2-3 sentences)",
  "key_points": ["array of 4 points"],
  "difficulty": "beginner|intermediate|advanced",
  "related_topics": ["array of 3 topics"]
}}
Return only valid JSON."""),
    ("human", "{topic}")
])

stream_chain = stream_prompt | llm | JsonOutputParser()

print("Streaming JSON output:\n")
partial_result = {}

for chunk in stream_chain.stream({"topic": "LangGraph multi-agent systems"}):
    partial_result = chunk   # JsonOutputParser yields partial dicts
    # Show what keys have been filled so far
    filled_keys = [k for k, v in chunk.items() if v]
    print(f"\rFilled: {filled_keys}", end="", flush=True)

print("\n\nFinal result:")
print(json.dumps(partial_result, indent=2))

# #### Multi-Schema Extraction

# Cell 9: Extract multiple different structures from one text

class Meeting(BaseModel):
    """Meeting details"""
    title:       str            = Field(description="Meeting title")
    date:        Optional[str]  = Field(description="Meeting date")
    time:        Optional[str]  = Field(description="Meeting time")
    attendees:   List[str]      = Field(description="List of attendees")
    agenda:      List[str]      = Field(description="Agenda items")
    action_items: List[str]     = Field(description="Action items decided")


class Budget(BaseModel):
    """Budget information"""
    total:       Optional[float]       = Field(description="Total budget amount")
    currency:    str                   = Field(default="USD")
    breakdown:   Dict[str, float]      = Field(
        default_factory=dict,
        description="Budget breakdown by category"
    )
    approved:    bool                  = Field(description="Is budget approved?")


class ProjectPlan(BaseModel):
    """Complete project extraction"""
    project_name: str             = Field(description="Project name")
    deadline:     Optional[str]   = Field(description="Project deadline")
    team_size:    Optional[int]   = Field(description="Number of team members")
    meeting:      Meeting         = Field(description="Meeting details")
    budget:       Budget          = Field(description="Budget details")
    risks:        List[str]       = Field(description="Identified risks")
    status:       Literal[
        "planning", "active", "on_hold", "completed"
    ] = Field(description="Project status")


structured_llm = llm.with_structured_output(ProjectPlan)

complex_text = """
Project Kickoff Meeting - AI Dashboard v2.0

Date: March 15, 2025 at 2:00 PM
Attendees: Raj (Tech Lead), Priya (Designer), Arjun (Backend), Meera (PM)

Agenda:
1. Review requirements
2. Assign tasks
3. Set milestones

We approved a budget of ₹8,00,000 with breakdown:
- Development: ₹5,00,000
- Design: ₹2,00,000
- Testing: ₹1,00,000

Action items:
- Raj to set up LangChain environment by March 18
- Priya to deliver wireframes by March 20
- Arjun to create API spec by March 19

The project deadline is April 30, 2025.
Main risks: API rate limits and team availability during holidays.
Team size: 4 members. Status: Planning phase.
"""

result = structured_llm.invoke(complex_text)

print(f"Project:    {result.project_name}")
print(f"Status:     {result.status}")
print(f"Deadline:   {result.deadline}")
print(f"Meeting:    {result.meeting.title}")
print(f"Attendees:  {result.meeting.attendees}")
print(f"Actions:    {result.meeting.action_items}")
print(f"Budget:     ₹{result.budget.total:,.0f}" if result.budget.total else "Budget: N/A")
print(f"Approved:   {result.budget.approved}")
print(f"Risks:      {result.risks}")

# #### Structured Output in Chains

# Cell 10: Structured outputs as intermediate steps in chains

from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Step 1: Extract structure
class CodeAnalysis(BaseModel):
    language:    str       = Field(description="Programming language")
    functions:   List[str] = Field(description="Function names found")
    has_tests:   bool      = Field(description="Does code have tests?")
    issues:      List[str] = Field(description="Issues or bugs found")
    complexity:  Literal["low", "medium", "high"] = Field(
        description="Code complexity"
    )

# Step 2: Generate recommendation based on structure
def generate_recommendation(analysis: CodeAnalysis) -> str:
    """Use structured analysis to generate targeted advice"""
    prompt = f"""Based on this code analysis:
- Language: {analysis.language}
- Functions: {analysis.functions}
- Has tests: {analysis.has_tests}
- Issues: {analysis.issues}
- Complexity: {analysis.complexity}

Provide 3 specific, actionable recommendations.
Be concise and technical."""

    return llm.invoke([HumanMessage(content=prompt)]).content


# Build the pipeline
analysis_llm  = llm.with_structured_output(CodeAnalysis)

def analyze_and_recommend(code: str) -> Dict:
    """Full pipeline: code → structured analysis → recommendations"""
    # Step 1: structured extraction
    analysis = analysis_llm.invoke(
        f"Analyze this code thoroughly:\n```\n{code}\n```"
    )

    # Step 2: targeted recommendations based on structure
    recommendations = generate_recommendation(analysis)

    return {
        "analysis":        analysis.model_dump(),
        "recommendations": recommendations
    }


# Test
sample_code = """
import requests

def get_user_data(user_id):
    response = requests.get(f"http://api.example.com/users/{user_id}")
    data = response.json()
    return data['name'], data['email']

def send_email(email, message):
    requests.post("http://api.example.com/email",
                  data={"to": email, "msg": message})

users = [1, 2, 3, 4, 5]
for uid in users:
    name, email = get_user_data(uid)
    send_email(email, f"Hello {name}!")
"""

result = analyze_and_recommend(sample_code)

print("📊 Code Analysis:")
print(json.dumps(result["analysis"], indent=2))
print(f"\n💡 Recommendations:\n{result['recommendations']}")

# #### Real-World Pipeline — Invoice Extractor

# Cell 11: Production example - extract invoice data

class LineItem(BaseModel):
    description: str   = Field(description="Item description")
    quantity:    float = Field(description="Quantity ordered")
    unit_price:  float = Field(description="Price per unit")
    total:       float = Field(description="Line total")

    @model_validator(mode="after")
    def validate_total(self):
        """Recalculate total if it doesn't match"""
        expected = round(self.quantity * self.unit_price, 2)
        if abs(self.total - expected) > 0.01:
            self.total = expected
        return self


class Invoice(BaseModel):
    invoice_number: str             = Field(description="Invoice ID")
    date:           str             = Field(description="Invoice date")
    vendor_name:    str             = Field(description="Vendor company name")
    client_name:    str             = Field(description="Client company name")
    line_items:     List[LineItem]  = Field(description="List of items")
    subtotal:       float           = Field(description="Subtotal before tax")
    tax_rate:       Optional[float] = Field(
        default=None, description="Tax rate as decimal e.g. 0.18 for 18%"
    )
    tax_amount:     Optional[float] = Field(description="Tax amount")
    total_amount:   float           = Field(description="Final total")
    currency:       str             = Field(default="INR")
    payment_terms:  Optional[str]   = Field(description="Payment terms")
    due_date:       Optional[str]   = Field(description="Payment due date")


invoice_llm = llm.with_structured_output(Invoice)

invoice_text = """
INVOICE #INV-2025-0342
Date: March 10, 2025
Due: March 25, 2025

From: TechSolutions Pvt Ltd
To: StartupCo India

Items:
1. LangChain Integration - 40 hours @ ₹3,500/hr = ₹1,40,000
2. RAG Pipeline Setup - 20 hours @ ₹3,500/hr = ₹70,000
3. LangGraph Multi-Agent - 30 hours @ ₹4,000/hr = ₹1,20,000
4. Documentation - 10 hours @ ₹2,000/hr = ₹20,000

Subtotal: ₹3,50,000
GST (18%): ₹63,000
Total: ₹4,13,000

Payment Terms: Net 15 days
"""

invoice = invoice_llm.invoke(invoice_text)

print(f"Invoice:      {invoice.invoice_number}")
print(f"Vendor:       {invoice.vendor_name}")
print(f"Client:       {invoice.client_name}")
print(f"Date:         {invoice.date}")
print(f"Due:          {invoice.due_date}")
print(f"\nLine Items:")
for item in invoice.line_items:
    print(f"  {item.description[:35]:35} | "
          f"Qty: {item.quantity:4} | "
          f"Rate: ₹{item.unit_price:,.0f} | "
          f"Total: ₹{item.total:,.0f}")
print(f"\nSubtotal: ₹{invoice.subtotal:,.0f}")
print(f"Tax:      ₹{invoice.tax_amount:,.0f}")
print(f"TOTAL:    ₹{invoice.total_amount:,.0f}")

# #### Structured Output Cheat Sheet

# METHOD                          BEST FOR
# ──────────────────────────────────────────────────────────────
# with_structured_output(Model)   Production code, typed access
#                                 result.field_name syntax
#                                 Auto-validation
# 
# JsonOutputParser()              Flexible schemas, prototyping
#                                 Dynamic field names
#                                 Streaming JSON
# 
# PydanticOutputParser            When you need format_instructions
# + OutputFixingParser            Auto-retry on parse failure
# 
# PYDANTIC FIELD TYPES:
#   str, int, float, bool         Basic types
#   List[str]                     Array of strings
#   Dict[str, float]              Key-value pairs
#   Optional[str]                 Nullable field
#   Literal["a","b","c"]          Enum constraint
#   ge=1, le=10                   Numeric bounds
#   min_length, max_length        String/list bounds
#   @field_validator              Custom validation logic
#   @model_validator              Cross-field validation
# 
# PRODUCTION RULES:
#   ✅ Always use Optional for fields that may not exist
#   ✅ Add validators for business logic constraints
#   ✅ Use Literal for fixed set of values
#   ✅ Set temperature=0 for extraction tasks
#   ✅ Use OutputFixingParser for unreliable models
#   ❌ Never trust raw .content for structured data
#   ❌ Don't use regex to parse LLM output
