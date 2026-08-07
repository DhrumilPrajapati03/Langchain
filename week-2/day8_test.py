# -*- coding: utf-8 -*-
"""
Auto-generated Python script from: day8_test.ipynb
"""

# ### Day 8: RAG Pipeline — Retrieval Augmented Generation

# Today everything clicks together. You'll connect Day 6 (loaders) + Day 7 (vectors) + LLM into a complete RAG system that can answer questions about any document with cited sources.

# #### 1. What RAG Actually Does

# Without RAG:
#   User: "What does our company refund policy say?"
#   LLM:  "I don't have access to your company documents" ❌
# 
# With RAG:
#   User: "What does our company refund policy say?"
#     ↓
#   Search vectorstore for relevant chunks
#     ↓
#   Found: "Refunds are processed within 7 days..." (from policy.pdf)
#     ↓
#   LLM reads chunk + answers question
#   Bot:  "According to your policy, refunds take 7 days..." ✅

# RAG = giving LLM eyes to read YOUR documents at query time

# #### 2. The RAG Architecture

# ── INDEXING (done once) ─────────────────────────────
#           │                                          │
#      Documents                                  VectorStore
#      (PDF/web/txt)  →  Loader → Splitter →  Embeddings → ChromaDB
#           │                                          │
#            ──────────────────────────────────────────
# 
#            ── RETRIEVAL + GENERATION (every query) ──
#           │                                          │
#       User Query                               Final Answer
#           │                                          │
#      Embed query  →  Search VectorStore  →  Top-K chunks
#                                                │
#                                     Prompt = query + chunks
#                                                │
#                                              LLM
#                                                │
#                                         Answer + Sources

# Cell 1: All imports for today
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import os, shutil

load_dotenv()

# LLM - Groq for quality answers
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# temperature=0 for RAG - we want factual, consistent answers

# Embeddings - local Ollama
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")

print("✅ All imports ready")

# #### Step 1 — Build the Knowledge Base

# Cell 2: Create a richer sample document for RAG
import os
sample_content = """
# The Complete Guide to LangChain

## Introduction
LangChain is an open-source framework released in 2022 by Harrison Chase.
It simplifies building applications powered by large language models (LLMs).
The framework supports Python and JavaScript.

## Core Components

### 1. Models
LangChain supports three types of models:
- LLMs: text-in, text-out (GPT-3 style)
- Chat Models: message-in, message-out (GPT-4 style) 
- Embedding Models: text-in, vector-out

### 2. Prompts
PromptTemplates allow dynamic prompt construction.
FewShotPromptTemplates include examples in the prompt.
ChatPromptTemplates handle multi-turn conversations.

### 3. Chains
Chains combine components using the pipe operator.
LCEL (LangChain Expression Language) is the modern way to build chains.
RunnableParallel runs multiple chains simultaneously.
RunnablePassthrough passes inputs unchanged through a chain.

### 4. Memory
ConversationBufferMemory stores full conversation history.
ConversationSummaryMemory stores a running summary (saves tokens).
VectorStoreMemory retrieves relevant past messages semantically.

### 5. Agents
Agents use LLMs to decide which tools to use.
ReAct agents follow a Reason → Act → Observe loop.
LangGraph enables multi-agent orchestration with state machines.

### 6. Document Loaders
PyPDFLoader loads PDF documents page by page.
WebBaseLoader scrapes and loads web pages.
CSVLoader loads tabular data row by row.
TextLoader handles plain text files.

### 7. Vector Stores
ChromaDB offers persistent storage with metadata filtering.
FAISS provides fast in-memory approximate nearest neighbor search.
Pinecone is a managed cloud vector database.

## RAG (Retrieval Augmented Generation)
RAG was introduced to solve the knowledge cutoff problem.
It retrieves relevant documents and injects them into the LLM prompt.
RAG is better than fine-tuning for frequently changing knowledge bases.
The typical RAG pipeline has two phases: indexing and retrieval.

## LangSmith
LangSmith is LangChain's observability platform.
It traces every chain execution for debugging.
LangSmith offers a free tier with 5000 traces per month.

## Performance Tips
Use async methods for production deployments.
Cache embeddings to avoid recomputing them.
Use MMR retrieval for diverse search results.
Chunk overlap should be 10-20% of chunk size.
Temperature=0 gives consistent factual answers in RAG.
"""

os.makedirs("rag_docs", exist_ok=True)
with open("rag_docs/langchain_guide.txt", "w", encoding="utf-8") as f:
    f.write(sample_content)

print("✅ Knowledge base document created")

# Cell 3: Index the document
def build_rag_index(file_path, db_path, chunk_size=500, overlap=50):
    """Load → Split → Embed → Store"""

    # Load
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # Store
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name="rag_knowledge"
    )

    print(f"✅ Indexed {len(chunks)} chunks from {file_path}")
    return vectorstore

vectorstore = build_rag_index(
    "rag_docs/langchain_guide.txt",
    "./rag_index"
)

# #### Step 2 — The RAG Prompt

# Cell 4: RAG prompt template - the most important prompt you'll write
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions 
based ONLY on the provided context.

Rules:
- Answer using ONLY information from the context below
- If the answer is not in the context, say "I don't have that information"
- Always cite which part of the context supports your answer
- Be concise and direct

Context:
{context}
"""),
    ("human", "{question}")
])

print("✅ RAG prompt ready")
print("\nPrompt variables needed:")
print("  {context}  ← retrieved chunks go here")
print("  {question} ← user's question goes here")

# #### Step 3 — Build the RAG Chain

# Cell 5: Simple RAG chain - the classic pattern
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    """Convert list of Documents to single string for prompt"""
    return "\n\n---\n\n".join(
        f"Source {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

# The RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,  # retrieve + format
        "question": RunnablePassthrough()    # pass question unchanged
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

print("✅ RAG chain built")
print("\nChain flow:")
print("  question → retriever → format_docs ┐")
print("  question → RunnablePassthrough     ┤→ prompt → llm → answer")

# Cell 6: Query your RAG system!
questions = [
    "Who created LangChain and when?",
    "What is the difference between ChromaDB and FAISS?",
    "What temperature should I use for RAG?",
    "What is LCEL?",
    "Does LangChain support JavaScript?"   
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print(f"A: {rag_chain.invoke(q)}")

# #### RAG with Sources — Professional Pattern

# Cell 7: Return answer AND source documents
rag_chain_with_sources = RunnableParallel(
    answer=(
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    ),
    sources=retriever   # also return raw source docs
)

result = rag_chain_with_sources.invoke(
    "What are the types of memory in LangChain?"
)

print("ANSWER:")
print(result["answer"])

print("\nSOURCE CHUNKS USED:")
for i, doc in enumerate(result["sources"]):
    print(f"\n  Source {i+1}:")
    print(f"  File: {doc.metadata.get('source', 'N/A')}")
    print(f"  Content: {doc.page_content[:150]}...")

# Cell 8: Stream RAG answers token by token
print("Q: What is RAG and why is it better than fine-tuning?\n")
print("A: ", end="", flush=True)

for chunk in rag_chain.stream(
    "What is RAG and why is it better than fine-tuning?"
):
    print(chunk, end="", flush=True)

print("\n")

# Cell 9: RAG that remembers conversation history
from langchain_core.messages import HumanMessage, AIMessage

CONVERSATIONAL_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant answering questions
about LangChain based ONLY on the provided context.

Context from documents:
{context}

If the answer isn't in the context, say so clearly.
Use conversation history to understand follow-up questions."""),
    ("placeholder", "{chat_history}"),
    ("human", "{question}")
])

class ConversationalRAG:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.chat_history = []

        self.chain = (
            {
                "context": (
                    lambda x: x["question"]
                ) | retriever | format_docs,
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"]
            }
            | CONVERSATIONAL_RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def chat(self, question: str) -> str:
        response = self.chain.invoke({
            "question": question,
            "chat_history": self.chat_history
        })

        # Update history
        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=response)
        ])

        return response

    def clear(self):
        self.chat_history = []
        print("🗑️ History cleared")


# Test conversational RAG
conv_rag = ConversationalRAG(retriever, llm)

# Multi-turn conversation
turns = [
    "What are the types of models in LangChain?",
    "Which of those is best for chat applications?",  # "those" refers to previous answer
    "What about embeddings - how are they different?",
    "Can you summarize what we just discussed?"
]

for question in turns:
    print(f"\n👤 {question}")
    answer = conv_rag.chat(question)
    print(f"🤖 {answer}")

# Cell 10: RAG over multiple documents
additional_content = """
# LangGraph Guide

## What is LangGraph?
LangGraph is a library for building stateful multi-agent applications.
It was built on top of LangChain and released in 2024.
LangGraph uses a graph-based approach with nodes and edges.

## Key Concepts

### StateGraph
StateGraph is the main class in LangGraph.
It defines the flow between different agents or steps.
Each node in the graph is a function that processes state.

### Nodes
Nodes are Python functions that take state and return updates.
Each node performs one specific task in the pipeline.

### Edges
Edges connect nodes and define the flow of execution.
Conditional edges allow dynamic routing based on state.

### Human in the Loop
LangGraph supports pausing execution for human approval.
This is critical for production agent deployments.

## When to Use LangGraph
Use LangGraph when you need:
- Multiple agents collaborating
- Complex branching logic
- Human approval steps
- Long-running workflows
- Retry and error handling logic
"""

with open("rag_docs/langgraph_guide.txt", "w") as f:
    f.write(additional_content)

# Load second document and ADD to existing vectorstore
loader2 = TextLoader("rag_docs/langgraph_guide.txt", encoding="utf-8")
docs2 = loader2.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks2 = splitter.split_documents(docs2)

# Add to existing Chroma collection
vectorstore.add_documents(chunks2)
print(f"✅ Added {len(chunks2)} chunks from langgraph_guide.txt")
print(f"   Total vectors: {vectorstore._collection.count()}")

# Cell 11: Query across both documents
multi_doc_questions = [
    "What is the difference between LangChain and LangGraph?",
    "When should I use LangGraph instead of regular chains?",
    "What are nodes and edges in LangGraph?",
]

for q in multi_doc_questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    result = rag_chain_with_sources.invoke(q)
    print(f"A: {result['answer']}")

    sources = set(
        doc.metadata.get('source', 'N/A')
        for doc in result['sources']
    )
    print(f"Sources: {sources}")

# Cell 12: Diagnose RAG quality issues

def diagnose_rag(vectorstore, query, k=5):
    """
    Debug tool - shows exactly what the retriever finds
    before the LLM sees it. Use this when RAG gives bad answers.
    """
    print(f"🔍 Diagnosing query: '{query}'")
    print("="*60)

    results = vectorstore.similarity_search_with_score(query, k=k)

    print(f"Top {k} retrieved chunks:\n")
    for i, (doc, score) in enumerate(results):
        print(f"Chunk {i+1} | Score: {score:.4f}")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"Content: {doc.page_content[:200]}")
        print("-"*40)

    print("\n💡 If top chunks don't contain the answer:")
    print("   → Reduce chunk_size (try 300)")
    print("   → Increase k (retrieve more chunks)")
    print("   → Rephrase query to match document language")
    print("   → Check document was actually indexed")


# Test the diagnostic tool
diagnose_rag(vectorstore, "What is conditional edges in LangGraph?")

# PATTERN                    WHEN TO USE
# ──────────────────────────────────────────────────────
# Basic RAG                  Single document, simple Q&A
# RAG with Sources           User needs to verify answers
# Conversational RAG         Multi-turn chat over docs
# Multi-doc RAG              Multiple knowledge sources
# Filtered RAG               Large collections by category
# 
# COMMON FIXES:
#   Bad answers?   → Check retrieved chunks with diagnose_rag()
#   Missing info?  → Increase k or decrease chunk_size
#   Too slow?      → Switch to FAISS, reduce k
#   Hallucinating? → Lower temperature, stricter system prompt
#   Cuts off mid?  → Increase chunk_overlap
