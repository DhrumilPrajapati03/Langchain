# -*- coding: utf-8 -*-
"""
Auto-generated Python script from: day6_test.ipynb
"""

# ### Day 6: Document Loaders & Text Splitters

# Today you learn how to feed any document into LangChain — PDFs, websites, text files, CSVs. This is the foundation of RAG (Project 2). By end of today your code can ingest any document and prepare it for AI processing.

# #### 1. The Big Picture — Why Document Loaders?

# LLMs have a context window limit — they can't read a 200-page PDF in one shot. The solution:

# Raw Document (PDF/Web/CSV)
#         ↓
#    Document Loader        ← Today Part 1
#         ↓
#   Raw text chunks
#         ↓
#    Text Splitter          ← Today Part 2
#         ↓
#   Small, overlapping chunks
#         ↓
#    Embeddings + VectorStore  ← Day 7
#         ↓
#    RAG Pipeline           ← Day 8

# Cell 1: Create sample files to work with
import os

os.makedirs("sample_docs", exist_ok=True)

# Create a sample text file
txt_content = """
LangChain Framework Overview
=============================

LangChain is an open-source framework designed to simplify 
the creation of applications using large language models (LLMs).

Core Components:
LangChain has several key components including chains, agents, 
memory, and document loaders. Each component serves a specific 
purpose in building AI applications.

Chains:
Chains allow you to combine multiple components together. 
The LangChain Expression Language (LCEL) uses the pipe operator 
to create clean, readable chains.

Agents:
Agents use LLMs to decide which actions to take. They can use 
tools like web search, calculators, and custom functions.

Memory:
Memory allows chatbots to remember previous conversations. 
This is crucial for building coherent multi-turn applications.

RAG (Retrieval Augmented Generation):
RAG combines document retrieval with LLM generation to answer 
questions about specific documents without fine-tuning.
"""

with open("sample_docs/sample.txt", "w") as f:
    f.write(txt_content)

# Create a sample CSV
csv_content = """name,role,expertise,years_experience
Alice Johnson,ML Engineer,PyTorch and model training,5
Bob Smith,LLM Developer,LangChain and RAG systems,3
Carol White,AI Researcher,Transformers and fine-tuning,7
David Brown,MLOps Engineer,Model deployment and monitoring,4
Eve Davis,Prompt Engineer,Prompt optimization and evaluation,2
"""

with open("sample_docs/sample.csv", "w") as f:
    f.write(csv_content)

print("✅ Sample files created!")

# Cell 2: TextLoader - simplest loader


from langchain_community.document_loaders import TextLoader

loader = TextLoader("sample_docs/sample.txt", encoding="utf-8")
documents = loader.load()

print(f"Documents loaded: {len(documents)}")
print(f"Type: {type(documents[0])}")
print(f"\n--- Content Preview ---")
print(documents[0].page_content[:300])
print(f"\n--- Metadata ---")
print(documents[0].metadata)

# Cell 3: Understanding the Document object
doc = documents[0]

print("Document has two parts:")
print(f"1. page_content (type: {type(doc.page_content)})")
print(f"   Length: {len(doc.page_content)} characters")
print(f"\n2. metadata (type: {type(doc.metadata)})")
print(f"   Keys: {list(doc.metadata.keys())}")
print(f"   Source: {doc.metadata.get('source', 'N/A')}")

# Cell 4: PDF Loader
# Download any PDF first - let's use a programmatically created one
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Install reportlab: pip install reportlab
def create_sample_pdf():
    c = canvas.Canvas("sample_docs/sample.pdf", pagesize=letter)
     
    # Page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Introduction to RAG Systems")
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, "RAG stands for Retrieval Augmented Generation.")
    c.drawString(100, 680, "It combines document search with LLM generation.")
    c.drawString(100, 660, "This approach is better than fine-tuning for:")
    c.drawString(120, 640, "1. Frequently updated knowledge bases")
    c.drawString(120, 620, "2. Private company documents")
    c.drawString(120, 600, "3. Reducing hallucinations")
    
    c.showPage()  # Page 2
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "RAG Architecture")
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, "Step 1: Load documents using Document Loaders")
    c.drawString(100, 680, "Step 2: Split into chunks using Text Splitters")
    c.drawString(100, 660, "Step 3: Embed chunks using Embedding Models")
    c.drawString(100, 640, "Step 4: Store in Vector Database")
    c.drawString(100, 620, "Step 5: Retrieve relevant chunks at query time")
    c.drawString(100, 600, "Step 6: Generate answer using LLM + context")
    
    c.save()
    print("✅ Sample PDF created!")

create_sample_pdf()

# Cell 5: Load the PDF
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample_docs/sample.pdf")
pages = loader.load()

print(f"Pages loaded: {len(pages)}")
print(f"\n--- Page 1 ---")
print(pages[0].page_content)
print(f"Metadata: {pages[0].metadata}")

print(f"\n--- Page 2 ---")
print(pages[1].page_content)
print(f"Metadata: {pages[1].metadata}")
# Notice: metadata includes page number automatically!

# Cell 6: WebBaseLoader - load any webpage
from langchain_community.document_loaders import WebBaseLoader
import bs4

# Load LangChain docs page
loader = WebBaseLoader(
    web_paths=["https://python.langchain.com/docs/introduction/"],
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=("markdown",)   # only grab main content, skip nav/footer
        )
    }
)

docs = loader.load()
print(f"Pages loaded: {len(docs)}")
print(f"Content length: {len(docs[0].page_content)} chars")
print(f"\nPreview:")
print(docs[0].page_content[:500])

# Cell 7: Load multiple URLs at once
urls = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
]

loader = WebBaseLoader(web_paths=urls)
docs = loader.load()

print(f"Total pages loaded: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\nPage {i+1}: {doc.metadata.get('source', 'N/A')}")
    print(f"  Length: {len(doc.page_content)} chars")
    print(f"  Preview: {doc.page_content[:100]}...")

# Cell 8: CSVLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    file_path="sample_docs/sample.csv",
    encoding="utf-8"
)
docs = loader.load()

print(f"Rows loaded: {len(docs)}")
print("\nEach row becomes a Document:")
for doc in docs:
    print(f"\n  Content: {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")

# Full Document (10,000 words)
#          ↓  chunk_size=500, overlap=50
#   [chunk1: words 1-500]
#   [chunk2: words 451-950]   ← 50 word overlap preserves context
#   [chunk3: words 901-1400]
#   ...
# 

# Cell 9: RecursiveCharacterTextSplitter - the go-to splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load document
loader = TextLoader("sample_docs/sample.txt", encoding="utf-8")
documents = loader.load()

# Split it
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,        # max characters per chunk
    chunk_overlap=40,      # characters shared between chunks
    length_function=len,   # how to measure chunk size
    separators=["\n\n", "\n", ". ", " ", ""]  # split priority order
)

chunks = splitter.split_documents(documents)

print(f"Original: 1 document, {len(documents[0].page_content)} chars")
print(f"After split: {len(chunks)} chunks")
print(f"\n--- Chunk 1 ---")
print(repr(chunks[0].page_content))
print(f"\n--- Chunk 2 ---")
print(repr(chunks[1].page_content))
print(f"\n--- Overlap visible? ---")
# Find where chunk 1 ends and chunk 2 starts
end_of_1 = chunks[0].page_content[-40:]
start_of_2 = chunks[1].page_content[:40]
print(f"End of chunk 1:   '{end_of_1}'")
print(f"Start of chunk 2: '{start_of_2}'")

# Cell 10: Visualize chunk sizes
chunk_sizes = [len(c.page_content) for c in chunks]

print("Chunk size distribution:")
print(f"  Min: {min(chunk_sizes)} chars")
print(f"  Max: {max(chunk_sizes)} chars")
print(f"  Avg: {sum(chunk_sizes)/len(chunk_sizes):.0f} chars")
print(f"\nAll chunks:")
for i, (chunk, size) in enumerate(zip(chunks, chunk_sizes)):
    bar = "█" * (size // 10)
    print(f"  Chunk {i+1:2d}: {bar} ({size} chars)")

# Cell 11: Metadata is preserved and enriched through splitting
print("Metadata after splitting:")
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1}:")
    print(f"  source: {chunk.metadata.get('source')}")
    # chunk index added automatically
    print(f"  All metadata: {chunk.metadata}")

# Cell 12: Splitting strategies comparison

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,  # general purpose - use this 90% of the time
    CharacterTextSplitter,           # splits on single separator only
    TokenTextSplitter,               # splits by tokens not characters
)

sample_text = documents[0].page_content

# Strategy 1: Recursive (best for general text)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=30
)
recursive_chunks = recursive_splitter.split_text(sample_text)

# Strategy 2: Character (simpler, less smart)
char_splitter = CharacterTextSplitter(
    separator="\n\n",     # only split on double newlines
    chunk_size=300,
    chunk_overlap=30
)
char_chunks = char_splitter.split_text(sample_text)

# Strategy 3: Token-based (most accurate for LLM limits)
token_splitter = TokenTextSplitter(
    chunk_size=100,        # 100 tokens per chunk
    chunk_overlap=10
)
token_chunks = token_splitter.split_text(sample_text)

print("Splitting strategy comparison:")
print(f"  RecursiveCharacter: {len(recursive_chunks)} chunks")
print(f"  Character:          {len(char_chunks)} chunks")
print(f"  Token-based:        {len(token_chunks)} chunks")
print(f"\n💡 Recommendation: Use RecursiveCharacterTextSplitter")
print(f"   for almost all use cases — it's the most context-aware.")

# Cell 13: Complete loader + splitter pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split(file_path: str, chunk_size: int = 500, overlap: int = 50):
    """
    Universal document ingestion pipeline.
    Returns list of chunks ready for embedding.
    """
    # Step 1: Load
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    print(f"✅ Loaded: {len(documents)} document(s)")

    # Step 2: Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into: {len(chunks)} chunks")
    print(f"   Chunk size: {chunk_size} chars | Overlap: {overlap} chars")

    return chunks


# Test on all our sample files
print("=== PDF ===")
pdf_chunks = load_and_split("sample_docs/sample.pdf", chunk_size=300)

print("\n=== TXT ===")
txt_chunks = load_and_split("sample_docs/sample.txt", chunk_size=200)

print("\n=== CSV ===")
csv_chunks = load_and_split("sample_docs/sample.csv", chunk_size=200)

print(f"\nTotal chunks ready for embedding: {len(pdf_chunks) + len(txt_chunks) + len(csv_chunks)}")

# DOCUMENT TYPE          CHUNK SIZE    OVERLAP    REASON
# ────────────────────────────────────────────────────────────────
# Books / Long docs       1000-2000     200       preserve paragraphs
# Research papers          500-1000     100       section coherence
# Web pages / articles     500-800      100       natural reading blocks
# FAQs / Short answers     200-400       50       one Q&A per chunk
# Code files               300-500       50       function-level chunks
# CSVs / Tabular           100-200       20       one record per chunk
# 
# GOLDEN RULE:
#   chunk_overlap = 10-20% of chunk_size
#   Too small chunks → lose context
#   Too large chunks → retrieve irrelevant info
#   Test with YOUR data — there's no universal perfect size
