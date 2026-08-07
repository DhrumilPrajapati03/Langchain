# -*- coding: utf-8 -*-
"""
Auto-generated Python script from: day7_test.ipynb
"""

# ### Day 7: Embeddings & Vector Stores

# Today is where RAG becomes real. You'll convert document chunks into vectors, store them in a database, and run semantic search — finding relevant content by meaning, not just keywords.

# #### 1. The Big Concept — What Are Embeddings?

# "I love programming"  →  [0.23, -0.81, 0.44, 0.12, ...]  (768 numbers)
# "I enjoy coding"      →  [0.21, -0.79, 0.41, 0.14, ...]  (very similar!)
# "I hate vegetables"   →  [-0.54, 0.33, -0.21, 0.67, ...]  (very different)

# Embeddings = meaning converted to numbers. Similar meanings → similar vectors. This is how semantic search works — find vectors closest to your query vector.

# Traditional search:  "coding tips" finds docs with word "coding"
# Semantic search:     "coding tips" also finds docs with "programming advice"
#                      because they mean the same thing

# Cell 3: Option B - HuggingFace embeddings (also free)
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings_hf = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",   # small, fast, good quality
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

test_text = "LangChain is a framework for building LLM applications"
vector_hf = embeddings_hf.embed_query(test_text)
print(f"Model: all-MiniLM-L6-v2 (HuggingFace)")
print(f"Vector dimensions: {len(vector_hf)}")
print(f"First 5 values: {vector_hf[:5]}")

# Cell 4: Embedding comparison
# Which one to use?
import time

text = "What is machine learning?"

# # Time Ollama
# start = time.time()
# v1 = embeddings_ollama.embed_query(text)
# ollama_time = time.time() - start

# Time HuggingFace
start = time.time()
v2 = embeddings_hf.embed_query(text)
hf_time = time.time() - start

print("Embedding Model Comparison:")
# print(f"  nomic-embed-text (Ollama): {len(v1)} dims | {ollama_time:.3f}s")
print(f"  all-MiniLM-L6-v2 (HF):    {len(v2)} dims | {hf_time:.3f}s")
print(f"\n💡 Recommendation: Use nomic-embed-text for this course")
print(f"   Higher dimensions = better semantic understanding")

# Use Ollama embeddings for rest of notebook
embeddings = embeddings_hf

# Cell 5: See how similarity works
import numpy as np

def cosine_similarity(v1, v2):
    """Measure similarity between two vectors (0=different, 1=identical)"""
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Test sentences
sentences = [
    "I love programming in Python",       # base sentence
    "Python coding is my passion",        # similar meaning
    "I enjoy writing software code",      # somewhat similar
    "Machine learning is fascinating",    # different topic, tech related
    "The weather is nice today",          # completely different
    "I hate programming",                 # opposite meaning
]

base = embeddings.embed_query(sentences[0])

print(f"Base: '{sentences[0]}'")
print(f"\nSimilarity scores:")
print("-" * 60)

for sentence in sentences[1:]:
    vec = embeddings.embed_query(sentence)
    sim = cosine_similarity(base, vec)
    bar = "█" * int(sim * 30)
    print(f"  {sim:.3f} {bar}")
    print(f"         '{sentence}'")

# #### ChromaDB — Persistent Vector Store

# Cell 6: ChromaDB setup
import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load and split a document (from Day 6)
loader = TextLoader("sample_docs/sample.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"✅ Prepared {len(chunks)} chunks")

# Cell 7: Create ChromaDB vector store
import shutil
import os

# Clean previous run if exists
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

# Create vector store - this embeds all chunks automatically
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",    # saves to disk!
    collection_name="langchain_docs"
)

print(f"✅ ChromaDB created!")
print(f"   Chunks stored: {vectorstore._collection.count()}")
print(f"   Location: ./chroma_db") 

# Cell 8: Query the vector store
query = "What are LangChain chains?"

results = vectorstore.similarity_search(
    query=query,
    k=3             # return top 3 most relevant chunks
)

print(f"Query: '{query}'")
print(f"Top {len(results)} results:\n")

for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(f"Content: {doc.page_content}")
    print(f"Source:  {doc.metadata.get('source', 'N/A')}")
    print()

# Cell 9: Similarity search WITH scores
results_with_scores = vectorstore.similarity_search_with_score(
    query=query,
    k=3
)

print(f"Query: '{query}'\n")
for doc, score in results_with_scores:
    print(f"Score: {score:.4f} (lower = more similar in Chroma)")
    print(f"Content: {doc.page_content[:150]}...")
    print()

# Cell 10: Load existing ChromaDB (persistence test)
# Reload from disk - embeddings are saved, no re-computation!
vectorstore_loaded = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="langchain_docs"
)

print(f"✅ Reloaded from disk!")
print(f"   Chunks: {vectorstore_loaded._collection.count()}")

# Query it - works exactly the same
results = vectorstore_loaded.similarity_search("What is memory in LangChain?", k=2)
for r in results:
    print(f"\n→ {r.page_content[:200]}")

# #### FAISS — In-Memory Vector Store

# Cell 11: FAISS - faster for search, no persistence by default
from langchain_community.vectorstores import FAISS

# Create FAISS store
faiss_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("✅ FAISS store created (in memory)")

# Search
results = faiss_store.similarity_search("How do agents work?", k=3)
print(f"\nFAISS results for 'How do agents work?':")
for i, r in enumerate(results):
    print(f"\n{i+1}. {r.page_content[:200]}")

# Cell 12: Save and load FAISS
faiss_store.save_local("./faiss_db")
print("✅ FAISS saved to disk")

# Load it back
faiss_loaded = FAISS.load_local(
    "./faiss_db",
    embeddings,
    allow_dangerous_deserialization=True  # required flag
)
print("✅ FAISS loaded from disk")
print(f"   Works: {len(faiss_loaded.similarity_search('test', k=1))} result")

# Cell 13: Head to head comparison
import time

queries = [
    "What is LangChain?",
    "How does memory work?",
    "What are chains used for?"
]

print("ChromaDB vs FAISS comparison")
print("="*50)

for query in queries:
    # ChromaDB
    start = time.time()
    chroma_results = vectorstore.similarity_search(query, k=2)
    chroma_time = time.time() - start

    # FAISS
    start = time.time()
    faiss_results = faiss_store.similarity_search(query, k=2)
    faiss_time = time.time() - start

    print(f"\nQuery: '{query}'")
    print(f"  ChromaDB: {chroma_time*1000:.1f}ms")
    print(f"  FAISS:    {faiss_time*1000:.1f}ms")
    print(f"  Same top result? {chroma_results[0].page_content[:50] == faiss_results[0].page_content[:50]}")

# When to use which:
# ┌─────────────────────────────────────────────────────┐
# │  ChromaDB  → persistent, filterable, production     │
# │              great for growing document collections │
# │                                                     │
# │  FAISS     → blazing fast search, in-memory         │
# │              great for fixed datasets, prototyping  │
# └─────────────────────────────────────────────────────┘

# Cell 14: Add metadata to chunks for filtered search
from langchain_core.documents import Document
import shutil

# Create documents with rich metadata
tech_docs = [
    Document(
        page_content="Python is great for data science and machine learning.",
        metadata={"source": "python_guide", "topic": "python", "level": "beginner"}
    ),
    Document(
        page_content="LangChain chains use the pipe operator for composition.",
        metadata={"source": "langchain_guide", "topic": "langchain", "level": "intermediate"}
    ),
    Document(
        page_content="FAISS provides fast approximate nearest neighbor search.",
        metadata={"source": "faiss_guide", "topic": "vectordb", "level": "advanced"}
    ),
    Document(
        page_content="Python decorators are functions that modify other functions.",
        metadata={"source": "python_guide", "topic": "python", "level": "intermediate"}
    ),
    Document(
        page_content="ChromaDB persists vector embeddings to disk automatically.",
        metadata={"source": "chroma_guide", "topic": "vectordb", "level": "beginner"}
    ),
]

# Clean and recreate
if os.path.exists("./chroma_filtered"):
    shutil.rmtree("./chroma_filtered")

filtered_store = Chroma.from_documents(
    documents=tech_docs,
    embedding=embeddings,
    persist_directory="./chroma_filtered"
)

# Search ALL docs
print("=== No filter ===")
results = filtered_store.similarity_search("Python usage", k=3)
for r in results:
    print(f"  [{r.metadata['topic']}] {r.page_content[:60]}")

# Filter by topic
print("\n=== Filter: topic=python ===")
results = filtered_store.similarity_search(
    "Python usage",
    k=3,
    filter={"topic": "python"}   # only search python docs
)
for r in results:
    print(f"  [{r.metadata['level']}] {r.page_content[:60]}")

# Filter by level
print("\n=== Filter: level=beginner ===")
results = filtered_store.similarity_search(
    "database",
    k=3,
    filter={"level": "beginner"}
)
for r in results:
    print(f"  [{r.metadata['topic']}] {r.page_content[:60]}")

# Cell 15: Convert vectorstore to retriever
# Retrievers are what LangChain chains actually use

retriever = vectorstore.as_retriever(
    search_type="similarity",   # or "mmr" for diversity
    search_kwargs={"k": 3}
)

# Use it
docs = retriever.invoke("What is RAG?")
print(f"Retrieved {len(docs)} docs for 'What is RAG?'")
for d in docs:
    print(f"\n→ {d.page_content[:200]}")

# Cell 16: MMR retriever - Maximum Marginal Relevance
# Balances relevance WITH diversity - avoids returning duplicate chunks

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,           # return 3 results
        "fetch_k": 10,    # consider top 10 candidates
        "lambda_mult": 0.7  # 1.0=pure relevance, 0.0=pure diversity
    }
)

query = "LangChain components"
sim_docs = retriever.invoke(query)
mmr_docs = mmr_retriever.invoke(query)

print("Similarity retriever results:")
for d in sim_docs:
    print(f"  → {d.page_content[:80]}...")

print("\nMMR retriever results (more diverse):")
for d in mmr_docs:
    print(f"  → {d.page_content[:80]}...")

# Cell 17: Complete Day 6 + Day 7 pipeline
from torch._functorch._aot_autograd.logging_utils import model_name
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import shutil, os

def build_vectorstore(file_path: str, collection_name: str):
    """
    Complete pipeline:
    Document → chunks → embeddings → vector store → retriever
    """
    print(f"📄 Loading: {file_path}")

    # Load
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()
    print(f"   Loaded {len(docs)} page(s)")

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"   Split into {len(chunks)} chunks")

    # Embed + Store
    db_path = f"./{collection_name}_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name=collection_name
    )
    print(f"   Stored {store._collection.count()} vectors")

    # Return retriever
    retriever = store.as_retriever(search_kwargs={"k": 3})
    print(f"✅ Retriever ready!\n")
    return retriever


# Build and test
retriever = build_vectorstore("sample_docs/sample.txt", "langchain_notes")

# Query it
test_queries = [
    "What is RAG?",
    "How do agents make decisions?",
    "What is the pipe operator used for?"
]

for query in test_queries:
    print(f"Q: {query}")
    results = retriever.invoke(query)
    print(f"A (from docs): {results[0].page_content[:200]}\n")

# TEXT           EMBEDDING MODEL          VECTOR
# "hello"   →   nomic-embed-text    →   [0.2, -0.5, ...]
#                     ↓
#               stored in VectorDB
#                     ↓
#          ChromaDB (persistent)
#          FAISS (fast, in-memory)
#                     ↓
#               as_retriever()
#                     ↓
#          retriever.invoke(query)
#                     ↓
#          returns top-k similar chunks
#                     ↓
#               feeds into LLM         ← Day 8 (RAG)
