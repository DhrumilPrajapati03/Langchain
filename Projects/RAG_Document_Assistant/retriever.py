# retriever.py
# VectorStore management and retrieval

import os
import shutil
from typing import List, Optional

from langchain_core.documents import Document
# from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ── Embeddings ────────────────────────────────────────────────
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
DB_PATH = "./vector_db"
COLLECTION = "rag_documents"


class DocumentRetriever:
    """
    Manages the vector store lifecycle:
    - Add documents
    - Search semantically
    - Persist to disk
    - Clear and rebuild
    """

    def __init__(self, reset: bool = False):
        if reset and os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
            print("🗑️  Vector store cleared")

        self.vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=EMBEDDINGS,
            collection_name=COLLECTION
        )
        count = self.vectorstore._collection.count()
        print(f"✅ Vector store ready | {count} chunks stored")

    def add_documents(self, chunks: List[Document]) -> int:
        """Add chunks to vectorstore"""
        self.vectorstore.add_documents(chunks)
        count = self.vectorstore._collection.count()
        print(f"✅ Added chunks | Total: {count}")
        return count

    def get_retriever(self, k: int = 4, use_mmr: bool = True):
        """
        Return a LangChain retriever.
        MMR balances relevance + diversity.
        """
        if use_mmr:
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": k * 3,
                    "lambda_mult": 0.7
                }
            )
        return self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

    def search(self, query: str, k: int = 4) -> List[Document]:
        """Direct semantic search with scores"""
        results = self.vectorstore.similarity_search_with_score(
            query, k=k
        )
        return results

    def is_empty(self) -> bool:
        return self.vectorstore._collection.count() == 0

    def get_count(self) -> int:
        return self.vectorstore._collection.count()

    def get_sources(self) -> List[str]:
        """Return list of unique source files indexed"""
        results = self.vectorstore.get()
        sources = set()
        for meta in results.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return list(sources)