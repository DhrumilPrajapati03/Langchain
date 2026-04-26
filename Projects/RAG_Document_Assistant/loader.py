# loader.py
# Handles all document loading and chunking

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Splitter config ───────────────────────────────────────────
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def load_pdf(path: str) -> List[Document]:
    loader = PyPDFLoader(path)
    return loader.load()


def load_txt(path: str) -> List[Document]:
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()


def load_csv(path: str) -> List[Document]:
    loader = CSVLoader(path, encoding="utf-8")
    return loader.load()


def load_url(url: str) -> List[Document]:
    loader = WebBaseLoader(web_paths=[url])
    return loader.load()


def load_document(source: str) -> List[Document]:
    """
    Auto-detect source type and load.
    Accepts: file path (pdf/txt/csv) or URL
    """
    if source.startswith("http://") or source.startswith("https://"):
        print(f"🌐 Loading URL: {source}")
        docs = load_url(source)

    elif not os.path.exists(source):
        raise FileNotFoundError(f"File not found: {source}")

    else:
        ext = Path(source).suffix.lower()
        print(f"📄 Loading {ext.upper()}: {source}")

        if ext == ".pdf":
            docs = load_pdf(source)
        elif ext == ".txt":
            docs = load_txt(source)
        elif ext == ".csv":
            docs = load_csv(source)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    print(f"   Loaded {len(docs)} page(s)")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks"""
    chunks = SPLITTER.split_documents(docs)
    print(f"   Split into {len(chunks)} chunks")
    return chunks


def load_and_split(source: str) -> List[Document]:
    """Full pipeline: load → split"""
    docs = load_document(source)
    chunks = split_documents(docs)
    return chunks


def load_multiple(sources: List[str]) -> List[Document]:
    """Load and split multiple sources"""
    all_chunks = []
    for source in sources:
        try:
            chunks = load_and_split(source)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"   ⚠️  Failed to load {source}: {e}")
    print(f"\n✅ Total chunks ready: {len(all_chunks)}")
    return all_chunks