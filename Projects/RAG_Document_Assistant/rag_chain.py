# rag_chain.py
# The brain - RAG chain with tools and memory

from typing import List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document

from ddgs import DDGS

load_dotenv()


# ── Tools ─────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """
    Search the web for current information not found in documents.
    Use when the document doesn't contain the answer,
    or when user asks about recent events.
    Input: a clear search query string.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No web results found."
        return "\n\n".join(
            f"Title: {r['title']}\nSummary: {r['body'][:250]}"
            for r in results
        )
    except Exception as e:
        return f"Search failed: {e}"


# ── Prompts ───────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are an intelligent document assistant.

Your job:
1. Answer questions using the provided document context
2. Always cite which source supports your answer  
3. If the document doesn't have the answer, say so clearly
4. Be concise, accurate, and helpful

Document Context:
{context}

Instructions:
- Base answers primarily on the document context above
- If context is insufficient, acknowledge the limitation
- Format citations as [Source: filename, page X]
- For follow-up questions, use conversation history"""


# ── RAG Chain Class ───────────────────────────────────────────

class RAGChain:
    """
    Complete RAG chain with:
    - Document context injection
    - Conversation memory
    - Web search fallback
    - Source citation
    """

    def __init__(self, retriever, use_local: bool = False):
        self.retriever = retriever
        self.chat_history: List = []

        # Choose LLM
        if use_local:
            self.llm = ChatOllama(model="llama3.2", temperature=0)
            self.backend = "Ollama (local)"
        else:
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0
            )
            self.backend = "Groq (cloud)"

        self.parser = StrOutputParser()
        print(f"✅ RAG Chain ready | Backend: {self.backend}")

    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved docs with source info"""
        if not docs:
            return "No relevant documents found."

        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f", page {page + 1}" if page != "" else ""
            formatted.append(
                f"[Source {i+1}: {source}{page_info}]\n"
                f"{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)

    def _retrieve(self, query: str) -> tuple:
        """Retrieve relevant chunks, return (formatted_str, raw_docs)"""
        docs = self.retriever.invoke(query)
        return self._format_docs(docs), docs

    def chat(self, question: str) -> dict:
        """
        Process one question. Returns:
        {answer, sources, used_web, backend}
        """
        # Step 1: Retrieve relevant context
        context_str, source_docs = self._retrieve(question)

        # Step 2: Build message list
        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT.format(
                context=context_str
            ))
        ]

        # Add conversation history (last 6 turns)
        messages.extend(self.chat_history[-6:])

        # Add current question
        messages.append(HumanMessage(content=question))

        # Step 3: Get LLM answer
        response = self.llm.invoke(messages)
        answer = self.parser.invoke(response)

        # Step 4: Web search fallback
        used_web = False
        fallback_phrases = [
            "don't have", "not in", "no information",
            "cannot find", "not mentioned", "outside"
        ]
        if any(phrase in answer.lower() for phrase in fallback_phrases):
            print("📡 Falling back to web search...")
            web_result = web_search.invoke(question)

            web_messages = messages + [
                AIMessage(content=answer),
                HumanMessage(content=f"""Web search results:
{web_result}

Using both the document context and these web results,
please provide a complete answer.""")
            ]
            response = self.llm.invoke(web_messages)
            answer = self.parser.invoke(response)
            used_web = True

        # Step 5: Save to memory
        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])

        # Step 6: Extract unique sources
        sources = list(set(
            doc.metadata.get("source", "Unknown")
            for doc in source_docs
        ))

        return {
            "answer": answer,
            "sources": sources,
            "source_docs": source_docs,
            "used_web": used_web,
            "backend": self.backend
        }

    def clear_memory(self):
        self.chat_history = []
        print("🗑️  Conversation memory cleared")

    def get_turn_count(self) -> int:
        return len(self.chat_history) // 2