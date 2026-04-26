# app.py
# Command-line interface

import os
from loader import load_and_split, load_multiple
from retriever import DocumentRetriever
from rag_chain import RAGChain
from dotenv import load_dotenv

load_dotenv()


def print_banner():
    print("""
╔══════════════════════════════════════════════╗
║      📚 RAG Document Assistant v2.0         ║
║      Built with LangChain + ChromaDB        ║
╚══════════════════════════════════════════════╝

Commands:
  /load <path>   → load a PDF, TXT, CSV, or URL
  /sources       → show indexed documents
  /clear         → clear conversation memory
  /reset         → clear entire vector store
  /local         → switch to Ollama (offline)
  /cloud         → switch to Groq (online)
  /quit          → exit
""")


def main():
    print_banner()

    # Init retriever and chain
    doc_retriever = DocumentRetriever(reset=False)
    rag = RAGChain(
        retriever=doc_retriever.get_retriever(k=4),
        use_local=False
    )

    if doc_retriever.is_empty():
        print("💡 No documents loaded yet.")
        print("   Use /load <filepath> to add documents\n")

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue

            # ── Commands ──────────────────────────────────────
            if user_input.lower() == "/quit":
                print("\n👋 Goodbye!\n")
                break

            elif user_input.lower().startswith("/load "):
                source = user_input[6:].strip()
                try:
                    chunks = load_and_split(source)
                    doc_retriever.add_documents(chunks)
                    # Refresh retriever after adding docs
                    rag.retriever = doc_retriever.get_retriever(k=4)
                    print(f"✅ Loaded and indexed: {source}")
                except Exception as e:
                    print(f"❌ Load failed: {e}")

            elif user_input.lower() == "/sources":
                sources = doc_retriever.get_sources()
                if sources:
                    print("\n📂 Indexed sources:")
                    for s in sources:
                        print(f"   - {s}")
                    print(f"\nTotal chunks: {doc_retriever.get_count()}")
                else:
                    print("No documents indexed yet")

            elif user_input.lower() == "/clear":
                rag.clear_memory()

            elif user_input.lower() == "/reset":
                doc_retriever = DocumentRetriever(reset=True)
                rag.retriever = doc_retriever.get_retriever(k=4)
                rag.clear_memory()
                print("✅ Vector store and memory reset")

            elif user_input.lower() == "/local":
                from langchain_ollama import ChatOllama
                rag.llm = ChatOllama(model="llama3.2", temperature=0)
                rag.backend = "Ollama (local)"
                print("✅ Switched to local Ollama")

            elif user_input.lower() == "/cloud":
                from langchain_groq import ChatGroq
                rag.llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0
                )
                rag.backend = "Groq (cloud)"
                print("✅ Switched to Groq cloud")

            # ── RAG Query ─────────────────────────────────────
            else:
                if doc_retriever.is_empty():
                    print("⚠️  No documents loaded. Use /load <path> first")
                    continue

                result = rag.chat(user_input)

                print(f"\n🤖 Answer:\n{result['answer']}")
                print(f"\n📎 Sources: {', '.join(result['sources'])}")
                if result["used_web"]:
                    print("🌐 (Web search was also used)")
                print(f"🔧 Backend: {result['backend']}")

        except KeyboardInterrupt:
            print("\n\n⚠️ Use /quit to exit cleanly")
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()