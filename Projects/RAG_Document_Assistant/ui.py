# ui.py
# Beautiful Gradio interface for the RAG assistant

import os
import gradio as gr
from dotenv import load_dotenv

from rag_chain import RAGChain
from loader import load_and_split
from retriever import DocumentRetriever

load_dotenv()

# ── Global state ──────────────────────────────────────────────
doc_retriever = DocumentRetriever(reset=False)
rag = RAGChain(
    retriever=doc_retriever.get_retriever(k=4),
    use_local=False
)


# ── Core functions ────────────────────────────────────────────

def upload_and_index(files):
    """Handle file uploads from Gradio"""
    if not files:
        return "⚠️ No files selected"

    results = []
    for file in files:
        try:
            chunks = load_and_split(file.name)
            doc_retriever.add_documents(chunks)
            rag.retriever = doc_retriever.get_retriever(k=4)
            fname = os.path.basename(file.name)
            results.append(f"✅ {fname} → {len(chunks)} chunks")
        except Exception as e:
            results.append(f"❌ {os.path.basename(file.name)}: {e}")

    return "\n".join(results)


def load_url(url: str):
    """Load a webpage into the knowledge base"""
    if not url.strip():
        return "⚠️ Enter a URL first"
    try:
        chunks = load_and_split(url.strip())
        doc_retriever.add_documents(chunks)
        rag.retriever = doc_retriever.get_retriever(k=4)
        return f"✅ URL loaded → {len(chunks)} chunks indexed"
    except Exception as e:
        return f"❌ Failed: {e}"


def chat(message, history):
    """Main chat function"""
    if not message.strip():
        return "", history

    if doc_retriever.is_empty():
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ No documents loaded yet. Please upload a PDF or enter a URL first."})
        return "", history

    result = rag.chat(message)

    # Format response with metadata
    answer = result["answer"]
    sources = result["sources"]
    used_web = result["used_web"]

    # Build response with source info
    response = answer
    if sources:
        source_names = [os.path.basename(s) for s in sources]
        response += f"\n\n📎 *Sources: {', '.join(source_names)}*"
    if used_web:
        response += "\n🌐 *Web search was also used*"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history


def get_sources():
    """Get list of indexed sources"""
    sources = doc_retriever.get_sources()
    count = doc_retriever.get_count()
    if not sources:
        return "No documents indexed yet"
    source_list = "\n".join(
        f"• {os.path.basename(s)}" for s in sources
    )
    return f"**{count} chunks** from:\n{source_list}"


def clear_memory():
    rag.clear_memory()
    return [], "✅ Conversation memory cleared"


def reset_all():
    global doc_retriever, rag
    doc_retriever = DocumentRetriever(reset=True)
    rag = RAGChain(
        retriever=doc_retriever.get_retriever(k=4),
        use_local=False
    )
    return [], "✅ Everything reset — vector store and memory cleared"


def switch_backend(choice):
    if choice == "Groq (cloud)":
        from langchain_groq import ChatGroq
        rag.llm = ChatGroq(
            model="llama-3.3-70b-versatile", temperature=0
        )
        rag.backend = "Groq (cloud)"
    else:
        from langchain_ollama import ChatOllama
        rag.llm = ChatOllama(model="llama3.2", temperature=0)
        rag.backend = "Ollama (local)"
    return f"✅ Switched to {choice}"


# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(
    title="RAG Document Assistant"
) as demo:

    gr.Markdown("""
    # 📚 RAG Document Assistant
    **Upload documents → Ask questions → Get cited answers**
    Built with LangChain + ChromaDB + Groq
    """)

    with gr.Row():

        # ── Left: Document Panel ──────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Knowledge Base")

            # File upload
            file_upload = gr.File(
                label="Upload PDF / TXT / CSV",
                file_types=[".pdf", ".txt", ".csv"],
                file_count="multiple"
            )
            upload_btn = gr.Button("📤 Index Documents", variant="primary")
            upload_status = gr.Markdown("No files uploaded yet")

            gr.Markdown("---")

            # URL loader
            url_input = gr.Textbox(
                label="Or load a URL",
                placeholder="https://example.com/article"
            )
            url_btn = gr.Button("🌐 Load URL", variant="secondary")
            url_status = gr.Markdown("")

            gr.Markdown("---")

            # Sources display
            sources_btn = gr.Button("📂 Show Indexed Sources")
            sources_display = gr.Markdown("")

            gr.Markdown("---")

            # Backend
            backend_radio = gr.Radio(
                choices=["Groq (cloud)", "Ollama (local)"],
                value="Groq (cloud)",
                label="🔧 LLM Backend"
            )
            backend_btn = gr.Button("Switch Backend")
            backend_status = gr.Markdown("")

        # ── Right: Chat Panel ─────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")

            chatbot_ui = gr.Chatbot(
                label="Conversation",
                height=480,
                layout="bubble",
                buttons=["copy"]
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask anything about your documents...",
                    show_label=False,
                    scale=5,
                    container=False
                )
                send_btn = gr.Button("Send 📤", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Memory", scale=1)
                reset_btn = gr.Button("🔄 Reset All", scale=1, variant="stop")

            # Example questions
            gr.Examples(
                examples=[
                    "Summarize the main topics in the document",
                    "What are the key points mentioned?",
                    "What does the document say about [topic]?",
                    "List the most important facts"
                ],
                inputs=msg_input,
                label="Example Questions"
            )

    # ── Event handlers ────────────────────────────────────────

    upload_btn.click(
        fn=upload_and_index,
        inputs=[file_upload],
        outputs=[upload_status]
    )

    url_btn.click(
        fn=load_url,
        inputs=[url_input],
        outputs=[url_status]
    )

    sources_btn.click(
        fn=get_sources,
        outputs=[sources_display]
    )

    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot_ui],
        outputs=[msg_input, chatbot_ui]
    )

    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot_ui],
        outputs=[msg_input, chatbot_ui]
    )

    clear_btn.click(
        fn=clear_memory,
        outputs=[chatbot_ui, sources_display]
    )

    reset_btn.click(
        fn=reset_all,
        outputs=[chatbot_ui, sources_display]
    )

    backend_btn.click(
        fn=switch_backend,
        inputs=[backend_radio],
        outputs=[backend_status]
    )


# ── Launch ────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        inbrowser=True,
        share=False
    )