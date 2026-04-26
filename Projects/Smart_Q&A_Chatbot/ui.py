# ui.py
# Gradio UI for Smart Q&A Chatbot

import gradio as gr
from chatbot import SmartChatbot
from personas import PERSONAS, list_personas
from dotenv import load_dotenv

load_dotenv()

# ── Global chatbot instance ───────────────────────────────────
bot = SmartChatbot(persona="default", use_local=False, stream=False)

# ── Core chat function ────────────────────────────────────────
def chat(user_message, history):
    """Called every time user sends a message"""
    if not user_message.strip():
        return "", history
    
    response = bot.chat(user_message)
    # Gradio 6.0 messages format: list of dicts
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})
    return "", history


def switch_persona(persona_key):
    """Switch chatbot persona"""
    bot.switch_persona(persona_key)
    bot.memory.clear()
    persona_name = PERSONAS[persona_key]["name"]
    return [], f"✅ Switched to **{persona_name}** — memory cleared"


def switch_backend(backend_choice):
    """Switch between Groq and Ollama"""
    if backend_choice == "Groq (cloud)":
        from langchain_groq import ChatGroq
        bot.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7
        )
        bot.backend = "Groq (cloud)"
    else:
        from langchain_ollama import ChatOllama
        bot.llm = ChatOllama(model="llama3.2", temperature=0.7)
        bot.backend = "Ollama (local)"

    return f"✅ Switched to **{backend_choice}**"


def clear_memory():
    """Clear conversation memory"""
    bot.memory.clear()
    return [], "🗑️ Memory cleared"


def get_stats():
    """Return chatbot stats as string"""
    return f"""
📊 **Chatbot Stats**
- **Backend:** {bot.backend}
- **Persona:** {bot.persona_data['name']}
- **Turns:** {bot.memory.get_turn_count()}
- **Messages in memory:** {len(bot.memory.get_messages())}
"""


# ── Build Gradio UI ───────────────────────────────────────────
persona_choices = list(PERSONAS.keys())

with gr.Blocks(title="Smart Q&A Chatbot") as demo:

    # ── Header ────────────────────────────────────────────────
    gr.Markdown("""
    # 🤖 Smart Q&A Chatbot
    **Built with LangChain + Groq + Gradio** | Week 1 Project
    """)

    with gr.Row():

        # ── Left: Chat Panel ──────────────────────────────────
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(
                label="Conversation",
                height=500,
                show_label=True
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    scale=5,
                    container=False
                )
                send_btn = gr.Button(
                    "Send 📤",
                    variant="primary",
                    scale=1
                )

            clear_btn = gr.Button("🗑️ Clear Memory", variant="secondary")

        # ── Right: Controls Panel ─────────────────────────────
        with gr.Column(scale=1):

            gr.Markdown("### ⚙️ Settings")

            # Persona selector
            persona_dropdown = gr.Dropdown(
                choices=persona_choices,
                value="default",
                label="🎭 Persona",
                info="Switch chatbot personality"
            )
            persona_btn = gr.Button("Apply Persona", variant="secondary")

            gr.Markdown("---")

            # Backend selector
            backend_radio = gr.Radio(
                choices=["Groq (cloud)", "Ollama (local)"],
                value="Groq (cloud)",
                label="🔧 Backend"
            )
            backend_btn = gr.Button("Switch Backend", variant="secondary")

            gr.Markdown("---")

            # Stats
            stats_btn = gr.Button("📊 Show Stats", variant="secondary")

            # Status display
            status_box = gr.Markdown(
                value="Ready to chat! 💬",
                label="Status"
            )

    # ── Persona descriptions ──────────────────────────────────
    with gr.Accordion("📋 Persona Descriptions", open=False):
        persona_info = ""
        for key, val in PERSONAS.items():
            persona_info += f"**[{key}] {val['name']}**\n"
            # first line of system prompt as description
            first_line = val['system_prompt'].split('\n')[0]
            persona_info += f"{first_line}\n\n"
        gr.Markdown(persona_info)

    # ── Event handlers ────────────────────────────────────────

    # Send on button click
    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot_ui],
        outputs=[msg_input, chatbot_ui]
    )

    # Send on Enter key
    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot_ui],
        outputs=[msg_input, chatbot_ui]
    )

    # Persona switch
    persona_btn.click(
        fn=switch_persona,
        inputs=[persona_dropdown],
        outputs=[chatbot_ui, status_box]
    )

    # Backend switch
    backend_btn.click(
        fn=switch_backend,
        inputs=[backend_radio],
        outputs=[status_box]
    )

    # Clear memory
    clear_btn.click(
        fn=clear_memory,
        outputs=[chatbot_ui, status_box]
    )

    # Stats
    stats_btn.click(
        fn=get_stats,
        outputs=[status_box]
    )


# ── Launch ────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css="""
            .chatbot-container { height: 500px; }
            .status-box { font-size: 0.9em; }
        """
    )