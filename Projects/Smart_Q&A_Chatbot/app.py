# app.py
# CLI interface - run this to use your chatbot

from chatbot import SmartChatbot
from personas import list_personas
from dotenv import load_dotenv

load_dotenv()


def print_banner():
    print("""
╔══════════════════════════════════════════╗
║        🤖 Smart Q&A Chatbot v1.0        ║
║         Built with LangChain + Groq     ║
╚══════════════════════════════════════════╝

Commands:
  /help      → show commands
  /persona   → switch personality
  /history   → show conversation
  /stats     → show stats
  /clear     → clear memory
  /local     → switch to local Ollama
  /cloud     → switch to Groq cloud
  /quit      → exit
""")


def print_help():
    print("""
📖 Commands:
  /help      → this menu
  /persona   → list and switch personas
  /history   → view full conversation
  /stats     → backend, turns, memory info
  /clear     → wipe conversation memory
  /local     → use Ollama (offline)
  /cloud     → use Groq (online, smarter)
  /quit      → exit chatbot
""")


def main():
    print_banner()

    # Start with default persona, Groq backend
    bot = SmartChatbot(
        persona="default",
        use_local=False,
        stream=True
    )

    print("💡 Type your message or a /command to begin.\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()

            # Skip empty input
            if not user_input:
                continue

            # ── Commands ──────────────────────────────
            if user_input.lower() == "/quit":
                print("\n👋 Goodbye! See you tomorrow for Day 6.\n")
                break

            elif user_input.lower() == "/help":
                print_help()

            elif user_input.lower() == "/history":
                bot.memory.show_history()

            elif user_input.lower() == "/stats":
                bot.get_stats()

            elif user_input.lower() == "/clear":
                bot.memory.clear()

            elif user_input.lower() == "/persona":
                list_personas()
                choice = input("Enter persona name: ").strip()
                bot.switch_persona(choice)

            elif user_input.lower() == "/local":
                bot.llm = __import__(
                    'langchain_ollama', fromlist=['ChatOllama']
                ).ChatOllama(model="llama3.2", temperature=0.7)
                bot.backend = "Ollama (local)"
                print("✅ Switched to local Ollama")

            elif user_input.lower() == "/cloud":
                bot.llm = __import__(
                    'langchain_groq', fromlist=['ChatGroq']
                ).ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.7
                )
                bot.backend = "Groq (cloud)"
                print("✅ Switched to Groq cloud")

            # ── Normal chat ───────────────────────────
            else:
                bot.chat(user_input)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Type /quit to exit cleanly.\n")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try again or type /quit to exit.\n")


if __name__ == "__main__":
    main()