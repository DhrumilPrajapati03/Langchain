# chatbot.py
# Core chatbot logic - the brain of the app

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from memory import ConversationMemory
from personas import get_persona

load_dotenv()


class SmartChatbot:
    """
    A smart chatbot with:
    - Persistent memory across turns
    - Swappable personas
    - Support for both Groq (cloud) and Ollama (local)
    - Streaming responses
    """

    def __init__(
        self,
        persona: str = "default",
        use_local: bool = False,
        stream: bool = True
    ):
        self.persona_data = get_persona(persona)
        self.memory = ConversationMemory(max_messages=20)
        self.stream = stream
        self.parser = StrOutputParser()

        # Choose LLM backend
        if use_local:
            self.llm = ChatOllama(
                model="llama3.2",
                temperature=0.7
            )
            self.backend = "Ollama (local)"
        else:
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.7
            )
            self.backend = "Groq (cloud)"

    def chat(self, user_input: str) -> str:
        """
        Process one turn of conversation.
        Returns the AI response as a string.
        """
        # 1. Add user message to memory
        self.memory.add_user_message(user_input)

        # 2. Build full message list:
        #    system prompt + full history (includes latest user msg)
        system = SystemMessage(content=self.persona_data["system_prompt"])
        all_messages = [system] + self.memory.get_messages()

        # 3. Get response (stream or invoke)
        if self.stream:
            response_text = self._stream_response(all_messages)
        else:
            response = self.llm.invoke(all_messages)
            response_text = self.parser.invoke(response)

        # 4. Save AI response to memory
        self.memory.add_ai_message(response_text)

        return response_text

    def _stream_response(self, messages) -> str:
        """Stream response token by token, return full text"""
        full_response = ""
        print(f"\n🤖 {self.persona_data['name']}: ", end="", flush=True)

        for chunk in self.llm.stream(messages):
            token = chunk.content
            print(token, end="", flush=True)
            full_response += token

        print()  # newline after streaming
        return full_response

    def switch_persona(self, persona: str):
        """Switch personality without losing memory"""
        self.persona_data = get_persona(persona)
        print(f"✅ Switched to persona: {self.persona_data['name']}")

    def get_stats(self):
        """Show chatbot stats"""
        print(f"""
📊 Chatbot Stats:
   Backend  : {self.backend}
   Persona  : {self.persona_data['name']}
   Turns    : {self.memory.get_turn_count()}
   Messages : {len(self.memory.get_messages())}
""")