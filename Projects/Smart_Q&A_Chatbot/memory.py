from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List

class ConversationMemory:
    """Manage conversation history for a chatbot.
    Stores messages and handles context window limits"""

    def __init__(self, max_messages:int = 20):
        self.messages: List = []
        self.max_messages = max_messages

    def add_user_message(self, content:str):
        """Add a human message to history"""
        self.messages.append(HumanMessage(content=content))
        self._trim()

    def add_ai_message(self, content:str):
        """Add an AI message to history"""
        self.messages.append(AIMessage(content=content))
        self._trim()

    def get_messages(self) -> List:
        """Return all stored messages"""
        return self.messages

    def clear(self):    
        """Clear all messages"""
        self.messages = []
        print("Memory cleaned")

    def _trim(self):
        """Trim messages to max_messages limit"""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def show_history(self):
        if not self.messages:
            print("No conversation history yet")
            return

        print("\n" + "="*50)
        print("CONVERSATION HISTORY")
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                print(f"\n👤 You: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"\n🤖 Bot: {msg.content}")
        print("="*50 + "\n")

    def get_turn_count(self) -> int:
        """How many exchanges have occured"""
        return len([m for m in self.messages if isinstance(m, HumanMessage)])
    