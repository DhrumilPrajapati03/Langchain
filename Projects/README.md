# 🤖 Smart Q&A Chatbot v1.1

A powerful, Persona-driven AI chatbot built with **LangChain**, **Groq**, and **Gradio**. This project supports both cloud-based LLMs (via Groq) and local LLMs (via Ollama), featuring persistent conversation memory and swappable behavioral profiles.

---

## 🌟 Key Features

- **🧠 Conversation Memory**: Remembers past interactions within a configurable context window (default: 20 messages).
- **🎭 Multi-Persona System**: Switch between different specialized AI personalities on the fly.
- **☁️ Cloud & 🏠 Local Support**: 
  - **Groq (Cloud)**: Uses `llama-3.3-70b-versatile` for high-speed, intelligent responses.
  - **Ollama (Local)**: Uses `llama3.2` for private, offline inference.
- **📟 Dual Interfaces**:
  - **CLI (Terminal)**: Lightweight streaming interface for command-line power users.
  - **Gradio (Web UI)**: Modern, responsive web interface for a rich visual experience.
- **⚡ Streaming Responses**: Real-time token generation for a more natural conversation feel (CLI).

---

## 🎭 Available Personas

| Persona | AI Name | Description |
| :--- | :--- | :--- |
| **`default`** | Aria | Friendly, helpful, and concise general assistant. |
| **`python_tutor`** | PyMentor | Expert Python tutor who explains "why" and gives challenges. |
| **`socratic`** | Socrates | Never gives direct answers; asks deep questions instead. |
| **`code_reviewer`** | CodeSensei | Brutally honest senior engineer analyzing code quality. |
| **`explain_like_5`** | Eli | Explains complex topics using simple words and emojis. 🎉 |

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+**
- **Groq API Key**: Get one from [Groq Cloud](https://console.groq.com/keys).
- **Ollama (Optional)**: If you want to run locally. Download at [ollama.com](https://ollama.com).

### 2. Installation
```bash
# Clone the repository
git clone <repository-url>
cd Langchain

# Setup virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_api_key_here
```

---

## 📖 How to Use

### Terminal Interface (CLI)
Run the chatbot directly in your terminal with streaming support:
```bash
python Projects/app.py
```
**Commands during chat:**
- `/persona` → Switch between available experts.
- `/history` → View the full conversation log.
- `/clear`   → Wipe memory and start fresh.
- `/stats`    → View backend info and turn count.
- `/quit`     → Exit the app.

### Web Interface (Gradio)
Launch a beautiful web interface in your browser:
```bash
python Projects/ui.py
```
*The app will automatically open at `http://localhost:7860`.*

---

## 🛠️ Built With

- **LangChain**: Core framework for LLM orchestration and memory.
- **Gradio**: High-performance web interface library.
- **Groq**: Lightning-fast cloud inference.
- **Ollama**: Seamless local model hosting.
- **Python-Dotenv**: Secure environment variable management.

---

## 👨‍💻 Author
**Dhrumil Prajapati**
*Week 1 Project - LangChain Deep Dive*
