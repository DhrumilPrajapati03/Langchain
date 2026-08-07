# 🦜 LangChain × LangGraph — Learning Journey

> A structured, hands-on repository documenting my journey from LangChain foundations to production-ready deep agents — covering LangChain, LangGraph, RAG, multi-agent systems, and trending AI engineering skills.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-7F77DD?style=flat-square)
![Notebooks](https://img.shields.io/badge/Notebooks-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## 📌 About This Repository

This repo is my private learning log for mastering the modern AI engineering stack — starting from LangChain basics, moving through RAG pipelines and agents, then into LangGraph-powered multi-agent systems, and finally deploying production-grade deep agents. Every section has dedicated notebooks, notes, and a capstone project.

**Learning philosophy:** Learn by building. Every concept is backed by a working notebook and every section ends with a real project inspired by production deployments at companies like LinkedIn, Replit, Uber, and Exa.

---

## 🗂️ Repository Structure

```
Langchain/
├── week-1/                          # LangChain Foundations
│   ├── day1_test.ipynb              # Python env, API keys, LangChain install
│   ├── day2_test.ipynb              # LLM APIs, tokens, completions
│   ├── day3_test.ipynb              # LCEL, RunnableSequence, pipelines
│   ├── day4_test.ipynb              # Prompt templates, few-shot, CoT
│   └── basic_agent.py              # Basic agent implementation
│
├── week-2/                          # Agents, Tools & RAG
│   ├── day6_test.ipynb              # Document loaders, text splitters
│   ├── day7_test.ipynb              # Embeddings, FAISS, Chroma
│   ├── day8_test.ipynb              # Full RAG pipeline with citations
│   ├── day9_test.ipynb              # Tool calling, custom tools, @tool
│   ├── chroma_db/                   # Chroma vector store data
│   ├── faiss_db/                    # FAISS vector store data
│   └── sample_docs/                 # Sample documents for RAG
│
├── week-3/                          # LangGraph & Multi-Agent Systems
│   ├── day11_test.ipynb             # ReAct agents, AgentExecutor
│   ├── day12_test.ipynb             # StateGraph, nodes, edges
│   ├── day13_test.ipynb             # Loops, branches, human-in-loop
│   └── day14_test.ipynb             # Supervisor pattern, subgraphs
│
├── week-4/                          # Deep Agents & Production
│   ├── day-17.ipynb                 # Evals, tracing, datasets (LangSmith)
│   ├── day-18.ipynb                 # Structured outputs, Pydantic
│   ├── day-19.ipynb                 # MCP, agent protocols, A2A
│   ├── mcp_servers/                 # MCP server implementations
│   └── eval_docs/                   # Evaluation datasets & documents
│
├── Projects/                        # Standalone project folders
│   ├── Smart_Q&A_Chatbot/
│   ├── RAG_Document_Assistant/
│   └── Autonomous Research Agent/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🗺️ Learning Roadmap

### Module 1 — LangChain Foundations

| # | Topic | Key Concepts | Status |
|---|-------|-------------|--------|
| 1 | Python & AI environment setup | venv, pip, Jupyter, dotenv | ✅ |
| 2 | LLM basics | Completions, tokens, cost, system/user/assistant | ✅ |
| 3 | LangChain chains | LCEL, RunnableSequence, RunnableParallel | ✅ |
| 4 | Prompt engineering | Few-shot, CoT, FewShotPromptTemplate | ✅ |
| 5 | 🛠 **Project: Smart Q&A Chatbot** | ConversationChain, streaming, Gradio | ✅ |

---

### Module 2 — Agents, Tools & RAG

| # | Topic | Key Concepts | Status |
|---|-------|-------------|--------|
| 1 | Document loaders & splitters | PyPDFLoader, WebBaseLoader, RecursiveCharacterTextSplitter | ✅ |
| 2 | Embeddings & vector stores | OpenAI embeddings, FAISS, Chroma | ✅ |
| 3 | RAG pipeline | RetrievalQA, contextual compression, reranking | ✅ |
| 4 | Tools & tool calling | @tool, AgentExecutor, custom tools | ✅ |
| 5 | 🛠 **Project: RAG Document Assistant** | Multi-PDF Q&A with source citations | ✅ |

---

### Module 3 — LangGraph & Multi-Agent Systems

| # | Topic | Key Concepts | Status |
|---|-------|-------------|--------|
| 1 | ReAct agents | Thought → Action → Observation, AgentExecutor | ✅ |
| 2 | LangGraph intro | StateGraph, nodes, edges, TypedDict | ✅ |
| 3 | LangGraph patterns | Loops, branches, human-in-loop, interrupt_before | ✅ |
| 4 | Multi-agent orchestration | Supervisor pattern, subgraphs, agent handoff | ✅ |
| 5 | 🛠 **Project: Deep Research Agent** | Planner → Tasks → Observer graph with Tavily | ✅ |

---

### Module 4 — Deep Agents & Production

| # | Topic | Key Concepts | Status |
|---|-------|-------------|--------|
| 1 | Advanced memory | LangMem, episodic/semantic memory, summarization | 🔄 |
| 2 | Evals & LangSmith | Tracing, evaluation datasets, custom evaluators | ✅ |
| 3 | Structured outputs | with_structured_output(), Pydantic, JSON mode | ✅ |
| 4 | MCP & agent protocols | Model Context Protocol, A2A communication | ✅ |
| 5 | Fine-tuning & prompting | LoRA, QLoRA, DSPy auto-optimization | ⬜ |
| 6 | Async & streaming | ainvoke, astream, WebSocket streaming | ⬜ |
| 7 | Vector DB deep dive | Pinecone, Weaviate, hybrid search, metadata filtering | ⬜ |
| 8 | Agent security | Prompt injection, NeMo Guardrails, LLM Guard | ⬜ |
| 9 | Deployment basics | LangServe, FastAPI, Docker | ⬜ |
| 10 | 🛠 **Project: Multi-Agent Coding Assistant** | Planner → Coder → Tester → Reviewer | ⬜ |

---

### Module 5 — Advanced Topics & Capstone

| # | Topic | Key Concepts | Status |
|---|-------|-------------|--------|
| 1 | Observability & monitoring | Token cost dashboards, latency alerts | ⬜ |
| 2 | CrewAI | Role-based agents, tasks, crews | ⬜ |
| 3 | AutoGen & Swarm | Microsoft AutoGen, OpenAI Swarm | ⬜ |
| 4 | Capstone planning | Architecture design, eval dataset, success metrics | ⬜ |
| 5 | 🏆 **Capstone: Full Agentic AI System** | End-to-end agent with RAG + tools + memory + LangSmith | ⬜ |

---

## 🛠️ Projects Built

### Project 1 — Smart Q&A Chatbot
Multi-turn chatbot with conversation memory and streaming responses.

**Stack:** LangChain · ChatPromptTemplate · ConversationChain · Gradio  
**Concepts:** LCEL, message history, streaming output  
**Inspired by:** [chat.langchain.com](https://chat.langchain.com) — LangChain's official reference chatbot

---

### Project 2 — RAG Document Assistant
Upload any PDF and ask questions — with cited source chunks for every answer.

**Stack:** LangChain · PyPDFLoader · FAISS · OpenAI Embeddings · RetrievalQA  
**Concepts:** RAG pipeline, semantic search, source attribution  
**Inspired by:** LangChain community spotlight — private PDF Q&A pattern

---

### Project 3 — Deep Research Agent
Autonomous research agent that searches the web, synthesises information, and outputs a structured report with citations.

**Stack:** LangGraph · Tavily Search · StateGraph · Structured output  
**Concepts:** Planner → parallel Tasks → Observer graph, conditional routing  
**Inspired by:** [Exa's production research agent](https://blog.langchain.com) — LangChain case study 2024

---

### Project 4 — Multi-Agent Coding Assistant
Natural language → working code with automated test generation and human review checkpoint before execution.

**Stack:** LangGraph · Multi-agent subgraphs · Human-in-loop · Code execution  
**Concepts:** Planner → Coder → Tester → Reviewer, interrupt_before  
**Inspired by:** Replit Agent & Uber code migration agent — LangGraph production deployments

---

### Capstone — Full Agentic AI System
End-to-end production agent with RAG, tools, long-term memory, LangSmith evaluation, and LangServe deployment.

**Stack:** LangGraph · LangChain · LangSmith · LangServe · FastAPI · Pinecone  
**Concepts:** Multi-agent orchestration, persistent memory, full eval pipeline, REST API deployment  
**Inspired by:** OpenRecovery's memory-first agent — Top 5 LangGraph agents of 2024

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- A virtual environment manager (`venv` or `conda`)
- API keys for OpenAI (or Anthropic/Groq), and optionally Tavily, Pinecone

### Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/DhrumilPrajapati03/Langchain.git
cd Langchain

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Add your API keys to .env

# 5. Launch Jupyter
jupyter notebook
```

### Environment Variables

Create a `.env` file in the root with the following:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...     # optional
TAVILY_API_KEY=tvly-...          # for web search tools
PINECONE_API_KEY=...             # for vector DB
LANGCHAIN_API_KEY=...            # for LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=langchain-learning
```

---

## 📦 Tech Stack

| Category | Tools |
|----------|-------|
| LLM frameworks | LangChain, LangGraph, LangSmith, LangServe |
| LLM providers | OpenAI GPT-4o, Anthropic Claude, Groq (Llama 3) |
| Vector stores | FAISS, Chroma, Pinecone, Weaviate |
| Agent frameworks | LangGraph, CrewAI, AutoGen, OpenAI Swarm |
| Search tools | Tavily, DuckDuckGo, Wikipedia |
| Fine-tuning | HuggingFace PEFT, LoRA, DSPy |
| Deployment | FastAPI, LangServe, Docker |
| Notebooks | Jupyter Lab |
| Security | NeMo Guardrails, LLM Guard |

---

## 📚 Key Learning Resources

- [LangChain Documentation](https://python.langchain.com) — official reference
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph) — graph-based agent framework
- [LangSmith](https://smith.langchain.com) — tracing, evals, and debugging
- [LangChain Blog](https://blog.langchain.com) — case studies and production patterns
- [Prompt Engineering Guide](https://www.promptingguide.ai) — comprehensive prompting reference
- [HuggingFace PEFT](https://huggingface.co/docs/peft) — fine-tuning with LoRA
- [CrewAI Docs](https://docs.crewai.com) — role-based multi-agent framework
- [Tavily API](https://tavily.com) — AI-optimised search for agents

---


## 🤝 Connect

If you're on the same learning journey, feel free to star this repo, open an issue, or connect on LinkedIn. Happy to discuss ideas, collaborate on projects, or share notes.

**GitHub:** [DhrumilPrajapati03](https://github.com/DhrumilPrajapati03)

---

## 📄 License

This repository is for educational purposes. All code and notebooks are open for reference and learning.

---

<p align="center">Built with curiosity, coffee, and a lot of <code>RecursionError: maximum depth exceeded</code></p>
