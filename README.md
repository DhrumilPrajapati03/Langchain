# 🦜 LangChain × LangGraph — 30-Day Learning Journey

> A structured, hands-on repository documenting my 30-day journey from LangChain foundations to production-ready deep agents — covering LangChain, LangGraph, RAG, multi-agent systems, and trending AI engineering skills.

![Progress](https://img.shields.io/badge/Progress-Week%203%20of%204-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-7F77DD?style=flat-square)
![Notebooks](https://img.shields.io/badge/Notebooks-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## 📌 About This Repository

This repo is my public learning log for mastering the modern AI engineering stack — starting from LangChain basics, moving through RAG pipelines and agents, then into LangGraph-powered multi-agent systems, and finally deploying production-grade deep agents. Every week has dedicated notebooks, notes, and a capstone project.

**Learning philosophy:** Learn by building. Every concept is backed by a working notebook and every week ends with a real project inspired by production deployments at companies like LinkedIn, Replit, Uber, and Exa.

---

## 🗂️ Repository Structure

```
Langchain/
├── week-1/                    # LangChain foundations
│   ├── day1_setup.ipynb       # Python env, API keys, LangChain install
│   ├── day2_llm_basics.ipynb  # LLM APIs, tokens, completions
│   ├── day3_chains.ipynb      # LCEL, RunnableSequence, pipelines
│   ├── day4_prompts.ipynb     # Prompt templates, few-shot, CoT
│   └── day5_project.ipynb     # 🛠 Project 1: Smart Q&A Chatbot
│
├── week-2/                    # Agents, tools & RAG
│   ├── day6_loaders.ipynb     # Document loaders, text splitters
│   ├── day7_embeddings.ipynb  # Embeddings, FAISS, Chroma
│   ├── day8_rag.ipynb         # Full RAG pipeline with citations
│   ├── day9_tools.ipynb       # Tool calling, custom tools, @tool
│   └── day10_project.ipynb    # 🛠 Project 2: RAG Document Assistant
│
├── week-3/                    # LangGraph & multi-agent systems
│   ├── day11_react.ipynb      # ReAct agents, AgentExecutor
│   ├── day12_langgraph.ipynb  # StateGraph, nodes, edges
│   ├── day13_patterns.ipynb   # Loops, branches, human-in-loop
│   ├── day14_multiagent.ipynb # Supervisor pattern, subgraphs
│   └── day15_project.ipynb    # 🛠 Project 3: Deep Research Agent
│
├── week-4/                    # Deep agents & production
│   ├── day16_memory.ipynb     # Long-term memory, episodic, semantic
│   ├── day17_langsmith.ipynb  # Evals, tracing, datasets
│   ├── day18_structured.ipynb # Structured outputs, Pydantic
│   ├── day19_mcp.ipynb        # MCP, agent protocols, A2A
│   ├── day20_finetuning.ipynb # LoRA, DSPy, prompt optimization
│   ├── day21_async.ipynb      # Async chains, streaming, websockets
│   ├── day22_vectordb.ipynb   # Pinecone, Weaviate, hybrid search
│   ├── day23_security.ipynb   # Prompt injection, guardrails
│   ├── day24_deploy.ipynb     # LangServe, FastAPI, Docker
│   ├── day25_project.ipynb    # 🛠 Project 4: Multi-Agent Coding Assistant
│   ├── day26_observability.ipynb # Monitoring, cost dashboards
│   ├── day27_crewai.ipynb     # CrewAI, role-based agents
│   ├── day28_autogen.ipynb    # AutoGen, OpenAI Swarm
│   ├── day29_capstone_plan.ipynb # Capstone architecture & design
│   └── day30_capstone.ipynb   # 🏆 Capstone: Full Agentic AI System
│
├── Projects/                  # Standalone project folders
│   ├── project1-chatbot/
│   ├── project2-rag-assistant/
│   ├── project3-research-agent/
│   ├── project4-coding-assistant/
│   └── capstone/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🗓️ 30-Day Learning Roadmap

### Week 1 — LangChain Foundations `Days 1–5`

| Day | Topic | Key Concepts | Status |
|-----|-------|-------------|--------|
| 1 | Python & AI environment setup | venv, pip, Jupyter, dotenv | ✅ |
| 2 | LLM basics | Completions, tokens, cost, system/user/assistant | ✅ |
| 3 | LangChain chains | LCEL, RunnableSequence, RunnableParallel | ✅ |
| 4 | Prompt engineering | Few-shot, CoT, FewShotPromptTemplate | ✅ |
| 5 | 🛠 **Project 1: Smart Q&A Chatbot** | ConversationChain, streaming, Gradio | ✅ |

### Week 2 — Agents, Tools & RAG `Days 6–10`

| Day | Topic | Key Concepts | Status |
|-----|-------|-------------|--------|
| 6 | Document loaders & splitters | PyPDFLoader, WebBaseLoader, RecursiveCharacterTextSplitter | ✅ |
| 7 | Embeddings & vector stores | OpenAI embeddings, FAISS, Chroma | ✅ |
| 8 | RAG pipeline | RetrievalQA, contextual compression, reranking | ✅ |
| 9 | Tools & tool calling | @tool, AgentExecutor, custom tools | ✅ |
| 10 | 🛠 **Project 2: RAG Document Assistant** | Multi-PDF Q&A with source citations | ✅ |

### Week 3 — LangGraph & Multi-Agent Systems `Days 11–15`

| Day | Topic | Key Concepts | Status |
|-----|-------|-------------|--------|
| 11 | ReAct agents | Thought → Action → Observation, AgentExecutor | ✅ |
| 12 | LangGraph intro | StateGraph, nodes, edges, TypedDict | 🔄 |
| 13 | LangGraph patterns | Loops, branches, human-in-loop, interrupt_before | ⬜ |
| 14 | Multi-agent orchestration | Supervisor pattern, subgraphs, agent handoff | ⬜ |
| 15 | 🛠 **Project 3: Deep Research Agent** | Planner → Tasks → Observer graph with Tavily | ⬜ |

### Week 4 — Deep Agents & Production `Days 16–30`

| Day | Topic | Key Concepts | Status |
|-----|-------|-------------|--------|
| 16 | Advanced memory | LangMem, episodic/semantic memory, summarization | ⬜ |
| 17 | Evals & LangSmith | Tracing, evaluation datasets, custom evaluators | ⬜ |
| 18 | Structured outputs | with_structured_output(), Pydantic, JSON mode | ⬜ |
| 19 | MCP & agent protocols | Model Context Protocol, A2A communication | ⬜ |
| 20 | Fine-tuning & prompting | LoRA, QLoRA, DSPy auto-optimization | ⬜ |
| 21 | Async & streaming | ainvoke, astream, WebSocket streaming | ⬜ |
| 22 | Vector DB deep dive | Pinecone, Weaviate, hybrid search, metadata filtering | ⬜ |
| 23 | Agent security | Prompt injection, NeMo Guardrails, LLM Guard | ⬜ |
| 24 | Deployment basics | LangServe, FastAPI, Docker | ⬜ |
| 25 | 🛠 **Project 4: Multi-Agent Coding Assistant** | Planner → Coder → Tester → Reviewer | ⬜ |
| 26 | Observability & monitoring | Token cost dashboards, latency alerts | ⬜ |
| 27 | CrewAI | Role-based agents, tasks, crews | ⬜ |
| 28 | AutoGen & Swarm | Microsoft AutoGen, OpenAI Swarm | ⬜ |
| 29 | Capstone planning | Architecture design, eval dataset, success metrics | ⬜ |
| 30 | 🏆 **Capstone: Full Agentic AI System** | End-to-end agent with RAG + tools + memory + LangSmith | ⬜ |

---

## 🛠️ Projects Built

### Project 1 — Smart Q&A Chatbot `Week 1`
Multi-turn chatbot with conversation memory and streaming responses.

**Stack:** LangChain · ChatPromptTemplate · ConversationChain · Gradio  
**Concepts:** LCEL, message history, streaming output  
**Inspired by:** [chat.langchain.com](https://chat.langchain.com) — LangChain's official reference chatbot

---

### Project 2 — RAG Document Assistant `Week 2`
Upload any PDF and ask questions — with cited source chunks for every answer.

**Stack:** LangChain · PyPDFLoader · FAISS · OpenAI Embeddings · RetrievalQA  
**Concepts:** RAG pipeline, semantic search, source attribution  
**Inspired by:** LangChain community spotlight — private PDF Q&A pattern

---

### Project 3 — Deep Research Agent `Week 3`
Autonomous research agent that searches the web, synthesises information, and outputs a structured report with citations.

**Stack:** LangGraph · Tavily Search · StateGraph · Structured output  
**Concepts:** Planner → parallel Tasks → Observer graph, conditional routing  
**Inspired by:** [Exa's production research agent](https://blog.langchain.com) — LangChain case study 2024

---

### Project 4 — Multi-Agent Coding Assistant `Week 4`
Natural language → working code with automated test generation and human review checkpoint before execution.

**Stack:** LangGraph · Multi-agent subgraphs · Human-in-loop · Code execution  
**Concepts:** Planner → Coder → Tester → Reviewer, interrupt_before  
**Inspired by:** Replit Agent & Uber code migration agent — LangGraph production deployments

---

### Capstone — Full Agentic AI System `Day 30`
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
TAVILY_API_KEY=tvly-...          # for web search tools (Week 3+)
PINECONE_API_KEY=...             # for vector DB (Week 4)
LANGCHAIN_API_KEY=...            # for LangSmith tracing (Week 4)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=30-day-langchain
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
