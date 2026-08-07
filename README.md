# 🦜 LangChain × LangGraph — Learning Roadmap

My personal learning journey from LangChain foundations to production-ready deep agents — covering LangChain, LangGraph, RAG, multi-agent systems, and AI engineering.

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

## 🛠️ Projects

| # | Project | Stack | Status |
|---|---------|-------|--------|
| 1 | **Smart Q&A Chatbot** — Multi-turn chatbot with memory & streaming | LangChain · Gradio | ✅ |
| 2 | **RAG Document Assistant** — PDF Q&A with source citations | LangChain · FAISS · PyPDFLoader | ✅ |
| 3 | **Deep Research Agent** — Autonomous web research with structured reports | LangGraph · Tavily · StateGraph | ✅ |
| 4 | **Multi-Agent Coding Assistant** — NL → code with auto testing & review | LangGraph · Multi-agent subgraphs | ⬜ |
| 5 | **Capstone: Full Agentic AI System** — Production agent with full stack | LangGraph · LangSmith · FastAPI | ⬜ |

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
| Security | NeMo Guardrails, LLM Guard |

---

## 📚 References

- [LangChain Docs](https://python.langchain.com)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph)
- [LangSmith](https://smith.langchain.com)
- [LangChain Blog](https://blog.langchain.com)
- [Prompt Engineering Guide](https://www.promptingguide.ai)
- [HuggingFace PEFT](https://huggingface.co/docs/peft)
- [CrewAI Docs](https://docs.crewai.com)
- [Tavily API](https://tavily.com)
