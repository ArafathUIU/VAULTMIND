# VaultMind 🔐
> Agentic Document Intelligence Platform

VaultMind is a multi-agent AI system that lets you upload documents and query them in natural language. Unlike a simple RAG chatbot, VaultMind uses a team of specialized agents — router, retriever, reasoner, and critic — orchestrated via LangGraph to produce accurate, hallucination-checked answers with full observability.

Built as a production-grade showcase of modern AI engineering: RAG pipelines, agentic workflows, LLM evaluation, and AI Ops.

---

## Architecture

User Query → Router Agent → Retriever Agent → Reasoning Agent → Critic Agent → Response
                                    ↑
                          Vector Store (FAISS / Pinecone)
                          Document Ingestion Pipeline

---

## Features

- Multi-agent orchestration with LangGraph
- PDF / DOCX / TXT ingestion with semantic chunking
- Vector search via FAISS (local) or Pinecone (cloud)
- Multi-LLM support: OpenAI, Gemini, Anthropic
- Streaming responses via FastAPI
- Hallucination checking via critic agent
- Full observability: latency, token cost, agent traces (LangSmith)
- Evaluation pipeline: RAGAS + DeepEval
- Dockerised for deployment

---

## Quickstart

git clone https://github.com/ArafathUIU/vaultmind
cd vaultmind
cp .env.example .env        # add your API keys
docker-compose up --build

API available at http://localhost:8000
Docs at http://localhost:8000/docs

---

## Tech Stack

| Layer         | Stack                              |
|---------------|------------------------------------|
| Agents        | LangGraph, LangChain               |
| LLMs          | OpenAI, Gemini, Anthropic          |
| Vector DB     | FAISS (local), Pinecone (cloud)    |
| Backend       | FastAPI, Python 3.11               |
| Observability | LangSmith, custom metrics          |
| Evaluation    | RAGAS, DeepEval                    |
| Infra         | Docker, GitHub Actions CI          |

---

## Roadmap

- [ ] Ingestion pipeline
- [ ] Vector store abstraction
- [ ] LangGraph agent orchestration
- [ ] FastAPI streaming endpoint
- [ ] Observability dashboard
- [ ] RAGAS evaluation suite
- [ ] Pinecone integration
- [ ] Frontend UI

---

## Author

Md. Arafath Hossain Akash — AI Engineer
github.com/ArafathUIU
