# VaultMind 🔐
> Agentic Document Intelligence Platform

VaultMind is a multi-agent AI system that lets you upload documents and query them in natural language. Unlike a simple RAG chatbot, VaultMind uses a team of specialized agents — router, retriever, reasoner, and critic — orchestrated via LangGraph to produce accurate, hallucination-checked answers with full observability.

Built as a production-grade showcase of modern AI engineering: RAG pipelines, agentic workflows, LLM evaluation, and AI Ops.

---

## Architecture

Document Upload -> Loader -> Chunker -> Embedder -> In-Memory Vector Store
User Query -> Router Agent -> Retriever Agent -> Reasoning Agent -> Critic Agent -> Response

---

## Features

- Multi-agent orchestration with LangGraph
- PDF / DOCX / TXT ingestion with semantic chunking
- Local vector search with a swappable vector-store abstraction
- Multi-LLM support: Groq, OpenAI, Gemini, Anthropic
- FastAPI upload, query, health, readiness, and reset endpoints
- Hallucination checking via critic agent
- Agent traces and latency metadata in API responses
- Professional browser UI served by FastAPI
- Dockerised for deployment

---

## Quickstart

git clone https://github.com/ArafathUIU/vaultmind
cd vaultmind
python -m venv .venv
.venv\Scripts\activate      # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env         # add GROQ_API_KEY or another LLM key
uvicorn api.main:app --reload

UI available at http://localhost:8000
API available at http://localhost:8000/docs
Docs at http://localhost:8000/docs

### API Endpoints

| Method | Path                | Purpose                         |
|--------|---------------------|---------------------------------|
| GET    | `/`                 | Served frontend UI              |
| GET    | `/health`           | Runtime health and store stats  |
| GET    | `/ready`            | Document-query readiness        |
| POST   | `/documents/upload` | Upload and index a document     |
| DELETE | `/documents`        | Clear the in-memory index       |
| POST   | `/query`            | Run the multi-agent pipeline    |

---

## Tech Stack

| Layer         | Stack                              |
|---------------|------------------------------------|
| Agents        | LangGraph, LangChain               |
| LLMs          | Groq, OpenAI, Gemini, Anthropic    |
| Vector DB     | In-memory vector store, FAISS-ready abstraction |
| Backend       | FastAPI, Python 3.11               |
| Frontend      | HTML, CSS, vanilla JavaScript      |
| Observability | LangSmith, custom metrics          |
| Evaluation    | RAGAS, DeepEval                    |
| Infra         | Docker, GitHub Actions CI          |

---

## Roadmap

- [x] Ingestion pipeline
- [x] Vector store abstraction
- [x] LangGraph agent orchestration
- [x] FastAPI upload/query endpoints
- [x] Frontend UI
- [ ] Observability dashboard
- [ ] RAGAS evaluation suite
- [ ] Pinecone integration

---

## Author

Md. Arafath Hossain Akash — AI Engineer
github.com/ArafathUIU
