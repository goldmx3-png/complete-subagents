# Complete Subagents - Modular RAG System

A production-ready RAG system built with **domain-based subagent architecture** using LangGraph, OpenRouter, and Qdrant.

## 🌟 Architecture Highlights

This project implements a **modular subagent system** where independent domain agents handle specific responsibilities:

```
┌─────────────────────────────────────────────────────────┐
│              ORCHESTRATOR AGENT                         │
│         (Intent Detection & Routing)                    │
└──────────────┬──────────────────────────────────────────┘
               │
    ┌──────────┴─────────┬──────────┬──────────┐
    ▼                    ▼          ▼          ▼
┌─────────┐        ┌─────────┐ ┌─────────┐ ┌─────────┐
│   RAG   │        │   API   │ │  MENU   │ │ SUPPORT │
│  AGENT  │        │  AGENT  │ │  AGENT  │ │  AGENT  │
└─────────┘        └─────────┘ └─────────┘ └─────────┘
```

### Domain Agents

1. **RAG Agent** - Document retrieval, query reformulation, answer generation
2. **API Agent** - Banking API integration, product classification, execution
3. **Menu Agent** - Intent classification, menu generation, navigation
4. **Support Agent** - FAQ search, ticket creation, knowledge base

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenRouter API key ([Get one here](https://openrouter.ai/keys))

### Setup

```bash
# 1. Clone/navigate to project
cd complete-subagents

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# 5. Start infrastructure (Qdrant + PostgreSQL)
docker-compose up -d

# 6. Start API server
python -m uvicorn src.api.routes:app --reload
```

### Access

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 📐 Project Structure

```
complete-subagents/
├── src/
│   ├── agents/                  # Domain agents
│   │   ├── base.py              # Base agent class
│   │   ├── orchestrator.py     # Main coordinator
│   │   ├── rag/                 # RAG Agent
│   │   ├── api/                 # API Agent
│   │   ├── menu/                # Menu Agent
│   │   ├── support/             # Support Agent
│   │   └── shared/              # Shared state & protocols
│   ├── llm/                     # LLM clients
│   ├── vectorstore/             # Qdrant + embeddings
│   ├── memory/                  # Conversation storage
│   ├── document_processing/     # PDF parsing
│   ├── api/                     # FastAPI routes
│   └── utils/                   # Utilities
├── tests/                       # Test suite
├── scripts/                     # Setup scripts
└── docs/                        # Documentation
```

## 🎯 Key Features

- **Modular Architecture**: Independent, testable domain agents
- **State Sharing**: Agents communicate via shared LangGraph state
- **Open-Source LLMs**: Uses OpenRouter for access to Llama, Mistral, Qwen
- **Self-Hosted Storage**: Qdrant (vectors) + PostgreSQL (conversations)
- **Streaming Responses**: Real-time answer generation
- **Conversation Memory**: Context-aware follow-up questions
- **Ambiguity Detection**: Clarifying questions when needed

## 📚 API Endpoints

### Chat
```bash
POST /chat
{
  "user_id": "user123",
  "message": "What is the Q4 revenue?",
  "conversation_id": "optional"
}
```

### Document Upload
```bash
POST /upload
FormData: {
  file: <pdf-file>,
  user_id: "user123"
}
```

### Health Check
```bash
GET /health
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific agent tests
pytest tests/test_rag_agent.py
pytest tests/test_orchestrator.py

# With coverage
pytest --cov=src tests/
```

## 🔧 Configuration

Key settings in `.env`:

```bash
# LLM
OPENROUTER_API_KEY=sk-or-v1-xxxxx
MAIN_MODEL=mistralai/magistral-small-2506
ROUTER_MODEL=mistralai/magistral-small-2506

# Vector Store
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=BAAI/bge-m3

# RAG Settings
MAX_CHUNKS_PER_QUERY=10
TOP_K_RETRIEVAL=20
AMBIGUITY_THRESHOLD=0.15
```

## 📖 Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - System design and agent communication
- [Agent Guide](docs/AGENTS.md) - Each agent's responsibility and API
- [API Reference](docs/API.md) - FastAPI endpoints

## 🤝 Contributing

Contributions welcome! Each agent is independent, making it easy to:
- Add new agents (e.g., Analytics Agent, Reporting Agent)
- Enhance existing agents
- Improve orchestration logic

## 📄 License

MIT License

---

**Built with 100% open-source models via OpenRouter**
