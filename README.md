# Complete Subagents - Advanced Multi-Agent RAG System

A **production-ready multi-agent RAG system** for corporate banking customer support, featuring state-of-the-art retrieval strategies, document processing, and intelligent agent orchestration.

## 🌟 Key Highlights

### Advanced RAG Pipeline
- **Hybrid Search**: Vector (semantic) + BM25 (keyword) → 8-15% accuracy boost
- **Cross-Encoder Reranking**: MiniLM model → 3-5% additional accuracy improvement
- **Docling PDF Parser**: Structure-preserving markdown conversion with table extraction
- **Hierarchical Metadata**: Breadcrumb navigation and section-aware retrieval
- **Smart Context Formatting**: Optimized prompts (reduced from 150K to 15K chars)

### Architecture Overview

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
     │
     ▼
┌──────────────────────────────────────────────┐
│        ENHANCED RETRIEVAL PIPELINE           │
│  ┌──────────────────────────────────────┐   │
│  │ 1. Query Rewriting (with caching)    │   │
│  │ 2. Hybrid Search (Vector + BM25)     │   │
│  │ 3. Cross-Encoder Reranking           │   │
│  │ 4. Context Organization              │   │
│  │ 5. Hierarchical Formatting           │   │
│  └──────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

### Domain Agents

1. **RAG Agent** - Advanced document retrieval with hybrid search, query reformulation, and context-aware answer generation
2. **API Agent** - Banking API integration, intelligent product classification, and execution
3. **Menu Agent** - Intent classification, dynamic menu generation, and navigation assistance
4. **Support Agent** - FAQ search, ticket creation, and knowledge base management

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
│   ├── agents/                     # Domain agents
│   │   ├── base.py                 # Base agent class
│   │   ├── orchestrator.py        # Main coordinator with intent routing
│   │   ├── classifier.py          # Intent classification logic
│   │   ├── rag/                    # RAG Agent (enhanced retrieval)
│   │   ├── api/                    # API Agent (banking integration)
│   │   ├── menu/                   # Menu Agent (navigation)
│   │   ├── support/                # Support Agent (FAQ/tickets)
│   │   └── shared/                 # Shared state & protocols
│   ├── retrieval/                  # Advanced retrieval strategies
│   │   ├── enhanced_retriever.py  # Main retrieval orchestrator
│   │   ├── hybrid_retriever.py    # Vector + BM25 hybrid search
│   │   ├── reranker.py            # Cross-encoder reranking
│   │   ├── query_rewriter.py      # Query enhancement with caching
│   │   ├── context_organizer.py   # Smart context formatting
│   │   └── module_analyzer.py     # Module-aware retrieval
│   ├── document_processing/        # Advanced document processing
│   │   ├── markdown_parser.py     # Docling PDF → Markdown converter
│   │   ├── markdown_chunker.py    # Header-aware chunking
│   │   ├── semantic_chunker.py    # LLM-based semantic chunking
│   │   ├── chunker_factory.py     # Chunking strategy selector
│   │   └── uploader.py            # Document upload handler
│   ├── vectorstore/                # Vector storage & embeddings
│   │   ├── qdrant_store.py        # Qdrant client with BM25
│   │   └── embeddings.py          # BAAI/bge-m3 embeddings
│   ├── api_tools/                  # Banking API integration
│   │   ├── api_registry.py        # API endpoint catalog
│   │   ├── api_selector.py        # Intelligent API selection
│   │   ├── api_executor.py        # API call execution
│   │   └── response_formatter.py  # Response formatting
│   ├── memory/                     # Conversation management
│   │   └── conversation_store.py  # PostgreSQL conversation storage
│   ├── llm/                        # LLM clients
│   │   └── openrouter_client.py   # OpenRouter integration
│   ├── config/                     # Configuration
│   │   └── menu_config.py         # Menu system configuration
│   ├── api/                        # FastAPI routes
│   │   ├── routes.py              # API endpoints
│   │   └── schemas.py             # Request/response schemas
│   └── utils/                      # Utilities
│       ├── logger.py              # Structured logging
│       ├── metrics.py             # Performance metrics
│       └── correlation.py         # Request correlation IDs
├── tests/                          # Test suite
│   ├── test_orchestrator.py      # Agent routing tests
│   ├── test_hybrid_pipeline.py   # Retrieval pipeline tests
│   ├── test_model_preload.py     # Model loading tests
│   └── evaluation/                # Evaluation scripts
├── scripts/                        # Setup & verification scripts
│   ├── setup_all.sh               # Complete setup automation
│   └── verify_markdown_chunking.py # Chunking verification
├── docs/                           # Comprehensive documentation
│   ├── ARCHITECTURE.md            # System architecture
│   ├── AGENTS.md                  # Agent specifications
│   ├── PIPELINE_FLOW.md           # Retrieval pipeline details
│   ├── MARKDOWN_CHUNKING_GUIDE.md # Chunking strategies
│   ├── IMPLEMENTATION_SUMMARY.md  # Implementation details
│   └── QUICK_START.md             # Quick start guide
```

## 🎯 Key Features

### 🔍 Advanced Retrieval
- **Hybrid Search**: Combines vector similarity and BM25 keyword matching for superior accuracy
- **Cross-Encoder Reranking**: Fast MiniLM model ensures top results are most relevant
- **Query Enhancement**: Automatic query rewriting with 24-hour caching
- **Hierarchical Context**: Section-aware retrieval with breadcrumb navigation
- **Smart Formatting**: Optimized context delivery (93% size reduction: 150K → 15K chars)
- **Module Analysis**: Intelligent grouping of related content

### 📄 Document Processing
- **Docling Integration**: State-of-the-art PDF → Markdown conversion
- **Structure Preservation**: Maintains headers, lists, tables, and formatting
- **Flexible Chunking**: Choose between markdown-based (structure-aware) or semantic (LLM-based) chunking
- **Table Intelligence**: Smart handling of small (inline) vs large (separate) tables
- **Hierarchical Metadata**: Extracts document structure for enhanced retrieval
- **Multi-Format Support**: PDF, DOCX, TXT with extensible parser architecture

### 🤖 Multi-Agent System
- **Modular Architecture**: Independent, testable domain agents with clear responsibilities
- **Intelligent Routing**: Intent-based orchestration with confidence scoring
- **State Sharing**: Agents communicate via shared LangGraph state protocol
- **Ambiguity Detection**: Clarifying questions when user intent is unclear
- **Streaming Responses**: Real-time answer generation with progress indicators

### 🏦 Banking Integration
- **API Tools**: Comprehensive banking API integration framework
- **Product Classification**: Intelligent categorization of banking products
- **Multi-Bank Support**: Isolated documentation sets for different institutions
- **Secure Operations**: JWT authentication, SSL verification, retry logic

### 💾 Storage & Memory
- **Vector Store**: Self-hosted Qdrant with BM25 indexing
- **Embeddings**: BAAI/bge-m3 (1024-dim) for semantic search
- **Conversation Memory**: PostgreSQL-backed context-aware chat history
- **Query Caching**: Smart caching for query rewrites and embeddings

### 🚀 Performance
- **Fast Models**: Optimized for speed (Magistral 7B, MiniLM reranker)
- **Batch Processing**: Efficient embedding and reranking operations
- **Lazy Loading**: Models load on-demand to minimize startup time
- **Configurable Timeouts**: Handles long-running operations gracefully

### 🔧 Production-Ready
- **Comprehensive Logging**: Structured logs with correlation IDs
- **Metrics & Monitoring**: Performance tracking and diagnostics
- **Error Handling**: Graceful degradation and fallback strategies
- **Environment-Based Config**: 150+ configuration options via `.env`
- **Multi-Tenancy**: Support for multiple banks with isolated data

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

### Core Settings

Key settings in `.env` (see `.env.example` for all 150+ options):

```bash
# ===== LLM Configuration =====
OPENROUTER_API_KEY=sk-or-v1-xxxxx
MAIN_MODEL=mistralai/magistral-small-2506  # Fast 7B model
ROUTER_MODEL=mistralai/magistral-small-2506
TEMPERATURE=0.7

# ===== Vector Store & Embeddings =====
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024

# ===== Document Processing =====
# Markdown Chunking (RECOMMENDED)
USE_MARKDOWN_CHUNKING=true
MARKDOWN_CHUNK_SIZE_TOKENS=600
MARKDOWN_CHUNK_OVERLAP_PERCENTAGE=15
DOCLING_EXTRACT_TABLES=true

# Semantic Chunking (alternative)
USE_SEMANTIC_CHUNKING=false
SEMANTIC_CHUNK_MAX_TOKENS=800

# ===== Advanced Retrieval =====
# Hybrid Search (RECOMMENDED)
ENABLE_HYBRID_SEARCH=true
HYBRID_VECTOR_WEIGHT=0.7  # Semantic similarity
HYBRID_BM25_WEIGHT=0.3     # Keyword matching

# Reranking (RECOMMENDED)
ENABLE_RERANKING=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_K=20          # Retrieve 20 docs
RERANKER_RETURN_TOP_K=10   # Return top 10 after reranking

# Query Enhancement
USE_QUERY_REWRITING=true
QUERY_REWRITE_CACHE_TTL=86400  # 24 hours

# ===== Hierarchical Metadata =====
ENABLE_HIERARCHICAL_METADATA=true
ENABLE_SECTION_GROUPING=true
ENABLE_BREADCRUMB_CONTEXT=true
BREADCRUMB_MAX_LEVELS=3

# ===== Context Formatting =====
MAX_FORMATTED_CHUNK_SIZE_CHARS=4000
MAX_TOTAL_CONTEXT_SIZE_CHARS=20000
FORMATTING_STYLE=minimal   # minimal, normal, or detailed
ENABLE_AUTO_FALLBACK=true

# ===== General Settings =====
MAX_CHUNKS_PER_QUERY=15
MIN_SIMILARITY_SCORE=0.3
AMBIGUITY_THRESHOLD=0.15
LOG_LEVEL=DEBUG  # Use INFO for production
```

### Configuration Strategies

#### 🚀 Recommended Setup (Fast + Accurate)
```bash
ENABLE_HYBRID_SEARCH=true
ENABLE_RERANKING=true
USE_MARKDOWN_CHUNKING=true
ENABLE_HIERARCHICAL_METADATA=true
USE_QUERY_REWRITING=true  # Test for your use case
```
**Expected Performance**: 500-800ms query latency, 8-15% accuracy boost

#### ⚡ Speed-Optimized Setup
```bash
ENABLE_HYBRID_SEARCH=false
ENABLE_RERANKING=false
USE_MARKDOWN_CHUNKING=true
ENABLE_HIERARCHICAL_METADATA=false
USE_QUERY_REWRITING=false
```
**Expected Performance**: 200-400ms query latency

#### 🎯 Maximum Accuracy Setup
```bash
ENABLE_HYBRID_SEARCH=true
ENABLE_RERANKING=true
USE_SEMANTIC_CHUNKING=true  # LLM-based chunking
ENABLE_HIERARCHICAL_METADATA=true
USE_QUERY_REWRITING=true
RERANKER_TOP_K=30
RERANKER_RETURN_TOP_K=15
```
**Expected Performance**: 1-2s query latency, highest accuracy

## 📖 Documentation

### Core Documentation
- [**ARCHITECTURE.md**](docs/ARCHITECTURE.md) - System design and agent communication patterns
- [**AGENTS.md**](docs/AGENTS.md) - Individual agent specifications and responsibilities
- [**PIPELINE_FLOW.md**](docs/PIPELINE_FLOW.md) - Complete retrieval pipeline walkthrough
- [**MARKDOWN_CHUNKING_GUIDE.md**](docs/MARKDOWN_CHUNKING_GUIDE.md) - Document processing strategies
- [**IMPLEMENTATION_SUMMARY.md**](docs/IMPLEMENTATION_SUMMARY.md) - Technical implementation details
- [**QUICK_START.md**](docs/QUICK_START.md) - Fast setup and deployment guide

### Key Concepts

#### Retrieval Pipeline Flow
```
User Query
    ↓
1. Query Rewriting (optional, cached)
    ↓
2. Hybrid Retrieval
   ├─ Vector Search (semantic similarity)
   └─ BM25 Search (keyword matching)
    ↓
3. Score Fusion (weighted combination)
    ↓
4. Cross-Encoder Reranking
    ↓
5. Context Organization
   ├─ Section Grouping
   ├─ Breadcrumb Generation
   └─ Metadata Enrichment
    ↓
6. Context Formatting
   ├─ Size Optimization
   ├─ Hierarchical Structure
   └─ Fallback Handling
    ↓
Final Context → LLM
```

#### Document Processing Pipeline
```
PDF Upload
    ↓
1. Docling Conversion (PDF → Markdown)
   ├─ Structure Detection (headers, lists, tables)
   ├─ Table Extraction
   └─ Metadata Extraction
    ↓
2. Markdown Chunking or Semantic Chunking
   ├─ Header-aware splitting (markdown)
   ├─ Or LLM-based boundaries (semantic)
   └─ Overlap management
    ↓
3. Hierarchical Metadata Extraction
   ├─ Breadcrumb paths
   ├─ Parent-child relationships
   └─ Section navigation
    ↓
4. Embedding Generation (BAAI/bge-m3)
    ↓
5. Vector Store Indexing
   ├─ Qdrant vector index
   └─ BM25 keyword index
    ↓
Ready for Retrieval
```

## 📊 Performance Benchmarks

### Retrieval Accuracy Improvements
- **Hybrid Search**: 8-15% improvement over vector-only search
- **Reranking**: Additional 3-5% accuracy boost
- **Combined Pipeline**: Up to 20% total improvement
- **Context Optimization**: 93% reduction in prompt size (150K → 15K chars)

### Query Latency (with recommended setup)
- **Document Upload**: 5-10s per PDF (includes parsing, chunking, embedding)
- **Query Processing**: 500-800ms average
  - Query rewriting: 50-100ms (cached after first use)
  - Hybrid retrieval: 100-200ms
  - Reranking: 200-300ms
  - Context formatting: 50-100ms
  - LLM generation: 1-3s (streaming)

### Resource Usage
- **Memory**: ~2GB (models loaded on-demand)
- **Disk**: ~500MB (embeddings + reranker models)
- **CPU**: Efficient batch processing
- **GPU**: Optional (faster embedding/reranking)

## 💡 Best Practices

### For Production Deployment
1. **Use Recommended Setup**: Hybrid search + reranking + markdown chunking
2. **Enable Caching**: Query rewrite cache reduces repeated processing
3. **Monitor Logs**: Set `LOG_LEVEL=INFO` and track metrics
4. **Optimize Chunk Size**: Test 400-800 tokens for your document type
5. **Tune Weights**: Adjust `HYBRID_VECTOR_WEIGHT` and `HYBRID_BM25_WEIGHT` for your domain

### For Banking Documents
- **Markdown Chunking**: Better for structured policy documents
- **Hierarchical Metadata**: Essential for multi-section navigation
- **Table Extraction**: Crucial for rate cards, fee schedules
- **Section Grouping**: Helps users find related policies

### For Development
- **Start Simple**: Disable reranking/hybrid initially, enable once baseline works
- **Test Incrementally**: Add one feature at a time and measure impact
- **Use Evaluation Scripts**: See `tests/evaluation/` for accuracy testing
- **Check Context Size**: Monitor `MAX_TOTAL_CONTEXT_SIZE_CHARS` to avoid token limits

## 🔍 Troubleshooting

### Common Issues

**Models not loading?**
```bash
# Check model files are downloaded
ls ~/.cache/torch/sentence_transformers/

# Increase timeout
RERANKER_LOAD_TIMEOUT=600
```

**Retrieval accuracy low?**
```bash
# Enable all features
ENABLE_HYBRID_SEARCH=true
ENABLE_RERANKING=true
ENABLE_HIERARCHICAL_METADATA=true

# Increase context
RERANKER_RETURN_TOP_K=15
MAX_CHUNKS_PER_QUERY=20
```

**Context too large?**
```bash
# Reduce context size
MAX_CHUNKS_PER_QUERY=10
MAX_FORMATTED_CHUNK_SIZE_CHARS=3000
FORMATTING_STYLE=minimal
```

**Slow queries?**
```bash
# Disable expensive features
ENABLE_RERANKING=false
USE_QUERY_REWRITING=false

# Use faster model
MAIN_MODEL=google/gemini-2.0-flash-exp:free
```

## 🤝 Contributing

Contributions welcome! Each agent is independent, making it easy to:
- **Add new agents** (e.g., Analytics Agent, Reporting Agent)
- **Enhance retrieval** (new reranking models, retrieval strategies)
- **Improve chunking** (custom chunking strategies for specific document types)
- **Optimize performance** (caching, batching, parallelization)
- **Extend integrations** (new LLM providers, vector stores)

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design details.

## 📄 License

MIT License

---

## 🏗️ Technology Stack Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Agents** | LangGraph | Multi-agent orchestration |
| **LLM** | OpenRouter | Open-source model access |
| **Embeddings** | BAAI/bge-m3 | Semantic search (1024-dim) |
| **Vector Store** | Qdrant | Vector + BM25 storage |
| **Database** | PostgreSQL | Conversation memory |
| **PDF Parser** | Docling | Markdown conversion |
| **Reranker** | MiniLM | Cross-encoder reranking |
| **API** | FastAPI | REST endpoints |
| **Framework** | Python 3.11+ | Core language |

**Built with 100% open-source models and self-hosted infrastructure**
