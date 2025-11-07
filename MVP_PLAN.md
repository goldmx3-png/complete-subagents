# MVP1 Project Plan - Minimal RAG System

## 🎯 **Objective**
Create a minimal viable product with essential RAG features only, removing all experimental/testing features.

---

## ✅ **Features to KEEP**

### 1. **Document Upload & Processing**
- ✅ Upload PDF, DOCX files via API
- ✅ Docling PDF → Markdown conversion
- ✅ Markdown-based chunking with header preservation
- ✅ Token-based chunking (600 tokens per chunk)
- ✅ Hierarchical metadata extraction (breadcrumbs, parent-child relationships)

### 2. **Storage & Retrieval**
- ✅ Qdrant vector store
- ✅ PostgreSQL for metadata
- ✅ Embeddings model (BAAI/bge-m3)
- ✅ Hybrid search (Vector + BM25)
- ✅ Reranking (cross-encoder)

### 3. **API Endpoints**
- ✅ POST /upload - Upload documents
- ✅ POST /chat - Query with retrieval + reranking
- ✅ GET /documents - List documents
- ✅ DELETE /document/{id} - Delete document
- ✅ GET /health - Health check

### 4. **Core Features**
- ✅ Conversation memory (PostgreSQL)
- ✅ Simple RAG agent (retrieve → rerank → format → LLM)
- ✅ Menu agent (fallback for greetings/off-topic)
- ✅ **NEW: Optimized formatting** (150K → 15K chars reduction)

---

## ❌ **Features to REMOVE**

### 1. **Agentic RAG (Not Required)**
- ❌ Query rewriting
- ❌ Iterative retrieval loops
- ❌ Document grading
- ❌ Complexity analysis
- ❌ Self-reflection
- ❌ Agentic workflow (LangGraph)

### 2. **Complex Routing**
- ❌ Support agent
- ❌ API agent (banking integration)
- ❌ Classifier agent (intent detection)
- ❌ Orchestrator with multi-agent routing

### 3. **Experimental Features**
- ❌ Semantic chunking
- ❌ Context organizer
- ❌ Module analyzer
- ❌ Section grouping (keep minimal formatting only)
- ❌ Query cache
- ❌ Parallel grading

### 4. **Testing/Debug Files**
- ❌ test_agentic_fix.py
- ❌ test_api_flow.py
- ❌ test_hierarchical_metadata.py
- ❌ AGENTIC_RAG_IMPLEMENTATION.md
- ❌ HIERARCHICAL_METADATA_IMPLEMENTATION.md
- ❌ etc.

---

## 📁 **Simplified Project Structure**

```
complete-subagents-mvp1/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   └── __init__.py              # Minimal config (20-30 env vars)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py                # Simplified API (no orchestrator)
│   │   └── schemas.py               # Request/response models
│   ├── document_processing/
│   │   ├── __init__.py
│   │   ├── uploader.py              # PDF/DOCX upload
│   │   ├── markdown_chunker.py      # Docling + chunking
│   │   └── hierarchical_metadata.py # Metadata extraction
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py             # Simple RAG retriever
│   │   ├── hybrid_retriever.py      # Vector + BM25
│   │   ├── reranker.py              # Cross-encoder reranking
│   │   └── enhanced_retriever.py    # With optimized formatting
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py                # OpenRouter client
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── qdrant_store.py          # Vector DB
│   │   └── embeddings.py            # Embedding model
│   ├── memory/
│   │   ├── __init__.py
│   │   └── conversation_store.py    # PostgreSQL store
│   └── utils/
│       ├── __init__.py
│       └── logger.py                # Simple logging
├── .env.example                      # ~30-40 lines (minimal)
├── requirements.txt                  # Core dependencies only
├── docker-compose.yml                # Qdrant + PostgreSQL
└── README.md                         # Simple setup guide
```

---

## 🔧 **Simplified Configuration (.env.example)**

**Reduced from 152 lines → ~40 lines**

```bash
# ===== LLM Configuration =====
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MAIN_MODEL=mistralai/magistral-small-2506
MAX_TOKENS=4096
TEMPERATURE=0.7

# ===== Vector Store =====
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents

# ===== Embeddings =====
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
EMBEDDING_DEVICE=cpu

# ===== Reranking =====
ENABLE_RERANKING=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_K=20
RERANKER_RETURN_TOP_K=5
RERANKER_DEVICE=cpu

# ===== Database =====
DATABASE_URL=postgresql://chatbot_user:changeme@localhost:5432/chatbot

# ===== Document Processing =====
UPLOAD_DIRECTORY=uploads
MAX_FILE_SIZE_MB=50
MARKDOWN_CHUNK_SIZE_TOKENS=600
MARKDOWN_CHUNK_OVERLAP_PERCENTAGE=15

# ===== Retrieval =====
ENABLE_HYBRID_SEARCH=true
HYBRID_VECTOR_WEIGHT=0.7
HYBRID_BM25_WEIGHT=0.3
TOP_K_RETRIEVAL=20

# ===== Hierarchical Metadata =====
ENABLE_HIERARCHICAL_METADATA=true
HIERARCHY_MAX_DEPTH=6

# ===== Context Formatting (NEW - Optimized) =====
MAX_FORMATTED_CHUNK_SIZE_CHARS=4000
MAX_TOTAL_CONTEXT_SIZE_CHARS=20000
BREADCRUMB_MAX_LEVELS=3
BREADCRUMB_MAX_LENGTH=80
FORMATTING_STYLE=minimal
ENABLE_AUTO_FALLBACK=true

# ===== API =====
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000
```

**Removed:**
- All Agentic RAG settings (10+ variables)
- Router/classifier settings
- Semantic chunking settings
- Query rewriting settings
- Complex retry/timeout settings

---

## 🔄 **Simplified API Flow**

### **Current (Complex):**
```
Request → Orchestrator → Classifier → Route Decision
  ├─→ AgenticRAG (query rewrite → retrieve → grade → refine)
  ├─→ RAG Agent (retrieve → format → LLM)
  ├─→ Support Agent
  ├─→ API Agent
  └─→ Menu Agent
```

### **MVP (Simple):**
```
Request → Simple Handler
  ├─→ Retrieve (hybrid search)
  ├─→ Rerank (top 5)
  ├─→ Format (optimized)
  └─→ LLM Response
```

**Single endpoint logic:**
1. Receive user query
2. Retrieve top 20 chunks (hybrid search)
3. Rerank to top 5
4. Format with minimal style (~15K chars)
5. Send to LLM
6. Return response

---

## 📝 **Files to Modify/Remove**

### **Files to DELETE:**
```
src/agents/
  ├── orchestrator.py                ❌ Delete
  ├── classifier.py                  ❌ Delete
  ├── agentic_rag/                   ❌ Delete entire folder
  ├── support/                       ❌ Delete entire folder
  ├── api/                           ❌ Delete entire folder
  ├── menu/                          ⚠️ Keep (fallback for greetings)
  └── rag/                           ✅ Keep & simplify

src/retrieval/
  ├── query_rewriter.py              ❌ Delete
  ├── context_organizer.py           ❌ Delete
  ├── module_analyzer.py             ❌ Delete
  ├── retriever.py                   ✅ Keep
  ├── hybrid_retriever.py            ✅ Keep
  ├── reranker.py                    ✅ Keep
  └── enhanced_retriever.py          ✅ Keep (has our fix!)

Root files:
  ├── test_agentic_fix.py            ❌ Delete
  ├── test_api_flow.py               ❌ Delete
  ├── test_hierarchical_metadata.py  ❌ Delete
  ├── test_formatting_fix.py         ⚠️ Keep (useful test)
  ├── test_formatting_simple.py      ✅ Keep (validation)
  ├── AGENTIC_*.md                   ❌ Delete
  ├── HIERARCHICAL_*.md              ❌ Delete
  └── problem.md                     ❌ Delete
```

### **Files to SIMPLIFY:**
```
src/api/routes.py
  - Remove orchestrator dependency
  - Direct retrieval → rerank → LLM flow
  - Keep: upload, chat, documents CRUD

src/agents/rag/agent.py
  - Remove agentic features
  - Simple: retrieve → format → generate

src/config/__init__.py
  - Remove 100+ unused settings
  - Keep only 30-40 essential ones
```

---

## 🧪 **Core Dependencies (requirements.txt)**

**Keep:**
```
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-multipart==0.0.6
qdrant-client==1.7.0
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
python-dotenv==1.0.0
docling==1.10.1
docling-core==1.5.1
sentence-transformers==2.2.2
rank-bm25==0.2.2
openai==1.5.0  # For OpenRouter
langchain==0.1.0
langchain-text-splitters==0.0.1
```

**Remove:**
```
langgraph==0.0.25               ❌ (agentic workflows)
langchain-community==0.0.13     ❌ (extra tools)
tiktoken==0.5.2                 ⚠️ Keep (token counting)
```

---

## 🚀 **Implementation Steps**

### **Phase 1: Create Branch & Structure**
1. Create branch `mvp1` from current branch
2. Delete unnecessary folders/files
3. Update .env.example (152 → 40 lines)
4. Update requirements.txt

### **Phase 2: Simplify API**
1. Refactor `src/api/routes.py`:
   - Remove orchestrator
   - Direct retrieval flow
   - Keep upload/CRUD endpoints

2. Create simple RAG handler:
   ```python
   async def simple_rag(query: str, user_id: str):
       # 1. Retrieve (hybrid search)
       chunks = await retriever.retrieve(query, top_k=20)

       # 2. Rerank
       if settings.enable_reranking:
           chunks = await reranker.rerank(query, chunks, top_k=5)

       # 3. Format (optimized - our fix!)
       context = retriever.format_context(chunks)

       # 4. Generate
       response = await llm.generate(query, context)

       return response
   ```

### **Phase 3: Clean Config**
1. Update `src/config/__init__.py`:
   - Remove all agentic settings
   - Remove classifier/router settings
   - Keep only core RAG settings

2. Validate all imports still work

### **Phase 4: Update Documentation**
1. Create simple README.md:
   - Quick start guide
   - API endpoints
   - Docker setup

2. Remove old implementation docs

### **Phase 5: Test & Validate**
1. Test upload → chunk → retrieve → rerank flow
2. Verify formatting optimization (15K chars)
3. Test all API endpoints
4. Push to branch `mvp1`

---

## 📊 **Expected Results**

### **Code Reduction:**
- **Lines of code:** ~5,000 → ~2,000 (60% reduction)
- **Config vars:** 152 → 40 (74% reduction)
- **Dependencies:** 25 → 15 (40% reduction)
- **Files:** 50+ → 25 (50% reduction)

### **Performance:**
- **Faster startup:** No agentic agent initialization
- **Lower latency:** Direct retrieval (no routing/grading)
- **Simpler debugging:** Single code path

### **Maintained Features:**
- ✅ Document upload (PDF, DOCX)
- ✅ Docling markdown conversion
- ✅ Hierarchical chunking with metadata
- ✅ Hybrid search (Vector + BM25)
- ✅ Reranking (cross-encoder)
- ✅ Optimized formatting (our 93% fix!)
- ✅ API endpoints
- ✅ Conversation memory

---

## ⚠️ **Trade-offs**

**What we lose:**
- No query rewriting (may miss some queries)
- No iterative refinement (single-pass only)
- No document quality grading (trust retrieval scores)
- No multi-agent routing (single RAG path)

**What we gain:**
- Much simpler codebase
- Faster responses
- Easier to maintain
- Easier to deploy
- Lower complexity

---

## 🎯 **Success Criteria**

1. ✅ Upload PDF → convert to markdown → chunk → store
2. ✅ Query → retrieve → rerank → format → LLM response
3. ✅ Context size < 20K chars (our formatting fix working)
4. ✅ API response time < 2 seconds
5. ✅ All core endpoints working
6. ✅ Clean codebase (no unused code)

---

## 📦 **Next Steps After Review**

Once you approve this plan, I will:

1. Create branch `mvp1` from current branch
2. Execute all deletion/simplification steps
3. Test the simplified system
4. Push to remote `claude/mvp1-011CUtP6zhGerfZ3We4Zk6nM`

**Estimated time:** 30-45 minutes

---

**Ready to proceed? Please review and let me know if you want any changes to the plan!**
