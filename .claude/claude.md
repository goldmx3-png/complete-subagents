# Project Overview

## Business Context

This project is a **multi-agent customer support system for corporate banking portals**. It provides intelligent, context-aware assistance to banking customers through a modular RAG (Retrieval-Augmented Generation) architecture.

### Key Characteristics

- **Domain**: Corporate banking customer support
- **Architecture**: Multi-agent system with specialized domain agents (RAG, API, Menu, Support)
- **Multi-tenancy**: Designed to serve multiple banks with different documentation sets
- **Status**: Development/Pre-production

### Critical Production Requirements

1. **Retrieval Accuracy**: High-precision document retrieval is paramount in production
2. **Knowledge Base Stability**: Document pipeline is a one-time process; knowledge base updates occur infrequently (6-12 month cycles when bank policies change)
3. **Multi-bank Support**: System must handle isolated documentation sets for different banking institutions which are deployed seperately.

## Technical Architecture

### Core Components

```
Orchestrator (Intent Detection & Routing)
    ├── RAG Agent (Document retrieval, query reformulation, answer generation)
    │   └── Agentic RAG (Multi-step retrieval with validation and refinement)
    ├── API Agent (Banking API integration, product classification)
    ├── Menu Agent (Intent classification, navigation assistance)
    └── Support Agent (FAQ search, ticket creation, knowledge base)
```

### Technology Stack

- **LLM Provider**: OpenRouter (open-source models)
- **Vector Store**: Qdrant (self-hosted)
- **Database**: PostgreSQL
- **Embeddings**: BAAI/bge-m3 (1024 dimensions)
- **Framework**: LangGraph for agent orchestration
- **API**: FastAPI

### Key Features

- Agentic RAG with iterative refinement
- Parallel document grading (9x faster)
- Query rewriting and caching
- Conversation memory with context
- Streaming responses
- Document relevance validation

---

## Development Guidelines

### 🔒 ALWAYS Required Actions

#### Configuration Management
- **ALWAYS** update both `.env.example` and `.env` when adding new configuration parameters
- **ALWAYS** document new environment variables with inline comments
- **ALWAYS** provide sensible defaults in `.env.example`

#### Change Approval Process
- **ALWAYS** explain your implementation plan before making significant changes
- **ALWAYS** get user approval before:
  - Adding new dependencies
  - Modifying core agent logic
  - Changing RAG retrieval strategies
  - Altering database schemas
  - Refactoring shared state management

#### Testing & Validation
- **ALWAYS** run existing tests after changes that could affect agents
- **ALWAYS** verify retrieval accuracy when modifying RAG components
- **ALWAYS** check that environment variables are correctly loaded

### 🔍 Code Quality Standards

#### Security
- Validate all user inputs
- Sanitize file uploads (check extensions, file size, content type)
- Use parameterized queries (SQLAlchemy ORM)
- Never log sensitive data (API keys, user credentials, PII)
- Keep dependencies updated

#### Performance
- Use async/await for I/O operations
- Batch database operations where possible
- Implement caching for frequently accessed data (query rewrites, embeddings)
- Monitor token usage in LLM calls

#### Maintainability
- Follow existing code structure and naming conventions
- Add docstrings to all public functions/classes
- Keep functions focused and single-purpose
- Use type hints consistently
- Document complex logic with inline comments

### 📁 File Organization

- Agent implementations: `src/agents/<agent_name>/agent.py`
- Shared protocols: `src/agents/shared/protocols.py`
- Utilities: `src/utils/`
- Tests: `tests/test_<component>.py`
- Configuration: `.env` (never commit), `.env.example` (commit)

### 🧪 Testing Strategy

- Unit tests for individual agent methods
- Integration tests for agent communication
- End-to-end tests for complete workflows
- Test files in `tests/` directory with clear naming

### 🚀 Development Workflow

1. Read and understand existing code before making changes
2. Propose implementation approach for review
3. Make changes following code quality standards
4. Update configuration files if needed
5. Test changes locally
6. Document any new features or behavior changes

---

## Research & Innovation

When implementing new features or solving complex problems:

1. **Use MCP (Model Context Protocol)** to fetch current industry best practices
2. Research latest techniques in:
   - RAG optimization (retrieval strategies, reranking)
   - Multi-agent orchestration
   - LLM prompt engineering
   - Vector similarity search optimization
3. Evaluate trade-offs between accuracy, latency, and cost
4. Document reasoning for architectural decisions

---

## Agent-Specific Considerations

### RAG Agent
- Retrieval quality > speed (accuracy is critical)
- Query rewriting should preserve user intent
- Document grading threshold: balance precision vs. recall
- Chunk size optimization for banking documents

### API Agent
- Validate API responses before presenting to user
- Handle rate limiting gracefully
- Log all banking API interactions for audit

### Menu Agent
- Keep menu structures simple and intuitive
- Provide clear navigation paths
- Handle ambiguous intents with clarifying questions

### Support Agent
- FAQ search should prioritize exact matches
- Ticket creation requires validation of required fields
- Knowledge base updates must maintain consistency

---

## Common Pitfalls to Avoid

1. **Don't** modify shared state protocols without understanding downstream impacts
2. **Don't** change embedding models (requires full knowledge base re-indexing)
3. **Don't** commit `.env` file with secrets
4. **Don't** bypass the orchestrator for agent communication
5. **Don't** assume configuration values - always use environment variables
6. **Don't** write hardcoded file paths (use configuration)

---

## Quick Reference

### Start Development Server
```bash
source venv/bin/activate
python -m uvicorn src.api.routes:app --reload
```

### Run Tests
```bash
pytest tests/
```

### Check Configuration
```bash
grep "^[A-Z]" .env | head -20
```

### Qdrant Collection Management
```bash
# View collections: http://localhost:6333/dashboard
# Access via Python: see examples in tests/
```

---

**Remember**: This system handles sensitive banking information. Prioritize security, accuracy, and reliability in all implementations.
