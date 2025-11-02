# Agentic RAG System

## Overview

This implementation features a state-of-the-art **Agentic RAG** system based on the latest patterns from LangGraph and LlamaIndex (via Context7). It provides self-correcting, adaptive retrieval with quality checking and iterative refinement.

## Key Features

### 🎯 Adaptive Routing
- **Intelligent Query Routing**: Automatically decides between vectorstore retrieval and direct generation
- **Strategy Selection**: Chooses between simple, multi-document, or multi-hop retrieval strategies
- **Context-Aware**: Considers conversation history for routing decisions

### 🔍 Self-Correcting Retrieval
- **Document Grading**: LLM-based relevance checking for retrieved documents
- **Query Rewriting**: Automatically reformulates queries when retrieval quality is poor
- **Retry Logic**: Up to 2 retries with improved queries before fallback

### ✅ Quality Validation
- **Hallucination Grading**: Ensures answers are grounded in source documents
- **Answer Quality Check**: Validates that answers actually address the question
- **Multi-Level Validation**: Documents → Generation → Answer quality

### 🧠 Multi-Step Reasoning
- **Query Planning**: Breaks down complex queries into sub-queries
- **Sequential Execution**: Executes query plans with dependency tracking
- **Result Synthesis**: Intelligently combines results from multiple steps

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Agentic RAG Workflow                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   1. Query Router      │
                    │   - vectorstore?       │
                    │   - direct generate?   │
                    └────────────────────────┘
                                 │
                  ┌──────────────┴───────────────┐
                  │                              │
         vectorstore route              direct generation
                  │                              │
                  ▼                              ▼
    ┌──────────────────────────┐    ┌──────────────────────┐
    │  2. Query Planner        │    │  Skip to Generate    │
    │  - simple / complex?     │    └──────────────────────┘
    │  - multi-step needed?    │
    └──────────────────────────┘
                  │
                  ▼
    ┌──────────────────────────┐
    │  3. Retrieve Documents   │
    │  - vectorstore search    │
    │  - context-aware         │
    └──────────────────────────┘
                  │
                  ▼
    ┌──────────────────────────┐
    │  4. Grade Documents      │
    │  - relevance check       │
    │  - score >= threshold?   │
    └──────────────────────────┘
                  │
      ┌───────────┴────────────┐
      │                        │
  relevant              not relevant
      │                        │
      ▼                        ▼
┌──────────────┐    ┌──────────────────────┐
│  Generate    │    │  5. Rewrite Query    │
│  Answer      │    │  - reformulate       │
└──────────────┘    │  - retry (max 2)     │
      │             └──────────────────────┘
      │                        │
      │                        └──────┐
      ▼                               │
┌──────────────────────────┐         │
│  6. Grade Answer         │◄────────┘
│  - hallucination check   │
│  - usefulness check      │
└──────────────────────────┘
                  │
      ┌───────────┴────────────┐
      │                        │
   passed                   failed
      │                        │
      ▼                        ▼
 ┌─────────┐          ┌──────────────┐
 │  DONE   │          │  Retry       │
 └─────────┘          │  (if < max)  │
                      └──────────────┘
```

## Components

### 1. **Graders** (`src/agents/rag/graders.py`)
- `DocumentGrader`: Checks relevance of retrieved documents
- `HallucinationGrader`: Validates answer grounding in sources
- `AnswerGrader`: Ensures answer addresses the question

### 2. **Routing** (`src/agents/rag/routing.py`)
- `QueryRouter`: Routes to vectorstore vs direct generation
- `RetrievalStrategyRouter`: Selects retrieval strategy

### 3. **Tools** (`src/agents/rag/tools.py`)
- `RetrieverTool`: Wraps retriever for agent use
- `ContextAwareRetrieverTool`: Handles conversation context

### 4. **Planner** (`src/agents/rag/planner.py`)
- `QueryPlanner`: Decomposes complex queries
- Multi-step execution with dependency tracking
- Result synthesis

### 5. **Workflow** (`src/agents/rag/agentic_workflow.py`)
- `AgenticRAGWorkflow`: Main LangGraph-based orchestrator
- State management with `AgenticRAGState`
- Conditional routing and retry logic

## Usage

### Enable Agentic RAG

**Option 1: Per-Request**
```python
state = {
    "query": "Compare transaction limits for USD and EUR",
    "user_id": "user123",
    "rag": {
        "use_agentic": True  # Enable for this request
    }
}

result = await rag_agent.execute(state)
```

**Option 2: Global Configuration**
```python
# In settings/config
settings.use_agentic_rag = True  # Enable globally
```

### Direct Usage

```python
from src.agents.rag.agentic_workflow import AgenticRAGWorkflow

# Initialize
workflow = AgenticRAGWorkflow()

# Run
result = await workflow.run(
    question="What are the validation rules for currency holidays?",
    user_id="user123",
    conversation_history=[...],
    max_retries=2
)

print(result["answer"])
print(result["metadata"])  # Routing decisions, quality scores, etc.
```

## Metadata Output

The agentic workflow provides rich metadata:

```python
{
    "answer": "Generated answer...",
    "documents": [...],
    "metadata": {
        "datasource": "vectorstore",  # or "generate_direct"
        "retrieval_strategy": "multi_document",
        "retry_count": 1,
        "documents_relevant": True,
        "relevance_score": 0.85,
        "answer_grounded": True,
        "answer_useful": True,
        "reformulated_queries": ["query v1", "query v2"]
    }
}
```

## Configuration

### Retry Settings
```python
# In agentic_workflow.py
await workflow.run(
    question="...",
    user_id="...",
    max_retries=2  # Adjust retry attempts
)
```

### Grading Thresholds
```python
# In graders.py - DocumentGrader.grade_documents()
threshold=0.6  # Minimum relevance score (0.0 - 1.0)
```

### Routing Prompts
Customize routing logic in `src/agents/rag/routing.py`:
- `QueryRouter._build_routing_prompt()`: Define routing criteria
- `RetrievalStrategyRouter.determine_strategy()`: Adjust complexity detection

## Examples

### Simple Query
```python
# User: "What is the cutoff time for payments?"
# → Routes to vectorstore → Simple retrieval → Generates answer
# → No retries needed if documents are relevant
```

### Complex Multi-Step Query
```python
# User: "Compare cutoff times for USD and EUR, then explain the validation process"
# → Routes to vectorstore
# → Planner creates 3-step plan:
#    1. Get USD cutoff times
#    2. Get EUR cutoff times
#    3. Get validation process
# → Executes steps → Synthesizes results
```

### Self-Correction Example
```python
# User: "What about the authorization matrix?"
# → Routes to vectorstore → Retrieves docs
# → Grades: relevance_score = 0.3 (too low)
# → Rewrites query: "authorization matrix rules and policies"
# → Retrieves again → relevance_score = 0.9
# → Generates answer → Validates no hallucination
```

## Performance Considerations

### LLM Calls
- **Standard RAG**: ~1 LLM call (generation)
- **Agentic RAG (best case)**: ~4 LLM calls (routing + grading + generation + validation)
- **Agentic RAG (with retries)**: Up to ~8 LLM calls

### Cost Optimization
- Uses `classifier_model` (faster/cheaper) for grading and routing
- Uses `main_model` for generation and planning
- Grades top 5 documents only (not all retrieved)

### Latency
- **Simple queries**: ~2-4 seconds (routing + retrieval + generation)
- **Complex queries with planning**: ~5-10 seconds
- **Queries with retries**: +2-3 seconds per retry

## Comparison: Standard vs Agentic RAG

| Feature | Standard RAG | Agentic RAG |
|---------|-------------|-------------|
| **Query Routing** | ❌ Always retrieves | ✅ Adaptive routing |
| **Quality Checking** | ❌ No validation | ✅ Multi-level grading |
| **Self-Correction** | ❌ Single attempt | ✅ Auto-retry with rewrite |
| **Hallucination Prevention** | ❌ No checking | ✅ Grounding validation |
| **Complex Queries** | ❌ Single retrieval | ✅ Multi-step planning |
| **Answer Quality** | ❌ No validation | ✅ Usefulness check |
| **LLM Calls** | ~1 | ~4-8 |
| **Latency** | Fast (~1-2s) | Moderate (~2-10s) |
| **Accuracy** | Good | Excellent |

## Troubleshooting

### Issue: Too many retries
**Solution**: Lower max_retries or adjust relevance threshold

### Issue: Slow responses
**Solution**:
- Reduce top_k for retrieval
- Skip grading for simple queries
- Use faster classifier_model

### Issue: Over-grading
**Solution**: Adjust grading prompts in `graders.py` to be less strict

## Future Enhancements

- [ ] Web search fallback integration
- [ ] Parallel document grading
- [ ] Streaming support for agentic workflow
- [ ] Fine-grained control over grading thresholds
- [ ] Cost tracking and optimization
- [ ] A/B testing framework

## References

- **LangGraph Adaptive RAG**: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/
- **LangGraph CRAG**: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/
- **LlamaIndex Query Planning**: https://docs.llamaindex.ai/en/stable/examples/workflow/planning_workflow/
- **Context7 Documentation**: Used for latest patterns and best practices

---

**Built with**: LangGraph, LlamaIndex patterns, Context7 latest techniques (2025)
