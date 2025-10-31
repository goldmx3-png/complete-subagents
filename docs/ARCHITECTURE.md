# Architecture Documentation

## System Overview

Complete Subagents implements a **domain-based subagent architecture** where independent agents handle specific domains of responsibility.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│              ORCHESTRATOR AGENT                          │
│         (Intent Detection & Routing)                     │
│                                                          │
│  • Rule-based fast path                                 │
│  • LLM-based intent classification                      │
│  • Agent coordination via LangGraph                     │
└──────────────┬──────────────────────────────────────────┘
               │
    ┌──────────┴─────────┬──────────┬──────────┐
    ▼                    ▼          ▼          ▼
┌─────────┐        ┌─────────┐ ┌─────────┐ ┌─────────┐
│   RAG   │        │   API   │ │  MENU   │ │ SUPPORT │
│  AGENT  │        │  AGENT  │ │  AGENT  │ │  AGENT  │
└─────────┘        └─────────┘ └─────────┘ └─────────┘
```

## Core Components

### 1. Orchestrator Agent
**Location**: `src/agents/orchestrator.py`

**Responsibilities**:
- Intent detection (rule-based + LLM-based)
- Agent routing
- State management
- Response aggregation

**Flow**:
1. Receive user query
2. Detect intent (fast rule-based first, LLM fallback)
3. Route to appropriate domain agent
4. Return agent response

### 2. Domain Agents

#### RAG Agent (`src/agents/rag/`)
**Handles**: Document retrieval, query reformulation, answer generation

**Process**:
1. Embed user query (BGE-M3)
2. Search vector store (Qdrant)
3. Check for ambiguity
4. Format context from top chunks
5. Generate answer using LLM

#### Menu Agent (`src/agents/menu/`)
**Handles**: Greetings, navigation, menu generation

**Process**:
1. Classify intent (greeting/help/menu)
2. Generate appropriate menu
3. Return menu with buttons

#### API Agent (`src/agents/api/`)
**Handles**: External API calls, banking operations

**Process**:
1. Classify product/service
2. Select appropriate APIs
3. Execute API calls
4. Format responses

#### Support Agent (`src/agents/support/`)
**Handles**: FAQ search, ticket creation, help queries

**Process**:
1. Search FAQ database
2. Create support tickets
3. Provide help documentation

## State Management

### Shared State (`AgentState`)
All agents communicate via shared state schema:

```python
AgentState:
  - query: str                      # User query
  - user_id: str                    # User ID
  - route: str                      # Routing decision
  - rag: RAGState                   # RAG agent state
  - api: APIState                   # API agent state
  - menu: MenuState                 # Menu agent state
  - support: SupportState           # Support agent state
  - final_response: str             # Final answer
```

### Domain-Specific States
Each agent has its own state namespace:
- **RAGState**: chunks, context, ambiguity info
- **APIState**: product, apis, responses
- **MenuState**: intent, buttons, menu type
- **SupportState**: faq results, tickets

## Communication Pattern

1. **State Sharing**: Agents read/write to shared state
2. **LangGraph Workflow**: Orchestrator manages agent execution flow
3. **Agent Independence**: Each agent is self-contained and testable

## Benefits

✅ **Modularity**: Each agent is independent
✅ **Scalability**: Easy to add new agents
✅ **Testability**: Test agents in isolation
✅ **Maintainability**: Clear boundaries and responsibilities
✅ **Performance**: Agents can run concurrently (future optimization)

## Future Enhancements

- **Analytics Agent**: Usage analytics and insights
- **Reporting Agent**: Generate reports from data
- **Multi-agent Collaboration**: Agents working together
- **Agent-to-Agent Communication**: Direct messaging between agents
