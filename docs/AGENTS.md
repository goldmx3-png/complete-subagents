# Agent Reference Guide

## Base Agent Interface

All agents implement the `BaseAgent` abstract class:

```python
class BaseAgent(ABC):
    async def can_handle(self, state: AgentState) -> bool
    async def execute(self, state: AgentState) -> AgentState
    def get_name(self) -> str
```

---

## RAG Agent

**File**: `src/agents/rag/agent.py`

### Purpose
Handles document retrieval and question answering using RAG (Retrieval-Augmented Generation).

### Capabilities
- Semantic search across document corpus
- Query reformulation for better retrieval
- Ambiguity detection and clarification
- Context-aware answer generation
- Citation support

### When to Use
- Questions about uploaded documents
- Information retrieval queries
- "What is..." or "Tell me about..." questions

### State Updates
```python
state["rag"] = {
    "chunks": [...],              # Retrieved chunks
    "context": "...",             # Formatted context
    "is_ambiguous": bool,         # Needs clarification?
    "disambiguation_options": [...]
}
```

---

## Menu Agent

**File**: `src/agents/menu/agent.py`

### Purpose
Provides fast-path navigation and interactive menus.

### Capabilities
- Intent classification (greeting/help/menu)
- Dynamic menu generation
- Button-based navigation
- Quick responses without LLM calls

### When to Use
- Greetings: "Hello", "Hi"
- Navigation: "Menu", "Help"
- Getting started flows

### State Updates
```python
state["menu"] = {
    "intent": "greeting",
    "menu_buttons": [...],
    "menu_type": "main",
    "is_menu": true
}
```

---

## API Agent

**File**: `src/agents/api/agent.py`

### Purpose
Integrates with external APIs and services.

### Capabilities
- Product/service classification
- API selection and execution
- Response formatting
- Error handling and retries

### When to Use
- Action queries: "Create", "Update", "Delete"
- Banking operations: "Get balance", "Transfer"
- External integrations

### State Updates
```python
state["api"] = {
    "product": "general",
    "selected_apis": [...],
    "api_responses": [...],
    "needs_clarification": false
}
```

---

## Support Agent

**File**: `src/agents/support/agent.py`

### Purpose
Provides help, FAQ support, and ticket management.

### Capabilities
- FAQ database search
- Support ticket creation
- Help documentation
- How-to guides

### When to Use
- "How do I..." questions
- Help requests
- Support queries
- Ticket creation

### State Updates
```python
state["support"] = {
    "faq_result": {...},
    "ticket_id": "TKT-123",
    "cached_response": "..."
}
```

---

## Adding New Agents

### Step 1: Create Agent Class
```python
from src.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    async def can_handle(self, state):
        # Check if agent should handle
        return state.get("route") == "CUSTOM"

    async def execute(self, state):
        # Agent logic here
        state["final_response"] = "Custom response"
        return state

    def get_name(self):
        return "CustomAgent"
```

### Step 2: Add to Orchestrator
```python
self.custom_agent = CustomAgent()

workflow.add_node("custom", self.custom_agent.execute)
workflow.add_edge("detect_intent", "custom")
```

### Step 3: Update State Schema
```python
class CustomState(TypedDict):
    custom_field: str

class AgentState(TypedDict):
    custom: CustomState  # Add to global state
```

---

## Agent Best Practices

1. **Single Responsibility**: Each agent handles one domain
2. **State Isolation**: Use domain-specific state namespaces
3. **Error Handling**: Graceful failures with fallback responses
4. **Logging**: Use self.logger for operations
5. **Testability**: Write unit tests for each agent
6. **Documentation**: Document capabilities and state updates
