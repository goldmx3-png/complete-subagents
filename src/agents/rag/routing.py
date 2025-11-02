"""
Agentic RAG Query Router - Intelligent routing for retrieval strategies
Based on latest LangGraph adaptive RAG patterns from Context7
"""

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from src.llm.openrouter_client import OpenRouterClient
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RouteDecision(BaseModel):
    """
    Routing decision for query processing

    Based on LangGraph adaptive RAG routing pattern
    """
    datasource: Literal["vectorstore", "generate_direct"] = Field(
        description="Route to vectorstore for retrieval or generate_direct for general knowledge"
    )
    reasoning: str = Field(
        description="Brief explanation of routing decision"
    )
    confidence: float = Field(
        description="Confidence score (0-1) for this routing decision",
        ge=0.0,
        le=1.0
    )


class QueryRouter:
    """
    Routes queries to appropriate data sources

    Based on LangGraph adaptive RAG pattern:
    - Analyzes query intent and complexity
    - Routes to vectorstore for document-based questions
    - Routes to direct generation for general knowledge questions
    - Provides confidence scores for routing decisions
    """

    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        self.llm = llm_client or OpenRouterClient(
            model=settings.classifier_model  # Use fast model for routing
        )

    async def route_query(
        self,
        question: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Route query to appropriate data source

        Args:
            question: User's question
            conversation_history: Optional conversation history for context

        Returns:
            {
                "datasource": Literal["vectorstore", "generate_direct"],
                "reasoning": str,
                "confidence": float
            }
        """
        logger.info(f"Routing query: '{question[:100]}...'")

        # Build context from conversation history if available
        context_str = ""
        if conversation_history:
            recent_exchanges = conversation_history[-4:]  # Last 2 exchanges
            context_lines = []
            for msg in recent_exchanges:
                role = msg.get("role", "")
                content = msg.get("content", "")[:100]
                if role in ["user", "assistant"]:
                    context_lines.append(f"{role}: {content}")
            if context_lines:
                context_str = "\n".join(context_lines)

        prompt = self._build_routing_prompt(question, context_str)

        try:
            decision = await self.llm.structured_output(
                prompt=prompt,
                response_model=RouteDecision,
                temperature=0
            )

            logger.info(
                f"Routing decision: {decision.datasource} "
                f"(confidence={decision.confidence:.2f}) - {decision.reasoning}"
            )

            return {
                "datasource": decision.datasource,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence
            }

        except Exception as e:
            logger.error(f"Routing error: {e}, defaulting to vectorstore")
            # Default to vectorstore on error (safer)
            return {
                "datasource": "vectorstore",
                "reasoning": f"Routing failed ({str(e)}), defaulting to vectorstore",
                "confidence": 0.5
            }

    def _build_routing_prompt(self, question: str, context: str = "") -> str:
        """Build the routing prompt"""

        prompt = f"""You are an expert at routing user questions to the appropriate data source.

Your organization has a vectorstore containing:
- Banking operations documentation
- Policy documents and procedures
- Technical specifications
- Transaction processing rules
- Compliance guidelines
- System documentation

**Route to 'vectorstore'** if the question is about:
- Specific banking policies, procedures, or rules
- Technical documentation or system details
- Transaction processing specifics
- Compliance requirements
- Operational guidelines
- Any domain-specific information that would be in documents

**Route to 'generate_direct'** if the question is:
- General knowledge that doesn't require specific documents
- Simple greetings or chitchat
- Capability questions ("what can you do?", "how do you work?")
- Math calculations or general reasoning
- Questions that can be answered without domain-specific documents

"""

        if context:
            prompt += f"""
Recent conversation context:
---
{context}
---

"""

        prompt += f"""Current question: {question}

Analyze the question and provide:
1. datasource: "vectorstore" or "generate_direct"
2. reasoning: Brief explanation (1-2 sentences)
3. confidence: Confidence score from 0.0 to 1.0

Consider:
- Does this require specific document knowledge?
- Is this a follow-up to previous questions?
- Can this be answered with general knowledge?
"""

        return prompt


class RetrievalStrategyRouter:
    """
    Routes to different retrieval strategies based on query complexity

    Advanced routing for:
    - Simple single-document queries
    - Complex multi-document queries
    - Multi-hop reasoning queries
    """

    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        self.llm = llm_client or OpenRouterClient(
            model=settings.classifier_model
        )

    async def determine_strategy(
        self,
        question: str
    ) -> Dict:
        """
        Determine retrieval strategy

        Args:
            question: User's question

        Returns:
            {
                "strategy": Literal["simple", "multi_document", "multi_hop"],
                "reasoning": str,
                "requires_planning": bool
            }
        """
        logger.info("Determining retrieval strategy")

        # Simple heuristics for now (can be enhanced with LLM)
        question_lower = question.lower()

        # Check for comparison keywords
        comparison_keywords = ["compare", "difference", "versus", "vs", "better", "worse"]
        has_comparison = any(kw in question_lower for kw in comparison_keywords)

        # Check for aggregation keywords
        aggregation_keywords = ["all", "total", "sum", "list", "what are the"]
        has_aggregation = any(kw in question_lower for kw in aggregation_keywords)

        # Check for multi-step keywords
        multi_step_keywords = ["then", "after that", "first", "second", "finally"]
        has_multi_step = any(kw in question_lower for kw in multi_step_keywords)

        # Check for "and" which often indicates multiple sub-questions
        has_multiple_parts = " and " in question_lower

        if has_multi_step or (has_comparison and has_aggregation):
            strategy = "multi_hop"
            reasoning = "Query requires multi-step reasoning across documents"
            requires_planning = True

        elif has_comparison or has_aggregation or has_multiple_parts:
            strategy = "multi_document"
            reasoning = "Query requires information from multiple documents"
            requires_planning = False

        else:
            strategy = "simple"
            reasoning = "Simple single-document query"
            requires_planning = False

        logger.info(f"Strategy: {strategy} - {reasoning}")

        return {
            "strategy": strategy,
            "reasoning": reasoning,
            "requires_planning": requires_planning
        }
