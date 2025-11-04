"""
Agentic RAG Agent

Main agent interface for agentic RAG.
Provides a clean API for the orchestrator to invoke agentic RAG workflows.
"""

from typing import Dict, Any, Optional, AsyncIterator, Tuple
from src.agents.base import BaseAgent
from src.agents.shared.state import AgentState, RAGState
from src.utils.logger import get_logger
from src.config import get_settings
from .workflow import get_agentic_rag_workflow
from .router import should_use_agentic_rag, analyze_query_complexity

logger = get_logger(__name__)
settings = get_settings()


class AgenticRAGAgent(BaseAgent):
    """
    Agentic RAG Agent

    Implements an intelligent RAG system with:
    - Document grading
    - Answer validation
    - Query rewriting
    - Iterative refinement
    """

    def __init__(self):
        """Initialize the agentic RAG agent"""
        super().__init__()
        self.workflow = get_agentic_rag_workflow()
        logger.info("AgenticRAGAgent initialized")

    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process a query using agentic RAG.

        Args:
            state: AgentState with query and context

        Returns:
            Dict with answer, sources, and metadata
        """
        query = state.get("query", "")
        user_id = state.get("user_id")
        conversation_id = state.get("conversation_id")

        logger.info(f"Processing query with agentic RAG: {query[:100]}")

        try:
            # Prepare workflow input
            workflow_input = {
                "query": query,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "iteration_count": 0,
                "max_iterations": settings.agentic_rag_max_iterations,
                "retrieved_docs": [],
                "relevant_docs": [],
                "retrieval_score": 0.0,
                "generated_answer": "",
                "context": "",
                "is_valid": False,
                "validation_issues": [],
                "confidence": 0.0,
                "rewritten_query": None,
                "final_answer": "",
                "sources": [],
                "metadata": {}
            }

            # Run the workflow
            result = self.workflow.invoke(workflow_input)

            # Extract final answer and metadata
            final_answer = result.get("generated_answer", "")
            sources = result.get("sources", [])
            confidence = result.get("confidence", 0.0)
            iterations = result.get("iteration_count", 0)

            logger.info(
                f"Agentic RAG complete: {iterations} iterations, confidence: {confidence:.2f}",
                extra={
                    "query": query[:50],
                    "iterations": iterations,
                    "confidence": confidence,
                    "answer_length": len(final_answer)
                }
            )

            return {
                "answer": final_answer,
                "sources": sources,
                "metadata": {
                    "agent_type": "agentic_rag",
                    "iterations": iterations,
                    "confidence": confidence,
                    "is_valid": result.get("is_valid", False),
                    "retrieval_score": result.get("retrieval_score", 0.0),
                    "relevant_docs_count": len(result.get("relevant_docs", [])),
                    "total_docs_retrieved": len(result.get("retrieved_docs", []))
                }
            }

        except Exception as e:
            logger.error(f"Error in agentic RAG processing: {e}", exc_info=True)
            return {
                "answer": "I apologize, but I encountered an error processing your query. Please try again.",
                "sources": [],
                "metadata": {
                    "agent_type": "agentic_rag",
                    "error": str(e)
                }
            }

    @staticmethod
    def should_use(query: str) -> bool:
        """
        Determine if agentic RAG should be used for this query.

        Args:
            query: User's query

        Returns:
            True if agentic RAG should be used
        """
        if not settings.agentic_rag_enabled:
            return False

        return should_use_agentic_rag(
            query=query,
            min_length=settings.agentic_rag_min_query_length
        )

    @staticmethod
    def analyze_query(query: str) -> Dict[str, Any]:
        """
        Analyze query complexity.

        Args:
            query: User's query

        Returns:
            Analysis dict with complexity info
        """
        return analyze_query_complexity(query)

    def get_name(self) -> str:
        """Get agent name"""
        return "AgenticRAGAgent"

    async def can_handle(self, state: AgentState) -> bool:
        """Check if agentic RAG agent should handle this"""
        return state.get("route") == "AGENTIC_RAG"

    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute agentic RAG pipeline (non-streaming).

        Args:
            state: AgentState with query and context

        Returns:
            Updated AgentState with response
        """
        self._log_start("Agentic RAG pipeline")

        # Initialize RAG state
        if "rag" not in state:
            state["rag"] = RAGState(
                chunks=[],
                context="",
                is_ambiguous=False,
                disambiguation_options=[],
                reformulated_query=None
            )

        try:
            result = self.process(state)

            # Update state with results
            state["final_response"] = result["answer"]
            state["rag"]["chunks"] = result.get("sources", [])
            state["rag"]["context"] = "\n\n".join([s.get("content", "") for s in result.get("sources", [])])

            # Store metadata
            if "metadata" not in state:
                state["metadata"] = {}
            state["metadata"].update(result["metadata"])

            self._log_complete("Agentic RAG pipeline", **result["metadata"])

        except Exception as e:
            self._log_error("Agentic RAG pipeline", e)
            state["error"] = str(e)
            state["final_response"] = "I encountered an error processing your request. Please try again."

        return state

    async def execute_stream(self, state: AgentState) -> AsyncIterator[Tuple[str, AgentState]]:
        """
        Execute agentic RAG pipeline with streaming.

        Note: Agentic RAG involves multiple iterations (retrieve -> grade -> generate -> validate),
        so we can't stream the intermediate generations. We stream the final answer only.

        Args:
            state: AgentState with query and context

        Yields:
            Tuple of (chunk_text, state)
        """
        self._log_start("Agentic RAG pipeline (streaming)")

        # Initialize RAG state
        if "rag" not in state:
            state["rag"] = RAGState(
                chunks=[],
                context="",
                is_ambiguous=False,
                disambiguation_options=[],
                reformulated_query=None
            )

        try:
            # Run the full agentic workflow
            result = self.process(state)

            # Update state with results
            answer = result["answer"]
            state["final_response"] = answer
            state["rag"]["chunks"] = result.get("sources", [])
            state["rag"]["context"] = "\n\n".join([s.get("content", "") for s in result.get("sources", [])])

            # Store metadata
            if "metadata" not in state:
                state["metadata"] = {}
            state["metadata"].update(result["metadata"])

            # Stream the final answer in chunks for better UX
            chunk_size = 50  # characters
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i+chunk_size]
                yield chunk, state

            logger.info(f"Streaming complete, total length={len(answer)}")
            self._log_complete("Agentic RAG pipeline (streaming)", **result["metadata"])

        except Exception as e:
            self._log_error("Agentic RAG pipeline (streaming)", e)
            error_msg = "I encountered an error while processing your request. Please try again."
            state["error"] = str(e)
            state["final_response"] = error_msg
            yield error_msg, state


# Create singleton instance
_agentic_rag_agent = None


def get_agentic_rag_agent() -> AgenticRAGAgent:
    """Get or create the agentic RAG agent instance"""
    global _agentic_rag_agent
    if _agentic_rag_agent is None:
        _agentic_rag_agent = AgenticRAGAgent()
    return _agentic_rag_agent
