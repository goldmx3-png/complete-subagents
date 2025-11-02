"""
Retrieval Tools for Agentic RAG
Wraps retriever as tools for agent decision-making
Based on LangGraph tool-calling patterns from Context7
"""

from typing import Dict, Optional, List
from src.retrieval.retriever import RAGRetriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrieverTool:
    """
    Wraps RAGRetriever as a tool for agent use

    Based on LangGraph pattern where agents call retrieval tools
    instead of always retrieving. This enables:
    - Agent decides when to retrieve
    - Agent can skip retrieval for general knowledge questions
    - Agent can retry with different queries
    """

    def __init__(self, retriever: Optional[RAGRetriever] = None):
        """
        Initialize retriever tool

        Args:
            retriever: RAGRetriever instance
        """
        self.retriever = retriever or RAGRetriever()
        self.name = "retrieve_documents"
        self.description = (
            "Search for relevant documents in the knowledge base. "
            "Use this when you need specific information from banking policies, "
            "procedures, technical documentation, or operational guidelines. "
            "Input should be a clear search query."
        )

    async def __call__(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        **kwargs
    ) -> Dict:
        """
        Execute retrieval

        Args:
            query: Search query
            user_id: User ID
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments

        Returns:
            Retrieval results
        """
        logger.info(f"RetrieverTool called: query='{query[:100]}...', top_k={top_k}")

        try:
            results = await self.retriever.retrieve(
                query=query,
                user_id=user_id,
                top_k=top_k,
                **kwargs
            )

            chunks = results.get("chunks", [])
            logger.info(f"Retrieved {len(chunks)} documents")

            return {
                "success": True,
                "chunks": chunks,
                "query": query,
                "count": len(chunks)
            }

        except Exception as e:
            logger.error(f"RetrieverTool error: {e}")
            return {
                "success": False,
                "chunks": [],
                "query": query,
                "count": 0,
                "error": str(e)
            }


class ContextAwareRetrieverTool:
    """
    Context-aware retrieval tool that handles conversation history

    Based on LangGraph conversation-aware RAG patterns
    """

    def __init__(self, retriever: Optional[RAGRetriever] = None):
        self.retriever = retriever or RAGRetriever()
        self.name = "retrieve_with_context"
        self.description = (
            "Search for relevant documents with conversation context awareness. "
            "Use this for follow-up questions that reference previous conversation. "
            "Automatically reformulates queries based on conversation history."
        )

    async def __call__(
        self,
        query: str,
        user_id: str,
        conversation_history: Optional[List[Dict]] = None,
        top_k: int = 5,
        **kwargs
    ) -> Dict:
        """
        Execute context-aware retrieval

        Args:
            query: Search query
            user_id: User ID
            conversation_history: Previous conversation
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments

        Returns:
            Retrieval results with reformulated query
        """
        logger.info(
            f"ContextAwareRetrieverTool called: query='{query[:100]}...', "
            f"history_length={len(conversation_history) if conversation_history else 0}"
        )

        try:
            if conversation_history:
                results = await self.retriever.retrieve_with_context(
                    query=query,
                    user_id=user_id,
                    conversation_history=conversation_history,
                    top_k=top_k,
                    **kwargs
                )
            else:
                results = await self.retriever.retrieve(
                    query=query,
                    user_id=user_id,
                    top_k=top_k,
                    **kwargs
                )

            chunks = results.get("chunks", [])
            reformulated_query = results.get("query", query)

            logger.info(
                f"Retrieved {len(chunks)} documents "
                f"(reformulated: {reformulated_query != query})"
            )

            return {
                "success": True,
                "chunks": chunks,
                "original_query": query,
                "reformulated_query": reformulated_query,
                "count": len(chunks)
            }

        except Exception as e:
            logger.error(f"ContextAwareRetrieverTool error: {e}")
            return {
                "success": False,
                "chunks": [],
                "original_query": query,
                "reformulated_query": query,
                "count": 0,
                "error": str(e)
            }


def create_retriever_tool(
    retriever: Optional[RAGRetriever] = None,
    context_aware: bool = False
) -> RetrieverTool | ContextAwareRetrieverTool:
    """
    Factory function to create retriever tool

    Args:
        retriever: Optional RAGRetriever instance
        context_aware: Whether to use context-aware retrieval

    Returns:
        RetrieverTool or ContextAwareRetrieverTool instance

    Example:
        ```python
        # Simple retrieval tool
        tool = create_retriever_tool()
        results = await tool(query="What is the cutoff time?", user_id="user123")

        # Context-aware retrieval tool
        context_tool = create_retriever_tool(context_aware=True)
        results = await context_tool(
            query="What about for USD?",
            user_id="user123",
            conversation_history=[...]
        )
        ```
    """
    if context_aware:
        return ContextAwareRetrieverTool(retriever=retriever)
    else:
        return RetrieverTool(retriever=retriever)


# Tool metadata for LangGraph integration
RETRIEVER_TOOL_METADATA = {
    "name": "retrieve_documents",
    "description": (
        "Search the knowledge base for relevant documents about banking policies, "
        "procedures, technical specifications, and operational guidelines. "
        "Returns chunks of relevant text with metadata."
    ),
    "parameters": {
        "query": {
            "type": "string",
            "description": "Clear search query for document retrieval",
            "required": True
        },
        "top_k": {
            "type": "integer",
            "description": "Number of documents to retrieve (default: 5)",
            "required": False,
            "default": 5
        }
    }
}

CONTEXT_AWARE_TOOL_METADATA = {
    "name": "retrieve_with_context",
    "description": (
        "Search the knowledge base with conversation context awareness. "
        "Automatically reformulates follow-up questions based on conversation history. "
        "Use this when the current question references previous exchanges."
    ),
    "parameters": {
        "query": {
            "type": "string",
            "description": "Current question or follow-up query",
            "required": True
        },
        "top_k": {
            "type": "integer",
            "description": "Number of documents to retrieve (default: 5)",
            "required": False,
            "default": 5
        }
    }
}
