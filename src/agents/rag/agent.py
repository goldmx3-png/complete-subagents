"""
RAG Agent - Advanced document retrieval and answer generation
Handles policy questions and general information queries
"""

from typing import List, Dict, Optional
from src.agents.base import BaseAgent
from src.agents.shared.state import AgentState, RAGState
from src.llm.openrouter_client import OpenRouterClient
from src.retrieval import RAGRetriever
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGAgent(BaseAgent):
    """
    Advanced RAG Agent with:
    - Query reformulation for follow-up questions
    - Ambiguity detection with clarification
    - Context-aware answer generation
    """

    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        llm_client: Optional[OpenRouterClient] = None
    ):
        super().__init__()
        self.retriever = retriever or RAGRetriever()
        self.llm = llm_client or OpenRouterClient(model=settings.main_model)

    async def can_handle(self, state: AgentState) -> bool:
        """Check if RAG agent should handle this"""
        return state.get("route") in ["RAG_ONLY", "RAG_THEN_API"]

    async def execute(self, state: AgentState) -> AgentState:
        """Execute advanced RAG pipeline"""
        self._log_start("RAG pipeline")

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
            query = state["query"]
            user_id = state["user_id"]
            conversation_history = state.get("conversation_history", [])

            logger.info(f"RAG pipeline: user={user_id}, query={query[:100]}")

            # Step 1: Retrieve with context-aware reformulation
            # This handles follow-up questions automatically
            if conversation_history:
                logger.info("Using context-aware retrieval for follow-up handling")
                retrieval_result = await self.retriever.retrieve_with_context(
                    query=query,
                    user_id=user_id,
                    conversation_history=conversation_history
                )
            else:
                retrieval_result = await self.retriever.retrieve(
                    query=query,
                    user_id=user_id
                )

            # Extract retrieval results
            chunks = retrieval_result.get("chunks", [])
            is_ambiguous = retrieval_result.get("is_ambiguous", False)
            disambiguation_options = retrieval_result.get("disambiguation_options", [])

            state["rag"]["chunks"] = chunks
            state["rag"]["is_ambiguous"] = is_ambiguous
            state["rag"]["disambiguation_options"] = disambiguation_options
            state["rag"]["reformulated_query"] = retrieval_result.get("query")

            logger.info(f"Retrieved {len(chunks)} chunks, ambiguous={is_ambiguous}")

            # Step 2: Handle ambiguity with natural clarification
            if is_ambiguous and disambiguation_options:
                clarification = self.retriever.format_disambiguation_question(
                    query=query,
                    options=disambiguation_options
                )
                state["final_response"] = clarification
                logger.info("Requesting clarification due to ambiguity")
                self._log_complete("RAG pipeline", status="needs_clarification")
                return state

            # Step 3: Check if we found relevant information
            if not chunks:
                state["final_response"] = "I don't have information about that in my knowledge base. Could you rephrase your question or ask about something else?"
                logger.warning("No relevant chunks found")
                self._log_complete("RAG pipeline", status="no_results")
                return state

            # Step 4: Format context for LLM
            context = self.retriever.format_context(
                chunks=chunks,
                max_chunks=settings.max_chunks_per_query
            )
            state["rag"]["context"] = context

            logger.info(f"Context formatted, length={len(context)}")

            # Step 5: Generate answer with context
            answer = await self._generate_answer(
                query=query,
                context=context,
                conversation_history=conversation_history
            )

            state["final_response"] = answer
            logger.info(f"Answer generated, length={len(answer)}")

            self._log_complete("RAG pipeline", chunks_retrieved=len(chunks))

        except Exception as e:
            self._log_error("RAG pipeline", e)
            logger.error(f"RAG agent error: {str(e)}", exc_info=True)
            state["error"] = str(e)
            state["final_response"] = "I encountered an error while retrieving information. Please try again."

        return state

    async def _generate_answer(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict]
    ) -> str:
        """
        Generate answer using LLM with retrieved context

        Args:
            query: User's question
            context: Retrieved document context
            conversation_history: Previous conversation turns

        Returns:
            Generated answer
        """
        system_prompt = """You are a helpful banking assistant. Answer questions naturally and conversationally based on the provided context.

Your role is to help users understand banking operations and functionalities.

Important guidelines:
- Use ONLY the information provided in the context
- Answer in a natural, conversational tone - like you're talking to a colleague
- Be direct and helpful without being overly formal
- If the context doesn't contain the answer, politely say "I don't know" or "I don't have that information"
- Don't make up information or make assumptions beyond what's in the context
- Keep responses concise and focused on the user's question"""

        messages = [{"role": "system", "content": system_prompt}]

        # Add recent conversation history (last 3-4 exchanges)
        for msg in conversation_history[-6:]:
            if msg.get("role") in ["user", "assistant"]:
                content = msg.get("content", "")
                # Truncate very long messages
                if len(content) > 500:
                    content = content[:500] + "..."
                messages.append({"role": msg["role"], "content": content})

        # Add current query with context
        user_message = f"""Context from documents:

{context}

---

Question: {query}

Please answer the question based ONLY on the context provided above."""

        messages.append({"role": "user", "content": user_message})

        try:
            answer = await self.llm.chat(
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens
            )
            return answer.strip()
        except Exception as e:
            logger.error(f"Answer generation error: {str(e)}")
            return "I encountered an issue generating an answer. Please try rephrasing your question."

    def get_name(self) -> str:
        return "RAGAgent"
