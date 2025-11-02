"""
RAG Agent - Advanced document retrieval and answer generation
Handles policy questions and general information queries
"""

from typing import List, Dict, Optional
from src.agents.base import BaseAgent
from src.agents.shared.state import AgentState, RAGState
from src.llm.openrouter_client import OpenRouterClient
from src.retrieval import RAGRetriever
from src.retrieval.context_organizer import auto_detect_structure, get_adaptive_system_prompt, format_context_note
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

    async def execute_stream(self, state: AgentState):
        """
        Execute RAG pipeline with streaming response

        Yields:
            Tuple of (chunk_text, state) where state is updated at the end
        """
        self._log_start("RAG pipeline (streaming)")

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

            logger.info(f"RAG pipeline (streaming): user={user_id}, query={query[:100]}")

            # Step 1: Retrieve with context-aware reformulation
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

            # Step 2: Handle ambiguity
            if is_ambiguous and disambiguation_options:
                clarification = self.retriever.format_disambiguation_question(
                    query=query,
                    options=disambiguation_options
                )
                state["final_response"] = clarification
                yield clarification, state
                return

            # Step 3: Format context (even if empty - LLM can handle capability questions)
            # No early exit for empty chunks - let LLM decide based on question type
            context = self.retriever.format_context(
                chunks=chunks,
                max_chunks=settings.max_chunks_per_query
            )
            state["rag"]["context"] = context

            # Step 5: Analyze retrieval quality
            if chunks:
                avg_score = sum(c.get("score", 0) for c in chunks[:5]) / min(len(chunks), 5)
                retrieval_quality = "high" if avg_score >= 0.7 else "medium" if avg_score >= 0.4 else "low"
                logger.info(f"Retrieval quality: {retrieval_quality} (avg_score={avg_score:.2f})")
            else:
                retrieval_quality = "none"
                logger.info("No chunks retrieved - relying on LLM general knowledge")

            # Step 6: Stream answer generation
            full_response = ""
            async for chunk in self._generate_answer_stream(
                query=query,
                context=context,
                conversation_history=conversation_history,
                retrieval_quality=retrieval_quality,
                chunks=chunks
            ):
                full_response += chunk
                yield chunk, state

            # Update state with final response
            state["final_response"] = full_response.strip()
            logger.info(f"Streaming complete, total length={len(full_response)}")

            self._log_complete("RAG pipeline (streaming)", chunks_retrieved=len(chunks))

        except Exception as e:
            self._log_error("RAG pipeline (streaming)", e)
            logger.error(f"RAG agent streaming error: {str(e)}", exc_info=True)
            error_msg = "I encountered an error while retrieving information. Please try again."
            state["error"] = str(e)
            state["final_response"] = error_msg
            yield error_msg, state

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

            # Step 3: Format context for LLM (even if empty - LLM can handle capability questions)
            # No early exit for empty chunks - let LLM decide based on question type
            if not chunks:
                logger.info("No chunks found - letting LLM handle from general knowledge")

            #Step 4: Format context for LLM
            context = self.retriever.format_context(
                chunks=chunks,
                max_chunks=settings.max_chunks_per_query
            )
            state["rag"]["context"] = context

            logger.info(f"Context formatted, length={len(context)}")

            # Step 5: Analyze retrieval quality for confidence-based generation
            if chunks:
                avg_score = sum(c.get("score", 0) for c in chunks[:5]) / min(len(chunks), 5)
                retrieval_quality = "high" if avg_score >= 0.7 else "medium" if avg_score >= 0.4 else "low"
                logger.info(f"Retrieval quality: {retrieval_quality} (avg_score={avg_score:.2f})")
            else:
                retrieval_quality = "none"
                logger.info("No chunks retrieved - relying on LLM general knowledge")

            # Step 6: Generate answer with context and quality indicator
            answer = await self._generate_answer(
                query=query,
                context=context,
                conversation_history=conversation_history,
                retrieval_quality=retrieval_quality,
                chunks=chunks
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
        conversation_history: List[Dict],
        retrieval_quality: str = "medium",
        chunks: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate answer using LLM with retrieved context

        Args:
            query: User's question
            context: Retrieved document context
            conversation_history: Previous conversation turns
            retrieval_quality: Quality of retrieval ("high", "medium", or "low")
            chunks: Optional retrieved chunks for structure detection

        Returns:
            Generated answer
        """
        # Detect structure and get adaptive prompt
        if chunks:
            structure = auto_detect_structure(chunks)
            system_prompt = get_adaptive_system_prompt(structure)
            context_note = format_context_note(structure)
        else:
            # Fallback to original prompt
            system_prompt = """You are a banking operations expert. Answer questions directly and confidently as if you personally know this information.

CRITICAL RULES - Response Style:
1. Answer naturally as an expert - NEVER mention "context", "documents", or "provided information"
2. Speak as if you inherently know this - use phrases like "The Cut-off master is..." not "Based on the context, it seems..."
3. NEVER say "based on the information provided" or similar phrases that expose you're reading documents
4. Be definitive and authoritative - you are the expert they're asking
5. NEVER suggest "consult documentation", "ask someone else", or "contact support"
6. NEVER use hedging phrases like "I don't have detailed information" unless you truly have zero information

CRITICAL RULES - Accuracy:
1. Answer using ONLY the factual information available to you (never invent facts)
2. If you don't have information about something, simply say "I don't have information about that"
3. Synthesize information to give complete, coherent explanations
4. When explaining technical terms, provide clear definitions as an expert would

Response Guidelines:
- Natural, conversational tone - like a knowledgeable colleague explaining something
- Direct and confident explanations
- Keep responses focused and concise
- No references to "context", "documents", "provided information", or "based on"
- Just answer as if you know it yourself"""
            context_note = ""

        messages = [{"role": "system", "content": system_prompt}]

        # Add recent conversation history (last 3-4 exchanges)
        for msg in conversation_history[-6:]:
            if msg.get("role") in ["user", "assistant"]:
                content = msg.get("content", "")
                # Truncate very long messages
                if len(content) > 500:
                    content = content[:500] + "..."
                messages.append({"role": msg["role"], "content": content})

        # Add current query with context and quality-based instructions
        quality_instruction = ""
        if retrieval_quality == "high":
            quality_instruction = "Answer with full confidence."
        elif retrieval_quality == "medium":
            quality_instruction = "Synthesize the available information to give a complete answer."
        else:
            quality_instruction = "Only answer if you have clear information about this."

        user_message = f"""Available knowledge:
{context_note}
{context}

---

Question: {query}

Instructions: {quality_instruction} Remember to answer naturally as an expert - never mention "context", "documents", or "provided information" in your response."""

        messages.append({"role": "user", "content": user_message})

        try:
            answer = await self.llm.chat(
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                stream=False  # Non-streaming by default
            )
            return answer.strip()
        except Exception as e:
            logger.error(f"Answer generation error: {str(e)}")
            return "I encountered an issue generating an answer. Please try rephrasing your question."

    async def _generate_answer_stream(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict],
        retrieval_quality: str = "medium",
        chunks: Optional[List[Dict]] = None
    ):
        """
        Generate answer using LLM with streaming

        Args:
            query: User's question
            context: Retrieved document context
            conversation_history: Previous conversation turns
            retrieval_quality: Quality of retrieval ("high", "medium", or "low")
            chunks: Optional retrieved chunks for structure detection

        Yields:
            Streamed content chunks
        """
        # Detect structure and get adaptive prompt
        if chunks:
            structure = auto_detect_structure(chunks)
            system_prompt = get_adaptive_system_prompt(structure)
            context_note = format_context_note(structure)
        else:
            # Fallback to original prompt
            system_prompt = """You are a banking operations expert. Answer questions directly and confidently as if you personally know this information.

CRITICAL RULES - Response Style:
1. Answer naturally as an expert - NEVER mention "context", "documents", or "provided information"
2. Speak as if you inherently know this - use phrases like "The Cut-off master is..." not "Based on the context, it seems..."
3. NEVER say "based on the information provided" or similar phrases that expose you're reading documents
4. Be definitive and authoritative - you are the expert they're asking
5. NEVER suggest "consult documentation", "ask someone else", or "contact support"
6. NEVER use hedging phrases like "I don't have detailed information" unless you truly have zero information

CRITICAL RULES - Accuracy:
1. Answer using ONLY the factual information available to you (never invent facts)
2. If you don't have information about something, simply say "I don't have information about that"
3. Synthesize information to give complete, coherent explanations
4. When explaining technical terms, provide clear definitions as an expert would

Response Guidelines:
- Natural, conversational tone - like a knowledgeable colleague explaining something
- Direct and confident explanations
- Keep responses focused and concise
- No references to "context", "documents", "provided information", or "based on"
- Just answer as if you know it yourself"""
            context_note = ""

        messages = [{"role": "system", "content": system_prompt}]

        # Add recent conversation history (last 3-4 exchanges)
        for msg in conversation_history[-6:]:
            if msg.get("role") in ["user", "assistant"]:
                content = msg.get("content", "")
                # Truncate very long messages
                if len(content) > 500:
                    content = content[:500] + "..."
                messages.append({"role": msg["role"], "content": content})

        # Add current query with context and quality-based instructions
        quality_instruction = ""
        if retrieval_quality == "high":
            quality_instruction = "Answer with full confidence."
        elif retrieval_quality == "medium":
            quality_instruction = "Synthesize the available information to give a complete answer."
        else:
            quality_instruction = "Only answer if you have clear information about this."

        user_message = f"""Available knowledge:
{context_note}
{context}

---

Question: {query}

Instructions: {quality_instruction} Remember to answer naturally as an expert - never mention "context", "documents", or "provided information" in your response."""

        messages.append({"role": "user", "content": user_message})

        try:
            # Stream from LLM
            async for chunk in await self.llm.chat(
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                stream=True
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming answer generation error: {str(e)}")
            yield "I encountered an issue generating an answer. Please try rephrasing your question."

    def get_name(self) -> str:
        return "RAGAgent"
