"""
LLM-based query rewriting for better retrieval and follow-up handling
Industry-standard approach for handling follow-up questions in RAG systems
"""

import re
from typing import List, Dict, Optional
from src.llm.openrouter_client import OpenRouterClient
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QueryRewriter:
    """
    Rewrites queries using LLM for:
    1. Adding domain-specific synonyms
    2. Reformulating follow-up questions into standalone queries
    """

    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        self.llm = llm_client or OpenRouterClient(model=settings.router_model)

    async def rewrite_query(self, query: str) -> str:
        """
        Rewrite query with banking-specific expansions

        Args:
            query: Original user query

        Returns:
            Expanded query with synonyms and related terms
        """
        try:
            rewritten = await self._llm_rewrite(query)
            return rewritten
        except Exception as e:
            logger.error(f"Query rewriting failed: {str(e)}")
            return query

    async def rewrite_with_conversation_history(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Reformulate query into a standalone query using conversation history
        This is the INDUSTRY-STANDARD approach for handling follow-up questions

        Args:
            query: Current user query (may be a follow-up)
            conversation_history: Recent conversation messages

        Returns:
            Standalone query with all necessary context

        Examples:
            History: "what is cutoff master?" → "cutoff master is a validation..."
            Query: "is this applicable for own transfers"
            Output: "is cutoff master applicable for own and within transfers"

            History: "tell me about payment limits" → "payment limits are..."
            Query: "how do I increase them?"
            Output: "how do I increase payment limits"
        """
        # If no history, use original query
        if not conversation_history or len(conversation_history) == 0:
            logger.debug("No conversation history")
            return query

        # Check if query is already standalone
        if self._is_standalone_query(query):
            logger.debug("Query is already standalone")
            return query

        # Reformulate with LLM
        logger.info(f"Reformulating query with history (length={len(conversation_history)})")
        try:
            reformulated = await self._llm_reformulate(query, conversation_history)
            logger.info(f"Reformulated: '{query[:50]}...' → '{reformulated[:50]}...'")
            return reformulated
        except Exception as e:
            logger.error(f"Reformulation failed: {str(e)}")
            return query

    def _is_standalone_query(self, query: str) -> bool:
        """
        Check if query is already standalone (doesn't need reformulation)

        A query needs reformulation if it contains:
        - Pronouns (this, that, it, they)
        - References (same, above, previous)
        - Vague questions without specific entities

        Args:
            query: User query

        Returns:
            True if query is standalone
        """
        query_lower = query.lower().strip()
        query_clean = re.sub(r'[^\w\s]', ' ', query_lower)

        # Follow-up indicators
        followup_indicators = [
            # Pronouns
            'this', 'that', 'these', 'those', 'it', 'they', 'them',
            # References
            'same', 'above', 'previous', 'mentioned',
            # Vague questions
            'is this', 'does this', 'can this', 'will this',
            'is that', 'does that', 'can that', 'will that',
            'how about', 'what about',
            # Continuation
            'also', 'and what about', 'what else'
        ]

        # Check for indicators
        for indicator in followup_indicators:
            if query_clean.startswith(indicator + ' '):
                return False
            if re.search(r'\b' + re.escape(indicator) + r'\b', query_clean):
                return False

        # Check for vague patterns
        vague_patterns = [
            r'^(explain|show|tell|describe|what is|what are|how|why|when|where)\s+(the\s+)?(flow|process|steps|procedure|workflow|method|way)(\s|$)',
            r'^how (does|do|can|to)\s+(i|we|you|it|they)',
            r'^what (is|are)\s+the\s+(flow|process|steps|procedure)',
            r'^(show|give|provide)\s+(me\s+)?(an?\s+)?(overview|summary|details)',
        ]

        for pattern in vague_patterns:
            if re.search(pattern, query_clean):
                logger.debug(f"Matched vague pattern: {pattern}")
                return False

        return True

    async def _llm_reformulate(
        self,
        query: str,
        conversation_history: List[Dict]
    ) -> str:
        """
        Use LLM to reformulate follow-up into standalone query

        Args:
            query: Current query
            conversation_history: Recent messages

        Returns:
            Reformulated standalone query
        """
        # Format conversation history (last 3 exchanges)
        recent_history = conversation_history[-6:]  # Last 3 user-assistant pairs

        history_text = ""
        for msg in recent_history:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Truncate long messages
            if len(content) > 300:
                content = content[:300] + "..."

            if role == "user":
                history_text += f"User: {content}\n"
            elif role == "assistant":
                history_text += f"Assistant: {content}\n"

        prompt = f"""You are a query reformulation assistant for a banking knowledge system. Convert follow-up questions into standalone queries that can be understood without conversation context.

Conversation history:
{history_text}

Current user question: "{query}"

**Instructions:**
1. Analyze conversation history to understand what the user is asking about
2. Identify pronouns (this, that, it, they) or references needing context
3. Replace them with actual entities/topics from conversation
4. Create standalone question with all necessary context
5. Keep question natural and concise
6. Preserve user's intent exactly - don't change what they're asking

**Examples:**

History: "what is cutoff master?" / "cutoff master is a validation mechanism..."
Question: "is this applicable for own transfers"
Standalone: "is cutoff master applicable for own and within transfers"

History: "tell me about payment limits" / "payment limits are..."
Question: "how do I increase them?"
Standalone: "how do I increase payment limits"

History: "what are the transaction fees" / "transaction fees are..."
Question: "does it apply to all currencies?"
Standalone: "do transaction fees apply to all currencies"

**Output ONLY the reformulated query, nothing else.**

Standalone query:"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3  # Low temperature for consistency
            )

            # Clean response
            reformulated = response.strip()

            # Remove quotes if added
            if reformulated.startswith('"') and reformulated.endswith('"'):
                reformulated = reformulated[1:-1]
            if reformulated.startswith("'") and reformulated.endswith("'"):
                reformulated = reformulated[1:-1]

            # Sanity check
            if len(reformulated) > len(query) * 3 or len(reformulated) < 3:
                logger.warning(f"Invalid reformulation: '{reformulated}'")
                return query

            return reformulated

        except Exception as e:
            logger.error(f"Reformulation error: {str(e)}")
            return query

    async def _llm_rewrite(self, query: str) -> str:
        """
        Use LLM to rewrite query with synonyms

        Args:
            query: Original query

        Returns:
            Expanded query
        """
        prompt = f"""You are a banking terminology expert. Expand this query with relevant synonyms and related terms commonly used in banking systems.

Original query: "{query}"

**Instructions:**
1. Keep the original query
2. Add banking synonyms (e.g., "transaction history" → add "transaction queue", "transaction log", "transaction status")
3. Add related terms (e.g., "check" → add "view", "monitor", "display")
4. If asking about status/history, include: queue, log, records, audit trail
5. If asking about transactions, include: in-progress, completed, failed, pending

**Output format:** Return expanded query as single line, no explanations.

Expanded query:"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )

            rewritten = response.strip()

            # Sanity check
            if len(rewritten) > len(query) * 5 or len(rewritten) < len(query):
                return query

            return rewritten

        except Exception as e:
            logger.error(f"Rewriting error: {str(e)}")
            return query


async def rewrite_query(query: str) -> str:
    """
    Convenience function for query rewriting

    Usage:
        rewritten = await rewrite_query("check payment")
        # Returns: "check payment view monitor status transaction"
    """
    rewriter = QueryRewriter()
    return await rewriter.rewrite_query(query)


async def reformulate_followup(
    query: str,
    conversation_history: List[Dict]
) -> str:
    """
    Convenience function for follow-up reformulation

    Usage:
        reformulated = await reformulate_followup(
            "is this applicable for transfers?",
            [{"role": "user", "content": "what is cutoff master?"},
             {"role": "assistant", "content": "cutoff master is..."}]
        )
        # Returns: "is cutoff master applicable for transfers?"
    """
    rewriter = QueryRewriter()
    return await rewriter.rewrite_with_conversation_history(query, conversation_history)
