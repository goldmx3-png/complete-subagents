"""
Advanced Query Enhancement - Multiple Reformulations
Based on best practices from RAG research and Exa search results

Features:
- Multi-perspective query generation
- Query decomposition for complex queries
- Expansion with related terms
- Hypothetical document embedding (HyDE)
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from src.llm.openrouter_client import OpenRouterClient
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QueryReformulations(BaseModel):
    """Multiple query reformulations"""
    original_query: str = Field(description="Original user query")
    reformulations: List[str] = Field(description="List of query reformulations")
    query_type: str = Field(description="Type of query: simple, complex, comparison, multi_part")
    reasoning: str = Field(description="Reasoning for the reformulations")


class QueryEnhancer:
    """
    Advanced query enhancement with multiple reformulation strategies

    Strategies:
    1. Multi-perspective: Generate queries from different angles
    2. Decomposition: Break complex queries into sub-queries
    3. Expansion: Add related terms and synonyms
    4. HyDE: Generate hypothetical answer for better retrieval
    """

    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        """
        Initialize query enhancer

        Args:
            llm_client: LLM client for query reformulation
        """
        self.llm = llm_client or OpenRouterClient(model=settings.classifier_model)

    async def enhance_query(
        self,
        query: str,
        strategy: str = "multi_perspective",
        num_variations: int = 3
    ) -> Dict:
        """
        Enhance query with multiple reformulations

        Args:
            query: Original user query
            strategy: Enhancement strategy (multi_perspective, decomposition, expansion, hyde)
            num_variations: Number of query variations to generate

        Returns:
            {
                "original_query": str,
                "enhanced_queries": List[str],
                "query_type": str,
                "strategy_used": str,
                "reasoning": str
            }
        """
        logger.info(f"Enhancing query with strategy: {strategy}")

        if strategy == "multi_perspective":
            result = await self._multi_perspective_queries(query, num_variations)
        elif strategy == "decomposition":
            result = await self._decompose_query(query)
        elif strategy == "expansion":
            result = await self._expand_query(query, num_variations)
        elif strategy == "hyde":
            result = await self._hyde_enhancement(query)
        else:
            logger.warning(f"Unknown strategy: {strategy}, falling back to multi_perspective")
            result = await self._multi_perspective_queries(query, num_variations)

        logger.info(f"Generated {len(result['enhanced_queries'])} enhanced queries")

        return result

    async def _multi_perspective_queries(
        self,
        query: str,
        num_variations: int = 3
    ) -> Dict:
        """
        Generate queries from multiple perspectives

        Example:
        Original: "What is the authorization matrix?"
        Variations:
        - "Explain the authorization matrix functionality"
        - "How does the authorization matrix work?"
        - "Authorization matrix features and workflow"
        """
        prompt = f"""Generate {num_variations} alternative ways to phrase this question, each from a different perspective or focus.

Original question: {query}

Requirements:
- Each variation should seek the same information but phrase it differently
- Use different angles: definition, functionality, workflow, use cases, etc.
- Keep variations concise and clear
- Ensure diversity in phrasing

Provide:
- reformulations: List of {num_variations} alternative queries
- query_type: simple, complex, comparison, or multi_part
- reasoning: Brief explanation of your approach
"""

        try:
            result = await self.llm.structured_output(
                prompt=prompt,
                response_model=QueryReformulations,
                temperature=0.7
            )

            return {
                "original_query": query,
                "enhanced_queries": [query] + result.reformulations,  # Include original
                "query_type": result.query_type,
                "strategy_used": "multi_perspective",
                "reasoning": result.reasoning
            }

        except Exception as e:
            logger.error(f"Multi-perspective generation error: {e}")
            return {
                "original_query": query,
                "enhanced_queries": [query],
                "query_type": "simple",
                "strategy_used": "multi_perspective",
                "reasoning": f"Error: {str(e)}"
            }

    async def _decompose_query(self, query: str) -> Dict:
        """
        Decompose complex query into atomic sub-queries

        Example:
        Original: "Compare authorization workflows for bulk vs single payments"
        Sub-queries:
        - "What is the authorization workflow for bulk payments?"
        - "What is the authorization workflow for single payments?"
        - "What are the differences between bulk and single payment authorization?"
        """
        prompt = f"""Analyze this query and break it down into atomic sub-questions if it's complex.

Question: {query}

If the query asks about multiple things, comparisons, or has multiple parts:
- Break it into clear sub-questions
- Each sub-question should be answerable independently
- Cover all aspects of the original question

If the query is simple and atomic:
- Return it as-is with one reformulation for clarity

Provide:
- reformulations: List of sub-queries (or simplified versions if query is simple)
- query_type: simple, complex, comparison, or multi_part
- reasoning: Why you broke it down this way (or why it's simple)
"""

        try:
            result = await self.llm.structured_output(
                prompt=prompt,
                response_model=QueryReformulations,
                temperature=0.3
            )

            return {
                "original_query": query,
                "enhanced_queries": [query] + result.reformulations,
                "query_type": result.query_type,
                "strategy_used": "decomposition",
                "reasoning": result.reasoning
            }

        except Exception as e:
            logger.error(f"Query decomposition error: {e}")
            return {
                "original_query": query,
                "enhanced_queries": [query],
                "query_type": "simple",
                "strategy_used": "decomposition",
                "reasoning": f"Error: {str(e)}"
            }

    async def _expand_query(self, query: str, num_variations: int = 3) -> Dict:
        """
        Expand query with related terms and synonyms

        Example:
        Original: "payment transaction limits"
        Expanded: "payment transaction limits thresholds maximum amounts daily ceiling caps restrictions"
        """
        prompt = f"""Expand this query by adding related terms, synonyms, and variations.

Original query: {query}

Generate {num_variations} expanded versions that include:
- Synonyms for key terms
- Related technical terms
- Common variations
- Domain-specific terminology

Keep expansions natural and focused.

Provide:
- reformulations: List of {num_variations} expanded queries
- query_type: simple, complex, comparison, or multi_part
- reasoning: Brief explanation of expansion approach
"""

        try:
            result = await self.llm.structured_output(
                prompt=prompt,
                response_model=QueryReformulations,
                temperature=0.5
            )

            return {
                "original_query": query,
                "enhanced_queries": [query] + result.reformulations,
                "query_type": result.query_type,
                "strategy_used": "expansion",
                "reasoning": result.reasoning
            }

        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return {
                "original_query": query,
                "enhanced_queries": [query],
                "query_type": "simple",
                "strategy_used": "expansion",
                "reasoning": f"Error: {str(e)}"
            }

    async def _hyde_enhancement(self, query: str) -> Dict:
        """
        Hypothetical Document Embeddings (HyDE)
        Generate a hypothetical answer, then use it for retrieval

        This helps retrieve documents that match the answer style, not just the question
        """
        prompt = f"""Generate a concise, technical answer to this question as if you were an expert:

Question: {query}

Requirements:
- Write 2-3 sentences as if answering the question
- Use technical terminology that would appear in documentation
- Be specific and detailed
- This hypothetical answer will be used to find real documentation

Provide:
- reformulations: List containing the hypothetical answer
- query_type: simple, complex, comparison, or multi_part
- reasoning: Brief explanation of the hypothetical answer approach
"""

        try:
            result = await self.llm.structured_output(
                prompt=prompt,
                response_model=QueryReformulations,
                temperature=0.3
            )

            # Combine original query with hypothetical answer
            enhanced_queries = [query] + result.reformulations

            return {
                "original_query": query,
                "enhanced_queries": enhanced_queries,
                "query_type": result.query_type,
                "strategy_used": "hyde",
                "reasoning": result.reasoning
            }

        except Exception as e:
            logger.error(f"HyDE enhancement error: {e}")
            return {
                "original_query": query,
                "enhanced_queries": [query],
                "query_type": "simple",
                "strategy_used": "hyde",
                "reasoning": f"Error: {str(e)}"
            }

    async def adaptive_enhance(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Adaptively choose enhancement strategy based on query characteristics

        Args:
            query: User query
            conversation_history: Optional conversation context

        Returns:
            Enhancement result with best strategy
        """
        query_lower = query.lower()

        # Detect query type and choose strategy
        if any(word in query_lower for word in ["compare", "difference", "vs", "versus", "better"]):
            # Comparison query -> decompose
            logger.info("Detected comparison query, using decomposition")
            return await self.enhance_query(query, strategy="decomposition")

        elif any(word in query_lower for word in ["what is", "explain", "describe", "define"]):
            # Definitional query -> multi-perspective
            logger.info("Detected definitional query, using multi_perspective")
            return await self.enhance_query(query, strategy="multi_perspective", num_variations=3)

        elif any(word in query_lower for word in ["how to", "steps", "process", "workflow"]):
            # Procedural query -> decomposition + expansion
            logger.info("Detected procedural query, using decomposition")
            return await self.enhance_query(query, strategy="decomposition")

        elif len(query.split()) > 15 or " and " in query_lower:
            # Complex query -> decompose
            logger.info("Detected complex query, using decomposition")
            return await self.enhance_query(query, strategy="decomposition")

        else:
            # Simple query -> multi-perspective for better recall
            logger.info("Default: using multi_perspective enhancement")
            return await self.enhance_query(query, strategy="multi_perspective", num_variations=2)


def merge_enhanced_results(results_list: List[List[Dict]]) -> List[Dict]:
    """
    Merge and deduplicate results from multiple enhanced queries

    Args:
        results_list: List of result lists from different queries

    Returns:
        Merged and deduplicated results, ranked by best score
    """
    # Use chunk_id or text hash for deduplication
    seen = {}

    for results in results_list:
        for result in results:
            chunk_id = result.get("chunk_id", str(hash(result.get("text", ""))))

            if chunk_id not in seen:
                seen[chunk_id] = result
            else:
                # Keep the one with higher score
                existing_score = seen[chunk_id].get("score", 0)
                new_score = result.get("score", 0)
                if new_score > existing_score:
                    seen[chunk_id] = result

    # Sort by score
    merged = list(seen.values())
    merged.sort(key=lambda x: x.get("score", 0), reverse=True)

    logger.info(f"Merged {sum(len(r) for r in results_list)} results into {len(merged)} unique results")

    return merged
