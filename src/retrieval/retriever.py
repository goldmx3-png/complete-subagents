"""
Advanced RAG Retriever with ambiguity detection and query enhancement
"""

import time
from typing import List, Dict, Optional, Tuple
from src.vectorstore.qdrant_store import QdrantStore
from src.vectorstore.embeddings import EmbeddingsModel
from src.retrieval.query_rewriter import QueryRewriter
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGRetriever:
    """
    Advanced retriever with:
    - Query rewriting (LLM-based)
    - Query expansion (dictionary-based)
    - Ambiguity detection
    - Context-aware retrieval
    """

    def __init__(
        self,
        vectorstore: Optional[QdrantStore] = None,
        embeddings: Optional[EmbeddingsModel] = None
    ):
        self.vectorstore = vectorstore or QdrantStore()
        self.embeddings = embeddings or EmbeddingsModel()
        self.query_rewriter = QueryRewriter()

    async def retrieve(
        self,
        query: str,
        user_id: str,
        doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Retrieve relevant chunks for query

        Args:
            query: User query
            user_id: User ID
            doc_id: Optional specific document ID
            top_k: Number of chunks to retrieve
            filters: Additional filters

        Returns:
            {
                "chunks": List of retrieved chunks,
                "is_ambiguous": bool,
                "disambiguation_options": List,
                "query": str,
                "expanded_query": str
            }
        """
        start_time = time.time()
        logger.info(f"RAG retrieval: user={user_id}, doc={doc_id}, query={query[:100]}")

        top_k = top_k or settings.top_k_retrieval

        # Step 1: LLM-based query rewriting (if enabled)
        rewritten_query = query
        if settings.use_query_rewriting:
            try:
                rewritten_query = await self.query_rewriter.rewrite_query(query)
                if rewritten_query != query:
                    logger.info(f"Query rewritten: '{query[:50]}...' → '{rewritten_query[:50]}...'")
            except Exception as e:
                logger.warning(f"Query rewriting failed: {str(e)}")
                rewritten_query = query

        # Step 2: Dictionary-based expansion
        expanded_query = self._expand_query(rewritten_query)
        if expanded_query != rewritten_query:
            logger.debug(f"Query expanded (+{len(expanded_query) - len(rewritten_query)} chars)")

        # Step 3: Generate embedding
        query_embedding = self.embeddings.embed_query(expanded_query)

        # Step 4: Search vector store with optional doc_id filter
        logger.info(f"Searching vector store (top_k={top_k}, doc_id={doc_id})")
        results = await self.vectorstore.search(
            query_vector=query_embedding,
            user_id=user_id,
            top_k=top_k,
            doc_id=doc_id
        )

        logger.info(f"Found {len(results)} results")

        # Step 5: Filter by minimum similarity score
        pre_filter_count = len(results)
        results = [
            r for r in results
            if r.get("score", 0) >= settings.min_similarity_score
        ]

        if len(results) < pre_filter_count:
            logger.info(f"Filtered by min_similarity_score: {pre_filter_count} → {len(results)}")

        # Step 7: Check for ambiguity
        is_ambiguous, disambiguation_options = self._check_ambiguity(results)

        if is_ambiguous:
            logger.info(f"Ambiguity detected: {len(disambiguation_options)} options")

        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"RAG retrieval complete in {duration_ms:.0f}ms: {len(results)} chunks, ambiguous={is_ambiguous}")

        return {
            "chunks": results,
            "is_ambiguous": is_ambiguous,
            "disambiguation_options": disambiguation_options,
            "query": query,
            "expanded_query": expanded_query,
            "top_k": top_k
        }

    async def retrieve_with_context(
        self,
        query: str,
        user_id: str,
        conversation_history: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict:
        """
        Retrieve with conversation context using LLM-based query reformulation
        This is the industry-standard approach for handling follow-up questions

        Args:
            query: User query
            user_id: User ID
            conversation_history: Previous messages
            **kwargs: Additional arguments for retrieve()

        Returns:
            Retrieval results
        """
        start_time = time.time()
        logger.info(f"Retrieving with context (history_length={len(conversation_history) if conversation_history else 0})")

        # Reformulate query into standalone query
        reformulated_query = await self.query_rewriter.rewrite_with_conversation_history(
            query=query,
            conversation_history=conversation_history
        )

        if reformulated_query != query:
            logger.info(f"Query reformulated: '{query[:50]}...' → '{reformulated_query[:50]}...'")

        # Retrieve using reformulated query
        result = await self.retrieve(
            query=reformulated_query,
            user_id=user_id,
            **kwargs
        )

        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Context-aware retrieval complete in {duration_ms:.0f}ms")

        return result

    def _expand_query(self, query: str) -> str:
        """
        Expand query with banking-specific related terms

        Args:
            query: Original query

        Returns:
            Expanded query with synonyms
        """
        query_lower = query.lower()

        # Banking terminology expansions
        expansions = {
            "payment": "payment transfer transaction fund remittance",
            "transaction": "transaction payment transfer fund remittance",
            "account": "account number debit credit balance",
            "balance": "balance amount funds available",
            "transfer": "transfer payment transaction remittance",
            "limit": "limit threshold maximum daily ceiling",
            "cutoff": "cutoff cut-off time deadline validation",
            "validation": "validation check verification verification",
            "currency": "currency currencies code exchange rate",
            "holiday": "holiday calendar cutoff validation payment date",

            # Status/history terms
            "transaction history": "transaction queue log status history audit records summary",
            "transaction status": "transaction queue status processing completed failed pending",
            "payment history": "payment queue log status records audit summary",
            "payment status": "payment queue status processing tracking",

            # Action verbs
            "check": "check view display show monitor query",
            "view": "view check display show monitor",
            "monitor": "monitor check view track status",
            "find": "find search locate view check",

            # Summary terms
            "summary": "summary overview details report statement record",

            # Queue/status
            "queue": "queue list status pending processing",
            "status": "status state condition progress",
            "history": "history log records audit trail",
            "log": "log history records audit trail",
        }

        # Find and apply expansions
        expanded_terms = []
        for term, expansion in expansions.items():
            if term in query_lower:
                expanded_terms.append(expansion)

        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"

        return query

    def _check_ambiguity(
        self,
        results: List[Dict]
    ) -> Tuple[bool, List[Dict]]:
        """
        Detect if results are ambiguous (from different contexts)

        Args:
            results: Search results

        Returns:
            (is_ambiguous, disambiguation_options)
        """
        if len(results) < 2:
            return False, []

        # Group by section/document
        sections = {}

        for result in results[:5]:  # Check top 5
            payload = result.get("payload", {})
            section = payload.get("section_title", "unknown")
            doc = payload.get("doc_id", "unknown")
            key = f"{doc}:{section}"

            if key not in sections:
                sections[key] = {
                    "doc_id": doc,
                    "section": section,
                    "chunks": [],
                    "max_score": 0.0
                }

            sections[key]["chunks"].append(result)
            sections[key]["max_score"] = max(
                sections[key]["max_score"],
                result.get("score", 0)
            )

        # Check if multiple sections with similar scores
        if len(sections) > 1:
            scores = [s["max_score"] for s in sections.values()]
            max_score = max(scores)
            min_score = min(scores)

            # Conservative ambiguity detection:
            # Only mark as ambiguous if:
            # 1. Multiple sections from different documents
            # 2. All scores are similarly low (< 0.55)
            # 3. Scores are very close (within threshold)
            unique_docs = set(s["doc_id"] for s in sections.values())
            good_sections = [s for s in sections.values() if s["max_score"] > 0.45]

            if (len(unique_docs) >= 3 and
                max_score < 0.55 and
                (max_score - min_score) < settings.ambiguity_threshold and
                len(good_sections) >= 3):

                # Create disambiguation options
                options = []
                for i, (key, section_data) in enumerate(sections.items(), 1):
                    first_chunk = section_data["chunks"][0]
                    payload = first_chunk.get("payload", {})

                    options.append({
                        "id": i,
                        "section": section_data["section"],
                        "doc_id": section_data["doc_id"],
                        "preview": payload.get("text", "")[:200] + "...",
                        "score": section_data["max_score"]
                    })

                return True, options

        return False, []

    def format_context(
        self,
        chunks: List[Dict],
        max_chunks: Optional[int] = None
    ) -> str:
        """
        Format retrieved chunks into context for LLM

        Args:
            chunks: Retrieved chunks
            max_chunks: Maximum chunks to include

        Returns:
            Formatted context string
        """
        max_chunks = max_chunks or settings.max_chunks_per_query

        # Take top chunks
        top_chunks = chunks[:max_chunks]

        # Format without citations - just clean text
        formatted_chunks = []
        for chunk in top_chunks:
            payload = chunk.get("payload", {})
            text = payload.get("text", "")
            formatted_chunks.append(text)

        return "\n\n".join(formatted_chunks)

    def format_disambiguation_question(
        self,
        query: str,
        options: List[Dict]
    ) -> str:
        """
        Format disambiguation question for user

        Args:
            query: Original query
            options: Disambiguation options

        Returns:
            Formatted question
        """
        question = f"I found multiple relevant sections for your question: \"{query}\"\n\nWhich one would you like to know about?\n\n"

        for opt in options:
            question += f"{opt['id']}. **{opt['section']}** (from {opt['doc_id']})\n"
            question += f"   Preview: {opt['preview']}\n\n"

        question += "Please specify the section number, or ask your question more specifically."

        return question


async def get_retriever() -> RAGRetriever:
    """
    Get retriever instance

    Usage:
        retriever = await get_retriever()
        results = await retriever.retrieve(query, user_id)
    """
    return RAGRetriever()
