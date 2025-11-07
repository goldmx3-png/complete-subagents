"""
Enhanced RAG Retriever with Hybrid Search and Reranking (2025)

Combines:
- Vector search (semantic)
- BM25 search (keyword matching)
- Cross-encoder reranking
- Query enhancement

Expected improvements:
- 8-15% accuracy boost from hybrid search
- 3-5% additional boost from reranking
- Better handling of complex banking queries
"""

import time
from typing import List, Dict, Optional
from src.vectorstore.qdrant_store import QdrantStore
from src.vectorstore.embeddings import EmbeddingsModel
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import get_reranker
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedRAGRetriever:
    """
    Enhanced retriever with hybrid search and reranking

    Features:
    - Hybrid search (vector + BM25) when enabled
    - Cross-encoder reranking when enabled
    - Backward compatible with existing code
    - Configurable via environment variables
    """

    def __init__(
        self,
        vectorstore: Optional[QdrantStore] = None,
        embeddings: Optional[EmbeddingsModel] = None
    ):
        self.vectorstore = vectorstore or QdrantStore()
        self.embeddings = embeddings or EmbeddingsModel()
        self.query_rewriter = QueryRewriter()

        # Initialize hybrid retriever if enabled
        self.hybrid_retriever = None
        if settings.enable_hybrid_search:
            self.hybrid_retriever = HybridRetriever(vectorstore=self.vectorstore)
            logger.info("Hybrid search enabled")

        # Initialize reranker if enabled
        self.reranker = None
        if settings.enable_reranking:
            self.reranker = get_reranker()
            logger.info("Reranking enabled")

    async def retrieve(
        self,
        query: str,
        user_id: str,
        doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Retrieve relevant chunks with optional hybrid search and reranking

        Args:
            query: User query
            user_id: User ID
            doc_id: Optional specific document ID
            top_k: Number of chunks to retrieve (final result count)
            filters: Additional filters

        Returns:
            {
                "chunks": List of retrieved chunks,
                "is_ambiguous": bool,
                "disambiguation_options": List,
                "query": str,
                "expanded_query": str,
                "retrieval_method": str,
                "timing": Dict
            }
        """
        start_time = time.time()
        timing = {}

        logger.info(f"Enhanced RAG retrieval: user={user_id}, doc={doc_id}, query={query[:100]}")

        # Determine final top_k
        final_top_k = top_k or settings.top_k_retrieval

        # If reranking is enabled, retrieve more candidates
        retrieval_top_k = settings.reranker_top_k if (settings.enable_reranking and self.reranker) else final_top_k

        # Step 1: Query enhancement
        rewrite_start = time.time()
        rewritten_query = query
        if settings.use_query_rewriting:
            try:
                rewritten_query = await self.query_rewriter.rewrite_query(query)
                if rewritten_query != query:
                    logger.info(f"Query rewritten: '{query[:50]}...' → '{rewritten_query[:50]}...'")
            except Exception as e:
                logger.warning(f"Query rewriting failed: {str(e)}")
                rewritten_query = query
        timing["query_rewriting"] = (time.time() - rewrite_start) * 1000

        # Step 2: Generate embedding
        embed_start = time.time()
        query_embedding = self.embeddings.embed_query(rewritten_query)
        timing["embedding"] = (time.time() - embed_start) * 1000

        # Step 3: Retrieval (Hybrid or Vector-only)
        retrieval_start = time.time()
        retrieval_method = "vector_only"

        if settings.enable_hybrid_search and self.hybrid_retriever:
            # Use hybrid search
            results = await self.hybrid_retriever.hybrid_search(
                query=rewritten_query,
                query_vector=query_embedding,
                user_id=user_id,
                top_k=retrieval_top_k,
                doc_id=doc_id,
                fusion_method="weighted"  # or "rrf"
            )
            retrieval_method = "hybrid"
            logger.info(f"Hybrid search returned {len(results)} results")
        else:
            # Fallback to vector-only search
            results = await self.vectorstore.search(
                query_vector=query_embedding,
                user_id=user_id,
                top_k=retrieval_top_k,
                doc_id=doc_id
            )
            logger.info(f"Vector search returned {len(results)} results")

        timing["retrieval"] = (time.time() - retrieval_start) * 1000

        # Step 4: Filter by minimum similarity score (only for vector-only search)
        # Skip filtering for hybrid search as fusion scores use different scale
        filter_start = time.time()
        if not (settings.enable_hybrid_search and self.hybrid_retriever):
            pre_filter_count = len(results)
            results = [
                r for r in results
                if r.get("score", 0) >= settings.min_similarity_score
            ]
            if len(results) < pre_filter_count:
                logger.info(f"Filtered by min_similarity_score: {pre_filter_count} → {len(results)}")
        else:
            logger.debug("Skipping min_similarity_score filter for hybrid search (fusion already handles quality)")
        timing["filtering"] = (time.time() - filter_start) * 1000

        # Step 5: Reranking (if enabled)
        if settings.enable_reranking and self.reranker and len(results) > 0:
            rerank_start = time.time()
            try:
                # Use reranker_return_top_k for final count (e.g., 5), not final_top_k (e.g., 20)
                results = await self.reranker.rerank(
                    query=rewritten_query,
                    documents=results,
                    top_k=settings.reranker_return_top_k
                )
                retrieval_method += "+reranked"
                logger.info(f"Reranking complete, returning top {len(results)}")
            except Exception as e:
                logger.error(f"Reranking failed: {str(e)}, using original results")
                results = results[:settings.reranker_return_top_k]
            timing["reranking"] = (time.time() - rerank_start) * 1000
        else:
            # No reranking, just take top-k
            results = results[:final_top_k]

        # Step 6: Check for ambiguity (from original retriever logic)
        is_ambiguous, disambiguation_options = self._check_ambiguity(results)

        if is_ambiguous:
            logger.info(f"Ambiguity detected: {len(disambiguation_options)} options")

        total_duration = (time.time() - start_time) * 1000
        timing["total"] = total_duration

        logger.info(
            f"Enhanced RAG retrieval complete in {total_duration:.0f}ms: "
            f"{len(results)} chunks, method={retrieval_method}"
        )

        return {
            "chunks": results,
            "is_ambiguous": is_ambiguous,
            "disambiguation_options": disambiguation_options,
            "query": query,
            "expanded_query": rewritten_query,
            "top_k": final_top_k,
            "retrieval_method": retrieval_method,
            "timing": timing
        }

    async def retrieve_with_context(
        self,
        query: str,
        user_id: str,
        conversation_history: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict:
        """
        Retrieve with conversation context

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

    def _check_ambiguity(self, results: List[Dict]) -> tuple:
        """
        Simplified ambiguity check
        (Full logic can be imported from original retriever if needed)

        Args:
            results: Search results

        Returns:
            (is_ambiguous, disambiguation_options)
        """
        # Disabled by default for simplicity
        # Can implement full logic from retriever.py if needed
        return False, []

    def format_context(
        self,
        chunks: List[Dict],
        max_chunks: Optional[int] = None,
        use_smart_organization: bool = True
    ) -> str:
        """
        Format retrieved chunks into context for LLM with hierarchical metadata.

        Args:
            chunks: Retrieved chunks
            max_chunks: Maximum chunks to include
            use_smart_organization: Whether to use smart organization

        Returns:
            Formatted context string with hierarchical breadcrumbs
        """
        max_chunks = max_chunks or settings.max_chunks_per_query
        top_chunks = chunks[:max_chunks]

        # Check if section grouping is enabled
        if settings.enable_section_grouping:
            return self._format_context_with_grouping(top_chunks)
        else:
            return self._format_context_simple(top_chunks)

    def _format_context_simple(self, chunks: List[Dict]) -> str:
        """
        Simple formatting without section grouping (backward compatible).

        Args:
            chunks: List of chunks to format

        Returns:
            Formatted context string
        """
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            payload = chunk.get("payload", {})
            text = payload.get("text", "")
            doc_id = payload.get("doc_id", "Document")
            score = chunk.get("score", 0.0)
            metadata = payload.get("metadata", {})

            # Try to get hierarchical context if available
            chunk_text = f"[Chunk {i}] From: {doc_id}"

            if settings.enable_breadcrumb_context:
                hierarchy = metadata.get("hierarchy", {})
                if hierarchy and "full_path" in hierarchy:
                    chunk_text += f"\nModule: {hierarchy['full_path']}"
                else:
                    # Fallback to old header_context
                    header_context = metadata.get("header_context", "")
                    section_title = payload.get("section_title", "")
                    if header_context:
                        chunk_text += f"\nSection: {header_context}"
                    elif section_title and section_title != "General Information":
                        chunk_text += f"\nSection: {section_title}"

            chunk_text += f"\nRelevance: {score:.2f}\n{text}"
            formatted_chunks.append(chunk_text)

        return "\n\n---\n\n".join(formatted_chunks)

    def _format_context_with_grouping(self, chunks: List[Dict]) -> str:
        """
        Format context with h1 > h2 level grouping for better module distinction.

        Args:
            chunks: List of chunks to format

        Returns:
            Formatted context string grouped by h1 > h2 hierarchy
        """
        from collections import defaultdict

        # Group chunks by h1 > h2 hierarchical path
        section_groups = defaultdict(list)
        module_stats = defaultdict(int)  # Track chunks per module

        for chunk in chunks:
            payload = chunk.get("payload", {})
            metadata = payload.get("metadata", {})
            hierarchy = metadata.get("hierarchy", {})

            # Determine grouping key using h1 > h2 hierarchy
            if hierarchy and "breadcrumbs" in hierarchy:
                breadcrumbs = hierarchy["breadcrumbs"]
                if len(breadcrumbs) >= 2:
                    # Use h1 > h2 format (skip "Document Start" if present)
                    h1 = breadcrumbs[1] if breadcrumbs[0].lower() in ['document start', 'document'] else breadcrumbs[0]
                    h2 = breadcrumbs[2] if len(breadcrumbs) > 2 else "General"
                    group_key = f"{h1} > {h2}"
                    module_stats[h1] += 1
                elif len(breadcrumbs) == 1:
                    group_key = breadcrumbs[0]
                    module_stats[breadcrumbs[0]] += 1
                else:
                    group_key = "General"
            elif hierarchy and "root_section" in hierarchy:
                # Fallback to old behavior
                root = hierarchy["root_section"]
                parent = hierarchy.get("parent_section", "")
                if parent and parent != root:
                    group_key = f"{root} > {parent}"
                else:
                    group_key = root
                module_stats[root] += 1
            elif "header_context" in metadata:
                # Fallback: use header_context
                header_context = metadata["header_context"]
                parts = header_context.split(" > ")
                if len(parts) >= 2:
                    group_key = f"{parts[0]} > {parts[1]}"
                    module_stats[parts[0]] += 1
                else:
                    group_key = parts[0] if parts else "General"
                    module_stats[group_key] += 1
            else:
                group_key = "General"

            section_groups[group_key].append(chunk)

        # Add module distribution header if cross-module query detected
        formatted_sections = []
        num_modules = len(module_stats)
        if num_modules > 1:
            module_summary = ", ".join([f"{module} ({count})"
                                       for module, count in sorted(module_stats.items(),
                                                                  key=lambda x: x[1],
                                                                  reverse=True)])
            formatted_sections.append(
                f"\n{'='*70}\n"
                f"📊 MODULE DISTRIBUTION: {module_summary}\n"
                f"{'='*70}\n"
            )

        # Format grouped sections (sort by relevance of best chunk in group)
        for section_name, section_chunks in sorted(section_groups.items(),
                                                    key=lambda x: max(c.get('score', 0) for c in x[1]),
                                                    reverse=True):
            section_header = f"\n{'='*60}\n📂 {section_name}\n{'='*60}\n"
            formatted_sections.append(section_header)

            for i, chunk in enumerate(section_chunks, 1):
                payload = chunk.get("payload", {})
                text = payload.get("text", "")
                doc_id = payload.get("doc_id", "Document")
                score = chunk.get("score", 0.0)
                metadata = payload.get("metadata", {})
                hierarchy = metadata.get("hierarchy", {})

                chunk_text = f"[{section_name} - Chunk {i}]"

                # Add full hierarchical path if available
                if settings.enable_breadcrumb_context and hierarchy:
                    if "full_path" in hierarchy:
                        chunk_text += f"\n📍 Location: {hierarchy['full_path']}"

                    # Add depth indicator
                    if "depth" in hierarchy:
                        chunk_text += f" (Level {hierarchy['depth']})"

                chunk_text += f"\n⚖️  Relevance: {score:.2f}\n\n{text}"
                formatted_sections.append(chunk_text)

        return "\n\n".join(formatted_sections)


async def get_enhanced_retriever() -> EnhancedRAGRetriever:
    """
    Get enhanced retriever instance

    Usage:
        retriever = await get_enhanced_retriever()
        results = await retriever.retrieve(query, user_id)
    """
    return EnhancedRAGRetriever()
