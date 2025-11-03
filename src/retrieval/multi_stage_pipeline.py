"""
Multi-Stage Retrieval Pipeline
Orchestrates advanced RAG techniques: hybrid search, query enhancement, reranking, and compression

Based on best practices from research:
Stage 1: Query Enhancement (multiple reformulations)
Stage 2: Broad Retrieval (hybrid search: vector + BM25)
Stage 3: Reranking (cross-encoder or MMR)
Stage 4: Contextual Compression (optional)
Stage 5: Final Selection

This provides the highest quality results while managing computational cost.
"""

from typing import List, Dict, Optional
import time
from src.retrieval.query_enhancement import QueryEnhancer, merge_enhanced_results
from src.retrieval.hybrid_retriever import HybridRetriever, BM25Retriever
from src.retrieval.reranker import RerankerPipeline
from src.retrieval.contextual_compression import ContextualCompressionRetriever
from src.retrieval.retriever import RAGRetriever
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultiStageRetrievalPipeline:
    """
    Advanced multi-stage retrieval pipeline

    Features:
    - Adaptive query enhancement based on query type
    - Hybrid search (dense + sparse fusion)
    - Advanced reranking (cross-encoder, LLM, or MMR)
    - Optional contextual compression
    - Comprehensive metrics tracking
    """

    def __init__(
        self,
        base_retriever: Optional[RAGRetriever] = None,
        enable_query_enhancement: bool = True,
        enable_hybrid_search: bool = True,
        enable_reranking: bool = True,
        enable_compression: bool = False,
        reranking_method: str = "mmr",
        compression_method: str = "embeddings"
    ):
        """
        Initialize multi-stage pipeline

        Args:
            base_retriever: Base RAG retriever
            enable_query_enhancement: Enable query enhancement stage
            enable_hybrid_search: Enable hybrid search (vector + BM25)
            enable_reranking: Enable reranking stage
            enable_compression: Enable contextual compression
            reranking_method: Reranking method (cross_encoder, llm, mmr)
            compression_method: Compression method (embeddings, llm)
        """
        self.base_retriever = base_retriever or RAGRetriever()

        # Stage 1: Query Enhancement
        self.enable_query_enhancement = enable_query_enhancement
        if self.enable_query_enhancement:
            self.query_enhancer = QueryEnhancer()
            logger.info("Query enhancement enabled")

        # Stage 2: Hybrid Search
        self.enable_hybrid_search = enable_hybrid_search
        if self.enable_hybrid_search:
            self.hybrid_retriever = HybridRetriever(
                dense_retriever=self.base_retriever,
                fusion_method=settings.hybrid_fusion_method,
                dense_weight=settings.hybrid_dense_weight,
                sparse_weight=settings.hybrid_sparse_weight
            )
            logger.info(f"Hybrid search enabled (fusion={settings.hybrid_fusion_method})")

        # Stage 3: Reranking
        self.enable_reranking = enable_reranking
        self.reranking_method = reranking_method
        if self.enable_reranking:
            self.reranker = RerankerPipeline(
                method=reranking_method,
                lambda_param=settings.mmr_lambda
            )
            logger.info(f"Reranking enabled (method={reranking_method})")

        # Stage 4: Contextual Compression
        self.enable_compression = enable_compression
        if self.enable_compression:
            self.compression_retriever = ContextualCompressionRetriever(
                base_retriever=self.base_retriever,
                compressor_type=compression_method,
                similarity_threshold=settings.compression_similarity_threshold
            )
            logger.info(f"Contextual compression enabled (method={compression_method})")

        # Metrics tracking
        self.metrics = {
            "queries_processed": 0,
            "avg_retrieval_time": 0.0,
            "avg_enhancement_time": 0.0,
            "avg_reranking_time": 0.0,
            "avg_results_before_rerank": 0.0,
            "avg_results_after_rerank": 0.0
        }

    async def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        enable_enhancement: Optional[bool] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Execute multi-stage retrieval pipeline

        Args:
            query: User query
            user_id: User ID
            top_k: Final number of results
            enable_enhancement: Override query enhancement setting
            conversation_history: Optional conversation context

        Returns:
            {
                "chunks": List[Dict],  # Final retrieved chunks
                "query": str,  # Original query
                "enhanced_queries": List[str],  # If enhancement enabled
                "metrics": Dict,  # Performance metrics
                "stages": Dict  # Results from each stage
            }
        """
        start_time = time.time()
        logger.info(f"Multi-stage retrieval: query='{query[:100]}...', top_k={top_k}")

        # Track metrics
        stage_metrics = {}
        stages_results = {}

        # Override enhancement if specified
        use_enhancement = enable_enhancement if enable_enhancement is not None else self.enable_query_enhancement

        # === STAGE 1: Query Enhancement ===
        enhanced_queries = [query]  # Default to original
        enhancement_info = {}

        if use_enhancement:
            enhancement_start = time.time()
            try:
                enhancement_result = await self.query_enhancer.adaptive_enhance(
                    query=query,
                    conversation_history=conversation_history
                )
                enhanced_queries = enhancement_result["enhanced_queries"]
                enhancement_info = {
                    "strategy": enhancement_result["strategy_used"],
                    "query_type": enhancement_result["query_type"],
                    "num_variations": len(enhanced_queries)
                }
                stage_metrics["enhancement_time"] = (time.time() - enhancement_start) * 1000
                logger.info(
                    f"Stage 1: Enhanced into {len(enhanced_queries)} queries "
                    f"(strategy={enhancement_info['strategy']})"
                )
            except Exception as e:
                logger.error(f"Query enhancement failed: {e}")
                enhanced_queries = [query]
                stage_metrics["enhancement_time"] = 0

            stages_results["enhancement"] = enhancement_info

        # === STAGE 2: Broad Retrieval (Hybrid or Standard) ===
        retrieval_start = time.time()
        all_results = []

        # Retrieve for each enhanced query
        for i, enhanced_query in enumerate(enhanced_queries):
            try:
                if self.enable_hybrid_search:
                    # Hybrid retrieval (vector + BM25)
                    results = await self.hybrid_retriever.retrieve(
                        query=enhanced_query,
                        top_k=top_k * 3  # Retrieve more for reranking
                    )
                else:
                    # Standard vector retrieval
                    result_dict = await self.base_retriever.retrieve(
                        query=enhanced_query,
                        user_id=user_id,
                        top_k=top_k * 3
                    )
                    results = result_dict.get("chunks", [])

                all_results.append(results)
                logger.debug(f"Query {i+1}/{len(enhanced_queries)}: Retrieved {len(results)} results")

            except Exception as e:
                logger.error(f"Retrieval failed for query {i+1}: {e}")
                all_results.append([])

        # Merge results from all enhanced queries
        merged_results = merge_enhanced_results(all_results)

        stage_metrics["retrieval_time"] = (time.time() - retrieval_start) * 1000
        stage_metrics["results_before_rerank"] = len(merged_results)

        logger.info(
            f"Stage 2: Retrieved {len(merged_results)} unique results "
            f"(hybrid={self.enable_hybrid_search})"
        )

        stages_results["retrieval"] = {
            "method": "hybrid" if self.enable_hybrid_search else "vector",
            "queries_used": len(enhanced_queries),
            "total_results": len(merged_results)
        }

        # === STAGE 3: Reranking ===
        reranked_results = merged_results

        if self.enable_reranking and merged_results:
            rerank_start = time.time()
            try:
                # Use original query for reranking (most relevant)
                reranked_results = await self.reranker.rerank(
                    query=query,
                    documents=merged_results,
                    top_k=top_k * 2  # Keep more than final top_k
                )
                stage_metrics["reranking_time"] = (time.time() - rerank_start) * 1000
                logger.info(
                    f"Stage 3: Reranked to {len(reranked_results)} results "
                    f"(method={self.reranking_method})"
                )
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                stage_metrics["reranking_time"] = 0

            stages_results["reranking"] = {
                "method": self.reranking_method,
                "results_after_rerank": len(reranked_results)
            }

        # === STAGE 4: Contextual Compression (Optional) ===
        compressed_results = reranked_results

        if self.enable_compression and reranked_results:
            compression_start = time.time()
            try:
                # Compress retrieved documents to only relevant parts
                compressed_results = await self.compression_retriever.compressor.compress_documents(
                    query=query,
                    documents=reranked_results
                )
                stage_metrics["compression_time"] = (time.time() - compression_start) * 1000
                logger.info(f"Stage 4: Compressed to {len(compressed_results)} results")
            except Exception as e:
                logger.error(f"Compression failed: {e}")
                stage_metrics["compression_time"] = 0

            stages_results["compression"] = {
                "enabled": True,
                "results_after_compression": len(compressed_results)
            }

        # === STAGE 5: Final Selection ===
        final_results = compressed_results[:top_k]

        # Calculate total time
        total_time = (time.time() - start_time) * 1000
        stage_metrics["total_time"] = total_time

        logger.info(
            f"Multi-stage retrieval complete: {len(final_results)} results in {total_time:.0f}ms"
        )

        # Update global metrics
        self._update_metrics(stage_metrics)

        return {
            "chunks": final_results,
            "query": query,
            "enhanced_queries": enhanced_queries if use_enhancement else [query],
            "metrics": stage_metrics,
            "stages": stages_results,
            "top_k": top_k
        }

    def _update_metrics(self, stage_metrics: Dict):
        """Update running metrics"""
        self.metrics["queries_processed"] += 1
        n = self.metrics["queries_processed"]

        # Running average for times
        if "enhancement_time" in stage_metrics:
            self.metrics["avg_enhancement_time"] = (
                (self.metrics["avg_enhancement_time"] * (n - 1) + stage_metrics["enhancement_time"]) / n
            )

        if "reranking_time" in stage_metrics:
            self.metrics["avg_reranking_time"] = (
                (self.metrics["avg_reranking_time"] * (n - 1) + stage_metrics["reranking_time"]) / n
            )

        self.metrics["avg_retrieval_time"] = (
            (self.metrics["avg_retrieval_time"] * (n - 1) + stage_metrics["total_time"]) / n
        )

        if "results_before_rerank" in stage_metrics:
            self.metrics["avg_results_before_rerank"] = (
                (self.metrics["avg_results_before_rerank"] * (n - 1) + stage_metrics["results_before_rerank"]) / n
            )

    def get_metrics(self) -> Dict:
        """Get pipeline performance metrics"""
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset metrics tracking"""
        self.metrics = {
            "queries_processed": 0,
            "avg_retrieval_time": 0.0,
            "avg_enhancement_time": 0.0,
            "avg_reranking_time": 0.0,
            "avg_results_before_rerank": 0.0,
            "avg_results_after_rerank": 0.0
        }
        logger.info("Metrics reset")


class AdaptiveMultiStagePipeline(MultiStageRetrievalPipeline):
    """
    Adaptive pipeline that adjusts stages based on query complexity and performance

    Features:
    - Automatically enables/disables stages based on query type
    - Monitors performance and adjusts parameters
    - Falls back to simpler pipeline if advanced stages fail
    """

    async def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Adaptive retrieval with intelligent stage selection

        Args:
            query: User query
            user_id: User ID
            top_k: Number of results
            conversation_history: Optional conversation context

        Returns:
            Retrieval results with adaptive stage selection
        """
        # Analyze query complexity
        query_lower = query.lower()
        word_count = len(query.split())

        # Simple queries: fast path (no enhancement)
        is_simple = (
            word_count < 5 and
            not any(word in query_lower for word in ["compare", "difference", "vs", "and"])
        )

        if is_simple:
            logger.info("Simple query detected, using fast path (no enhancement)")
            return await super().retrieve(
                query=query,
                user_id=user_id,
                top_k=top_k,
                enable_enhancement=False,
                conversation_history=conversation_history
            )

        # Complex queries: full pipeline
        logger.info("Complex query detected, using full pipeline")
        return await super().retrieve(
            query=query,
            user_id=user_id,
            top_k=top_k,
            enable_enhancement=True,
            conversation_history=conversation_history
        )
