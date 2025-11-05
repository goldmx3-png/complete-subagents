"""
Cross-Encoder Reranker for improving retrieval precision
Uses BAAI/bge-reranker-large for accurate relevance scoring
"""

from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    Cross-encoder reranker for final relevance scoring

    Features:
    - Batch processing for efficiency
    - Caching of loaded model
    - Configurable top-k selection
    """

    def __init__(self):
        self.model = None
        self.model_name = settings.reranker_model_v2
        self.device = settings.reranker_device_v2
        self.batch_size = settings.reranker_batch_size

    def _load_model(self):
        """Lazy load the reranker model"""
        if self.model is None:
            try:
                logger.info(f"Loading reranker model: {self.model_name}")
                self.model = CrossEncoder(
                    self.model_name,
                    max_length=512,
                    device=self.device
                )
                logger.info(f"Reranker model loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load reranker model: {str(e)}")
                raise

    def preload(self):
        """
        Preload the reranker model (for startup initialization)

        Call this during service startup to avoid first-request latency
        """
        self._load_model()
        logger.info("Reranker model preloaded successfully")

    async def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank documents using cross-encoder

        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of documents to return (uses settings if None)

        Returns:
            Reranked documents with updated scores
        """
        if not documents:
            return []

        # Load model if needed
        self._load_model()

        top_k = top_k or settings.reranker_return_top_k

        try:
            # Prepare query-document pairs
            pairs = []
            for doc in documents:
                text = doc.get("payload", {}).get("text", "")
                pairs.append([query, text])

            logger.info(f"Reranking {len(pairs)} documents")

            # Get scores from cross-encoder
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )

            # Update documents with new scores
            reranked_docs = []
            for doc, score in zip(documents, scores):
                reranked_doc = doc.copy()
                reranked_doc["score"] = float(score)
                reranked_doc["original_score"] = doc.get("score", 0.0)
                reranked_doc["search_type"] = "reranked"
                reranked_docs.append(reranked_doc)

            # Sort by new score and return top-k
            reranked_docs.sort(key=lambda x: x["score"], reverse=True)

            logger.info(f"Reranking complete, returning top {top_k} documents")
            return reranked_docs[:top_k]

        except Exception as e:
            logger.error(f"Reranking error: {str(e)}")
            # Fallback: return original documents
            return documents[:top_k]

    async def rerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[Dict]],
        top_k: Optional[int] = None
    ) -> List[List[Dict]]:
        """
        Rerank multiple query-document sets

        Args:
            queries: List of queries
            documents_list: List of document lists
            top_k: Number of documents to return per query

        Returns:
            List of reranked document lists
        """
        results = []
        for query, docs in zip(queries, documents_list):
            reranked = await self.rerank(query, docs, top_k)
            results.append(reranked)
        return results


# Global reranker instance (lazy loaded)
_reranker_instance = None


def get_reranker() -> Reranker:
    """
    Get or create global reranker instance

    Returns:
        Reranker instance
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance


def preload_reranker():
    """
    Preload the global reranker instance and model

    Call this during service startup to load the model into memory
    """
    logger.info("Preloading reranker model at startup...")
    reranker = get_reranker()
    reranker.preload()
    logger.info("✓ Reranker model preloaded and ready")
