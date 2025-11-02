"""
Document re-ranking for improved relevance
Reorders retrieved documents using cross-encoder models or LLM-based scoring
"""

from typing import List, Dict, Optional
import numpy as np
from src.vectorstore.embeddings import get_embeddings
from src.llm.openrouter_client import OpenRouterClient
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Re-rank using cross-attention between query and document

    More accurate than bi-encoder (separate embeddings) but slower
    """

    def __init__(self):
        """Initialize cross-encoder reranker"""
        # Use same embedding model for now (can be upgraded to dedicated cross-encoder)
        self.embeddings = get_embeddings()

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Re-rank documents by relevance

        Args:
            query: Search query
            documents: Retrieved documents
            top_k: Optional limit on returned documents

        Returns:
            Re-ranked documents
        """
        if not documents:
            return documents

        # Calculate relevance scores
        scored_docs = []
        query_embedding = np.array(self.embeddings.embed_query(query))

        for doc in documents:
            text = doc.get("text", "")

            # Get document embedding
            doc_embedding = np.array(self.embeddings.embed_query(text))

            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)

            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(similarity)
            scored_docs.append(doc_copy)

        # Sort by rerank score
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Limit to top_k
        if top_k:
            scored_docs = scored_docs[:top_k]

        logger.info(f"Re-ranked {len(documents)} documents (cross-encoder)")

        return scored_docs

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        norm1 = emb1 / np.linalg.norm(emb1)
        norm2 = emb2 / np.linalg.norm(emb2)
        return float(np.dot(norm1, norm2))


class LLMReranker:
    """
    Re-rank using LLM to score document relevance

    Most accurate but slowest method
    """

    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        """
        Initialize LLM reranker

        Args:
            llm_client: LLM client (uses classifier model)
        """
        self.llm = llm_client or OpenRouterClient(model=settings.classifier_model)

    async def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Re-rank documents using LLM scoring

        Args:
            query: Search query
            documents: Retrieved documents
            top_k: Optional limit on returned documents

        Returns:
            Re-ranked documents
        """
        if not documents:
            return documents

        # Score each document
        scored_docs = []

        for doc in documents:
            try:
                score = await self._score_document(query, doc)
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = score
                scored_docs.append(doc_copy)

            except Exception as e:
                logger.error(f"Scoring error: {str(e)}")
                # Keep original score
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = doc.get("score", 0.5)
                scored_docs.append(doc_copy)

        # Sort by rerank score
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Limit to top_k
        if top_k:
            scored_docs = scored_docs[:top_k]

        logger.info(f"Re-ranked {len(documents)} documents (LLM)")

        return scored_docs

    async def _score_document(self, query: str, doc: Dict) -> float:
        """
        Score a single document's relevance to the query

        Args:
            query: Search query
            doc: Document to score

        Returns:
            Relevance score (0-1)
        """
        text = doc.get("text", "")[:1000]  # Limit for speed

        prompt = f"""Rate how relevant the following document is to the query on a scale of 0 to 10.
Return only a single number (0-10), no explanation.

Query: {query}

Document:
{text}

Relevance score (0-10):"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )

            # Parse score
            score_str = response.strip()
            score = float(score_str) / 10.0  # Normalize to 0-1

            return max(0.0, min(1.0, score))  # Clamp to [0, 1]

        except Exception as e:
            logger.error(f"Score parsing error: {str(e)}")
            return 0.5  # Default mid-score


class MMRReranker:
    """
    Maximal Marginal Relevance (MMR) re-ranking

    Balances relevance and diversity to avoid redundant results
    """

    def __init__(self, lambda_param: float = 0.7):
        """
        Initialize MMR reranker

        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        """
        self.lambda_param = lambda_param
        self.embeddings = get_embeddings()

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Re-rank using MMR

        Args:
            query: Search query
            documents: Retrieved documents
            top_k: Number of documents to return

        Returns:
            Re-ranked diverse documents
        """
        if not documents:
            return documents

        top_k = top_k or len(documents)

        # Get embeddings
        query_embedding = np.array(self.embeddings.embed_query(query))

        doc_texts = [doc.get("text", "") for doc in documents]
        doc_embeddings = [
            np.array(self.embeddings.embed_query(text))
            for text in doc_texts
        ]

        # Calculate relevance scores
        relevance_scores = [
            self._cosine_similarity(query_embedding, doc_emb)
            for doc_emb in doc_embeddings
        ]

        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(documents)))

        for _ in range(min(top_k, len(documents))):
            if not remaining_indices:
                break

            # Calculate MMR scores for remaining documents
            mmr_scores = []
            for idx in remaining_indices:
                relevance = relevance_scores[idx]

                # Calculate max similarity to already selected documents
                if selected_indices:
                    similarities_to_selected = [
                        self._cosine_similarity(
                            doc_embeddings[idx],
                            doc_embeddings[selected_idx]
                        )
                        for selected_idx in selected_indices
                    ]
                    max_similarity = max(similarities_to_selected)
                else:
                    max_similarity = 0

                # MMR formula
                mmr_score = (
                    self.lambda_param * relevance -
                    (1 - self.lambda_param) * max_similarity
                )

                mmr_scores.append((idx, mmr_score))

            # Select document with highest MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Build result
        reranked_docs = []
        for idx in selected_indices:
            doc = documents[idx].copy()
            doc["rerank_score"] = relevance_scores[idx]
            doc["mmr_selected"] = True
            reranked_docs.append(doc)

        logger.info(f"Re-ranked {len(documents)} documents using MMR (diversity)")

        return reranked_docs

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        norm1 = emb1 / (np.linalg.norm(emb1) + 1e-10)
        norm2 = emb2 / (np.linalg.norm(emb2) + 1e-10)
        return float(np.dot(norm1, norm2))


class RerankerPipeline:
    """
    Configurable re-ranking pipeline

    Supports multiple reranking strategies
    """

    def __init__(
        self,
        method: str = "cross_encoder",
        llm_client: Optional[OpenRouterClient] = None,
        lambda_param: float = 0.7
    ):
        """
        Initialize reranker pipeline

        Args:
            method: 'cross_encoder', 'llm', or 'mmr'
            llm_client: Optional LLM client for LLM reranking
            lambda_param: Lambda for MMR (relevance vs diversity)
        """
        self.method = method

        if method == "cross_encoder":
            self.reranker = CrossEncoderReranker()
        elif method == "llm":
            self.reranker = LLMReranker(llm_client=llm_client)
        elif method == "mmr":
            self.reranker = MMRReranker(lambda_param=lambda_param)
        else:
            raise ValueError(f"Invalid reranking method: {method}")

    async def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Re-rank documents

        Args:
            query: Search query
            documents: Retrieved documents
            top_k: Number of documents to return

        Returns:
            Re-ranked documents
        """
        if self.method == "llm":
            # Async reranking
            return await self.reranker.rerank(query, documents, top_k)
        else:
            # Sync reranking
            return self.reranker.rerank(query, documents, top_k)
