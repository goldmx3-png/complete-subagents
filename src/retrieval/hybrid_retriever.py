"""
Hybrid retrieval combining dense vector search + sparse BM25
Based on best practices from LangChain and LlamaIndex
"""

from typing import List, Dict, Optional
import math
from collections import Counter
import re
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    """
    BM25 (Best Matching 25) sparse retrieval algorithm

    Complements dense vector search by capturing exact keyword matches
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25 retriever

        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Document length normalization parameter (0.75 typical)
        """
        self.k1 = k1
        self.b = b
        self.corpus = []  # List of documents
        self.doc_freqs = []  # Document frequencies
        self.idf = {}  # Inverse document frequencies
        self.avg_doc_len = 0
        self.doc_lens = []

    def index_documents(self, documents: List[Dict]):
        """
        Index documents for BM25 retrieval

        Args:
            documents: List of document dicts with 'text' field
        """
        self.corpus = documents

        # Tokenize all documents
        tokenized_docs = []
        total_len = 0

        for doc in documents:
            tokens = self._tokenize(doc.get("text", ""))
            tokenized_docs.append(tokens)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len

        # Calculate average document length
        self.avg_doc_len = total_len / len(documents) if documents else 0

        # Calculate document frequencies and IDF
        df = Counter()
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1

        self.doc_freqs = tokenized_docs

        # Calculate IDF for each term
        num_docs = len(documents)
        for term, freq in df.items():
            idf = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = idf

        logger.info(f"Indexed {num_docs} documents for BM25 search")

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search using BM25 algorithm

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of documents with scores
        """
        if not self.corpus:
            logger.warning("BM25 corpus is empty")
            return []

        query_tokens = self._tokenize(query)

        # Calculate BM25 scores for all documents
        scores = []
        for idx, doc_tokens in enumerate(self.doc_freqs):
            score = self._calculate_bm25_score(query_tokens, doc_tokens, idx)
            scores.append((idx, score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Get top-k results
        results = []
        for idx, score in scores[:top_k]:
            if score > 0:  # Only include documents with positive scores
                doc = self.corpus[idx].copy()
                doc["score"] = score
                doc["retrieval_method"] = "bm25"
                results.append(doc)

        return results

    def _calculate_bm25_score(
        self,
        query_tokens: List[str],
        doc_tokens: List[str],
        doc_idx: int
    ) -> float:
        """Calculate BM25 score for a document"""
        score = 0.0
        doc_len = self.doc_lens[doc_idx]

        # Count term frequencies in document
        doc_term_freqs = Counter(doc_tokens)

        for term in query_tokens:
            if term not in self.idf:
                continue

            # Get term frequency in document
            tf = doc_term_freqs.get(term, 0)

            # BM25 formula
            idf = self.idf[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_len / self.avg_doc_len)
            )

            score += idf * (numerator / denominator)

        return score

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]{2,}\b', text)
        return tokens


class HybridRetriever:
    """
    Hybrid retriever combining dense vector search and sparse BM25

    Fusion methods:
    - RRF (Reciprocal Rank Fusion): Combines rankings
    - Weighted: Weighted combination of scores
    """

    def __init__(
        self,
        dense_retriever=None,
        bm25_retriever: Optional[BM25Retriever] = None,
        fusion_method: str = "rrf",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever

        Args:
            dense_retriever: Dense vector retriever
            bm25_retriever: BM25 sparse retriever
            fusion_method: 'rrf' or 'weighted'
            dense_weight: Weight for dense scores (if weighted fusion)
            sparse_weight: Weight for sparse scores (if weighted fusion)
        """
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.fusion_method = fusion_method
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def index_documents(self, documents: List[Dict]):
        """Index documents in both retrievers"""
        # Index in BM25
        self.bm25_retriever.index_documents(documents)

        # Dense retriever should be indexed separately (in vector DB)
        logger.info(f"Indexed {len(documents)} documents in hybrid retriever")

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Hybrid retrieval

        Args:
            query: Search query
            top_k: Final number of results
            dense_top_k: Number of dense results to retrieve
            sparse_top_k: Number of sparse results to retrieve

        Returns:
            Fused results
        """
        # Retrieve more candidates from each method
        dense_top_k = dense_top_k or top_k * 2
        sparse_top_k = sparse_top_k or top_k * 2

        # Get dense results (vector search)
        dense_results = []
        if self.dense_retriever:
            try:
                dense_results = await self.dense_retriever.retrieve(
                    query=query,
                    user_id="hybrid",  # Placeholder
                    top_k=dense_top_k
                )
                dense_results = dense_results.get("chunks", [])

                # Mark as dense
                for result in dense_results:
                    result["retrieval_method"] = "dense"

            except Exception as e:
                logger.error(f"Dense retrieval error: {str(e)}")

        # Get sparse results (BM25)
        sparse_results = self.bm25_retriever.search(query, top_k=sparse_top_k)

        # Fuse results
        if self.fusion_method == "rrf":
            fused_results = self._reciprocal_rank_fusion(
                dense_results, sparse_results, top_k
            )
        else:  # weighted
            fused_results = self._weighted_fusion(
                dense_results, sparse_results, top_k
            )

        logger.info(
            f"Hybrid retrieval: {len(dense_results)} dense + "
            f"{len(sparse_results)} sparse → {len(fused_results)} fused"
        )

        return fused_results

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        top_k: int,
        k: int = 60
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF)

        RRF score = sum(1 / (k + rank))

        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            top_k: Number of final results
            k: RRF constant (usually 60)

        Returns:
            Fused and ranked results
        """
        # Build a dict of chunk_id -> document
        doc_scores = {}
        doc_map = {}

        # Add dense results
        for rank, doc in enumerate(dense_results):
            chunk_id = doc.get("chunk_id", str(hash(doc.get("text", ""))))
            rrf_score = 1.0 / (k + rank + 1)

            if chunk_id not in doc_scores:
                doc_scores[chunk_id] = 0
                doc_map[chunk_id] = doc

            doc_scores[chunk_id] += rrf_score

        # Add sparse results
        for rank, doc in enumerate(sparse_results):
            chunk_id = doc.get("chunk_id", str(hash(doc.get("text", ""))))
            rrf_score = 1.0 / (k + rank + 1)

            if chunk_id not in doc_scores:
                doc_scores[chunk_id] = 0
                doc_map[chunk_id] = doc

            doc_scores[chunk_id] += rrf_score

        # Sort by RRF score
        sorted_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results
        results = []
        for chunk_id, score in sorted_ids[:top_k]:
            doc = doc_map[chunk_id].copy()
            doc["score"] = score
            doc["retrieval_method"] = "hybrid_rrf"
            results.append(doc)

        return results

    def _weighted_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Weighted score fusion

        Final score = dense_weight * dense_score + sparse_weight * sparse_score

        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            top_k: Number of final results

        Returns:
            Fused and ranked results
        """
        # Normalize scores to [0, 1]
        dense_results = self._normalize_scores(dense_results)
        sparse_results = self._normalize_scores(sparse_results)

        # Build score dict
        doc_scores = {}
        doc_map = {}

        # Add dense results
        for doc in dense_results:
            chunk_id = doc.get("chunk_id", str(hash(doc.get("text", ""))))
            score = doc.get("score", 0) * self.dense_weight

            doc_scores[chunk_id] = score
            doc_map[chunk_id] = doc

        # Add sparse results
        for doc in sparse_results:
            chunk_id = doc.get("chunk_id", str(hash(doc.get("text", ""))))
            score = doc.get("score", 0) * self.sparse_weight

            if chunk_id in doc_scores:
                doc_scores[chunk_id] += score
            else:
                doc_scores[chunk_id] = score
                doc_map[chunk_id] = doc

        # Sort by combined score
        sorted_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results
        results = []
        for chunk_id, score in sorted_ids[:top_k]:
            doc = doc_map[chunk_id].copy()
            doc["score"] = score
            doc["retrieval_method"] = "hybrid_weighted"
            results.append(doc)

        return results

    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """Normalize scores to [0, 1]"""
        if not results:
            return results

        scores = [r.get("score", 0) for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores are the same
            for r in results:
                r["score"] = 1.0
        else:
            # Normalize to [0, 1]
            for r in results:
                score = r.get("score", 0)
                r["score"] = (score - min_score) / (max_score - min_score)

        return results
