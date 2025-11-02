"""
Contextual compression for retrieved documents
Reduces retrieved content to only query-relevant parts
Based on LangChain's contextual compression patterns
"""

from typing import List, Dict, Optional
import numpy as np
from src.llm.openrouter_client import OpenRouterClient
from src.vectorstore.embeddings import get_embeddings
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMChainExtractor:
    """
    Use LLM to extract query-relevant parts from documents

    More accurate but slower than embedding-based filtering
    """

    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        """
        Initialize LLM extractor

        Args:
            llm_client: LLM client (uses classifier model for speed)
        """
        self.llm = llm_client or OpenRouterClient(model=settings.classifier_model)

    async def compress_documents(
        self,
        query: str,
        documents: List[Dict]
    ) -> List[Dict]:
        """
        Extract query-relevant parts from documents using LLM

        Args:
            query: Search query
            documents: Retrieved documents

        Returns:
            Compressed documents with only relevant parts
        """
        compressed_docs = []

        for doc in documents:
            try:
                compressed_text = await self._extract_relevant_text(query, doc)

                if compressed_text:
                    compressed_doc = doc.copy()
                    compressed_doc["text"] = compressed_text
                    compressed_doc["compressed"] = True
                    compressed_docs.append(compressed_doc)

            except Exception as e:
                logger.error(f"Compression error: {str(e)}")
                # Keep original on error
                compressed_docs.append(doc)

        logger.info(
            f"Compressed {len(documents)} docs using LLM extraction, "
            f"kept {len(compressed_docs)}"
        )

        return compressed_docs

    async def _extract_relevant_text(self, query: str, doc: Dict) -> str:
        """Extract relevant text from a single document"""
        text = doc.get("text", "")

        if len(text) < 200:
            return text  # Too short to compress

        prompt = f"""Given the following question and context, extract only the parts that are relevant to answering the question.
Preserve complete sentences. If nothing is relevant, return "NOT_RELEVANT".

Question: {query}

Context:
{text}

Relevant text:"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500
            )

            extracted = response.strip()

            if extracted == "NOT_RELEVANT" or len(extracted) < 20:
                return ""  # Not relevant

            return extracted

        except Exception as e:
            logger.error(f"Extraction error: {str(e)}")
            return text  # Return original on error


class EmbeddingsFilter:
    """
    Filter documents based on embedding similarity

    Faster than LLM-based extraction, filters out low-similarity docs
    """

    def __init__(self, similarity_threshold: float = 0.76):
        """
        Initialize embeddings filter

        Args:
            similarity_threshold: Minimum cosine similarity (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.embeddings = get_embeddings()

    async def compress_documents(
        self,
        query: str,
        documents: List[Dict]
    ) -> List[Dict]:
        """
        Filter documents by embedding similarity

        Args:
            query: Search query
            documents: Retrieved documents

        Returns:
            Filtered documents above similarity threshold
        """
        if not documents:
            return []

        # Get query embedding
        query_embedding = np.array(self.embeddings.embed_query(query))

        # Get document embeddings
        doc_texts = [doc.get("text", "") for doc in documents]
        doc_embeddings = self.embeddings.embed_documents(doc_texts)
        doc_embeddings = np.array(doc_embeddings)

        # Calculate similarities
        similarities = self._cosine_similarity(query_embedding, doc_embeddings)

        # Filter by threshold
        filtered_docs = []
        for doc, sim in zip(documents, similarities):
            if sim >= self.similarity_threshold:
                filtered_doc = doc.copy()
                filtered_doc["compression_score"] = float(sim)
                filtered_docs.append(filtered_doc)

        logger.info(
            f"Filtered {len(documents)} docs by embedding similarity "
            f"(threshold={self.similarity_threshold}), kept {len(filtered_docs)}"
        )

        return filtered_docs

    def _cosine_similarity(
        self,
        query_emb: np.ndarray,
        doc_embs: np.ndarray
    ) -> np.ndarray:
        """Calculate cosine similarity"""
        # Normalize
        query_norm = query_emb / np.linalg.norm(query_emb)

        doc_norms = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)

        # Dot product
        similarities = np.dot(doc_norms, query_norm)

        return similarities


class ContextualCompressionRetriever:
    """
    Retriever with contextual compression

    Combines retrieval with compression to return only relevant content
    """

    def __init__(
        self,
        base_retriever,
        compressor_type: str = "embeddings",
        llm_client: Optional[OpenRouterClient] = None,
        similarity_threshold: float = 0.76
    ):
        """
        Initialize compression retriever

        Args:
            base_retriever: Base retriever
            compressor_type: 'embeddings' or 'llm'
            llm_client: Optional LLM client for LLM compression
            similarity_threshold: Threshold for embeddings filter
        """
        self.base_retriever = base_retriever

        if compressor_type == "embeddings":
            self.compressor = EmbeddingsFilter(
                similarity_threshold=similarity_threshold
            )
        elif compressor_type == "llm":
            self.compressor = LLMChainExtractor(llm_client=llm_client)
        else:
            raise ValueError(f"Invalid compressor_type: {compressor_type}")

        self.compressor_type = compressor_type

    async def retrieve(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 10
    ) -> Dict:
        """
        Retrieve and compress documents

        Args:
            query: Search query
            user_id: User ID
            top_k: Number of documents to return

        Returns:
            Dict with compressed chunks
        """
        # Retrieve more candidates
        retrieval_top_k = top_k * 2 if self.compressor_type == "embeddings" else top_k

        try:
            # Base retrieval
            results = await self.base_retriever.retrieve(
                query=query,
                user_id=user_id,
                top_k=retrieval_top_k
            )

            chunks = results.get("chunks", [])

            if not chunks:
                return results

            # Compress
            compressed_chunks = await self.compressor.compress_documents(
                query=query,
                documents=chunks
            )

            # Limit to top_k
            compressed_chunks = compressed_chunks[:top_k]

            results["chunks"] = compressed_chunks
            results["compression_applied"] = True
            results["compressor_type"] = self.compressor_type

            logger.info(
                f"Contextual compression: {len(chunks)} → {len(compressed_chunks)} chunks"
            )

            return results

        except Exception as e:
            logger.error(f"Compression retrieval error: {str(e)}")
            # Fallback to base retrieval
            return await self.base_retriever.retrieve(
                query=query,
                user_id=user_id,
                top_k=top_k
            )
