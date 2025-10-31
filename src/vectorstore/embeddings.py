"""Embeddings model (simplified)"""

from typing import List
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingsModel:
    """BGE-M3 embeddings model"""

    def __init__(self):
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
        logger.info("Embedding model loaded")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        embeddings = self.model.encode(texts, batch_size=settings.embedding_batch_size, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]


# Global instance
_embeddings: EmbeddingsModel = None


def get_embeddings() -> EmbeddingsModel:
    """Get or create embeddings model"""
    global _embeddings
    if _embeddings is None:
        _embeddings = EmbeddingsModel()
    return _embeddings
