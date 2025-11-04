"""Qdrant vector store client (simplified)"""

from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantStore:
    """Qdrant vector database client"""

    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self.collection_name = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure collection exists"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=settings.embedding_dimension, distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Collection setup error: {e}")

    async def search(self, query_vector: List[float], user_id: str, top_k: int = 20, doc_id: Optional[str] = None) -> List[Dict]:
        """Search vectors with optional doc_id filter (user_id kept for API compatibility but not used in filtering)"""
        try:
            # Build filter conditions (knowledge base is shared across all users)
            filter_conditions = []

            # Add doc_id filter if provided
            if doc_id:
                filter_conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))

            # Search with optional filter
            if filter_conditions:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=Filter(must=filter_conditions),
                    limit=top_k
                )
            else:
                # No filter needed, search all documents
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k
                )
            return [{
                "id": r.id,
                "score": r.score,
                "payload": r.payload
            } for r in results]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    async def upsert(self, points: List[Dict], user_id: str):
        """Upsert points"""
        try:
            qdrant_points = [
                PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload={**p.get("payload", {}), "user_id": user_id}
                )
                for p in points
            ]
            self.client.upsert(collection_name=self.collection_name, points=qdrant_points)
            logger.info(f"Upserted {len(points)} points")
        except Exception as e:
            logger.error(f"Upsert error: {e}")

    async def upsert_chunks(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
        user_id: str,
        doc_id: str
    ):
        """
        Upsert document chunks with embeddings

        Args:
            chunks: List of chunk dicts
            embeddings: List of embedding vectors
            user_id: User ID
            doc_id: Document ID
        """
        import uuid

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = {
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "user_id": user_id,
                    "doc_id": doc_id,
                    "chunk_id": chunk.get("chunk_id"),
                    "text": chunk.get("text"),
                    "chunk_type": chunk.get("chunk_type"),
                    "page": chunk.get("page_numbers", [None])[0] if chunk.get("page_numbers") else None,
                    "metadata": chunk.get("metadata", {})
                }
            }
            points.append(point)

        await self.upsert(points, user_id)

    async def delete_document(self, doc_id: str, user_id: str):
        """
        Delete all chunks for a document

        Args:
            doc_id: Document ID
            user_id: User ID
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                    ]
                )
            )
            logger.info(f"Deleted document: doc_id={doc_id}, user_id={user_id}")
        except Exception as e:
            logger.error(f"Delete document error: {e}")
            raise

    async def get_document_count(self, user_id: str, doc_id: str) -> int:
        """
        Get chunk count for a document

        Args:
            user_id: User ID
            doc_id: Document ID

        Returns:
            Number of chunks
        """
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                    ]
                )
            )
            return result.count
        except Exception as e:
            logger.error(f"Get document count error: {e}")
            return 0
