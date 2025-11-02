"""
Multi-vector retrieval supporting summaries, chunks, and hypothetical questions
Based on LangChain's MultiVectorRetriever pattern
"""

from typing import List, Dict, Optional
import uuid
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultiVectorStore:
    """
    Storage for multi-vector retrieval

    Stores:
    - Child documents (indexed): summaries, chunks, questions
    - Parent documents (retrieved): full original documents
    """

    def __init__(self):
        """Initialize multi-vector store"""
        self.child_index = {}  # chunk_id -> child document
        self.parent_store = {}  # parent_id -> parent document
        self.child_to_parent = {}  # child_id -> parent_id

    def add_documents(
        self,
        parent_docs: List[Dict],
        child_docs: List[Dict]
    ):
        """
        Add parent and child documents

        Args:
            parent_docs: Full parent documents
            child_docs: Child documents (summaries/chunks/questions) with parent_id in metadata
        """
        # Store parent documents
        for parent in parent_docs:
            parent_id = parent.get("chunk_id") or str(uuid.uuid4())
            parent["chunk_id"] = parent_id
            self.parent_store[parent_id] = parent

        # Index child documents
        for child in child_docs:
            child_id = child.get("chunk_id") or str(uuid.uuid4())
            child["chunk_id"] = child_id

            parent_id = child.get("metadata", {}).get("parent_id")
            if not parent_id:
                logger.warning(f"Child {child_id} has no parent_id")
                continue

            self.child_index[child_id] = child
            self.child_to_parent[child_id] = parent_id

        logger.info(
            f"Added {len(parent_docs)} parents and {len(child_docs)} children "
            f"to multi-vector store"
        )

    def get_parent_documents(self, child_ids: List[str]) -> List[Dict]:
        """
        Retrieve parent documents from child IDs

        Args:
            child_ids: List of child document IDs

        Returns:
            List of parent documents
        """
        parent_ids = set()
        for child_id in child_ids:
            parent_id = self.child_to_parent.get(child_id)
            if parent_id:
                parent_ids.add(parent_id)

        parents = []
        for parent_id in parent_ids:
            parent = self.parent_store.get(parent_id)
            if parent:
                parents.append(parent)

        return parents


class MultiVectorRetriever:
    """
    Multi-vector retriever

    Strategy:
    1. Index multiple representations: summaries, chunks, hypothetical questions
    2. Search on these representations
    3. Return full parent documents for better context
    """

    def __init__(
        self,
        base_retriever=None,
        multi_vector_store: Optional[MultiVectorStore] = None
    ):
        """
        Initialize multi-vector retriever

        Args:
            base_retriever: Base retriever for child documents
            multi_vector_store: Storage for parent-child relationships
        """
        self.base_retriever = base_retriever
        self.multi_vector_store = multi_vector_store or MultiVectorStore()

    def index_with_summaries(
        self,
        parent_documents: List[Dict],
        summaries: List[str]
    ):
        """
        Index documents using their summaries

        Args:
            parent_documents: Full documents
            summaries: Summary for each document (same order)
        """
        if len(parent_documents) != len(summaries):
            raise ValueError("Number of documents and summaries must match")

        child_docs = []
        for i, (parent, summary) in enumerate(zip(parent_documents, summaries)):
            parent_id = parent.get("chunk_id") or f"parent_{i}"
            parent["chunk_id"] = parent_id

            # Create summary child document
            summary_doc = {
                "chunk_id": f"{parent_id}_summary",
                "text": summary,
                "chunk_type": "summary",
                "metadata": {
                    "parent_id": parent_id,
                    "representation_type": "summary"
                }
            }
            child_docs.append(summary_doc)

        self.multi_vector_store.add_documents(parent_documents, child_docs)

        logger.info(f"Indexed {len(parent_documents)} documents with summaries")

    def index_with_questions(
        self,
        parent_documents: List[Dict],
        questions_per_doc: List[List[str]]
    ):
        """
        Index documents using hypothetical questions

        Args:
            parent_documents: Full documents
            questions_per_doc: List of questions for each document
        """
        if len(parent_documents) != len(questions_per_doc):
            raise ValueError("Number of documents and question lists must match")

        child_docs = []
        for i, (parent, questions) in enumerate(zip(parent_documents, questions_per_doc)):
            parent_id = parent.get("chunk_id") or f"parent_{i}"
            parent["chunk_id"] = parent_id

            # Create question child documents
            for j, question in enumerate(questions):
                question_doc = {
                    "chunk_id": f"{parent_id}_question_{j}",
                    "text": question,
                    "chunk_type": "question",
                    "metadata": {
                        "parent_id": parent_id,
                        "representation_type": "question",
                        "question_index": j
                    }
                }
                child_docs.append(question_doc)

        self.multi_vector_store.add_documents(parent_documents, child_docs)

        logger.info(
            f"Indexed {len(parent_documents)} documents with "
            f"{sum(len(q) for q in questions_per_doc)} questions"
        )

    def index_with_chunks(
        self,
        parent_documents: List[Dict],
        chunks_per_doc: List[List[Dict]]
    ):
        """
        Index documents using child chunks

        Args:
            parent_documents: Full documents
            chunks_per_doc: List of chunks for each document
        """
        if len(parent_documents) != len(chunks_per_doc):
            raise ValueError("Number of documents and chunk lists must match")

        child_docs = []
        for i, (parent, chunks) in enumerate(zip(parent_documents, chunks_per_doc)):
            parent_id = parent.get("chunk_id") or f"parent_{i}"
            parent["chunk_id"] = parent_id

            # Add parent_id to chunks
            for chunk in chunks:
                if "metadata" not in chunk:
                    chunk["metadata"] = {}
                chunk["metadata"]["parent_id"] = parent_id
                chunk["metadata"]["representation_type"] = "chunk"
                child_docs.append(chunk)

        self.multi_vector_store.add_documents(parent_documents, child_docs)

        logger.info(
            f"Indexed {len(parent_documents)} documents with "
            f"{sum(len(c) for c in chunks_per_doc)} chunks"
        )

    def index_with_all(
        self,
        parent_documents: List[Dict],
        summaries: List[str],
        questions_per_doc: List[List[str]],
        chunks_per_doc: List[List[Dict]]
    ):
        """
        Index with summaries + questions + chunks

        Args:
            parent_documents: Full documents
            summaries: Summary for each document
            questions_per_doc: Questions for each document
            chunks_per_doc: Chunks for each document
        """
        n_docs = len(parent_documents)
        if not (len(summaries) == len(questions_per_doc) == len(chunks_per_doc) == n_docs):
            raise ValueError("All lists must have the same length")

        child_docs = []

        for i in range(n_docs):
            parent = parent_documents[i]
            parent_id = parent.get("chunk_id") or f"parent_{i}"
            parent["chunk_id"] = parent_id

            # Add summary
            summary_doc = {
                "chunk_id": f"{parent_id}_summary",
                "text": summaries[i],
                "chunk_type": "summary",
                "metadata": {
                    "parent_id": parent_id,
                    "representation_type": "summary"
                }
            }
            child_docs.append(summary_doc)

            # Add questions
            for j, question in enumerate(questions_per_doc[i]):
                question_doc = {
                    "chunk_id": f"{parent_id}_question_{j}",
                    "text": question,
                    "chunk_type": "question",
                    "metadata": {
                        "parent_id": parent_id,
                        "representation_type": "question"
                    }
                }
                child_docs.append(question_doc)

            # Add chunks
            for chunk in chunks_per_doc[i]:
                if "metadata" not in chunk:
                    chunk["metadata"] = {}
                chunk["metadata"]["parent_id"] = parent_id
                chunk["metadata"]["representation_type"] = "chunk"
                child_docs.append(chunk)

        self.multi_vector_store.add_documents(parent_documents, child_docs)

        total_children = (
            len(summaries) +
            sum(len(q) for q in questions_per_doc) +
            sum(len(c) for c in chunks_per_doc)
        )

        logger.info(
            f"Indexed {n_docs} documents with {total_children} total representations"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        representation_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve parent documents using multi-vector search

        Args:
            query: Search query
            top_k: Number of parent documents to return
            representation_filter: Optional filter ('summary', 'question', 'chunk')

        Returns:
            List of parent documents
        """
        if not self.base_retriever:
            raise ValueError("Base retriever not configured")

        # Search child documents
        try:
            child_results = await self.base_retriever.retrieve(
                query=query,
                user_id="multi_vector",
                top_k=top_k * 3  # Retrieve more children
            )
            child_chunks = child_results.get("chunks", [])

            # Filter by representation type if specified
            if representation_filter:
                child_chunks = [
                    c for c in child_chunks
                    if c.get("metadata", {}).get("representation_type") == representation_filter
                ]

            # Get child IDs
            child_ids = [c.get("chunk_id") for c in child_chunks if c.get("chunk_id")]

            # Retrieve parent documents
            parent_docs = self.multi_vector_store.get_parent_documents(child_ids)

            # Limit to top_k
            parent_docs = parent_docs[:top_k]

            logger.info(
                f"Multi-vector retrieval: {len(child_chunks)} children → "
                f"{len(parent_docs)} parents"
            )

            return parent_docs

        except Exception as e:
            logger.error(f"Multi-vector retrieval error: {str(e)}")
            return []
