"""
Document upload and processing pipeline
"""

from typing import Dict, Optional
from pathlib import Path
import uuid
import shutil
from src.document_processing.parser import DocumentParser
from src.document_processing.chunker import DocumentChunker
from src.document_processing.chunker_factory import ChunkerFactory
from src.vectorstore.embeddings import get_embeddings
from src.vectorstore.qdrant_store import QdrantStore
from src.config import settings
from src.utils.logger import get_logger
from src.utils.metrics import timed_operation

# Initialize logger
logger = get_logger(__name__)


class DocumentUploader:
    """Handle document upload and processing"""

    def __init__(
        self,
        parser: Optional[DocumentParser] = None,
        chunker = None,  # Can be any chunker type
        embeddings = None,
        vectorstore: Optional[QdrantStore] = None,
        upload_dir: Optional[str] = None,
        use_markdown_parser: Optional[bool] = None
    ):
        """
        Initialize document uploader

        Args:
            parser: Document parser instance (uses config-based selection if None)
            chunker: Document chunker instance (uses ChunkerFactory if None)
            embeddings: Embeddings instance
            vectorstore: Vector store instance
            upload_dir: Directory to save uploaded files (defaults to settings)
            use_markdown_parser: Override config to force markdown parser (None = use config)
        """
        # Determine which parser to use
        self.use_markdown = use_markdown_parser if use_markdown_parser is not None else settings.use_markdown_chunking

        if self.use_markdown and parser is None:
            # Use MarkdownDocumentParser when markdown chunking is enabled
            try:
                from src.document_processing.markdown_parser import MarkdownDocumentParser
                self.parser = MarkdownDocumentParser()
                logger.info("Using MarkdownDocumentParser with docling")
            except ImportError as e:
                logger.warning(f"Failed to import MarkdownDocumentParser: {e}. Falling back to DocumentParser")
                self.parser = DocumentParser()
                self.use_markdown = False
        else:
            self.parser = parser or DocumentParser()

        # Use ChunkerFactory for intelligent chunker selection
        self.chunker = chunker or ChunkerFactory.create_chunker()
        self.chunker_name = ChunkerFactory.get_active_chunker_name()

        self.embeddings = embeddings or get_embeddings()
        self.vectorstore = vectorstore or QdrantStore()
        self.upload_dir = Path(upload_dir or settings.upload_directory)
        self.upload_dir.mkdir(exist_ok=True)

        logger.info(f"DocumentUploader initialized: parser={'MarkdownDocumentParser' if self.use_markdown else 'DocumentParser'}, chunker={self.chunker_name}")

    async def upload_document(
        self,
        file_path: str,
        user_id: str,
        doc_id: Optional[str] = None,
        extract_tables: bool = True
    ) -> Dict:
        """
        Upload and process document

        Args:
            file_path: Path to file
            user_id: User ID
            doc_id: Optional document ID (generated if not provided)
            extract_tables: Whether to extract tables

        Returns:
            Document processing result
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate doc_id if not provided
        if not doc_id:
            doc_id = str(uuid.uuid4())

        # Check file extension
        file_ext = file_path.suffix.lower()
        allowed_extensions = ["." + ext for ext in settings.allowed_upload_extensions.split(",")]

        if file_ext not in allowed_extensions:
            raise ValueError(f"File type {file_ext} not allowed. Allowed: {allowed_extensions}")

        # Check file size
        file_size = file_path.stat().st_size
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024

        if file_size > max_size_bytes:
            raise ValueError(f"File size {file_size} exceeds maximum {max_size_bytes}")

        logger.info(f"Starting document processing: file={file_path.name}, doc_id={doc_id}, user_id={user_id}, size={file_size} bytes")

        with timed_operation(logger, "Document upload and processing", file_name=file_path.name, file_size=file_size) as metrics:
            # Step 1: Copy file to upload directory
            logger.debug(f"Copying file to upload directory")
            dest_path = self.upload_dir / f"{doc_id}_{file_path.name}"
            shutil.copy2(file_path, dest_path)

            # Step 2: Parse document
            logger.info(f"Parsing document: type={file_ext}, parser={'markdown' if self.use_markdown else 'standard'}")
            if file_ext == ".pdf":
                parsed_doc = self.parser.parse_pdf(str(dest_path), extract_tables=extract_tables)
            elif file_ext == ".txt":
                parsed_doc = self.parser.parse_text_file(str(dest_path))
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            # Handle different parser outputs
            if self.use_markdown:
                num_pages = parsed_doc.get("metadata", {}).get("num_pages", 1)
                logger.info(f"Document parsed: {num_pages} pages (markdown format)")
            else:
                num_pages = parsed_doc.get("num_pages", 1)
                logger.info(f"Document parsed: {num_pages} pages")

            # Step 3: Chunk document
            logger.info(f"Chunking document with {self.chunker_name}...")

            # Handle different chunker interfaces
            if self.chunker_name == "MarkdownChunker":
                # MarkdownChunker expects (doc_data, doc_id, user_id)
                chunks = self.chunker.chunk_document(parsed_doc, doc_id, user_id)
            else:
                # Legacy chunkers expect (parsed_doc, doc_id)
                chunks = self.chunker.chunk_document(parsed_doc, doc_id)
                # Add user_id to chunks if not present
                for chunk in chunks:
                    if "user_id" not in chunk:
                        chunk["user_id"] = user_id

            logger.info(f"Created {len(chunks)} chunks")

            # Step 4: Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embeddings.embed_documents(chunk_texts)
            logger.info(f"Generated {len(embeddings)} embeddings")

            # Step 5: Upload to vector store
            logger.info("Uploading to vector store...")
            await self.vectorstore.upsert_chunks(
                chunks=chunks,
                embeddings=embeddings,
                user_id=user_id,
                doc_id=doc_id
            )

            # Add metrics
            metrics['num_chunks'] = len(chunks)
            metrics['num_pages'] = num_pages
            metrics['num_text_chunks'] = len([c for c in chunks if c["chunk_type"] in ["text", "text_with_table"]])
            metrics['num_table_chunks'] = len([c for c in chunks if "table" in c["chunk_type"]])
            metrics['chunker_type'] = self.chunker_name
            metrics['parser_type'] = 'markdown' if self.use_markdown else 'standard'

        logger.info(f"Document processing complete: doc_id={doc_id}, chunks={len(chunks)}")

        return {
            "document_id": doc_id,
            "filename": file_path.name,
            "status": "completed",
            "chunks_created": len(chunks),
            "message": f"Successfully processed {file_path.name} with {len(chunks)} chunks"
        }

    async def delete_document(self, doc_id: str, user_id: str) -> Dict:
        """
        Delete document from vector store and file system

        Args:
            doc_id: Document ID
            user_id: User ID

        Returns:
            Deletion result
        """
        logger.info(f"Deleting document: doc_id={doc_id}, user_id={user_id}")

        # Delete from vector store
        await self.vectorstore.delete_document(doc_id, user_id)
        logger.debug("Document deleted from vector store")

        # Delete file from upload directory
        file_count = 0
        for file_path in self.upload_dir.glob(f"{doc_id}_*"):
            file_path.unlink()
            file_count += 1
            logger.debug(f"Deleted file: {file_path}")

        logger.info(f"Document deletion complete: doc_id={doc_id}, files_deleted={file_count}")

        return {
            "document_id": doc_id,
            "status": "deleted"
        }

    async def get_document_info(self, doc_id: str, user_id: str) -> Dict:
        """
        Get document information

        Args:
            doc_id: Document ID
            user_id: User ID

        Returns:
            Document info
        """
        # Get chunk count
        chunk_count = await self.vectorstore.get_document_count(user_id, doc_id)

        # Find file
        file_path = None
        for path in self.upload_dir.glob(f"{doc_id}_*"):
            file_path = path
            break

        if file_path:
            file_info = {
                "filename": file_path.name.split("_", 1)[1] if "_" in file_path.name else file_path.name,
                "file_size": file_path.stat().st_size,
                "file_path": str(file_path)
            }
        else:
            file_info = {}

        return {
            "document_id": doc_id,
            "chunks_count": chunk_count,
            **file_info
        }
