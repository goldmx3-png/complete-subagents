"""
Token-based chunking for better context preservation with complex documents
Optimized for banking documents with tables and multi-step procedures
"""

from typing import List, Dict
import re
import tiktoken
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TokenBasedChunker:
    """
    Token-based document chunker with table preservation

    Features:
    - Token counting using tiktoken (more accurate than character-based)
    - Configurable chunk sizes (400-800 tokens)
    - Percentage-based overlap (10-20%)
    - Table preservation (keeps tables whole)
    - Sentence boundary awareness
    """

    def __init__(
        self,
        chunk_size_tokens: int = None,
        chunk_overlap_percentage: int = None,
        preserve_tables: bool = None
    ):
        """
        Initialize token-based chunker

        Args:
            chunk_size_tokens: Target chunk size in tokens
            chunk_overlap_percentage: Overlap as percentage (0-100)
            preserve_tables: Keep tables as complete chunks
        """
        self.chunk_size_tokens = chunk_size_tokens or settings.chunk_size_tokens
        self.chunk_overlap_percentage = chunk_overlap_percentage or settings.chunk_overlap_percentage
        self.preserve_tables = preserve_tables if preserve_tables is not None else settings.preserve_tables

        # Calculate overlap in tokens
        self.chunk_overlap_tokens = int(
            self.chunk_size_tokens * (self.chunk_overlap_percentage / 100.0)
        )

        # Initialize tokenizer (cl100k_base is used by GPT-4, good general tokenizer)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken, falling back to estimate: {str(e)}")
            self.tokenizer = None

        logger.info(
            f"Token chunker initialized: size={self.chunk_size_tokens} tokens, "
            f"overlap={self.chunk_overlap_percentage}% ({self.chunk_overlap_tokens} tokens), "
            f"preserve_tables={self.preserve_tables}"
        )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback estimate: ~4 characters per token
            return len(text) // 4

    def chunk_document(self, parsed_doc: Dict, doc_id: str) -> List[Dict]:
        """
        Chunk parsed document using token-based approach

        Args:
            parsed_doc: Parsed document dict from parser
            doc_id: Document ID

        Returns:
            List of chunk dicts
        """
        chunks = []
        chunk_id = 0

        # Process text elements
        for element in parsed_doc.get("text_elements", []):
            text = element["text"]
            element_type = element["type"]
            metadata = element.get("metadata", {})
            page_num = metadata.get("page_number", 0)

            # Chunk this element
            element_chunks = self._chunk_text_by_tokens(
                text=text,
                element_type=element_type,
                page_num=page_num,
                metadata=metadata
            )

            for chunk_text in element_chunks:
                token_count = self.count_tokens(chunk_text)
                chunks.append({
                    "chunk_id": f"{doc_id}_chunk_{chunk_id}",
                    "text": chunk_text,
                    "chunk_type": "text",
                    "section_title": metadata.get("section", ""),
                    "page_numbers": [page_num] if page_num else [],
                    "parent_chunk_id": None,
                    "metadata": {
                        **metadata,
                        "token_count": token_count
                    }
                })
                chunk_id += 1

        # Process table elements (preserve as whole chunks if enabled)
        for table in parsed_doc.get("table_elements", []):
            table_text = table["text"]
            metadata = table.get("metadata", {})
            page_num = metadata.get("page_number", 0)

            if self.preserve_tables:
                # Keep table as single chunk
                token_count = self.count_tokens(table_text)
                chunks.append({
                    "chunk_id": f"{doc_id}_table_{chunk_id}",
                    "text": table_text,
                    "chunk_type": "table",
                    "section_title": "Table",
                    "page_numbers": [page_num] if page_num else [],
                    "parent_chunk_id": None,
                    "metadata": {
                        **metadata,
                        "token_count": token_count,
                        "preserved": True
                    }
                })
                chunk_id += 1
            else:
                # Chunk large tables
                table_chunks = self._chunk_text_by_tokens(
                    text=table_text,
                    element_type="table",
                    page_num=page_num,
                    metadata=metadata
                )
                for chunk_text in table_chunks:
                    token_count = self.count_tokens(chunk_text)
                    chunks.append({
                        "chunk_id": f"{doc_id}_table_{chunk_id}",
                        "text": chunk_text,
                        "chunk_type": "table",
                        "section_title": "Table",
                        "page_numbers": [page_num] if page_num else [],
                        "parent_chunk_id": None,
                        "metadata": {
                            **metadata,
                            "token_count": token_count
                        }
                    })
                    chunk_id += 1

        logger.info(f"Chunked document into {len(chunks)} chunks (avg {self._avg_tokens(chunks):.0f} tokens)")
        return chunks

    def _chunk_text_by_tokens(
        self,
        text: str,
        element_type: str,
        page_num: int,
        metadata: Dict
    ) -> List[str]:
        """
        Chunk text based on token count

        Args:
            text: Text to chunk
            element_type: Type of element
            page_num: Page number
            metadata: Element metadata

        Returns:
            List of text chunks
        """
        # Count tokens in full text
        total_tokens = self.count_tokens(text)

        # If text is short enough, return as-is
        if total_tokens <= self.chunk_size_tokens:
            return [text]

        # Split into sentences
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence is longer than chunk_size, split it
            if sentence_tokens > self.chunk_size_tokens:
                # Add current chunk if not empty
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_chunk_tokens = 0

                # Split long sentence
                sub_chunks = self._split_long_sentence_by_tokens(sentence)
                chunks.extend(sub_chunks)
            else:
                # Check if adding this sentence exceeds chunk size
                if current_chunk_tokens + sentence_tokens > self.chunk_size_tokens:
                    # Save current chunk
                    chunks.append(" ".join(current_chunk_sentences))

                    # Start new chunk with overlap
                    overlap_sentences = self._get_overlap_sentences_by_tokens(
                        current_chunk_sentences,
                        self.chunk_overlap_tokens
                    )
                    current_chunk_sentences = overlap_sentences + [sentence]
                    current_chunk_tokens = sum(
                        self.count_tokens(s) for s in current_chunk_sentences
                    )
                else:
                    current_chunk_sentences.append(sentence)
                    current_chunk_tokens += sentence_tokens

        # Add remaining chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Enhanced sentence splitter for banking documents
        # Handles: Mr., Dr., amounts like $1.5M, etc.
        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])'
        sentences = re.split(pattern, text)

        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]

        # Handle list items and numbered items
        processed_sentences = []
        for sentence in sentences:
            # Check if this is a list item
            if re.match(r'^\d+\.\s+', sentence) or re.match(r'^[•\-\*]\s+', sentence):
                processed_sentences.append(sentence)
            else:
                processed_sentences.append(sentence)

        return processed_sentences

    def _split_long_sentence_by_tokens(self, sentence: str) -> List[str]:
        """
        Split a long sentence into smaller chunks based on tokens

        Args:
            sentence: Long sentence

        Returns:
            List of chunks
        """
        # Split on natural breaks: commas, semicolons, conjunctions
        parts = re.split(r'([,;:]|\s+and\s+|\s+or\s+|\s+but\s+)', sentence)

        chunks = []
        current_chunk = ""
        current_tokens = 0

        for part in parts:
            part_tokens = self.count_tokens(part)

            if current_tokens + part_tokens > self.chunk_size_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
                current_tokens = part_tokens
            else:
                current_chunk += part
                current_tokens += part_tokens

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _get_overlap_sentences_by_tokens(
        self,
        sentences: List[str],
        overlap_tokens: int
    ) -> List[str]:
        """
        Get sentences for overlap based on token count

        Args:
            sentences: List of sentences
            overlap_tokens: Target overlap size in tokens

        Returns:
            List of overlap sentences
        """
        overlap_sentences = []
        total_tokens = 0

        # Take sentences from the end
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if total_tokens + sentence_tokens > overlap_tokens:
                break
            overlap_sentences.insert(0, sentence)
            total_tokens += sentence_tokens

        return overlap_sentences

    def _avg_tokens(self, chunks: List[Dict]) -> float:
        """Calculate average tokens per chunk"""
        if not chunks:
            return 0
        total = sum(chunk.get("metadata", {}).get("token_count", 0) for chunk in chunks)
        return total / len(chunks)

    def chunk_text_simple(self, text: str, doc_id: str) -> List[Dict]:
        """
        Simple chunking for plain text

        Args:
            text: Text to chunk
            doc_id: Document ID

        Returns:
            List of chunk dicts
        """
        chunks = self._chunk_text_by_tokens(
            text=text,
            element_type="text",
            page_num=1,
            metadata={}
        )

        return [
            {
                "chunk_id": f"{doc_id}_chunk_{i}",
                "text": chunk,
                "chunk_type": "text",
                "section_title": "",
                "page_numbers": [1],
                "parent_chunk_id": None,
                "metadata": {"token_count": self.count_tokens(chunk)}
            }
            for i, chunk in enumerate(chunks)
        ]
