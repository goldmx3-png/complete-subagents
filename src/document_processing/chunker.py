"""
Text chunking with semantic preservation
"""

from typing import List, Dict
import re
from src.config import settings


class DocumentChunker:
    """Chunk documents while preserving semantic boundaries"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize chunker

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk_document(self, parsed_doc: Dict, doc_id: str) -> List[Dict]:
        """
        Chunk parsed document

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

            # Get page number from metadata
            page_num = metadata.get("page_number", 0)

            # Chunk this element
            element_chunks = self._chunk_text(
                text=text,
                element_type=element_type,
                page_num=page_num,
                metadata=metadata
            )

            for chunk_text in element_chunks:
                chunks.append({
                    "chunk_id": f"{doc_id}_chunk_{chunk_id}",
                    "text": chunk_text,
                    "chunk_type": "text",
                    "section_title": metadata.get("section", ""),
                    "page_numbers": [page_num] if page_num else [],
                    "parent_chunk_id": None,
                    "metadata": metadata
                })
                chunk_id += 1

        # Process table elements separately (don't chunk tables)
        for table in parsed_doc.get("table_elements", []):
            table_text = table["text"]
            metadata = table.get("metadata", {})
            page_num = metadata.get("page_number", 0)

            chunks.append({
                "chunk_id": f"{doc_id}_table_{chunk_id}",
                "text": table_text,
                "chunk_type": "table",
                "section_title": "Table",
                "page_numbers": [page_num] if page_num else [],
                "parent_chunk_id": None,
                "metadata": metadata
            })
            chunk_id += 1

        return chunks

    def _chunk_text(
        self,
        text: str,
        element_type: str,
        page_num: int,
        metadata: Dict
    ) -> List[str]:
        """
        Chunk text while preserving semantic boundaries

        Args:
            text: Text to chunk
            element_type: Type of element
            page_num: Page number
            metadata: Element metadata

        Returns:
            List of text chunks
        """
        # If text is short enough, return as-is
        if len(text) <= self.chunk_size:
            return [text]

        # Split into sentences
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If single sentence is longer than chunk_size, split it
            if sentence_length > self.chunk_size:
                # Add current chunk if not empty
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split long sentence
                sub_chunks = self._split_long_sentence(sentence)
                chunks.extend(sub_chunks)
            else:
                # Check if adding this sentence exceeds chunk size
                if current_length + sentence_length > self.chunk_size:
                    # Save current chunk
                    chunks.append(" ".join(current_chunk))

                    # Start new chunk with overlap
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk,
                        self.chunk_overlap
                    )
                    current_chunk = overlap_sentences + [sentence]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length

        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitter (can be improved with spaCy/NLTK)
        # Split on '. ', '! ', '? ' followed by capital letter or end
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)

        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        Split a long sentence into smaller chunks

        Args:
            sentence: Long sentence

        Returns:
            List of chunks
        """
        # Split on commas, semicolons, etc.
        parts = re.split(r'[,;:]', sentence)

        chunks = []
        current_chunk = ""

        for part in parts:
            if len(current_chunk) + len(part) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += part + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _get_overlap_sentences(
        self,
        sentences: List[str],
        overlap_size: int
    ) -> List[str]:
        """
        Get sentences for overlap

        Args:
            sentences: List of sentences
            overlap_size: Target overlap size in characters

        Returns:
            List of overlap sentences
        """
        overlap_sentences = []
        total_length = 0

        # Take sentences from the end
        for sentence in reversed(sentences):
            if total_length + len(sentence) > overlap_size:
                break
            overlap_sentences.insert(0, sentence)
            total_length += len(sentence)

        return overlap_sentences

    def chunk_text_simple(self, text: str, doc_id: str) -> List[Dict]:
        """
        Simple chunking for plain text

        Args:
            text: Text to chunk
            doc_id: Document ID

        Returns:
            List of chunk dicts
        """
        chunks = self._chunk_text(
            text=text,
            element_type="Text",
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
                "metadata": {}
            }
            for i, chunk in enumerate(chunks)
        ]
