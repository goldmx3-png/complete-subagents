"""
Semantic chunking using LLM to detect logical document boundaries
Optimized for banking documents with complex structure
"""

from typing import List, Dict
import re
import tiktoken
from openai import AsyncOpenAI
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SemanticChunker:
    """
    LLM-based semantic chunking that respects document structure

    Features:
    - Detects logical section boundaries using Mistral
    - Respects headers, lists, procedures
    - Maintains min/max token constraints
    - Preserves tables as complete units
    """

    def __init__(
        self,
        min_tokens: int = None,
        max_tokens: int = None,
        model: str = None,
        preserve_tables: bool = None
    ):
        """
        Initialize semantic chunker

        Args:
            min_tokens: Minimum chunk size in tokens
            max_tokens: Maximum chunk size in tokens
            model: LLM model for structure detection
            preserve_tables: Keep tables whole
        """
        self.min_tokens = min_tokens or settings.semantic_chunk_min_tokens
        self.max_tokens = max_tokens or settings.semantic_chunk_max_tokens
        self.model = model or settings.semantic_chunk_model
        self.preserve_tables = preserve_tables if preserve_tables is not None else settings.preserve_tables

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken: {str(e)}")
            self.tokenizer = None

        # Initialize OpenRouter client (Mistral via OpenRouter)
        self.client = AsyncOpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key
        )

        logger.info(
            f"Semantic chunker initialized: {self.min_tokens}-{self.max_tokens} tokens, "
            f"model={self.model}"
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4  # Fallback estimate

    async def chunk_document(self, parsed_doc: Dict, doc_id: str) -> List[Dict]:
        """
        Chunk parsed document using semantic boundaries

        Args:
            parsed_doc: Parsed document dict
            doc_id: Document ID

        Returns:
            List of chunk dicts
        """
        chunks = []
        chunk_id = 0

        # Process text elements with semantic chunking
        for element in parsed_doc.get("text_elements", []):
            text = element["text"]
            metadata = element.get("metadata", {})
            page_num = metadata.get("page_number", 0)

            # Use semantic chunking for longer texts
            element_chunks = await self._chunk_text_semantically(
                text=text,
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
                        "token_count": token_count,
                        "chunking_method": "semantic"
                    }
                })
                chunk_id += 1

        # Process tables (preserve as whole)
        for table in parsed_doc.get("table_elements", []):
            table_text = table["text"]
            metadata = table.get("metadata", {})
            page_num = metadata.get("page_number", 0)
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

        logger.info(f"Semantic chunking complete: {len(chunks)} chunks")
        return chunks

    async def _chunk_text_semantically(
        self,
        text: str,
        metadata: Dict
    ) -> List[str]:
        """
        Chunk text using LLM-detected boundaries

        Args:
            text: Text to chunk
            metadata: Element metadata

        Returns:
            List of semantic chunks
        """
        token_count = self.count_tokens(text)

        # If text is short, return as-is
        if token_count <= self.max_tokens:
            return [text]

        # If text is very short, just return it
        if token_count <= self.min_tokens:
            return [text]

        # Use LLM to detect logical boundaries
        try:
            boundaries = await self._detect_boundaries(text)

            # Split text at boundaries
            chunks = self._split_at_boundaries(text, boundaries)

            # Ensure chunks meet size constraints
            chunks = self._enforce_size_constraints(chunks)

            return chunks

        except Exception as e:
            logger.error(f"Semantic chunking failed: {str(e)}, falling back to simple split")
            # Fallback to simple splitting
            return self._simple_split(text)

    async def _detect_boundaries(self, text: str) -> List[int]:
        """
        Use LLM to detect logical section boundaries

        Args:
            text: Text to analyze

        Returns:
            List of character positions for boundaries
        """
        prompt = f"""Analyze this banking document text and identify logical section boundaries.
Look for:
- Topic changes
- New procedures or steps
- Transition between concepts
- Headers or section markers
- List transitions

Return ONLY the character positions (as integers) where logical breaks should occur, separated by commas.
If no clear boundaries exist, return "NONE".

Text:
{text[:2000]}... [truncated if longer]

Character positions:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a document analysis expert. Respond only with character positions or 'NONE'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )

            result = response.choices[0].message.content.strip()

            if result == "NONE" or not result:
                return []

            # Parse positions
            positions = []
            for pos in result.split(","):
                try:
                    positions.append(int(pos.strip()))
                except ValueError:
                    continue

            # Sort and filter valid positions
            positions = sorted([p for p in positions if 0 < p < len(text)])

            return positions

        except Exception as e:
            logger.error(f"Boundary detection failed: {str(e)}")
            return []

    def _split_at_boundaries(self, text: str, boundaries: List[int]) -> List[str]:
        """
        Split text at detected boundaries

        Args:
            text: Input text
            boundaries: List of split positions

        Returns:
            List of text chunks
        """
        if not boundaries:
            return [text]

        chunks = []
        prev_pos = 0

        for pos in boundaries:
            chunk = text[prev_pos:pos].strip()
            if chunk:
                chunks.append(chunk)
            prev_pos = pos

        # Add final chunk
        final_chunk = text[prev_pos:].strip()
        if final_chunk:
            chunks.append(final_chunk)

        return chunks

    def _enforce_size_constraints(self, chunks: List[str]) -> List[str]:
        """
        Ensure all chunks meet min/max size constraints

        Args:
            chunks: Input chunks

        Returns:
            Size-compliant chunks
        """
        result = []
        current_chunk = ""
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = self.count_tokens(chunk)

            # If chunk is too large, split it
            if chunk_tokens > self.max_tokens:
                if current_chunk:
                    result.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0

                # Split large chunk
                sub_chunks = self._simple_split(chunk)
                result.extend(sub_chunks)

            # If chunk is too small, accumulate
            elif chunk_tokens < self.min_tokens:
                if current_tokens + chunk_tokens > self.max_tokens:
                    # Current accumulated chunk is full
                    if current_chunk:
                        result.append(current_chunk)
                    current_chunk = chunk
                    current_tokens = chunk_tokens
                else:
                    # Add to current chunk
                    current_chunk += "\n\n" + chunk if current_chunk else chunk
                    current_tokens += chunk_tokens

            # Chunk is good size
            else:
                if current_chunk:
                    result.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0
                result.append(chunk)

        # Add any remaining accumulated chunk
        if current_chunk:
            result.append(current_chunk)

        return result

    def _simple_split(self, text: str) -> List[str]:
        """
        Simple fallback splitting by sentences

        Args:
            text: Text to split

        Returns:
            List of chunks
        """
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
