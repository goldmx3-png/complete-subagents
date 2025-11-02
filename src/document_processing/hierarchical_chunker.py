"""
Hierarchical chunking with parent-child relationships for better context retrieval
Based on LangChain/LlamaIndex best practices
"""

from typing import List, Dict, Optional, Tuple
import re
import uuid
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HierarchicalChunker:
    """
    Advanced hierarchical chunking that creates parent-child relationships

    Features:
    - Multi-level chunking (parent chunks with child chunks)
    - Index small chunks for retrieval, return large parent chunks for context
    - Preserves semantic boundaries
    - Auto-detects structure (sections, paragraphs)
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400,
        chunk_overlap: int = 50
    ):
        """
        Initialize hierarchical chunker

        Args:
            parent_chunk_size: Size of parent chunks (retrieved for context)
            child_chunk_size: Size of child chunks (indexed for search)
            chunk_overlap: Overlap between chunks
        """
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document_hierarchical(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Create hierarchical chunks with parent-child relationships

        Args:
            text: Document text
            doc_id: Document ID
            metadata: Optional metadata

        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        metadata = metadata or {}

        # Step 1: Detect document structure
        structure = self._detect_structure(text)

        # Step 2: Create parent chunks
        parent_chunks = self._create_parent_chunks(
            text=text,
            doc_id=doc_id,
            structure=structure,
            metadata=metadata
        )

        # Step 3: Create child chunks from each parent
        child_chunks = []
        for parent in parent_chunks:
            children = self._create_child_chunks(
                parent_text=parent["text"],
                parent_id=parent["chunk_id"],
                parent_metadata=parent["metadata"]
            )
            child_chunks.extend(children)

        logger.info(
            f"Created {len(parent_chunks)} parent chunks and "
            f"{len(child_chunks)} child chunks for doc {doc_id}"
        )

        return parent_chunks, child_chunks

    def _detect_structure(self, text: str) -> Dict:
        """
        Auto-detect document structure

        Returns:
            Dict with detected structural elements
        """
        structure = {
            'sections': [],
            'section_positions': [],
            'has_headers': False
        }

        lines = text.split('\n')

        for i, line in enumerate(lines):
            # Detect headers (numbered, all caps, or followed by dashes/equals)
            if self._is_header(line, lines, i):
                structure['sections'].append({
                    'title': line.strip(),
                    'position': i,
                    'line_number': i
                })
                structure['has_headers'] = True

        return structure

    def _is_header(self, line: str, all_lines: List[str], index: int) -> bool:
        """Check if line is a header"""
        line = line.strip()

        if not line or len(line) > 100:
            return False

        # Numbered headers: "1.2.3 Title"
        if re.match(r'^\d+(\.\d+)*\s+[A-Z]', line):
            return True

        # ALL CAPS headers (at least 3 chars, max 80)
        if line.isupper() and 3 <= len(line) <= 80:
            return True

        # Markdown style
        if line.startswith('#'):
            return True

        # Check if next line is underline (===  or ---)
        if index + 1 < len(all_lines):
            next_line = all_lines[index + 1].strip()
            if next_line and all(c in '=-' for c in next_line) and len(next_line) >= 3:
                return True

        return False

    def _create_parent_chunks(
        self,
        text: str,
        doc_id: str,
        structure: Dict,
        metadata: Dict
    ) -> List[Dict]:
        """Create large parent chunks"""
        parent_chunks = []

        if structure['has_headers'] and structure['sections']:
            # Chunk by sections
            parent_chunks = self._chunk_by_sections(
                text, doc_id, structure, metadata
            )
        else:
            # No clear structure - use size-based chunking
            parent_chunks = self._chunk_by_size(
                text, doc_id, self.parent_chunk_size, metadata, is_parent=True
            )

        return parent_chunks

    def _chunk_by_sections(
        self,
        text: str,
        doc_id: str,
        structure: Dict,
        metadata: Dict
    ) -> List[Dict]:
        """Chunk document by detected sections"""
        chunks = []
        lines = text.split('\n')
        sections = structure['sections']

        for i, section in enumerate(sections):
            # Get text from this section to next section (or end)
            start_line = section['position']
            end_line = sections[i + 1]['position'] if i + 1 < len(sections) else len(lines)

            section_text = '\n'.join(lines[start_line:end_line])

            # If section is too large, split it
            if len(section_text) > self.parent_chunk_size * 1.5:
                sub_chunks = self._chunk_by_size(
                    section_text,
                    doc_id,
                    self.parent_chunk_size,
                    {**metadata, 'section': section['title']},
                    is_parent=True
                )
                chunks.extend(sub_chunks)
            else:
                chunk_id = f"{doc_id}_parent_{uuid.uuid4().hex[:8]}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": section_text,
                    "chunk_type": "parent",
                    "section_title": section['title'],
                    "metadata": {
                        **metadata,
                        'section': section['title'],
                        'chunk_level': 'parent'
                    }
                })

        return chunks

    def _chunk_by_size(
        self,
        text: str,
        doc_id: str,
        chunk_size: int,
        metadata: Dict,
        is_parent: bool = False
    ) -> List[Dict]:
        """Create chunks by size with overlap"""
        chunks = []

        # Split into sentences
        sentences = self._split_sentences(text)

        current_chunk = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunk_type = "parent" if is_parent else "child"
                chunk_id = f"{doc_id}_{chunk_type}_{chunk_index}"

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "chunk_type": chunk_type,
                    "metadata": {
                        **metadata,
                        'chunk_level': chunk_type,
                        'chunk_index': chunk_index
                    }
                })

                # Create overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, self.chunk_overlap
                )
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
                chunk_index += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_type = "parent" if is_parent else "child"
            chunk_id = f"{doc_id}_{chunk_type}_{chunk_index}"

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "chunk_type": chunk_type,
                "metadata": {
                    **metadata,
                    'chunk_level': chunk_type,
                    'chunk_index': chunk_index
                }
            })

        return chunks

    def _create_child_chunks(
        self,
        parent_text: str,
        parent_id: str,
        parent_metadata: Dict
    ) -> List[Dict]:
        """Create smaller child chunks from a parent chunk"""
        child_chunks = []

        # Split into sentences
        sentences = self._split_sentences(parent_text)

        current_chunk = []
        current_length = 0
        child_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.child_chunk_size and current_chunk:
                # Save child chunk
                chunk_text = ' '.join(current_chunk)
                chunk_id = f"{parent_id}_child_{child_index}"

                child_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "chunk_type": "child",
                    "parent_chunk_id": parent_id,  # Link to parent
                    "metadata": {
                        **parent_metadata,
                        'chunk_level': 'child',
                        'child_index': child_index,
                        'parent_id': parent_id
                    }
                })

                # Create overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, self.chunk_overlap
                )
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
                child_index += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add remaining child chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = f"{parent_id}_child_{child_index}"

            child_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "chunk_type": "child",
                "parent_chunk_id": parent_id,
                "metadata": {
                    **parent_metadata,
                    'chunk_level': 'child',
                    'child_index': child_index,
                    'parent_id': parent_id
                }
            })

        return child_chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Split on sentence boundaries
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_sentences(
        self,
        sentences: List[str],
        overlap_size: int
    ) -> List[str]:
        """Get sentences for overlap"""
        overlap_sentences = []
        total_length = 0

        for sentence in reversed(sentences):
            if total_length + len(sentence) > overlap_size:
                break
            overlap_sentences.insert(0, sentence)
            total_length += len(sentence)

        return overlap_sentences
