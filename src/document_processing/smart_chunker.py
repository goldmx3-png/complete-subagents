"""
Smart chunking with automatic metadata extraction
Based on adaptive RAG patterns for better structure detection
"""

from typing import List, Dict, Optional
import re
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SmartDocumentChunker:
    """
    Smart chunker that automatically detects and extracts:
    - Headers and sections
    - Subsections
    - Page numbers
    - Document structure
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize smart chunker

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def smart_chunk_with_metadata(
        self,
        text: str,
        doc_id: str,
        initial_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Automatically extract structure while chunking

        Args:
            text: Text to chunk
            doc_id: Document ID
            initial_metadata: Initial metadata (e.g., from parser)

        Returns:
            List of chunks with rich metadata
        """
        initial_metadata = initial_metadata or {}

        chunks = []
        lines = text.split('\n')

        current_chunk = ""
        current_metadata = {
            'section': initial_metadata.get('section'),
            'subsection': None,
            'page': initial_metadata.get('page_number'),
            'doc_id': doc_id
        }

        chunk_id = 0

        for line in lines:
            # Detect headers (common patterns)
            if self.is_header(line):
                section_title = self.clean_header(line)
                current_metadata['section'] = section_title
                logger.debug(f"Detected section: {section_title}")

            # Detect sub-headers
            if self.is_subheader(line):
                subsection_title = self.clean_header(line)
                current_metadata['subsection'] = subsection_title
                logger.debug(f"Detected subsection: {subsection_title}")

            # Detect page numbers
            page_match = re.search(r'Page\s+(\d+)|^\s*(\d+)\s*$', line)
            if page_match:
                page_num = page_match.group(1) or page_match.group(2)
                current_metadata['page'] = int(page_num)

            current_chunk += line + "\n"

            # Chunk when size reached
            if len(current_chunk) >= self.chunk_size:
                chunks.append({
                    'chunk_id': f"{doc_id}_chunk_{chunk_id}",
                    'text': current_chunk.strip(),
                    'chunk_type': 'text',
                    'section_title': current_metadata.get('section', ''),
                    'subsection': current_metadata.get('subsection'),
                    'page_numbers': [current_metadata.get('page')] if current_metadata.get('page') else [],
                    'parent_chunk_id': None,
                    'metadata': current_metadata.copy()
                })

                chunk_id += 1

                # Overlap - keep last portion
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text

        # Last chunk
        if current_chunk.strip():
            chunks.append({
                'chunk_id': f"{doc_id}_chunk_{chunk_id}",
                'text': current_chunk.strip(),
                'chunk_type': 'text',
                'section_title': current_metadata.get('section', ''),
                'subsection': current_metadata.get('subsection'),
                'page_numbers': [current_metadata.get('page')] if current_metadata.get('page') else [],
                'parent_chunk_id': None,
                'metadata': current_metadata.copy()
            })

        logger.info(f"Smart chunking created {len(chunks)} chunks with metadata")
        return chunks

    def is_header(self, line: str) -> bool:
        """
        Detect if line is a header

        Args:
            line: Text line

        Returns:
            True if line is a header
        """
        line = line.strip()

        if not line:
            return False

        return (
            # Numbered headers: "1.2.3 Title"
            bool(re.match(r'^\d+(\.\d+)*\s+[A-Z]', line)) or
            # ALL CAPS headers (at least 3 chars, max 100)
            (line.isupper() and 3 < len(line) < 100 and not line.isdigit()) or
            # Markdown style
            line.startswith('#') or
            # Title Case with length constraints (likely headers)
            (line.istitle() and 10 < len(line) < 80 and ':' not in line)
        )

    def is_subheader(self, line: str) -> bool:
        """
        Detect sub-headers

        Args:
            line: Text line

        Returns:
            True if line is a subheader
        """
        line = line.strip()

        if not line:
            return False

        return (
            # Multi-level numbering: 1.1.1
            bool(re.match(r'^\d+\.\d+\.\d+', line)) or
            # Bold indicators (if preserved)
            line.startswith('**') or
            # Indented numbered sections
            bool(re.match(r'^\s+\d+\.', line)) or
            # Letter-based subsections: a) b) c)
            bool(re.match(r'^[a-z]\)', line))
        )

    def clean_header(self, line: str) -> str:
        """
        Extract clean header text

        Args:
            line: Header line

        Returns:
            Cleaned header text
        """
        line = line.strip()

        # Remove markdown
        line = re.sub(r'^#+\s*', '', line)

        # Remove numbering
        line = re.sub(r'^\d+(\.\d+)*\s*', '', line)

        # Remove bold
        line = re.sub(r'\*\*', '', line)

        # Remove letter-based markers
        line = re.sub(r'^[a-z]\)\s*', '', line)

        return line.strip()

    def chunk_document_smart(
        self,
        parsed_doc: Dict,
        doc_id: str
    ) -> List[Dict]:
        """
        Smart chunk a parsed document

        Args:
            parsed_doc: Parsed document from parser
            doc_id: Document ID

        Returns:
            List of chunks with metadata
        """
        chunks = []

        # Process text elements with smart chunking
        for element in parsed_doc.get("text_elements", []):
            text = element["text"]
            metadata = element.get("metadata", {})

            element_chunks = self.smart_chunk_with_metadata(
                text=text,
                doc_id=doc_id,
                initial_metadata=metadata
            )

            chunks.extend(element_chunks)

        # Process table elements separately (don't chunk tables)
        for table in parsed_doc.get("table_elements", []):
            table_text = table["text"]
            metadata = table.get("metadata", {})

            chunks.append({
                "chunk_id": f"{doc_id}_table_{len(chunks)}",
                "text": table_text,
                "chunk_type": "table",
                "section_title": metadata.get("section", "Table"),
                "page_numbers": [metadata.get("page_number")] if metadata.get("page_number") else [],
                "parent_chunk_id": None,
                "metadata": metadata
            })

        logger.info(f"Smart chunked document {doc_id}: {len(chunks)} total chunks")
        return chunks
