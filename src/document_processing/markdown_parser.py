"""
Markdown Document Parser using docling for advanced PDF to Markdown conversion.

This parser uses docling to convert PDFs to structured markdown format,
preserving document hierarchy, tables, and formatting for improved RAG performance.
"""

import re
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tiktoken

from docling.document_converter import DocumentConverter
from src.config import settings


class MarkdownDocumentParser:
    """
    Parse PDF documents using docling with markdown output.

    Features:
    - Converts PDFs to structured markdown
    - Preserves headers, lists, formatting
    - Smart table handling (inline vs separate)
    - Extracts metadata for enhanced retrieval
    """

    def __init__(
        self,
        extract_tables: bool = None,
        extract_images: bool = None,
        table_size_threshold: int = None,
    ):
        """
        Initialize the MarkdownDocumentParser.

        Args:
            extract_tables: Whether to extract tables (default from config)
            extract_images: Whether to extract images (default from config)
            table_size_threshold: Token threshold for table size classification
        """
        self.extract_tables = extract_tables if extract_tables is not None else settings.docling_extract_tables
        self.extract_images = extract_images if extract_images is not None else settings.docling_extract_images
        self.table_size_threshold = table_size_threshold if table_size_threshold is not None else settings.markdown_table_size_threshold

        # Initialize docling converter
        self.converter = DocumentConverter()

        # Initialize tokenizer for measuring text
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def parse_pdf(self, file_path: str) -> Dict:
        """
        Parse a PDF file and return structured markdown content.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary containing:
                - markdown_content: Full markdown text
                - metadata: Document metadata (pages, file info, etc.)
                - table_elements: List of table dicts with metadata
                - text_elements: List of text section dicts with metadata
                - file_hash: MD5 hash of the file

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            Exception: If docling conversion fails
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        # Compute file hash for tracking
        file_hash = self._compute_file_hash(file_path)

        try:
            # Convert PDF to docling document
            result = self.converter.convert(file_path)
            doc = result.document

            # Export to markdown
            markdown_content = doc.export_to_markdown()

            # Get page count from docling document (if available)
            try:
                num_pages = len(doc.pages) if hasattr(doc, 'pages') else 1
            except:
                num_pages = 1  # Fallback

            # Analyze and extract tables from markdown
            table_elements, text_with_markers = self._analyze_tables(markdown_content)

            # Extract text sections with headers
            text_elements = self._extract_text_sections(text_with_markers)

            # Build hierarchical structure for enhanced metadata
            hierarchy_structure = self._build_hierarchical_structure(text_elements)

            # Extract document metadata
            metadata = {
                "file_name": file_path_obj.name,
                "file_path": str(file_path_obj.absolute()),
                "file_hash": file_hash,
                "num_pages": num_pages,
                "total_chars": len(markdown_content),
                "total_tokens": len(self.tokenizer.encode(markdown_content)),
                "num_tables": len(table_elements),
                "num_sections": len(text_elements),
            }

            return {
                "markdown_content": markdown_content,
                "metadata": metadata,
                "table_elements": table_elements,
                "text_elements": text_elements,
                "hierarchy_structure": hierarchy_structure,  # NEW: Full hierarchical tree
                "file_hash": file_hash,
            }

        except Exception as e:
            raise Exception(f"Docling conversion failed for {file_path}: {str(e)}")

    def _analyze_tables(self, markdown: str) -> Tuple[List[Dict], str]:
        """
        Analyze markdown tables and classify them as inline or separate.

        Args:
            markdown: Full markdown content

        Returns:
            Tuple of (table_elements list, text_with_markers string)
            - table_elements: List of dicts with table data and metadata
            - text_with_markers: Markdown with large tables replaced by markers
        """
        table_elements = []
        text_with_markers = markdown

        # Regex to find markdown tables (pipe-delimited format)
        table_pattern = r'(\|.+\|[\r\n]+(?:\|[-:\s|]+\|[\r\n]+)?(?:\|.+\|[\r\n]+)*)'

        matches = list(re.finditer(table_pattern, markdown, re.MULTILINE))

        for idx, match in enumerate(matches):
            table_text = match.group(1).strip()
            table_start = match.start()
            table_end = match.end()

            # Calculate token size
            token_count = len(self.tokenizer.encode(table_text))

            # Classify table size
            is_large = token_count >= self.table_size_threshold

            # Determine table type
            if is_large:
                table_type = "table_large"
                # Replace large table with marker in text
                marker = f"\n\n[TABLE_{idx}_LARGE]\n\n"
                text_with_markers = text_with_markers.replace(table_text, marker, 1)
            else:
                table_type = "table_inline"

            # Extract table rows for additional metadata
            rows = [row.strip() for row in table_text.split('\n') if row.strip()]
            num_rows = len([r for r in rows if not re.match(r'^\|[\s:-]+\|$', r)])  # Exclude separator rows
            num_cols = len(rows[0].split('|')) - 2 if rows else 0  # -2 for leading/trailing pipes

            table_element = {
                "content": table_text,
                "chunk_type": table_type,
                "token_count": token_count,
                "char_count": len(table_text),
                "is_large": is_large,
                "position": table_start,
                "table_index": idx,
                "num_rows": num_rows,
                "num_cols": num_cols,
                "metadata": {
                    "table_id": f"table_{idx}",
                    "size_classification": "large" if is_large else "small",
                },
            }

            table_elements.append(table_element)

        return table_elements, text_with_markers

    def _build_hierarchical_structure(self, text_elements: List[Dict]) -> Dict:
        """
        Build complete hierarchical tree from text sections for enhanced metadata.

        This method creates a full document tree structure tracking parent-child
        relationships, sibling sections, and positional context for each section.

        Args:
            text_elements: List of text section dictionaries from _extract_text_sections

        Returns:
            Dictionary mapping section index to hierarchical metadata:
            {
                section_idx: {
                    "full_path": "Section 1 > Subsection 1.1 > Details",
                    "breadcrumbs": ["Section 1", "Subsection 1.1", "Details"],
                    "depth": 3,
                    "parent_idx": 1,  # Index of parent section
                    "root_idx": 0,    # Index of root section
                    "children_indices": [3, 4],
                    "sibling_indices": [2],
                    "position_in_doc": "middle",  # "intro", "middle", "conclusion"
                }
            }
        """
        if not text_elements:
            return {}

        hierarchy_map = {}
        hierarchy_stack = []  # Stack to track current path: [(idx, level, text), ...]

        for idx, element in enumerate(text_elements):
            level = element["header_level"]
            header_text = element["header_text"]

            # Pop stack until we find the parent level
            while hierarchy_stack and hierarchy_stack[-1][1] >= level:
                hierarchy_stack.pop()

            # Build breadcrumb path
            breadcrumbs = [item[2] for item in hierarchy_stack]
            breadcrumbs.append(header_text)

            # Determine parent and root
            parent_idx = hierarchy_stack[-1][0] if hierarchy_stack else None
            root_idx = hierarchy_stack[0][0] if hierarchy_stack else idx

            # Determine position in document
            if idx == 0:
                position = "intro"
            elif idx == len(text_elements) - 1:
                position = "conclusion"
            else:
                position = "middle"

            # Initialize hierarchy metadata
            hierarchy_map[idx] = {
                "full_path": " > ".join(breadcrumbs),
                "breadcrumbs": breadcrumbs,
                "depth": len(breadcrumbs),
                "level": level,
                "parent_idx": parent_idx,
                "root_idx": root_idx,
                "parent_section": breadcrumbs[-2] if len(breadcrumbs) > 1 else None,
                "root_section": breadcrumbs[0] if breadcrumbs else None,
                "children_indices": [],
                "sibling_indices": [],
                "position_in_doc": position,
                "section_index": idx,
            }

            # Update parent's children list
            if parent_idx is not None:
                hierarchy_map[parent_idx]["children_indices"].append(idx)

            # Add to stack for future children
            hierarchy_stack.append((idx, level, header_text))

        # Second pass: identify siblings (same parent, same level)
        for idx in hierarchy_map:
            parent_idx = hierarchy_map[idx]["parent_idx"]
            current_level = hierarchy_map[idx]["level"]

            # Find siblings: sections with same parent and level
            for other_idx in hierarchy_map:
                if other_idx != idx:
                    if (hierarchy_map[other_idx]["parent_idx"] == parent_idx and
                        hierarchy_map[other_idx]["level"] == current_level):
                        hierarchy_map[idx]["sibling_indices"].append(other_idx)

        # Third pass: add navigation hints
        for idx in hierarchy_map:
            siblings = hierarchy_map[idx]["sibling_indices"]
            if siblings:
                # Sort siblings by index
                siblings_sorted = sorted(siblings)
                current_pos = siblings_sorted.index(idx) if idx in siblings_sorted else -1

                # Add previous/next section hints
                if current_pos > 0:
                    prev_idx = siblings_sorted[current_pos - 1]
                    hierarchy_map[idx]["previous_section_idx"] = prev_idx
                    hierarchy_map[idx]["previous_section"] = hierarchy_map[prev_idx]["full_path"]

                if current_pos < len(siblings_sorted) - 1:
                    next_idx = siblings_sorted[current_pos + 1]
                    hierarchy_map[idx]["next_section_idx"] = next_idx
                    hierarchy_map[idx]["next_section"] = hierarchy_map[next_idx]["full_path"]

        return hierarchy_map

    def _extract_text_sections(self, markdown: str) -> List[Dict]:
        """
        Extract text sections from markdown, grouping by headers.

        Args:
            markdown: Markdown content (possibly with table markers)

        Returns:
            List of text section dictionaries with metadata
        """
        text_elements = []

        # Split by headers while preserving header hierarchy
        # Regex to match markdown headers (# Header, ## Header, etc.)
        header_pattern = r'^(#{1,6})\s+(.+)$'

        lines = markdown.split('\n')
        current_section = {
            "header_level": 0,
            "header_text": "Document Start",
            "content": [],
            "line_start": 0,
        }

        for line_num, line in enumerate(lines):
            header_match = re.match(header_pattern, line)

            if header_match:
                # Save previous section if it has content
                if current_section["content"]:
                    section_text = '\n'.join(current_section["content"]).strip()
                    if section_text:  # Only add non-empty sections
                        token_count = len(self.tokenizer.encode(section_text))
                        text_elements.append({
                            "content": section_text,
                            "chunk_type": "text",
                            "header_level": current_section["header_level"],
                            "header_text": current_section["header_text"],
                            "token_count": token_count,
                            "char_count": len(section_text),
                            "line_start": current_section["line_start"],
                            "line_end": line_num - 1,
                            "metadata": {
                                "section_header": current_section["header_text"],
                                "header_depth": current_section["header_level"],
                            },
                        })

                # Start new section
                header_level = len(header_match.group(1))
                header_text = header_match.group(2).strip()
                current_section = {
                    "header_level": header_level,
                    "header_text": header_text,
                    "content": [line],  # Include header in content
                    "line_start": line_num,
                }
            else:
                # Add line to current section
                current_section["content"].append(line)

        # Don't forget the last section
        if current_section["content"]:
            section_text = '\n'.join(current_section["content"]).strip()
            if section_text:
                token_count = len(self.tokenizer.encode(section_text))
                text_elements.append({
                    "content": section_text,
                    "chunk_type": "text",
                    "header_level": current_section["header_level"],
                    "header_text": current_section["header_text"],
                    "token_count": token_count,
                    "char_count": len(section_text),
                    "line_start": current_section["line_start"],
                    "line_end": len(lines) - 1,
                    "metadata": {
                        "section_header": current_section["header_text"],
                        "header_depth": current_section["header_level"],
                    },
                })

        return text_elements

    def _compute_file_hash(self, file_path: str) -> str:
        """
        Compute MD5 hash of a file for deduplication.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash string
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def parse_pdf_to_markdown(self, file_path: str) -> str:
        """
        Simple helper to get just the markdown content.

        Args:
            file_path: Path to the PDF file

        Returns:
            Markdown string
        """
        result = self.parse_pdf(file_path)
        return result["markdown_content"]

    def parse_text_file(self, file_path: str) -> Dict:
        """
        Parse a plain text file (for compatibility with DocumentUploader).

        Note: Text files are returned as-is with minimal processing.
        No markdown conversion is performed since they're already plain text.

        Args:
            file_path: Path to the text file

        Returns:
            Dictionary with text content and metadata
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        # Read text content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Compute file hash
        file_hash = self._compute_file_hash(file_path)

        # Calculate token count
        token_count = len(self.tokenizer.encode(content))

        # Create simple structure similar to parse_pdf
        metadata = {
            "file_name": file_path_obj.name,
            "file_path": str(file_path_obj.absolute()),
            "file_hash": file_hash,
            "total_chars": len(content),
            "total_tokens": token_count,
            "num_tables": 0,
            "num_sections": 1,
        }

        return {
            "markdown_content": content,
            "metadata": metadata,
            "table_elements": [],
            "text_elements": [{
                "content": content,
                "chunk_type": "text",
                "header_level": 0,
                "header_text": "Text File",
                "token_count": token_count,
                "char_count": len(content),
                "line_start": 0,
                "line_end": len(content.split('\n')),
                "metadata": {
                    "section_header": "Text File",
                    "header_depth": 0,
                },
            }],
            "file_hash": file_hash,
        }

    def get_document_stats(self, file_path: str) -> Dict:
        """
        Get statistics about a document without full parsing.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with document statistics
        """
        result = self.parse_pdf(file_path)
        return {
            "file_name": result["metadata"]["file_name"],
            "total_tokens": result["metadata"]["total_tokens"],
            "total_chars": result["metadata"]["total_chars"],
            "num_tables": result["metadata"]["num_tables"],
            "num_sections": result["metadata"]["num_sections"],
            "large_tables": len([t for t in result["table_elements"] if t["is_large"]]),
            "inline_tables": len([t for t in result["table_elements"] if not t["is_large"]]),
        }
