"""
Document parser using pdfplumber (simplified version)
"""

from typing import List, Dict, Optional
import pdfplumber
from pathlib import Path
import hashlib


class DocumentParser:
    """Parse PDF documents with pdfplumber"""

    def __init__(self):
        """Initialize document parser"""
        pass

    def parse_pdf(self, file_path: str, extract_tables: bool = True) -> Dict:
        """
        Parse PDF file using pdfplumber

        Args:
            file_path: Path to PDF file
            extract_tables: Whether to extract tables

        Returns:
            Parsed document dict
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file metadata
        file_size = file_path.stat().st_size
        file_hash = self._compute_file_hash(file_path)

        # Parse with pdfplumber
        text_elements = []
        table_elements = []
        title_elements = []

        with pdfplumber.open(file_path) as pdf:
            num_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text()
                if text:
                    text_elements.append({
                        "type": "Text",
                        "text": text,
                        "metadata": {
                            "page_number": page_num,
                            "source": "pdfplumber"
                        }
                    })

                # Extract tables if requested
                if extract_tables:
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if table:
                            markdown_table = self._table_to_markdown(table)
                            table_elements.append({
                                "type": "Table",
                                "text": markdown_table,
                                "metadata": {
                                    "page_number": page_num,
                                    "table_index": table_idx,
                                    "rows": len(table),
                                    "cols": len(table[0]) if table else 0
                                }
                            })

        return {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_size": file_size,
            "file_hash": file_hash,
            "num_pages": num_pages,
            "text_elements": text_elements,
            "table_elements": table_elements,
            "title_elements": title_elements,
            "all_elements": text_elements + table_elements
        }


    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """
        Convert table to markdown format

        Args:
            table: Table as list of lists

        Returns:
            Markdown formatted table
        """
        if not table:
            return ""

        # Clean empty cells
        table = [[cell if cell else "" for cell in row] for row in table]

        # Header
        header = "| " + " | ".join(str(cell) for cell in table[0]) + " |"

        # Separator
        separator = "| " + " | ".join(["---"] * len(table[0])) + " |"

        # Rows
        rows = []
        for row in table[1:]:
            rows.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join([header, separator] + rows)

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute MD5 hash of file

        Args:
            file_path: Path to file

        Returns:
            MD5 hash string
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def parse_text_file(self, file_path: str) -> Dict:
        """
        Parse plain text file

        Args:
            file_path: Path to text file

        Returns:
            Parsed document dict
        """
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        file_size = file_path.stat().st_size
        file_hash = self._compute_file_hash(file_path)

        return {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_size": file_size,
            "file_hash": file_hash,
            "num_pages": 1,
            "text_elements": [{"type": "Text", "text": text, "metadata": {}}],
            "table_elements": [],
            "title_elements": []
        }

    def parse_docx(self, file_path: str) -> Dict:
        """
        Parse DOCX file - requires markdown parser with docling.

        This is a stub method. DOCX parsing is only supported with MarkdownDocumentParser.
        To enable DOCX support, set USE_MARKDOWN_CHUNKING=true in your configuration.

        Args:
            file_path: Path to DOCX file

        Raises:
            NotImplementedError: Always raised as DOCX requires markdown parser
        """
        raise NotImplementedError(
            "DOCX file parsing requires the MarkdownDocumentParser with docling. "
            "Please set USE_MARKDOWN_CHUNKING=true in your configuration to enable DOCX support."
        )
