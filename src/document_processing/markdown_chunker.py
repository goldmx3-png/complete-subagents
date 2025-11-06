"""
Markdown-aware Document Chunker using LangChain splitters.

This chunker implements a two-stage splitting strategy:
1. Split by markdown headers (h1, h2, h3, h4) to respect document structure
2. Apply token-based size constraints using recursive splitting

This approach preserves semantic coherence while ensuring chunks fit within
token limits for optimal RAG performance.
"""

from typing import List, Dict, Optional
import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.config import settings


class MarkdownChunker:
    """
    Chunk markdown documents with header-aware splitting and size constraints.

    Features:
    - Respects markdown header hierarchy (h1-h4)
    - Enforces token-based size limits
    - Preserves section context in metadata
    - Handles inline vs separate table chunks
    - Compatible with existing chunk interface
    """

    def __init__(
        self,
        chunk_size_tokens: int = None,
        chunk_overlap_percentage: int = None,
        table_size_threshold: int = None,
        preserve_headers: bool = None,
    ):
        """
        Initialize the MarkdownChunker.

        Args:
            chunk_size_tokens: Target chunk size in tokens (default from config)
            chunk_overlap_percentage: Overlap percentage (default from config)
            table_size_threshold: Token threshold for table classification (default from config)
            preserve_headers: Whether to preserve header hierarchy (default from config)
        """
        self.chunk_size_tokens = chunk_size_tokens if chunk_size_tokens is not None else settings.markdown_chunk_size_tokens
        self.chunk_overlap_percentage = chunk_overlap_percentage if chunk_overlap_percentage is not None else settings.markdown_chunk_overlap_percentage
        self.table_size_threshold = table_size_threshold if table_size_threshold is not None else settings.markdown_table_size_threshold
        self.preserve_headers = preserve_headers if preserve_headers is not None else settings.markdown_preserve_headers

        # Calculate overlap in tokens
        self.chunk_overlap_tokens = int(self.chunk_size_tokens * (self.chunk_overlap_percentage / 100))

        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Stage 1: Header-based splitter
        self.headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]

        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=not self.preserve_headers,  # Keep headers in content if preserving
        )

        # Stage 2: Token-based recursive splitter
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.chunk_size_tokens,
            chunk_overlap=self.chunk_overlap_tokens,
        )

    def chunk_document(self, doc_data: Dict, doc_id: str, user_id: str) -> List[Dict]:
        """
        Chunk a parsed markdown document into retrieval-optimized chunks.

        Args:
            doc_data: Document data from MarkdownDocumentParser.parse_pdf()
            doc_id: Unique document identifier
            user_id: User identifier

        Returns:
            List of chunk dictionaries ready for embedding and storage
        """
        chunks = []

        # Get hierarchical structure if available
        hierarchy_structure = doc_data.get("hierarchy_structure", {})
        text_elements = doc_data.get("text_elements", [])

        # Process text elements with markdown-aware splitting
        markdown_content = doc_data.get("markdown_content", "")
        text_chunks = self._chunk_markdown_content(
            markdown_content, doc_id, user_id, hierarchy_structure, text_elements
        )
        chunks.extend(text_chunks)

        # Process table elements
        table_elements = doc_data.get("table_elements", [])
        table_chunks = self._process_table_elements(table_elements, doc_id, user_id)
        chunks.extend(table_chunks)

        # Enrich chunks with section-level hierarchy information
        if hierarchy_structure and text_elements:
            chunks = self._enrich_chunk_hierarchy(chunks, hierarchy_structure, text_elements)

        # Add chunk IDs and sequence numbers
        for idx, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"{doc_id}_chunk_{idx}"
            chunk["sequence_number"] = idx

        return chunks

    def _chunk_markdown_content(
        self,
        markdown: str,
        doc_id: str,
        user_id: str,
        hierarchy_structure: Dict = None,
        text_elements: List[Dict] = None
    ) -> List[Dict]:
        """
        Apply two-stage splitting to markdown content.

        Stage 1: Split by headers to preserve document structure
        Stage 2: Apply size constraints with recursive splitting

        Args:
            markdown: Markdown text to chunk
            doc_id: Document identifier
            user_id: User identifier
            hierarchy_structure: Optional hierarchical structure from parser
            text_elements: Optional text elements from parser

        Returns:
            List of chunk dictionaries with metadata
        """
        if hierarchy_structure is None:
            hierarchy_structure = {}
        if text_elements is None:
            text_elements = []
        chunks = []

        try:
            # Stage 1: Split by markdown headers
            md_header_splits = self.markdown_splitter.split_text(markdown)

            # Stage 2: Apply size constraints
            final_splits = self.text_splitter.split_documents(md_header_splits)

            for idx, doc in enumerate(final_splits):
                # Extract header hierarchy from metadata
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}

                # Build section hierarchy
                section_hierarchy = {}
                for level in ["h1", "h2", "h3", "h4"]:
                    if level in metadata:
                        section_hierarchy[level] = metadata[level]

                # Calculate token count
                token_count = len(self.tokenizer.encode(doc.page_content))

                # Determine if chunk contains table markers
                has_inline_tables = "[TABLE_" in doc.page_content and "_LARGE]" not in doc.page_content
                has_table_markers = "[TABLE_" in doc.page_content

                chunk_dict = {
                    "text": doc.page_content,
                    "user_id": user_id,
                    "doc_id": doc_id,
                    "chunk_type": "text_with_table" if has_inline_tables else "text",
                    "metadata": {
                        "section_hierarchy": section_hierarchy,
                        "token_count": token_count,
                        "char_count": len(doc.page_content),
                        "chunking_method": "markdown_header_recursive",
                        "has_inline_tables": has_inline_tables,
                        "has_table_markers": has_table_markers,
                        "header_context": self._format_header_context(section_hierarchy),
                    },
                }

                chunks.append(chunk_dict)

        except Exception as e:
            # Fallback: If header splitting fails, use recursive splitting only
            print(f"Warning: Header splitting failed, falling back to recursive splitting. Error: {e}")
            fallback_splits = self.text_splitter.split_text(markdown)

            for idx, text in enumerate(fallback_splits):
                token_count = len(self.tokenizer.encode(text))

                chunk_dict = {
                    "text": text,
                    "user_id": user_id,
                    "doc_id": doc_id,
                    "chunk_type": "text",
                    "metadata": {
                        "token_count": token_count,
                        "char_count": len(text),
                        "chunking_method": "markdown_fallback_recursive",
                        "fallback_reason": "header_splitting_failed",
                    },
                }

                chunks.append(chunk_dict)

        return chunks

    def _process_table_elements(self, table_elements: List[Dict], doc_id: str, user_id: str) -> List[Dict]:
        """
        Process table elements from MarkdownDocumentParser.

        Large tables become separate chunks.
        Small tables are already inline in the markdown content.

        Args:
            table_elements: List of table dictionaries from parser
            doc_id: Document identifier
            user_id: User identifier

        Returns:
            List of chunk dictionaries for large tables only
        """
        chunks = []

        for table_elem in table_elements:
            # Only create separate chunks for large tables
            if table_elem.get("is_large", False):
                chunk_dict = {
                    "text": table_elem["content"],
                    "user_id": user_id,
                    "doc_id": doc_id,
                    "chunk_type": "table_large",
                    "metadata": {
                        "token_count": table_elem["token_count"],
                        "char_count": table_elem["char_count"],
                        "table_index": table_elem["table_index"],
                        "num_rows": table_elem["num_rows"],
                        "num_cols": table_elem["num_cols"],
                        "chunking_method": "markdown_table_extraction",
                        "size_classification": "large",
                    },
                }

                chunks.append(chunk_dict)

        return chunks

    def _enrich_chunk_hierarchy(
        self,
        chunks: List[Dict],
        hierarchy_structure: Dict,
        text_elements: List[Dict]
    ) -> List[Dict]:
        """
        Enrich chunks with detailed hierarchical metadata.

        Maps each chunk back to its source section and adds full hierarchical context
        including breadcrumbs, depth, parent/child relationships, and navigation hints.

        Args:
            chunks: List of chunk dictionaries to enrich
            hierarchy_structure: Hierarchical structure from parser
            text_elements: Text elements from parser with section info

        Returns:
            Enriched chunks with hierarchical metadata
        """
        for chunk in chunks:
            # Skip non-text chunks (tables)
            if chunk["chunk_type"] not in ["text", "text_with_table"]:
                continue

            # Get existing section_hierarchy from chunk metadata
            section_hierarchy = chunk["metadata"].get("section_hierarchy", {})

            # Try to find matching section in hierarchy_structure
            # Match by comparing the header context
            best_match_idx = None
            best_match_score = 0

            for idx, hierarchy_meta in hierarchy_structure.items():
                # Compare headers at each level
                match_score = 0
                for level in ["h1", "h2", "h3", "h4"]:
                    if level in section_hierarchy:
                        level_num = int(level[1])  # Extract number from "h1", "h2", etc.
                        if (len(hierarchy_meta["breadcrumbs"]) >= level_num and
                            hierarchy_meta["breadcrumbs"][level_num - 1] == section_hierarchy[level]):
                            match_score += 1

                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_idx = idx

            # If we found a good match, enrich with hierarchy metadata
            if best_match_idx is not None and best_match_score > 0:
                hierarchy_meta = hierarchy_structure[best_match_idx]

                # Add enhanced hierarchy metadata
                chunk["metadata"]["hierarchy"] = {
                    "full_path": hierarchy_meta["full_path"],
                    "breadcrumbs": hierarchy_meta["breadcrumbs"],
                    "depth": hierarchy_meta["depth"],
                    "level": hierarchy_meta["level"],
                    "parent_section": hierarchy_meta.get("parent_section"),
                    "root_section": hierarchy_meta.get("root_section"),
                    "has_children": len(hierarchy_meta.get("children_indices", [])) > 0,
                    "has_siblings": len(hierarchy_meta.get("sibling_indices", [])) > 0,
                    "position_in_doc": hierarchy_meta.get("position_in_doc", "middle"),
                    "section_index": hierarchy_meta["section_index"],
                }

                # Add navigation hints if available
                if "previous_section" in hierarchy_meta:
                    chunk["metadata"]["hierarchy"]["previous_section"] = hierarchy_meta["previous_section"]
                if "next_section" in hierarchy_meta:
                    chunk["metadata"]["hierarchy"]["next_section"] = hierarchy_meta["next_section"]

                # Store sibling and children section names (not indices)
                if hierarchy_meta.get("sibling_indices"):
                    sibling_names = []
                    for sib_idx in hierarchy_meta["sibling_indices"]:
                        if sib_idx in hierarchy_structure:
                            sibling_names.append(hierarchy_structure[sib_idx]["breadcrumbs"][-1])
                    chunk["metadata"]["hierarchy"]["sibling_sections"] = sibling_names

                if hierarchy_meta.get("children_indices"):
                    children_names = []
                    for child_idx in hierarchy_meta["children_indices"]:
                        if child_idx in hierarchy_structure:
                            children_names.append(hierarchy_structure[child_idx]["breadcrumbs"][-1])
                    chunk["metadata"]["hierarchy"]["children_sections"] = children_names

        return chunks

    def _format_header_context(self, section_hierarchy: Dict) -> str:
        """
        Format section hierarchy into a readable context string.

        Args:
            section_hierarchy: Dictionary with h1-h4 headers

        Returns:
            Formatted context string like "Introduction > Background > History"
        """
        if not section_hierarchy:
            return ""

        # Order by header level
        context_parts = []
        for level in ["h1", "h2", "h3", "h4"]:
            if level in section_hierarchy:
                context_parts.append(section_hierarchy[level])

        return " > ".join(context_parts)

    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """
        Calculate statistics for a set of chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_tokens": 0,
                "avg_chars": 0,
                "text_chunks": 0,
                "table_chunks": 0,
                "chunks_with_tables": 0,
            }

        total_tokens = sum(c["metadata"].get("token_count", 0) for c in chunks)
        total_chars = sum(c["metadata"].get("char_count", 0) for c in chunks)

        text_chunks = len([c for c in chunks if c["chunk_type"] in ["text", "text_with_table"]])
        table_chunks = len([c for c in chunks if c["chunk_type"] == "table_large"])
        chunks_with_tables = len([c for c in chunks if c["metadata"].get("has_inline_tables", False)])

        return {
            "total_chunks": len(chunks),
            "avg_tokens": int(total_tokens / len(chunks)) if chunks else 0,
            "avg_chars": int(total_chars / len(chunks)) if chunks else 0,
            "text_chunks": text_chunks,
            "table_chunks": table_chunks,
            "chunks_with_tables": chunks_with_tables,
            "min_tokens": min(c["metadata"].get("token_count", 0) for c in chunks) if chunks else 0,
            "max_tokens": max(c["metadata"].get("token_count", 0) for c in chunks) if chunks else 0,
        }
