"""
Chunker Factory for selecting the appropriate document chunker based on configuration.

This factory resolves the issue where configuration flags weren't being respected
and DocumentUploader always defaulted to the basic DocumentChunker.
"""

from typing import Optional
from src.config import settings
from src.document_processing.chunker import DocumentChunker
from src.document_processing.token_chunker import TokenBasedChunker
from src.document_processing.semantic_chunker import SemanticChunker


class ChunkerFactory:
    """
    Factory class for creating document chunkers based on configuration.

    Selection priority:
    1. USE_MARKDOWN_CHUNKING → MarkdownChunker
    2. USE_SEMANTIC_CHUNKING → SemanticChunker
    3. USE_TOKEN_BASED_CHUNKING → TokenBasedChunker
    4. Default → DocumentChunker (character-based)
    """

    @staticmethod
    def create_chunker():
        """
        Create and return the appropriate chunker based on configuration flags.

        Returns:
            A chunker instance configured according to environment settings.

        Note:
            This method checks flags in priority order and returns the first
            enabled chunker. If no flags are set, returns basic DocumentChunker.
        """
        # Priority 1: Markdown-based chunking (when implemented)
        if settings.use_markdown_chunking:
            # Import here to avoid circular dependencies
            try:
                from src.document_processing.markdown_chunker import MarkdownChunker
                return MarkdownChunker(
                    chunk_size_tokens=settings.markdown_chunk_size_tokens,
                    chunk_overlap_percentage=settings.markdown_chunk_overlap_percentage,
                    table_size_threshold=settings.markdown_table_size_threshold,
                    preserve_headers=settings.markdown_preserve_headers,
                )
            except ImportError:
                print("Warning: MarkdownChunker not yet implemented, falling back to next option")
                # Fall through to next option

        # Priority 2: Semantic chunking (LLM-based boundary detection)
        if settings.use_semantic_chunking:
            return SemanticChunker(
                min_tokens=settings.semantic_chunk_min_tokens,
                max_tokens=settings.semantic_chunk_max_tokens,
                model_name=settings.semantic_chunk_model,
            )

        # Priority 3: Token-based chunking (recommended for production)
        if settings.use_token_based_chunking:
            return TokenBasedChunker(
                chunk_size_tokens=settings.chunk_size_tokens,
                chunk_overlap_percentage=settings.chunk_overlap_percentage,
                preserve_tables=settings.preserve_tables,
            )

        # Default: Character-based chunking (legacy)
        return DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    @staticmethod
    def get_active_chunker_name() -> str:
        """
        Return the name of the currently active chunker based on configuration.

        Returns:
            String identifier of the active chunker type.
        """
        if settings.use_markdown_chunking:
            return "MarkdownChunker"
        elif settings.use_semantic_chunking:
            return "SemanticChunker"
        elif settings.use_token_based_chunking:
            return "TokenBasedChunker"
        else:
            return "DocumentChunker"

    @staticmethod
    def get_chunker_info() -> dict:
        """
        Get detailed information about the active chunker and its configuration.

        Returns:
            Dictionary with chunker type and relevant configuration parameters.
        """
        chunker_name = ChunkerFactory.get_active_chunker_name()

        info = {
            "chunker_type": chunker_name,
            "configuration": {},
        }

        if chunker_name == "MarkdownChunker":
            info["configuration"] = {
                "chunk_size_tokens": settings.markdown_chunk_size_tokens,
                "chunk_overlap_percentage": settings.markdown_chunk_overlap_percentage,
                "table_size_threshold": settings.markdown_table_size_threshold,
                "preserve_headers": settings.markdown_preserve_headers,
            }
        elif chunker_name == "SemanticChunker":
            info["configuration"] = {
                "min_tokens": settings.semantic_chunk_min_tokens,
                "max_tokens": settings.semantic_chunk_max_tokens,
                "model_name": settings.semantic_chunk_model,
            }
        elif chunker_name == "TokenBasedChunker":
            info["configuration"] = {
                "chunk_size_tokens": settings.chunk_size_tokens,
                "chunk_overlap_percentage": settings.chunk_overlap_percentage,
                "preserve_tables": settings.preserve_tables,
            }
        else:  # DocumentChunker
            info["configuration"] = {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            }

        return info
