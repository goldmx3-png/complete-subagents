"""
Chunker Factory for selecting the appropriate document chunker based on configuration.

This factory resolves the issue where configuration flags weren't being respected
and DocumentUploader always defaulted to the basic DocumentChunker.
"""

from typing import Optional
from src.config import settings
from src.document_processing.semantic_chunker import SemanticChunker


class ChunkerFactory:
    """
    Factory class for creating document chunkers based on configuration.

    Selection priority:
    1. USE_MARKDOWN_CHUNKING → MarkdownChunker (default)
    2. USE_SEMANTIC_CHUNKING → SemanticChunker
    """

    @staticmethod
    def create_chunker():
        """
        Create and return the appropriate chunker based on configuration flags.

        Returns:
            A chunker instance configured according to environment settings.

        Raises:
            ValueError: If no valid chunker is configured.

        Note:
            This method checks flags in priority order and returns the first
            enabled chunker.
        """
        # Priority 1: Markdown-based chunking (default)
        if settings.use_markdown_chunking:
            from src.document_processing.markdown_chunker import MarkdownChunker
            return MarkdownChunker(
                chunk_size_tokens=settings.markdown_chunk_size_tokens,
                chunk_overlap_percentage=settings.markdown_chunk_overlap_percentage,
                table_size_threshold=settings.markdown_table_size_threshold,
                preserve_headers=settings.markdown_preserve_headers,
            )

        # Priority 2: Semantic chunking (LLM-based boundary detection)
        if settings.use_semantic_chunking:
            return SemanticChunker(
                min_tokens=settings.semantic_chunk_min_tokens,
                max_tokens=settings.semantic_chunk_max_tokens,
                model_name=settings.semantic_chunk_model,
            )

        # No chunker configured
        raise ValueError(
            "No chunker configured. Please enable either USE_MARKDOWN_CHUNKING or USE_SEMANTIC_CHUNKING in your .env file."
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
        else:
            return "None"

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

        return info
