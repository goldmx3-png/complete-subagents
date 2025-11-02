"""
Metadata extraction for document enrichment
Extracts titles, summaries, keywords, and hypothetical questions using LLM
"""

from typing import List, Dict, Optional
import re
from src.llm.openrouter_client import OpenRouterClient
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetadataExtractor:
    """
    Extract metadata from document chunks to improve retrieval

    Features:
    - Title extraction
    - Summary generation
    - Keyword extraction
    - Hypothetical question generation (for better semantic search)
    """

    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        """
        Initialize metadata extractor

        Args:
            llm_client: Optional LLM client (uses classifier model for speed)
        """
        self.llm = llm_client or OpenRouterClient(model=settings.classifier_model)

    async def extract_metadata(
        self,
        chunk: Dict,
        extract_title: bool = True,
        extract_summary: bool = True,
        extract_keywords: bool = True,
        extract_questions: bool = False
    ) -> Dict:
        """
        Extract all metadata for a chunk

        Args:
            chunk: Chunk dict with 'text' field
            extract_title: Whether to extract title
            extract_summary: Whether to extract summary
            extract_keywords: Whether to extract keywords
            extract_questions: Whether to generate hypothetical questions

        Returns:
            Updated chunk with metadata
        """
        text = chunk.get("text", "")
        if not text or len(text) < 50:
            return chunk

        metadata = chunk.get("metadata", {})

        try:
            # Extract title (if not already present)
            if extract_title and not metadata.get("title"):
                title = await self._extract_title(text)
                metadata["title"] = title

            # Extract summary
            if extract_summary and not metadata.get("summary"):
                summary = await self._extract_summary(text)
                metadata["summary"] = summary

            # Extract keywords
            if extract_keywords and not metadata.get("keywords"):
                keywords = await self._extract_keywords(text)
                metadata["keywords"] = keywords

            # Generate hypothetical questions
            if extract_questions and not metadata.get("hypothetical_questions"):
                questions = await self._generate_questions(text)
                metadata["hypothetical_questions"] = questions

            chunk["metadata"] = metadata

        except Exception as e:
            logger.error(f"Metadata extraction error: {str(e)}")
            # Return chunk unchanged on error

        return chunk

    async def extract_metadata_batch(
        self,
        chunks: List[Dict],
        **extraction_options
    ) -> List[Dict]:
        """
        Extract metadata for multiple chunks

        Args:
            chunks: List of chunk dicts
            **extraction_options: Options for extract_metadata

        Returns:
            List of enriched chunks
        """
        enriched_chunks = []

        for i, chunk in enumerate(chunks):
            try:
                enriched = await self.extract_metadata(chunk, **extraction_options)
                enriched_chunks.append(enriched)

                if (i + 1) % 10 == 0:
                    logger.info(f"Enriched {i + 1}/{len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Error enriching chunk {i}: {str(e)}")
                enriched_chunks.append(chunk)  # Use original on error

        return enriched_chunks

    async def _extract_title(self, text: str) -> str:
        """
        Extract a concise title for the chunk

        Args:
            text: Chunk text

        Returns:
            Generated title
        """
        prompt = f"""Generate a concise, descriptive title (5-10 words) for the following text.
The title should capture the main topic or concept.

Text:
{text[:500]}

Title:"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50
            )
            title = response.strip().strip('"').strip("'")
            return title

        except Exception as e:
            logger.error(f"Title extraction error: {str(e)}")
            # Fallback: use first sentence or first 100 chars
            sentences = re.split(r'[.!?]', text)
            if sentences:
                return sentences[0][:100].strip()
            return text[:100].strip()

    async def _extract_summary(self, text: str) -> str:
        """
        Generate a concise summary of the chunk

        Args:
            text: Chunk text

        Returns:
            Generated summary
        """
        prompt = f"""Summarize the following text in 1-2 sentences.
Focus on the key information and main points.

Text:
{text}

Summary:"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            return response.strip()

        except Exception as e:
            logger.error(f"Summary extraction error: {str(e)}")
            # Fallback: return first 200 chars
            return text[:200].strip() + "..."

    async def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract relevant keywords/keyphrases

        Args:
            text: Chunk text

        Returns:
            List of keywords
        """
        prompt = f"""Extract 5-10 important keywords or key phrases from the following text.
Return as a comma-separated list.

Text:
{text[:800]}

Keywords:"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )

            # Parse keywords
            keywords = [k.strip() for k in response.split(',')]
            keywords = [k for k in keywords if k and len(k) > 2][:10]
            return keywords

        except Exception as e:
            logger.error(f"Keyword extraction error: {str(e)}")
            # Fallback: simple word frequency
            return self._simple_keyword_extraction(text)

    async def _generate_questions(self, text: str) -> List[str]:
        """
        Generate hypothetical questions that this chunk could answer

        This improves retrieval by allowing semantic search on questions
        rather than just the content.

        Args:
            text: Chunk text

        Returns:
            List of hypothetical questions
        """
        prompt = f"""Generate 2-3 specific questions that the following text could answer.
The questions should be natural and specific to the content.
Return as a numbered list.

Text:
{text}

Questions:"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=200
            )

            # Parse questions
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                # Remove numbering (1., 2., etc.)
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                if line and line.endswith('?'):
                    questions.append(line)

            return questions[:3]

        except Exception as e:
            logger.error(f"Question generation error: {str(e)}")
            return []

    def _simple_keyword_extraction(self, text: str, top_n: int = 10) -> List[str]:
        """
        Fallback simple keyword extraction using word frequency

        Args:
            text: Input text
            top_n: Number of keywords to return

        Returns:
            List of keywords
        """
        # Remove common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        # Tokenize and count
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        word_freq = {}

        for word in words:
            if word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top N
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
