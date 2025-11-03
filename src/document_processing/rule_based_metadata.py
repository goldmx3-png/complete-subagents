"""
Rule-based metadata extraction for document chunks
Fast, deterministic metadata extraction without LLM dependency
"""

from typing import List, Dict
import re
from collections import Counter
import string
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RuleBasedMetadataExtractor:
    """
    Fast rule-based metadata extraction for document chunks

    Extracts:
    - Title: First sentence or heading
    - Summary: First 2-3 sentences
    - Keywords: TF-IDF based keyword extraction

    No LLM calls required - instant processing
    """

    def __init__(self):
        """Initialize rule-based metadata extractor"""
        # Common stopwords for keyword extraction
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'them', 'their', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'just', 'about', 'if', 'out', 'up', 'down', 'then', 'once'
        }

    def extract_metadata(
        self,
        chunk: Dict,
        extract_title: bool = True,
        extract_summary: bool = True,
        extract_keywords: bool = True
    ) -> Dict:
        """
        Extract metadata for a single chunk

        Args:
            chunk: Chunk dict with 'text' field
            extract_title: Whether to extract title
            extract_summary: Whether to extract summary
            extract_keywords: Whether to extract keywords

        Returns:
            Updated chunk with metadata
        """
        text = chunk.get("text", "")
        if not text or len(text) < 20:
            return chunk

        metadata = chunk.get("metadata", {})

        try:
            # Extract title
            if extract_title and not metadata.get("title"):
                title = self._extract_title(text)
                metadata["title"] = title

            # Extract summary
            if extract_summary and not metadata.get("summary"):
                summary = self._extract_summary(text)
                metadata["summary"] = summary

            # Extract keywords
            if extract_keywords and not metadata.get("keywords"):
                keywords = self._extract_keywords(text)
                metadata["keywords"] = keywords

            chunk["metadata"] = metadata

        except Exception as e:
            logger.error(f"Rule-based metadata extraction error: {str(e)}")

        return chunk

    def extract_metadata_batch(
        self,
        chunks: List[Dict],
        **extraction_options
    ) -> List[Dict]:
        """
        Extract metadata for multiple chunks

        Fast synchronous processing - no batching or delays needed

        Args:
            chunks: List of chunk dicts
            **extraction_options: Options for extract_metadata

        Returns:
            List of enriched chunks
        """
        enriched_chunks = []

        for chunk in chunks:
            try:
                enriched_chunk = self.extract_metadata(chunk, **extraction_options)
                enriched_chunks.append(enriched_chunk)
            except Exception as e:
                logger.error(f"Error enriching chunk: {str(e)}")
                enriched_chunks.append(chunk)  # Use original on error

        logger.info(f"Rule-based metadata extraction complete: {len(enriched_chunks)} chunks processed")
        return enriched_chunks

    def _extract_title(self, text: str, max_length: int = 100) -> str:
        """
        Extract title from text

        Strategy:
        1. Look for heading patterns (##, numbered sections, ALL CAPS lines)
        2. Fall back to first sentence
        3. Limit to max_length characters

        Args:
            text: Input text
            max_length: Maximum title length

        Returns:
            Extracted title
        """
        lines = text.strip().split('\n')

        # Check first few lines for headings
        for i, line in enumerate(lines[:3]):
            line = line.strip()
            if not line:
                continue

            # Markdown heading
            if line.startswith('#'):
                title = re.sub(r'^#+\s*', '', line)
                return title[:max_length]

            # Numbered section (e.g., "1. Introduction" or "1.1 Overview")
            if re.match(r'^\d+[\.\)]\s+', line):
                title = re.sub(r'^\d+[\.\)]\s+', '', line)
                return title[:max_length]

            # ALL CAPS line (likely a heading)
            if line.isupper() and len(line.split()) <= 10:
                return line[:max_length]

            # Line ending with colon (likely a section header)
            if line.endswith(':') and len(line.split()) <= 10:
                return line[:-1][:max_length]

        # Fall back to first sentence
        sentences = self._split_sentences(text)
        if sentences:
            return sentences[0][:max_length]

        # Last resort: first N characters
        return text[:max_length].strip()

    def _extract_summary(self, text: str, num_sentences: int = 2) -> str:
        """
        Extract summary from text

        Strategy: Return first 2-3 sentences

        Args:
            text: Input text
            num_sentences: Number of sentences to include

        Returns:
            Summary text
        """
        sentences = self._split_sentences(text)

        if not sentences:
            return text[:200]

        # Take first N sentences
        summary_sentences = sentences[:num_sentences]
        summary = ' '.join(summary_sentences)

        # Limit to reasonable length
        if len(summary) > 500:
            summary = summary[:500] + '...'

        return summary

    def _extract_keywords(self, text: str, top_n: int = 8) -> List[str]:
        """
        Extract keywords using frequency and basic TF-IDF

        Strategy:
        1. Tokenize and filter stopwords
        2. Score by frequency
        3. Boost multi-word phrases
        4. Return top N keywords

        Args:
            text: Input text
            top_n: Number of keywords to return

        Returns:
            List of keywords
        """
        # Extract single words
        words = self._tokenize(text)

        # Count word frequencies
        word_freq = Counter(words)

        # Extract phrases (2-3 word combinations that appear together)
        phrases = self._extract_phrases(text)
        phrase_freq = Counter(phrases)

        # Combine words and phrases with scoring
        candidates = []

        # Add words
        for word, freq in word_freq.most_common(top_n * 2):
            if len(word) >= 3:  # Skip very short words
                # Score based on frequency and length
                score = freq * (1 + len(word) * 0.1)
                candidates.append((word, score))

        # Add phrases (with boost)
        for phrase, freq in phrase_freq.most_common(top_n):
            if freq >= 2:  # Only phrases that appear multiple times
                score = freq * 2.0  # Phrases get higher weight
                candidates.append((phrase, score))

        # Sort by score and return top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        keywords = [word for word, score in candidates[:top_n]]

        return keywords

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words

        Args:
            text: Input text

        Returns:
            List of tokens (words)
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation except hyphens in words
        text = re.sub(r'[^\w\s-]', ' ', text)

        # Extract words (alphanumeric + hyphen)
        words = re.findall(r'\b[a-z][a-z0-9-]*\b', text)

        # Filter stopwords and short words
        words = [w for w in words if w not in self.stopwords and len(w) >= 3]

        return words

    def _extract_phrases(self, text: str, min_freq: int = 2) -> List[str]:
        """
        Extract common multi-word phrases

        Args:
            text: Input text
            min_freq: Minimum frequency to consider

        Returns:
            List of phrases
        """
        # Clean and prepare text
        text = text.lower()
        text = re.sub(r'[^\w\s-]', ' ', text)

        # Extract bigrams and trigrams
        words = text.split()
        phrases = []

        # Bigrams
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if (w1 not in self.stopwords and w2 not in self.stopwords and
                len(w1) >= 3 and len(w2) >= 3):
                phrases.append(f"{w1} {w2}")

        # Trigrams
        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i + 1], words[i + 2]
            # Allow middle stopword (e.g., "point of sale")
            if (w1 not in self.stopwords and w3 not in self.stopwords and
                len(w1) >= 3 and len(w3) >= 3):
                phrases.append(f"{w1} {w2} {w3}")

        return phrases

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting on common delimiters
        # Handle abbreviations like "Dr.", "Mr.", etc.
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Sr|Jr|vs|etc|e\.g|i\.e)\.', r'\1<PERIOD>', text)

        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)

        # Restore periods
        sentences = [s.replace('<PERIOD>', '.') for s in sentences]

        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences
