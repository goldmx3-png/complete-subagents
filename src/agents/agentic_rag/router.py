"""
Router Module

Decides whether to use simple RAG (fast path) or agentic RAG (precision path)
based on query complexity.
"""

from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Query patterns that indicate simple queries
SIMPLE_PATTERNS = [
    "what is",
    "who is",
    "when",
    "where",
    "define",
    "meaning of",
]

# Query patterns that indicate complex queries
COMPLEX_PATTERNS = [
    "compare",
    "difference between",
    "differences between",
    "different for",  # e.g., "how is it different for bulk and single"
    "vs",
    "versus",
    "how",  # Generic how question (often requires explanation)
    "how to",
    "how do i",
    "how can i",
    "what are the steps",
    "procedure for",
    "process for",
    "why",
    "explain",
    "advantages and disadvantages",
    "pros and cons",
]


def should_use_agentic_rag(
    query: str,
    min_length: int = 8,
    enable_pattern_matching: bool = True
) -> bool:
    """
    Determine whether to use agentic RAG based on query characteristics.

    Args:
        query: User's query
        min_length: Minimum word count to consider complex
        enable_pattern_matching: Whether to use pattern matching

    Returns:
        True if agentic RAG should be used, False for simple RAG
    """
    query_lower = query.lower().strip()
    word_count = len(query_lower.split())

    # Pattern matching (most reliable)
    # IMPORTANT: Check complex patterns FIRST because a query can have both simple and complex patterns
    # Example: "what is auth matrix and how is it different for bulk vs single?"
    #   - Contains "what is" (simple) BUT also "how", "different for" (complex)
    #   - Should use agentic RAG because it requires comparison/explanation
    if enable_pattern_matching:
        # Check for complex patterns FIRST
        complex_matches = [pattern for pattern in COMPLEX_PATTERNS if pattern in query_lower]
        if complex_matches:
            logger.info(
                f"Routing to AGENTIC RAG (complex patterns detected: {complex_matches[:2]})",
                extra={"query": query[:100]}
            )
            return True

        # Only check simple patterns if no complex patterns found
        simple_matches = [pattern for pattern in SIMPLE_PATTERNS if pattern in query_lower]
        if simple_matches:
            logger.info(
                f"Routing to SIMPLE RAG (only simple patterns detected: {simple_matches[:2]})",
                extra={"query": query[:100]}
            )
            return False

    # Length-based routing
    if word_count <= 5:
        logger.info(
            f"Routing to SIMPLE RAG (short query, {word_count} words)",
            extra={"query": query[:100]}
        )
        return False

    if word_count >= min_length:
        logger.info(
            f"Routing to AGENTIC RAG (long query, {word_count} words)",
            extra={"query": query[:100]}
        )
        return True

    # Multiple questions indicator
    if query.count("?") > 1:
        logger.info(
            f"Routing to AGENTIC RAG (multiple questions)",
            extra={"query": query[:100]}
        )
        return True

    # Default to simple RAG for borderline cases
    logger.info(
        f"Routing to SIMPLE RAG (default for borderline case)",
        extra={"query": query[:100], "word_count": word_count}
    )
    return False


def analyze_query_complexity(query: str) -> Dict[str, Any]:
    """
    Analyze query and return detailed complexity assessment.

    Args:
        query: User's query

    Returns:
        Dict with complexity analysis:
            - complexity_level: "simple" | "medium" | "complex"
            - reasons: List of reasons for the classification
            - confidence: Confidence score (0-1)
            - recommended_strategy: "simple_rag" | "agentic_rag"
    """
    query_lower = query.lower().strip()
    word_count = len(query_lower.split())
    reasons = []
    complexity_score = 0.0

    # Pattern analysis
    simple_pattern_matches = sum(1 for p in SIMPLE_PATTERNS if p in query_lower)
    complex_pattern_matches = sum(1 for p in COMPLEX_PATTERNS if p in query_lower)

    if simple_pattern_matches > 0:
        reasons.append(f"Contains {simple_pattern_matches} simple pattern(s)")
        complexity_score -= 0.3 * simple_pattern_matches

    if complex_pattern_matches > 0:
        reasons.append(f"Contains {complex_pattern_matches} complex pattern(s)")
        complexity_score += 0.4 * complex_pattern_matches

    # Length analysis
    if word_count <= 5:
        reasons.append("Short query (≤5 words)")
        complexity_score -= 0.2
    elif word_count >= 10:
        reasons.append("Long query (≥10 words)")
        complexity_score += 0.3

    # Multi-question analysis
    question_count = query.count("?")
    if question_count > 1:
        reasons.append(f"Contains {question_count} questions")
        complexity_score += 0.4

    # Determine complexity level
    if complexity_score <= -0.2:
        complexity_level = "simple"
        recommended_strategy = "simple_rag"
    elif complexity_score >= 0.3:
        complexity_level = "complex"
        recommended_strategy = "agentic_rag"
    else:
        complexity_level = "medium"
        # Medium queries go to simple RAG by default (balance speed/quality)
        recommended_strategy = "simple_rag"

    # Calculate confidence
    confidence = min(abs(complexity_score) / 0.5, 1.0)

    return {
        "complexity_level": complexity_level,
        "reasons": reasons,
        "confidence": confidence,
        "recommended_strategy": recommended_strategy,
        "word_count": word_count,
        "complexity_score": complexity_score
    }


def get_routing_decision(
    query: str,
    user_preference: str = "auto",
    override: bool = False
) -> Dict[str, Any]:
    """
    Get comprehensive routing decision with analysis.

    Args:
        query: User's query
        user_preference: "auto", "simple", or "agentic"
        override: Whether user preference overrides auto-detection

    Returns:
        Routing decision with analysis
    """
    # Analyze query
    analysis = analyze_query_complexity(query)

    # Handle user preference
    if override and user_preference in ["simple", "agentic"]:
        strategy = f"{user_preference}_rag"
        logger.info(f"User override: forcing {strategy}")
        return {
            **analysis,
            "recommended_strategy": strategy,
            "routing_reason": "user_override"
        }

    # Auto-routing
    use_agentic = should_use_agentic_rag(query)
    analysis["recommended_strategy"] = "agentic_rag" if use_agentic else "simple_rag"
    analysis["routing_reason"] = "auto_detection"

    return analysis
