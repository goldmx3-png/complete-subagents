"""
Answer Validator Module

Validates generated answers for:
- Grounding in retrieved context (no hallucinations)
- Completeness (addresses full query)
- Accuracy (no banking misinformation)
"""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage
from src.llm.openrouter_client import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Validation prompt template
VALIDATION_PROMPT = """You are an answer validator for a banking support system.

Your task is to validate if the generated answer is accurate, complete, and grounded in the provided context.

Query: {query}

Retrieved Context:
{context}

Generated Answer:
{answer}

Validation Checklist:
1. **Grounding**: Is the answer based ONLY on information from the context? (No made-up facts)
2. **Completeness**: Does the answer fully address all parts of the query?
3. **Accuracy**: Are all banking details (rates, policies, procedures) correct?

Respond in the following format:
GROUNDING: yes/no
COMPLETENESS: yes/no
ACCURACY: yes/no
ISSUES: [list any specific problems, or "none"]
OVERALL: valid/invalid

Your validation:"""


def validate_answer(
    query: str,
    context: str,
    answer: str
) -> Dict[str, Any]:
    """
    Validate a generated answer against the query and context.

    Args:
        query: Original user query
        context: Retrieved context documents (concatenated)
        answer: Generated answer to validate

    Returns:
        Validation result dict with:
            - is_valid: bool
            - issues: List[str]
            - grounding_check: bool
            - completeness_check: bool
            - accuracy_check: bool
            - confidence: float
    """
    logger.info("Validating generated answer")

    try:
        # Get LLM client with low temperature for consistent validation
        llm = get_llm(model_type="main")

        # Format validation prompt
        prompt = VALIDATION_PROMPT.format(
            query=query,
            context=context[:3000],  # Limit context length
            answer=answer
        )

        # Get validation response - use sync wrapper for async chat
        from src.utils.async_helpers import run_async_in_new_loop

        messages = [{"role": "user", "content": prompt}]

        # Run async chat in sync context using helper
        validation_text = run_async_in_new_loop(llm.chat(messages))

        validation_text = validation_text.strip()

        # Parse validation response
        result = parse_validation_response(validation_text)

        # Log validation result
        logger.info(
            f"Validation complete: {'VALID' if result['is_valid'] else 'INVALID'}",
            extra={
                "is_valid": result["is_valid"],
                "issues": result["issues"],
                "grounding": result["grounding_check"],
                "completeness": result["completeness_check"],
                "accuracy": result["accuracy_check"]
            }
        )

        return result

    except Exception as e:
        logger.error(f"Error during validation: {e}")
        # On error, assume invalid to be safe
        return {
            "is_valid": False,
            "issues": [f"Validation error: {str(e)}"],
            "grounding_check": False,
            "completeness_check": False,
            "accuracy_check": False,
            "confidence": 0.0,
            "raw_response": ""
        }


def parse_validation_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the LLM validation response into structured format.

    Args:
        response_text: Raw LLM response

    Returns:
        Parsed validation result
    """
    # Convert to lowercase for easier parsing
    text_lower = response_text.lower()

    # Parse individual checks
    grounding_check = "grounding: yes" in text_lower
    completeness_check = "completeness: yes" in text_lower
    accuracy_check = "accuracy: yes" in text_lower

    # Parse overall result
    is_valid = "overall: valid" in text_lower

    # Extract issues
    issues = []
    if "issues:" in text_lower:
        issues_section = response_text.split("ISSUES:")[-1].split("OVERALL:")[0].strip()
        if "none" not in issues_section.lower():
            # Split by newlines or commas
            issues = [
                issue.strip()
                for issue in issues_section.replace("\n", ",").split(",")
                if issue.strip() and issue.strip() != "-"
            ]

    # Calculate confidence score
    checks_passed = sum([grounding_check, completeness_check, accuracy_check])
    confidence = checks_passed / 3.0

    return {
        "is_valid": is_valid and checks_passed >= 2,  # Need at least 2/3 checks
        "issues": issues,
        "grounding_check": grounding_check,
        "completeness_check": completeness_check,
        "accuracy_check": accuracy_check,
        "confidence": confidence,
        "raw_response": response_text
    }


def validate_with_early_exit(
    query: str,
    context: str,
    answer: str,
    retrieval_score: float = 0.0,
    enable_early_exit: bool = True
) -> Dict[str, Any]:
    """
    Validate answer with optional early exit for high-confidence cases.

    Args:
        query: Original query
        context: Retrieved context
        answer: Generated answer
        retrieval_score: Average retrieval confidence score
        enable_early_exit: Whether to skip validation for high-confidence cases

    Returns:
        Validation result
    """
    # Early exit optimization: skip validation if retrieval score is very high
    if enable_early_exit and retrieval_score > 0.9:
        logger.info(
            f"Early exit: High retrieval score ({retrieval_score:.2f}), skipping validation"
        )
        return {
            "is_valid": True,
            "issues": [],
            "grounding_check": True,
            "completeness_check": True,
            "accuracy_check": True,
            "confidence": retrieval_score,
            "raw_response": "Early exit - high confidence",
            "early_exit": True
        }

    # Otherwise, perform full validation
    result = validate_answer(query, context, answer)
    result["early_exit"] = False
    return result


def get_validation_feedback(validation_result: Dict[str, Any]) -> str:
    """
    Convert validation result to human-readable feedback.

    Args:
        validation_result: Output from validate_answer

    Returns:
        Feedback string for debugging/logging
    """
    if validation_result["is_valid"]:
        return "Answer is valid and meets all quality criteria."

    feedback_parts = ["Answer validation failed:"]

    if not validation_result["grounding_check"]:
        feedback_parts.append("- Not properly grounded in context (possible hallucination)")

    if not validation_result["completeness_check"]:
        feedback_parts.append("- Does not fully address the query")

    if not validation_result["accuracy_check"]:
        feedback_parts.append("- Contains potential inaccuracies")

    if validation_result["issues"]:
        feedback_parts.append(f"- Specific issues: {', '.join(validation_result['issues'])}")

    return "\n".join(feedback_parts)
