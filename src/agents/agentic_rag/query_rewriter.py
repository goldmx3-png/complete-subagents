"""
Query Rewriter Module

Rewrites queries for better semantic search when initial retrieval fails.
Optimizes queries for banking document retrieval.
"""

from typing import Optional, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from src.llm.openrouter_client import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Query rewriting prompt
REWRITE_PROMPT = """You are a query optimization expert for a banking document search system.

Original Query: {original_query}

Context: The initial retrieval did not find relevant documents. The query may be too vague, use informal language, or lack banking-specific terminology.

Your task: Rewrite the query to be more specific and optimized for semantic search in banking documents.

Guidelines:
1. Use formal banking terminology
2. Be more specific about what information is needed
3. Include relevant keywords (account, policy, procedure, rate, etc.)
4. Expand abbreviations
5. Keep it concise (1-2 sentences max)

Examples:
- "what's the rate" → "What is the interest rate for savings accounts?"
- "how open account" → "What is the procedure to open a new checking account?"
- "transfer money" → "How do I transfer money between my accounts?"
- "card benefits" → "What are the benefits and features of the credit card?"

IMPORTANT: Respond with ONLY the rewritten query. Do not include any explanations, introductions, or formatting.

Rewritten Query:"""


# Cache for common query rewrites
QUERY_REWRITE_CACHE = {
    "what's the rate": "What is the interest rate for savings accounts",
    "how open account": "What is the procedure to open a checking account",
    "transfer money": "How do I transfer money between accounts",
    "card benefits": "What are the credit card benefits and features",
    "loan apply": "What is the process to apply for a personal loan",
    "account types": "What are the different types of bank accounts available",
    "fees charges": "What are the account fees and service charges",
    "online banking": "How do I set up and use online banking services",
}


def rewrite_query(
    original_query: str,
    context: Optional[str] = None,
    use_cache: bool = True
) -> str:
    """
    Rewrite a query for better retrieval.

    Args:
        original_query: The original user query
        context: Optional context about why the query failed
        use_cache: Whether to use the query rewrite cache

    Returns:
        Rewritten query string
    """
    logger.info(f"Rewriting query: {original_query}")

    # Check cache first
    if use_cache:
        query_lower = original_query.lower().strip()
        if query_lower in QUERY_REWRITE_CACHE:
            cached_rewrite = QUERY_REWRITE_CACHE[query_lower]
            logger.info(f"Using cached rewrite: {cached_rewrite}")
            return cached_rewrite

    try:
        # Get LLM client
        llm = get_llm(model_type="main")

        # Format prompt
        prompt = REWRITE_PROMPT.format(original_query=original_query)

        # Add context if provided
        if context:
            prompt += f"\n\nAdditional Context: {context[:500]}"

        # Get rewritten query - use sync wrapper for async chat
        from src.utils.async_helpers import run_async_in_new_loop

        messages = [{"role": "user", "content": prompt}]

        # Run async chat in sync context using helper
        rewritten_query = run_async_in_new_loop(llm.chat(messages))

        # Clean up the response (remove quotes, extra whitespace, extra text)
        rewritten_query = rewritten_query.strip('"\'').strip()

        # If the response contains multiple lines or extra formatting, extract just the query
        lines = [line.strip() for line in rewritten_query.split('\n') if line.strip()]
        if len(lines) > 0:
            # Take the first substantial line that looks like a query
            for line in lines:
                # Skip lines that are headers or formatting
                if not line.startswith('**') and not line.startswith('#') and not line.startswith('Here'):
                    rewritten_query = line.strip('"\'*').strip()
                    break

        logger.info(
            f"Query rewritten successfully",
            extra={
                "original": original_query,
                "rewritten": rewritten_query
            }
        )

        return rewritten_query

    except Exception as e:
        logger.error(f"Error rewriting query: {e}", exc_info=True)
        # Fallback: return original query
        return original_query


def rewrite_query_with_feedback(
    original_query: str,
    retrieved_docs: list,
    issues: list
) -> str:
    """
    Rewrite query based on specific feedback about retrieval issues.

    Args:
        original_query: Original query
        retrieved_docs: Documents that were retrieved but not relevant
        issues: List of issues identified (e.g., "too vague", "missing keywords")

    Returns:
        Rewritten query
    """
    logger.info("Rewriting query with feedback")

    try:
        llm = get_llm(model_type="main")

        # Build feedback context
        feedback_context = f"Issues with original query: {', '.join(issues)}"

        if retrieved_docs:
            doc_previews = "\n".join([
                f"- {doc.get('content', '')[:100]}..."
                for doc in retrieved_docs[:3]
            ])
            feedback_context += f"\n\nRetrieved documents (not relevant):\n{doc_previews}"

        # Rewrite with feedback
        prompt = f"""{REWRITE_PROMPT.format(original_query=original_query)}

Feedback on why retrieval failed:
{feedback_context}

Consider this feedback when rewriting the query.

Rewritten Query:"""

        # Get rewritten query - use sync wrapper for async chat
        from src.utils.async_helpers import run_async_in_new_loop

        messages = [{"role": "user", "content": prompt}]

        # Run async chat in sync context using helper
        rewritten_query = run_async_in_new_loop(llm.chat(messages))

        rewritten_query = rewritten_query.strip().strip('"\'')

        logger.info(
            f"Query rewritten with feedback",
            extra={
                "original": original_query,
                "rewritten": rewritten_query,
                "feedback": issues
            }
        )

        return rewritten_query

    except Exception as e:
        logger.error(f"Error rewriting query with feedback: {e}")
        # Fallback to simple rewrite
        return rewrite_query(original_query, use_cache=False)


def expand_banking_abbreviations(query: str) -> str:
    """
    Expand common banking abbreviations in the query.

    Args:
        query: Query with potential abbreviations

    Returns:
        Query with expanded terms
    """
    abbreviations = {
        "acct": "account",
        "a/c": "account",
        "txn": "transaction",
        "intl": "international",
        "dom": "domestic",
        "apr": "annual percentage rate",
        "apy": "annual percentage yield",
        "atm": "automated teller machine",
        "ach": "automated clearing house",
        "eft": "electronic funds transfer",
        "pin": "personal identification number",
        "cd": "certificate of deposit",
        "ira": "individual retirement account",
        "roi": "return on investment",
    }

    expanded_query = query.lower()
    for abbr, full in abbreviations.items():
        # Replace whole words only
        expanded_query = expanded_query.replace(f" {abbr} ", f" {full} ")
        if expanded_query.startswith(f"{abbr} "):
            expanded_query = f"{full} " + expanded_query[len(abbr)+1:]
        if expanded_query.endswith(f" {abbr}"):
            expanded_query = expanded_query[:-len(abbr)-1] + f" {full}"

    return expanded_query


def add_query_context(query: str) -> str:
    """
    Add implicit banking context to short queries.

    Args:
        query: Short query

    Returns:
        Query with added context
    """
    # Detect short queries that need context
    words = query.split()

    if len(words) <= 3:
        # Add banking context based on keywords
        if any(word in query.lower() for word in ["rate", "rates", "interest"]):
            return f"What is the {query} for banking products"
        elif any(word in query.lower() for word in ["fee", "fees", "charge"]):
            return f"What are the {query} for bank services"
        elif any(word in query.lower() for word in ["open", "close", "apply"]):
            return f"How do I {query} a bank account or service"

    return query
