"""
Document Grader Module

Evaluates retrieved documents for relevance to the query.
Implements both sequential and parallel grading for performance optimization.
"""

import asyncio
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm.openrouter_client import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Grading prompt template
GRADING_PROMPT = """You are a document relevance grader for a banking support system.

Your task is to evaluate if a retrieved document is relevant to answer the user's query.

Query: {query}

Document:
{document}

Instructions:
- Respond with ONLY "yes" if the document contains information that helps answer the query
- Respond with ONLY "no" if the document is not relevant or off-topic
- Be strict: only mark as relevant if it directly addresses the query

Your response (yes/no):"""


def grade_single_document(query: str, document: Dict[str, Any], llm=None) -> Dict[str, Any]:
    """
    Grade a single document for relevance.

    Args:
        query: User's query
        document: Document dict with 'content' and 'metadata'
        llm: LLM client instance (optional, will be created if not provided)

    Returns:
        Document with added 'is_relevant' field
    """
    try:
        # Extract content
        content = document.get("content", document.get("page_content", ""))

        # Format prompt
        prompt = GRADING_PROMPT.format(
            query=query,
            document=content[:1000]  # Limit to first 1000 chars for efficiency
        )

        # Get LLM response - use sync wrapper for async chat
        from src.utils.async_helpers import run_async_in_new_loop

        messages = [{"role": "user", "content": prompt}]

        # Define async function to create LLM and call chat in the new thread
        async def grade_with_llm():
            llm_client = get_llm(model_type="main")
            return await llm_client.chat(messages)

        # Run async chat in sync context using helper
        response = run_async_in_new_loop(grade_with_llm)

        # Parse response
        answer = response.strip().lower()
        is_relevant = "yes" in answer

        # Add relevance field to document
        graded_doc = {
            **document,
            "is_relevant": is_relevant,
            "grading_reason": response
        }

        logger.debug(
            f"Document graded: {is_relevant}",
            extra={"query": query[:50], "doc_preview": content[:100]}
        )

        return graded_doc

    except Exception as e:
        logger.error(f"Error grading document: {e}")
        # On error, assume relevant to be safe
        return {**document, "is_relevant": True, "grading_error": str(e)}


async def grade_single_document_async(
    query: str,
    document: Dict[str, Any],
    llm
) -> Dict[str, Any]:
    """
    Async version of grade_single_document for parallel processing.

    Args:
        query: User's query
        document: Document dict
        llm: LLM client instance

    Returns:
        Graded document
    """
    # Run in executor to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        grade_single_document,
        query,
        document,
        llm
    )


def grade_documents(
    query: str,
    documents: List[Dict[str, Any]],
    threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Grade documents sequentially (slower but simpler).

    Args:
        query: User's query
        documents: List of retrieved documents
        threshold: Minimum relevance threshold (not used in binary grading)

    Returns:
        List of relevant documents only
    """
    logger.info(f"Grading {len(documents)} documents sequentially")

    # Get LLM client
    llm = get_llm(model_type="main")

    # Grade each document
    graded_docs = []
    for i, doc in enumerate(documents):
        graded_doc = grade_single_document(query, doc, llm)
        graded_docs.append(graded_doc)
        logger.debug(f"Graded document {i+1}/{len(documents)}: {graded_doc['is_relevant']}")

    # Filter to relevant only
    relevant_docs = [doc for doc in graded_docs if doc["is_relevant"]]

    logger.info(
        f"Grading complete: {len(relevant_docs)}/{len(documents)} relevant",
        extra={
            "total_docs": len(documents),
            "relevant_docs": len(relevant_docs),
            "relevance_rate": len(relevant_docs) / len(documents) if documents else 0
        }
    )

    return relevant_docs


async def grade_documents_parallel(
    query: str,
    documents: List[Dict[str, Any]],
    threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Grade documents in parallel for better performance (recommended).

    Args:
        query: User's query
        documents: List of retrieved documents
        threshold: Minimum relevance threshold (not used in binary grading)

    Returns:
        List of relevant documents only
    """
    logger.info(f"Grading {len(documents)} documents in parallel")

    # Get LLM client
    llm = get_llm(model_type="main")

    # Grade all documents concurrently
    tasks = [
        grade_single_document_async(query, doc, llm)
        for doc in documents
    ]

    graded_docs = await asyncio.gather(*tasks)

    # Filter to relevant only
    relevant_docs = [doc for doc in graded_docs if doc["is_relevant"]]

    logger.info(
        f"Parallel grading complete: {len(relevant_docs)}/{len(documents)} relevant",
        extra={
            "total_docs": len(documents),
            "relevant_docs": len(relevant_docs),
            "relevance_rate": len(relevant_docs) / len(documents) if documents else 0
        }
    )

    return relevant_docs


def grade_documents_sync_wrapper(
    query: str,
    documents: List[Dict[str, Any]],
    threshold: float = 0.0,
    use_parallel: bool = True
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper that automatically chooses parallel or sequential grading.

    Args:
        query: User's query
        documents: List of retrieved documents
        threshold: Minimum relevance threshold
        use_parallel: Whether to use parallel grading (recommended)

    Returns:
        List of relevant documents
    """
    if use_parallel:
        try:
            # Try to get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run parallel grading
            return loop.run_until_complete(
                grade_documents_parallel(query, documents, threshold)
            )
        except Exception as e:
            logger.warning(f"Parallel grading failed, falling back to sequential: {e}")
            return grade_documents(query, documents, threshold)
    else:
        return grade_documents(query, documents, threshold)
