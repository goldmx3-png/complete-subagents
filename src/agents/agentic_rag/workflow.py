"""
Agentic RAG Workflow using LangGraph

Implements a multi-step RAG workflow with:
- Document retrieval and grading
- Answer generation
- Self-reflection and validation
- Query rewriting for iterative improvement
"""

from typing import TypedDict, List, Dict, Any, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from src.vectorstore.embeddings import get_embeddings
from src.llm.openrouter_client import get_llm
from src.utils.logger import get_logger
from src.config import get_settings

from .grader import grade_documents_sync_wrapper
from .validator import validate_answer, validate_with_early_exit
from .query_rewriter import rewrite_query

logger = get_logger(__name__)
settings = get_settings()


# State schema for the agentic RAG workflow
class AgenticRAGState(TypedDict):
    """State for agentic RAG workflow"""
    # Input
    query: str
    user_id: Optional[str]
    conversation_id: Optional[str]

    # Retrieval
    retrieved_docs: List[Dict[str, Any]]
    relevant_docs: List[Dict[str, Any]]
    retrieval_score: float

    # Generation
    generated_answer: str
    context: str

    # Validation
    is_valid: bool
    validation_issues: List[str]
    confidence: float

    # Iteration control
    iteration_count: int
    rewritten_query: Optional[str]
    max_iterations: int

    # Final output
    final_answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


def retrieve_documents_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    Node 1: Retrieve documents from Qdrant vector DB.
    """
    query = state.get("rewritten_query") or state["query"]
    iteration = state.get("iteration_count", 0)

    logger.info(f"[Iteration {iteration}] Retrieving documents for query: {query[:100]}")

    try:
        # Import here to avoid circular dependencies
        from src.retrieval.retriever import RAGRetriever
        from src.utils.async_helpers import run_async_in_new_loop

        # Create retriever
        retriever = RAGRetriever()

        # Retrieve documents
        top_k = settings.agentic_rag_retrieval_top_k
        user_id = state.get("user_id", "default")

        # Run async retrieve in sync context using helper
        result = run_async_in_new_loop(
            retriever.retrieve(query=query, user_id=user_id, top_k=top_k)
        )

        chunks = result.get("chunks", [])

        # Calculate average retrieval score
        if chunks:
            avg_score = sum(c.get("score", 0.0) for c in chunks) / len(chunks)
        else:
            avg_score = 0.0

        logger.info(
            f"Retrieved {len(chunks)} documents with avg score {avg_score:.3f}",
            extra={"query": query[:50], "doc_count": len(chunks), "avg_score": avg_score}
        )

        return {
            "retrieved_docs": chunks,
            "retrieval_score": avg_score
        }

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}", exc_info=True)
        return {
            "retrieved_docs": [],
            "retrieval_score": 0.0
        }


def grade_documents_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    Node 2: Grade retrieved documents for relevance.
    """
    query = state.get("rewritten_query") or state["query"]
    documents = state["retrieved_docs"]
    iteration = state.get("iteration_count", 0)

    logger.info(f"[Iteration {iteration}] Grading {len(documents)} documents")

    try:
        # Grade documents (uses parallel grading by default)
        use_parallel = getattr(settings, "ENABLE_PARALLEL_GRADING", True)
        relevant_docs = grade_documents_sync_wrapper(
            query=query,
            documents=documents,
            use_parallel=use_parallel
        )

        logger.info(
            f"Grading complete: {len(relevant_docs)}/{len(documents)} relevant",
            extra={
                "total": len(documents),
                "relevant": len(relevant_docs),
                "relevance_rate": len(relevant_docs) / len(documents) if documents else 0
            }
        )

        return {"relevant_docs": relevant_docs}

    except Exception as e:
        logger.error(f"Error grading documents: {e}")
        # On error, use all documents
        return {"relevant_docs": documents}


def generate_answer_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    Node 3: Generate answer from relevant documents.
    """
    query = state.get("rewritten_query") or state["query"]
    relevant_docs = state["relevant_docs"]
    iteration = state.get("iteration_count", 0)

    logger.info(f"[Iteration {iteration}] Generating answer")

    try:
        # Prepare context from relevant documents
        context_parts = []
        sources = []

        for i, doc in enumerate(relevant_docs[:5], 1):  # Limit to top 5
            content = doc.get("content", doc.get("page_content", ""))
            metadata = doc.get("metadata", {})

            context_parts.append(f"[Source {i}]\n{content}")
            sources.append({
                "index": i,
                "content": content[:200],  # Preview
                "metadata": metadata
            })

        context = "\n\n".join(context_parts)

        # Generate answer
        llm = get_llm(model_type="main")

        prompt = f"""You are a banking support assistant. Answer the query using ONLY the provided context.

Rules:
- Be accurate and concise
- Cite sources using [Source N] format
- If information is incomplete, state what's available
- Never make up banking information
- Use professional banking terminology

Context:
{context}

Query: {query}

Answer:"""

        # Get generated answer - use sync wrapper for async chat
        from src.utils.async_helpers import run_async_in_new_loop

        messages = [{"role": "user", "content": prompt}]

        # Run async chat in sync context using helper
        generated_answer = run_async_in_new_loop(llm.chat(messages))

        generated_answer = generated_answer.strip()

        logger.info(
            f"Answer generated ({len(generated_answer)} chars)",
            extra={"answer_length": len(generated_answer)}
        )

        return {
            "generated_answer": generated_answer,
            "context": context,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {
            "generated_answer": "I apologize, but I encountered an error generating the answer.",
            "context": "",
            "sources": []
        }


def validate_answer_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    Node 4: Validate generated answer for quality.
    """
    query = state.get("rewritten_query") or state["query"]
    answer = state["generated_answer"]
    context = state["context"]
    retrieval_score = state.get("retrieval_score", 0.0)
    iteration = state.get("iteration_count", 0)

    logger.info(f"[Iteration {iteration}] Validating answer")

    try:
        # Validate answer (with early exit if retrieval score is high)
        enable_early_exit = getattr(settings, "ENABLE_EARLY_EXIT", True)

        validation_result = validate_with_early_exit(
            query=query,
            context=context,
            answer=answer,
            retrieval_score=retrieval_score,
            enable_early_exit=enable_early_exit
        )

        logger.info(
            f"Validation complete: {'VALID' if validation_result['is_valid'] else 'INVALID'}",
            extra={
                "is_valid": validation_result["is_valid"],
                "confidence": validation_result["confidence"],
                "issues": validation_result["issues"]
            }
        )

        return {
            "is_valid": validation_result["is_valid"],
            "validation_issues": validation_result["issues"],
            "confidence": validation_result["confidence"]
        }

    except Exception as e:
        logger.error(f"Error validating answer: {e}")
        # On error, assume invalid
        return {
            "is_valid": False,
            "validation_issues": [f"Validation error: {str(e)}"],
            "confidence": 0.0
        }


def rewrite_query_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    Node 5: Rewrite query for better retrieval.
    """
    original_query = state["query"]
    iteration = state.get("iteration_count", 0)
    issues = state.get("validation_issues", [])

    logger.info(f"[Iteration {iteration}] Rewriting query")

    try:
        # Rewrite query
        enable_cache = getattr(settings, "ENABLE_QUERY_CACHE", True)
        rewritten = rewrite_query(
            original_query=original_query,
            context=f"Issues: {', '.join(issues)}",
            use_cache=enable_cache
        )

        logger.info(
            f"Query rewritten",
            extra={"original": original_query[:50], "rewritten": rewritten[:50]}
        )

        return {
            "rewritten_query": rewritten,
            "iteration_count": iteration + 1
        }

    except Exception as e:
        logger.error(f"Error rewriting query: {e}")
        return {
            "rewritten_query": original_query,  # Fallback
            "iteration_count": iteration + 1
        }


# Conditional edge functions

def decide_after_grading(state: AgenticRAGState) -> Literal["generate", "rewrite"]:
    """
    Decide whether we have enough relevant documents to generate an answer.
    """
    relevant_count = len(state["relevant_docs"])
    min_required = getattr(settings, "AGENTIC_RAG_MIN_RELEVANT_DOCS", 2)
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", getattr(settings, "AGENTIC_RAG_MAX_ITERATIONS", 3))

    # Check if we've hit max iterations - if so, generate anyway to avoid infinite loop
    if iteration >= max_iterations:
        logger.info(f"Max iterations ({max_iterations}) reached with {relevant_count} documents, proceeding to generate")
        return "generate"

    if relevant_count >= min_required:
        logger.info(f"Sufficient documents ({relevant_count}), proceeding to generate")
        return "generate"
    else:
        logger.info(f"Insufficient documents ({relevant_count}/{min_required}), rewriting query")
        return "rewrite"


def decide_after_validation(
    state: AgenticRAGState
) -> Literal[END, "rewrite", "generate"]:
    """
    Decide next step based on validation result.
    """
    is_valid = state["is_valid"]
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", getattr(settings, "AGENTIC_RAG_MAX_ITERATIONS", 3))
    issues = state.get("validation_issues", [])

    # Check if max iterations reached
    if iteration >= max_iterations:
        logger.info(f"Max iterations ({max_iterations}) reached, ending")
        return END

    # Check if answer is valid
    if is_valid:
        logger.info("Answer is valid, ending successfully")
        return END

    # Determine issue type
    has_hallucination = any("hallucination" in issue.lower() or "grounding" in issue.lower() for issue in issues)
    has_incompleteness = any("incomplete" in issue.lower() or "completeness" in issue.lower() for issue in issues)

    if has_hallucination:
        logger.info("Hallucination detected, regenerating answer")
        return "generate"
    elif has_incompleteness:
        logger.info("Answer incomplete, rewriting query")
        return "rewrite"
    else:
        logger.info("Generic validation failure, rewriting query")
        return "rewrite"


# Build the workflow

def create_agentic_rag_workflow():
    """
    Create and compile the agentic RAG workflow.
    """
    logger.info("Creating agentic RAG workflow")

    # Create state graph
    builder = StateGraph(AgenticRAGState)

    # Add nodes
    builder.add_node("retrieve", retrieve_documents_node)
    builder.add_node("grade", grade_documents_node)
    builder.add_node("generate", generate_answer_node)
    builder.add_node("validate", validate_answer_node)
    builder.add_node("rewrite", rewrite_query_node)

    # Define edges
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")

    # After grading: sufficient docs → generate, insufficient → rewrite
    builder.add_conditional_edges(
        "grade",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )

    # After rewriting: loop back to retrieve
    builder.add_edge("rewrite", "retrieve")

    # After generating: validate
    builder.add_edge("generate", "validate")

    # After validation: valid → END, issues → rewrite or regenerate
    builder.add_conditional_edges(
        "validate",
        decide_after_validation,
        {
            END: END,
            "rewrite": "rewrite",
            "generate": "generate"
        }
    )

    # Compile the graph
    graph = builder.compile()

    logger.info("Agentic RAG workflow created successfully")

    return graph


# Singleton instance
_agentic_rag_graph = None


def get_agentic_rag_workflow():
    """Get or create the agentic RAG workflow instance."""
    global _agentic_rag_graph
    if _agentic_rag_graph is None:
        _agentic_rag_graph = create_agentic_rag_workflow()
    return _agentic_rag_graph
