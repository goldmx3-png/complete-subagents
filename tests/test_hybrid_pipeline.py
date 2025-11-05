"""
Test script to validate hybrid search integration in the RAG pipeline

This script tests the full query -> retrieval -> response pipeline
with hybrid search enabled.

Usage:
    python tests/test_hybrid_pipeline.py
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.rag.agent import RAGAgent
from src.agents.shared.state import AgentState, RAGState
from src.config import settings


async def test_hybrid_search_pipeline():
    """Test RAG agent with hybrid search"""

    print("=" * 80)
    print("HYBRID SEARCH PIPELINE TEST")
    print("=" * 80)
    print()

    # Show configuration
    print("Configuration:")
    print(f"  - Hybrid Search: {settings.enable_hybrid_search}")
    print(f"  - Reranking: {settings.enable_reranking}")
    print(f"  - Vector Weight: {settings.hybrid_vector_weight}")
    print(f"  - BM25 Weight: {settings.hybrid_bm25_weight}")
    print(f"  - Top K: {settings.top_k_retrieval}")
    print()

    # Test queries
    test_queries = [
        "What are the payment cutoff times?",
        "How do I check my transaction history?",
        "What is the daily transfer limit for international payments?",
    ]

    # Initialize RAG agent
    print("Initializing RAG Agent with EnhancedRAGRetriever...")
    agent = RAGAgent()
    print("✓ RAG Agent initialized")
    print()

    # Test each query
    for i, query in enumerate(test_queries, 1):
        print("-" * 80)
        print(f"Test {i}/{len(test_queries)}: {query}")
        print("-" * 80)

        # Create state
        state = AgentState(
            query=query,
            user_id="test_user",
            conversation_history=[],
            route="RAG_ONLY",
            rag=RAGState(
                chunks=[],
                context="",
                is_ambiguous=False,
                disambiguation_options=[],
                reformulated_query=None
            )
        )

        try:
            # Execute RAG agent
            print(f"\nExecuting RAG pipeline...")
            result_state = await agent.execute(state)

            # Show results
            rag_state = result_state.get("rag", {})
            chunks = rag_state.get("chunks", [])

            print(f"\n✓ Retrieval complete")
            print(f"  - Retrieved chunks: {len(chunks)}")

            if chunks:
                print(f"\n  Top 3 chunks:")
                for idx, chunk in enumerate(chunks[:3], 1):
                    score = chunk.get("score", 0)
                    payload = chunk.get("payload", {})
                    text = payload.get("text", "")[:100]
                    print(f"    {idx}. Score: {score:.3f}")
                    print(f"       Text: {text}...")
                    print()

            # Show answer
            answer = result_state.get("final_response", "")
            print(f"Generated Answer:")
            print(f"  {answer[:200]}...")
            print()

        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()
    print("Check the logs above for:")
    print("  1. 'Hybrid search enabled' message")
    print("  2. 'Initializing BM25 index' message")
    print("  3. 'Hybrid search: vector=X, bm25=Y' messages")
    print("  4. 'method=hybrid' in retrieval completion logs")
    print()


async def test_comparison():
    """Compare hybrid vs vector-only search"""

    print("=" * 80)
    print("HYBRID vs VECTOR-ONLY COMPARISON")
    print("=" * 80)
    print()

    query = "What are the payment cutoff times?"

    from src.retrieval.enhanced_retriever import EnhancedRAGRetriever
    from src.retrieval.retriever import RAGRetriever

    # Test with enhanced retriever (hybrid)
    print("1. Testing with EnhancedRAGRetriever (Hybrid Search)...")
    enhanced_retriever = EnhancedRAGRetriever()

    try:
        result = await enhanced_retriever.retrieve(
            query=query,
            user_id="test_user",
            top_k=5
        )

        chunks = result.get("chunks", [])
        method = result.get("retrieval_method", "unknown")
        timing = result.get("timing", {})

        print(f"   Method: {method}")
        print(f"   Results: {len(chunks)} chunks")
        print(f"   Timing: {timing.get('total', 0):.0f}ms")
        if chunks:
            avg_score = sum(c.get("score", 0) for c in chunks) / len(chunks)
            print(f"   Avg Score: {avg_score:.3f}")
        print()

    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        print()

    # Test with old retriever (vector-only)
    print("2. Testing with RAGRetriever (Vector-Only)...")
    old_retriever = RAGRetriever()

    try:
        result = await old_retriever.retrieve(
            query=query,
            user_id="test_user",
            top_k=5
        )

        chunks = result.get("chunks", [])

        print(f"   Method: vector_only")
        print(f"   Results: {len(chunks)} chunks")
        if chunks:
            avg_score = sum(c.get("score", 0) for c in chunks) / len(chunks)
            print(f"   Avg Score: {avg_score:.3f}")
        print()

    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        print()

    print("=" * 80)


if __name__ == "__main__":
    print("\n🔍 Testing Hybrid Search Pipeline\n")

    # Run main test
    asyncio.run(test_hybrid_search_pipeline())

    # Optional: Run comparison
    print("\nRun comparison test? (y/n): ", end="")
    response = input().strip().lower()
    if response == 'y':
        asyncio.run(test_comparison())
