"""
Test script to verify model preloading works correctly

This can be run independently or you can check the API startup logs
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_preload_embeddings():
    """Test embedding model preloading"""
    print("\n" + "=" * 80)
    print("TEST 1: Embedding Model Preload")
    print("=" * 80)

    try:
        from src.vectorstore.embeddings import EmbeddingsModel
        import time

        start = time.time()
        embeddings = EmbeddingsModel()
        elapsed = (time.time() - start) * 1000

        print(f"✓ Embedding model loaded in {elapsed:.0f}ms")
        print(f"  Model: {settings.embedding_model}")
        print(f"  Device: {settings.embedding_device}")

        # Test embedding
        test_text = "Test query for embedding"
        embedding = embeddings.embed_query(test_text)
        print(f"  Embedding dimension: {len(embedding)}")
        print("✓ Embedding test successful")

    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        import traceback
        traceback.print_exc()


def test_preload_reranker():
    """Test reranker model preloading"""
    print("\n" + "=" * 80)
    print("TEST 2: Reranker Model Preload")
    print("=" * 80)

    if not settings.enable_reranking:
        print("⚠ Reranking disabled in config - skipping test")
        print(f"  Set ENABLE_RERANKING=true in .env to test")
        return

    try:
        from src.retrieval.reranker import preload_reranker
        import time

        start = time.time()
        preload_reranker()
        elapsed = (time.time() - start) * 1000

        print(f"✓ Reranker model loaded in {elapsed:.0f}ms")
        print(f"  Model: {settings.reranker_model_v2}")
        print(f"  Device: {settings.reranker_device_v2}")

        # Test reranking
        from src.retrieval.reranker import get_reranker
        reranker = get_reranker()

        test_docs = [
            {
                "id": "1",
                "score": 0.8,
                "payload": {"text": "Payment cutoff times are 3 PM for same-day processing."}
            },
            {
                "id": "2",
                "score": 0.6,
                "payload": {"text": "The weather is sunny today."}
            }
        ]

        import asyncio
        result = asyncio.run(reranker.rerank(
            query="What are payment cutoff times?",
            documents=test_docs,
            top_k=2
        ))

        print(f"  Reranked {len(result)} documents")
        print(f"  Top score after reranking: {result[0]['score']:.3f}")
        print("✓ Reranking test successful")

    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        import traceback
        traceback.print_exc()


def test_startup_sequence():
    """Simulate the FastAPI startup sequence"""
    print("\n" + "=" * 80)
    print("TEST 3: Simulated API Startup Sequence")
    print("=" * 80)

    print("\nThis simulates what happens when you run:")
    print("  uvicorn src.api.routes:app --reload")
    print()

    import time
    total_start = time.time()

    # Simulate startup event
    print("=" * 80)
    print("SERVICE STARTUP - Preloading models...")
    print("=" * 80)

    # Preload embedding model
    try:
        from src.vectorstore.embeddings import EmbeddingsModel
        print("Preloading embedding model...")
        start = time.time()
        embeddings = EmbeddingsModel()
        elapsed = (time.time() - start) * 1000
        print(f"✓ Embedding model loaded in {elapsed:.0f}ms: {settings.embedding_model}")
    except Exception as e:
        print(f"✗ Failed to preload embedding model: {str(e)}")

    # Preload reranker model if enabled
    if settings.enable_reranking:
        try:
            from src.retrieval.reranker import preload_reranker
            start = time.time()
            preload_reranker()
            elapsed = (time.time() - start) * 1000
            print(f"✓ Reranker model loaded in {elapsed:.0f}ms")
        except Exception as e:
            print(f"✗ Failed to preload reranker: {str(e)}")
    else:
        print("Reranking disabled - skipping reranker preload")

    total_elapsed = (time.time() - total_start) * 1000
    print("=" * 80)
    print(f"✓ SERVICE READY - All models preloaded in {total_elapsed:.0f}ms")
    print("=" * 80)


if __name__ == "__main__":
    print("\n🚀 Model Preloading Test Suite\n")

    # Show current config
    print("Current Configuration:")
    print(f"  ENABLE_HYBRID_SEARCH: {settings.enable_hybrid_search}")
    print(f"  ENABLE_RERANKING: {settings.enable_reranking}")
    print(f"  Embedding Model: {settings.embedding_model}")
    print(f"  Reranker Model: {settings.reranker_model_v2}")
    print()

    # Run tests
    test_preload_embeddings()
    test_preload_reranker()
    test_startup_sequence()

    print("\n" + "=" * 80)
    print("All Tests Complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Start the API server: uvicorn src.api.routes:app --reload")
    print("  2. Watch the startup logs for model preloading messages")
    print("  3. Make a request and notice NO delay for model loading")
    print()
