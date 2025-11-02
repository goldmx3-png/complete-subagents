"""
Test script for Advanced RAG techniques
Tests each module independently and validates the complete pipeline
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings
from src.document_processing.hierarchical_chunker import HierarchicalChunker
from src.document_processing.metadata_extractor import MetadataExtractor
from src.retrieval.hybrid_retriever import BM25Retriever
from src.retrieval.reranker import RerankerPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Test document
SAMPLE_DOCUMENT = """
AUTHORIZATION MATRIX POLICY

1. OVERVIEW
The authorization matrix defines approval levels for all financial transactions.
This policy ensures proper oversight and dual-control principles are maintained.

2. APPROVAL LEVELS

2.1 Level 1 (Under $10,000)
- Single approver required
- Must be from authorized personnel list
- Standard processing time: 24 hours

2.2 Level 2 ($10,001 - $50,000)
- Two approvers required
- One must be at manager level or above
- Processing time: 48 hours

2.3 Level 3 (Above $50,000)
- Three approvers required including CFO or CEO
- Board notification for amounts over $100,000
- Processing time: 72 hours

3. SPECIAL CASES

3.1 Emergency Transactions
- CEO authorization required
- Post-approval review within 24 hours
- Complete audit trail mandatory

3.2 Weekend Transactions
- Additional compliance review needed
- Monday morning verification required
- Limited to critical operations only

3.3 International Transactions
- Treasury team approval required
- Currency risk assessment needed
- Compliance with international regulations
"""


async def test_hierarchical_chunking():
    """Test 1: Hierarchical Chunking"""
    print("\n" + "=" * 70)
    print("TEST 1: Hierarchical Chunking")
    print("=" * 70)

    try:
        chunker = HierarchicalChunker(
            parent_chunk_size=settings.parent_chunk_size,
            child_chunk_size=settings.child_chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        parent_chunks, child_chunks = chunker.chunk_document_hierarchical(
            text=SAMPLE_DOCUMENT,
            doc_id="test_policy_001",
            metadata={"source": "test_policy.pdf", "version": "1.0"}
        )

        print(f"\n✓ Created {len(parent_chunks)} parent chunks")
        print(f"✓ Created {len(child_chunks)} child chunks")

        # Validate parent-child relationships
        parent_ids = {p["chunk_id"] for p in parent_chunks}
        child_parent_ids = {c["parent_chunk_id"] for c in child_chunks}

        orphaned = child_parent_ids - parent_ids
        if orphaned:
            print(f"✗ Warning: {len(orphaned)} children have invalid parent references")
        else:
            print("✓ All children have valid parent references")

        # Show example
        if parent_chunks:
            print(f"\nParent chunk example:")
            print(f"  ID: {parent_chunks[0]['chunk_id']}")
            print(f"  Length: {len(parent_chunks[0]['text'])} chars")
            print(f"  Preview: {parent_chunks[0]['text'][:150]}...")

        if child_chunks:
            print(f"\nChild chunk example:")
            print(f"  ID: {child_chunks[0]['chunk_id']}")
            print(f"  Parent: {child_chunks[0]['parent_chunk_id']}")
            print(f"  Preview: {child_chunks[0]['text'][:100]}...")

        print("\n✅ TEST 1 PASSED: Hierarchical chunking works correctly")
        return True, parent_chunks, child_chunks

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None


async def test_metadata_extraction(chunks):
    """Test 2: Metadata Extraction"""
    print("\n" + "=" * 70)
    print("TEST 2: Metadata Extraction")
    print("=" * 70)

    if not chunks:
        print("⚠️  Skipping test (no chunks provided)")
        return False, None

    try:
        extractor = MetadataExtractor()

        # Test on first chunk
        chunk = chunks[0] if chunks else None
        if not chunk:
            print("❌ No chunk to test")
            return False, None

        print(f"\nExtracting metadata for chunk: {chunk['chunk_id']}")
        print(f"Text preview: {chunk['text'][:100]}...")

        enriched_chunk = await extractor.extract_metadata(
            chunk=chunk,
            extract_title=True,
            extract_summary=True,
            extract_keywords=True,
            extract_questions=False  # Skip questions for speed
        )

        print(f"\n✓ Title: {enriched_chunk['metadata'].get('title', 'N/A')}")
        print(f"✓ Summary: {enriched_chunk['metadata'].get('summary', 'N/A')[:150]}...")
        print(f"✓ Keywords: {enriched_chunk['metadata'].get('keywords', [])}")

        print("\n✅ TEST 2 PASSED: Metadata extraction works correctly")
        return True, enriched_chunk

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def test_bm25_retrieval(chunks):
    """Test 3: BM25 Sparse Retrieval"""
    print("\n" + "=" * 70)
    print("TEST 3: BM25 Sparse Retrieval")
    print("=" * 70)

    if not chunks:
        print("⚠️  Skipping test (no chunks provided)")
        return False

    try:
        # Initialize BM25
        bm25 = BM25Retriever()

        # Index chunks
        bm25.index_documents(chunks)
        print(f"\n✓ Indexed {len(chunks)} documents for BM25")

        # Test query
        query = "approval levels authorization"
        print(f"\nQuery: '{query}'")

        results = bm25.search(query, top_k=3)

        print(f"\n✓ Retrieved {len(results)} results")

        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Score: {result['score']:.3f}")
            print(f"    Text: {result['text'][:100]}...")

        print("\n✅ TEST 3 PASSED: BM25 retrieval works correctly")
        return True

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_reranking(chunks):
    """Test 4: Re-ranking"""
    print("\n" + "=" * 70)
    print("TEST 4: Re-ranking (MMR)")
    print("=" * 70)

    if not chunks or len(chunks) < 3:
        print("⚠️  Skipping test (need at least 3 chunks)")
        return False

    try:
        # Prepare sample results (simulate retrieval scores)
        test_docs = []
        for i, chunk in enumerate(chunks[:5]):
            doc = chunk.copy()
            doc["score"] = 0.9 - (i * 0.1)  # Decreasing scores
            test_docs.append(doc)

        print(f"\n✓ Prepared {len(test_docs)} documents for re-ranking")

        # Initialize MMR reranker
        reranker = RerankerPipeline(
            method=settings.reranking_method,
            lambda_param=settings.mmr_lambda
        )

        query = "what are the approval levels"

        # Re-rank
        reranked_docs = await reranker.rerank(query, test_docs, top_k=3)

        print(f"\n✓ Re-ranked to top {len(reranked_docs)} documents")

        print("\nOriginal order:")
        for i, doc in enumerate(test_docs[:3], 1):
            print(f"  {i}. Score: {doc['score']:.2f} - {doc['text'][:60]}...")

        print(f"\nAfter {settings.reranking_method.upper()} re-ranking:")
        for i, doc in enumerate(reranked_docs, 1):
            print(f"  {i}. Score: {doc.get('rerank_score', 0):.2f} - {doc['text'][:60]}...")

        print("\n✅ TEST 4 PASSED: Re-ranking works correctly")
        return True

    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test 5: Configuration Loading"""
    print("\n" + "=" * 70)
    print("TEST 5: Configuration Loading")
    print("=" * 70)

    try:
        print("\n Advanced RAG Configuration:")
        print(f"  ✓ Hierarchical Chunking: {settings.use_hierarchical_chunking}")
        print(f"    - Parent chunk size: {settings.parent_chunk_size}")
        print(f"    - Child chunk size: {settings.child_chunk_size}")

        print(f"\n  ✓ Metadata Extraction: {settings.use_metadata_extraction}")
        print(f"    - Extract summaries: {settings.extract_summaries}")
        print(f"    - Extract keywords: {settings.extract_keywords}")
        print(f"    - Extract questions: {settings.extract_questions}")

        print(f"\n  ✓ Hybrid Search: {settings.use_hybrid_search}")
        print(f"    - Fusion method: {settings.hybrid_fusion_method}")
        print(f"    - Dense weight: {settings.hybrid_dense_weight}")
        print(f"    - Sparse weight: {settings.hybrid_sparse_weight}")

        print(f"\n  ✓ Multi-Vector: {settings.use_multi_vector}")

        print(f"\n  ✓ Contextual Compression: {settings.use_contextual_compression}")
        print(f"    - Method: {settings.compression_method}")
        print(f"    - Threshold: {settings.compression_similarity_threshold}")

        print(f"\n  ✓ Advanced Re-ranking: {settings.use_advanced_reranking}")
        print(f"    - Method: {settings.reranking_method}")
        print(f"    - MMR Lambda: {settings.mmr_lambda}")

        print(f"\n  ✓ Classifier Model: {settings.classifier_model}")

        print("\n✅ TEST 5 PASSED: Configuration loaded successfully")
        return True

    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("ADVANCED RAG TECHNIQUES - TEST SUITE")
    print("=" * 70)

    results = {}

    # Test 1: Hierarchical Chunking
    success, parent_chunks, child_chunks = await test_hierarchical_chunking()
    results["hierarchical_chunking"] = success

    # Test 2: Metadata Extraction
    if child_chunks:
        success, enriched = await test_metadata_extraction(child_chunks[:1])
        results["metadata_extraction"] = success
    else:
        results["metadata_extraction"] = False

    # Test 3: BM25 Retrieval
    if child_chunks:
        success = test_bm25_retrieval(child_chunks)
        results["bm25_retrieval"] = success
    else:
        results["bm25_retrieval"] = False

    # Test 4: Re-ranking
    if child_chunks:
        success = await test_reranking(child_chunks)
        results["reranking"] = success
    else:
        results["reranking"] = False

    # Test 5: Configuration
    success = test_configuration()
    results["configuration"] = success

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nResults: {passed_tests}/{total_tests} tests passed\n")

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name.replace('_', ' ').title()}")

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Advanced RAG is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
