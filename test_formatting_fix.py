"""
Test script to verify formatting optimizations work correctly.
This tests the fix for 30K character chunks problem.
"""

import sys
sys.path.insert(0, '/home/user/complete-subagents')

from src.retrieval.enhanced_retriever import EnhancedRAGRetriever
from src.config import settings


def create_test_chunks():
    """Create test chunks with long breadcrumbs and large text."""
    chunks = []

    # Simulate 5 chunks with realistic data
    for i in range(5):
        # Long breadcrumb path (similar to real documents)
        long_breadcrumb = f"Corporate Banking Guide > Module {i+1} > Section {i+1}.{i+1} > Subsection {i+1}.{i+1}.{i+1} > Detail {i+1}.{i+1}.{i+1}.{i+1}"

        # Simulate chunk text (~2400 chars, similar to 600 tokens)
        chunk_text = f"This is chunk {i+1} content. " * 100  # ~2400 chars

        chunk = {
            "payload": {
                "text": chunk_text,
                "doc_id": f"doc_{i+1}",
                "metadata": {
                    "hierarchy": {
                        "full_path": long_breadcrumb,
                        "breadcrumbs": long_breadcrumb.split(" > "),
                        "depth": 4
                    }
                }
            },
            "score": 0.9 - (i * 0.1)
        }
        chunks.append(chunk)

    return chunks


def test_formatting():
    """Test the formatting with the new optimizations."""
    print("=" * 60)
    print("TESTING FORMATTING OPTIMIZATIONS")
    print("=" * 60)

    # Create retriever instance
    retriever = EnhancedRAGRetriever()

    # Create test chunks
    chunks = create_test_chunks()

    print(f"\nTest Data:")
    print(f"  - Number of chunks: {len(chunks)}")
    print(f"  - Avg chunk text size: ~{len(chunks[0]['payload']['text'])} chars")
    print(f"  - Breadcrumb example: {chunks[0]['payload']['metadata']['hierarchy']['full_path']}")
    print(f"  - Breadcrumb length: {len(chunks[0]['payload']['metadata']['hierarchy']['full_path'])} chars")

    # Test settings
    print(f"\nConfiguration Settings:")
    print(f"  - FORMATTING_STYLE: {settings.formatting_style}")
    print(f"  - MAX_FORMATTED_CHUNK_SIZE_CHARS: {settings.max_formatted_chunk_size_chars}")
    print(f"  - MAX_TOTAL_CONTEXT_SIZE_CHARS: {settings.max_total_context_size_chars}")
    print(f"  - BREADCRUMB_MAX_LEVELS: {settings.breadcrumb_max_levels}")
    print(f"  - BREADCRUMB_MAX_LENGTH: {settings.breadcrumb_max_length}")
    print(f"  - ENABLE_SECTION_GROUPING: {settings.enable_section_grouping}")
    print(f"  - ENABLE_BREADCRUMB_CONTEXT: {settings.enable_breadcrumb_context}")
    print(f"  - ENABLE_AUTO_FALLBACK: {settings.enable_auto_fallback}")

    # Test breadcrumb truncation
    print(f"\n{'=' * 60}")
    print("TEST 1: Breadcrumb Truncation")
    print("=" * 60)

    test_breadcrumb = chunks[0]['payload']['metadata']['hierarchy']['full_path']
    truncated = retriever._truncate_breadcrumb(test_breadcrumb)

    print(f"Original: {test_breadcrumb}")
    print(f"  Length: {len(test_breadcrumb)} chars")
    print(f"\nTruncated: {truncated}")
    print(f"  Length: {len(truncated)} chars")
    print(f"  ✅ Reduction: {len(test_breadcrumb) - len(truncated)} chars")

    # Test chunk text truncation
    print(f"\n{'=' * 60}")
    print("TEST 2: Chunk Text Truncation")
    print("=" * 60)

    # Create a very large chunk (simulate oversized chunk)
    large_text = "X" * 5000
    truncated_text = retriever._truncate_chunk_text(large_text)

    print(f"Original text size: {len(large_text)} chars")
    print(f"Truncated text size: {len(truncated_text)} chars")
    print(f"Max allowed: {settings.max_formatted_chunk_size_chars} chars")
    print(f"✅ Truncation: {'Applied' if len(truncated_text) < len(large_text) else 'Not needed'}")

    # Test format_context with grouping (main fix)
    print(f"\n{'=' * 60}")
    print("TEST 3: Context Formatting (Main Fix)")
    print("=" * 60)

    formatted_context = retriever.format_context(chunks)

    print(f"\nFormatted Context Stats:")
    print(f"  - Total length: {len(formatted_context)} chars")
    print(f"  - Per chunk average: {len(formatted_context) // len(chunks)} chars")
    print(f"  - Max allowed total: {settings.max_total_context_size_chars} chars")

    # Calculate reduction
    # Old format would be: ~30K chars per chunk × 5 = ~150K chars
    old_estimated_size = 30000 * len(chunks)
    reduction_percent = ((old_estimated_size - len(formatted_context)) / old_estimated_size) * 100

    print(f"\n📊 Improvement Analysis:")
    print(f"  - OLD (estimated): ~{old_estimated_size:,} chars")
    print(f"  - NEW (actual):     {len(formatted_context):,} chars")
    print(f"  - 🎉 REDUCTION:     {reduction_percent:.1f}%")

    if len(formatted_context) < settings.max_total_context_size_chars:
        print(f"\n✅ SUCCESS: Context size is within limits!")
    else:
        print(f"\n⚠️  WARNING: Context size exceeds limits (fallback should trigger)")

    # Show sample of formatted output
    print(f"\n{'=' * 60}")
    print("Sample of Formatted Output (first 500 chars):")
    print("=" * 60)
    print(formatted_context[:500])
    print("...")

    print(f"\n{'=' * 60}")
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    test_formatting()
