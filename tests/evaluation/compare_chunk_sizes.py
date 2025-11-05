"""
Simple A/B testing script for comparing chunk sizes
Tests 400, 600, and 800 token chunk sizes
"""

import asyncio
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.document_processing.token_chunker import TokenBasedChunker


def load_test_document(file_path: str = None) -> str:
    """
    Load a test document for chunking

    If no file provided, uses sample banking text
    """
    if file_path and Path(file_path).exists():
        with open(file_path, 'r') as f:
            return f.read()

    # Sample banking document text
    return """
    Payment Processing Guidelines

    1. Payment Cutoff Times
    All domestic payments must be submitted before 5:00 PM EST for same-day processing.
    International wire transfers have a cutoff time of 3:00 PM EST.
    ACH transfers submitted after cutoff will be processed the next business day.

    2. Daily Transfer Limits
    Standard accounts: $10,000 per day
    Premium accounts: $50,000 per day
    Business accounts: Custom limits based on agreement

    Contact your relationship manager to request limit increases.
    Additional documentation may be required for limits above $100,000.

    3. Transaction Status Codes
    The following status codes are used to track payment progress:
    - PENDING: Payment submitted and awaiting processing
    - VALIDATED: Payment has passed all validation checks
    - PROCESSING: Payment is being processed by the clearing system
    - COMPLETED: Payment successfully delivered to beneficiary
    - FAILED: Payment rejected due to validation errors
    - CANCELLED: Payment cancelled by user before processing

    4. Beneficiary Management
    To add a new beneficiary, follow these steps:
    Step 1: Navigate to the Beneficiaries section
    Step 2: Click "Add New Beneficiary"
    Step 3: Enter beneficiary details (name, account number, bank code)
    Step 4: Verify details carefully - incorrect information will cause payment failures
    Step 5: Submit for approval (may require second authorization for business accounts)

    Beneficiary information is validated against bank databases.
    Invalid account numbers will be rejected during validation.
    International beneficiaries require additional SWIFT/BIC codes.

    5. Currency Exchange
    Supported currencies include:
    USD - US Dollar
    EUR - Euro
    GBP - British Pound
    JPY - Japanese Yen
    CHF - Swiss Franc
    AUD - Australian Dollar
    CAD - Canadian Dollar

    Exchange rates are updated every 15 minutes during business hours.
    Rates are locked at time of transaction submission.
    Additional fees may apply for exotic currency pairs.

    6. Holiday Processing
    Payments are not processed on bank holidays.
    The system accepts submissions on holidays but processes them the next business day.
    Refer to the holiday calendar for specific dates.
    Payment cutoff times may be earlier on days before holidays.

    7. Validation Rules
    All payments are subject to validation checks:
    - Account number format validation
    - Beneficiary bank code verification
    - Available balance check
    - Daily limit verification
    - Fraud detection screening

    Failed validations will result in payment rejection.
    Error codes and descriptions are provided for all validation failures.
    """


def analyze_chunks(chunks: list, chunk_size: int) -> dict:
    """Analyze chunking results"""
    if not chunks:
        return {
            "chunk_size": chunk_size,
            "num_chunks": 0,
            "avg_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0
        }

    token_counts = [c.get("metadata", {}).get("token_count", 0) for c in chunks]

    return {
        "chunk_size": chunk_size,
        "num_chunks": len(chunks),
        "avg_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0
    }


async def test_chunk_size(text: str, chunk_size: int) -> dict:
    """Test chunking with specific token size"""
    print(f"\nTesting chunk size: {chunk_size} tokens")
    print("-" * 60)

    # Create chunker
    chunker = TokenBasedChunker(
        chunk_size_tokens=chunk_size,
        chunk_overlap_percentage=15,
        preserve_tables=True
    )

    # Chunk text
    chunks = chunker.chunk_text_simple(text, doc_id="test_doc")

    # Analyze
    stats = analyze_chunks(chunks, chunk_size)

    print(f"Number of chunks: {stats['num_chunks']}")
    print(f"Average tokens per chunk: {stats['avg_tokens']:.1f}")
    print(f"Token range: {stats['min_tokens']} - {stats['max_tokens']}")

    # Show first chunk sample
    if chunks:
        first_chunk = chunks[0]["text"]
        print(f"\nFirst chunk preview:")
        print(f"{first_chunk[:200]}...")

    return stats


async def main():
    """Compare different chunk sizes"""
    print("\n" + "="*60)
    print("CHUNK SIZE COMPARISON TEST")
    print("="*60)

    # Load test document
    print("\nLoading test document...")
    text = load_test_document()
    print(f"Document length: {len(text)} characters")

    # Test different chunk sizes
    chunk_sizes = [400, 600, 800]
    results = []

    for size in chunk_sizes:
        stats = await test_chunk_size(text, size)
        results.append(stats)

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60 + "\n")

    print(f"{'Chunk Size':<15} {'Num Chunks':<15} {'Avg Tokens':<15} {'Min-Max':<20}")
    print("-" * 65)
    for r in results:
        print(f"{r['chunk_size']:<15} {r['num_chunks']:<15} {r['avg_tokens']:<15.1f} "
              f"{r['min_tokens']}-{r['max_tokens']}")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("\n400 tokens:")
    print("  - More chunks, better precision")
    print("  - Good for FAQs and simple queries")
    print("  - Faster retrieval")

    print("\n600 tokens:")
    print("  - Balanced approach (RECOMMENDED for complex docs)")
    print("  - Good context for multi-step procedures")
    print("  - Best for banking documents with tables")

    print("\n800 tokens:")
    print("  - Maximum context per chunk")
    print("  - Better for long-form policy documents")
    print("  - May reduce precision for specific queries")

    print("\n" + "="*60)

    # Save results
    output_dir = "tests/evaluation/results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/chunk_size_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}/chunk_size_comparison.json")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
