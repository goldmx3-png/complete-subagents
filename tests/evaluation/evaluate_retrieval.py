"""
Simple evaluation script for retrieval accuracy
Compares different retrieval configurations
"""

import asyncio
import json
import time
import csv
from pathlib import Path
from typing import List, Dict
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.retrieval.enhanced_retriever import EnhancedRAGRetriever
from src.config import settings


def load_test_queries(file_path: str = "tests/evaluation/test_queries.json") -> List[Dict]:
    """Load test queries from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


async def evaluate_single_query(retriever, query_data: Dict, user_id: str = "test_user") -> Dict:
    """
    Evaluate a single query

    Returns:
        {
            "query_id": int,
            "query": str,
            "category": str,
            "num_results": int,
            "avg_score": float,
            "top_score": float,
            "retrieval_method": str,
            "retrieval_time_ms": float,
            "total_time_ms": float
        }
    """
    try:
        start_time = time.time()

        # Retrieve
        result = await retriever.retrieve(
            query=query_data["query"],
            user_id=user_id,
            top_k=5
        )

        total_time = (time.time() - start_time) * 1000

        chunks = result.get("chunks", [])
        scores = [c.get("score", 0.0) for c in chunks]

        return {
            "query_id": query_data["id"],
            "query": query_data["query"],
            "category": query_data["category"],
            "num_results": len(chunks),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "top_score": max(scores) if scores else 0.0,
            "retrieval_method": result.get("retrieval_method", "unknown"),
            "retrieval_time_ms": result.get("timing", {}).get("retrieval", 0),
            "total_time_ms": total_time
        }
    except Exception as e:
        print(f"Error evaluating query {query_data['id']}: {str(e)}")
        return {
            "query_id": query_data["id"],
            "query": query_data["query"],
            "category": query_data["category"],
            "num_results": 0,
            "avg_score": 0.0,
            "top_score": 0.0,
            "retrieval_method": "error",
            "retrieval_time_ms": 0,
            "total_time_ms": 0
        }


async def run_evaluation(
    config_name: str,
    enable_hybrid: bool,
    enable_reranking: bool,
    output_dir: str = "tests/evaluation/results"
) -> Dict:
    """
    Run evaluation with specific configuration

    Returns:
        Summary statistics
    """
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"Hybrid Search: {enable_hybrid}")
    print(f"Reranking: {enable_reranking}")
    print(f"{'='*60}\n")

    # Override settings for this run
    original_hybrid = settings.enable_hybrid_search
    original_rerank = settings.enable_reranking

    settings.enable_hybrid_search = enable_hybrid
    settings.enable_reranking = enable_reranking

    # Load test queries
    queries = load_test_queries()

    # Initialize retriever
    retriever = EnhancedRAGRetriever()

    # Evaluate each query
    results = []
    for i, query_data in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Evaluating: {query_data['query'][:60]}...")
        result = await evaluate_single_query(retriever, query_data)
        results.append(result)
        print(f"  → {result['num_results']} results, "
              f"avg_score={result['avg_score']:.3f}, "
              f"time={result['total_time_ms']:.0f}ms\n")

        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)

    # Calculate summary statistics
    avg_results = sum(r["num_results"] for r in results) / len(results)
    avg_score = sum(r["avg_score"] for r in results) / len(results)
    avg_top_score = sum(r["top_score"] for r in results) / len(results)
    avg_time = sum(r["total_time_ms"] for r in results) / len(results)

    summary = {
        "config_name": config_name,
        "enable_hybrid": enable_hybrid,
        "enable_reranking": enable_reranking,
        "num_queries": len(queries),
        "avg_results_per_query": avg_results,
        "avg_score": avg_score,
        "avg_top_score": avg_top_score,
        "avg_time_ms": avg_time
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY - {config_name}")
    print(f"{'='*60}")
    print(f"Queries evaluated: {summary['num_queries']}")
    print(f"Avg results per query: {summary['avg_results_per_query']:.1f}")
    print(f"Avg relevance score: {summary['avg_score']:.3f}")
    print(f"Avg top score: {summary['avg_top_score']:.3f}")
    print(f"Avg retrieval time: {summary['avg_time_ms']:.0f}ms")
    print(f"{'='*60}\n")

    # Save detailed results to CSV
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = f"{output_dir}/{config_name.lower().replace(' ', '_')}_results.csv"

    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"Detailed results saved to: {csv_path}")

    # Restore original settings
    settings.enable_hybrid_search = original_hybrid
    settings.enable_reranking = original_rerank

    return summary


async def main():
    """Run all evaluation configurations"""
    print("\n" + "="*60)
    print("RAG RETRIEVAL EVALUATION")
    print("="*60)

    # Test configurations
    configs = [
        ("Baseline (Vector Only)", False, False),
        ("Hybrid Search", True, False),
        ("Hybrid + Reranking", True, True),
        ("Vector + Reranking", False, True)
    ]

    summaries = []

    for config_name, enable_hybrid, enable_reranking in configs:
        summary = await run_evaluation(
            config_name=config_name,
            enable_hybrid=enable_hybrid,
            enable_reranking=enable_reranking
        )
        summaries.append(summary)

    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60 + "\n")

    print(f"{'Configuration':<25} {'Avg Score':<12} {'Avg Top':<12} {'Avg Time':<12}")
    print("-" * 60)
    for s in summaries:
        print(f"{s['config_name']:<25} {s['avg_score']:<12.3f} {s['avg_top_score']:<12.3f} {s['avg_time_ms']:<12.0f}")

    # Calculate improvements
    baseline = summaries[0]
    print(f"\n{'Configuration':<25} {'Score Δ':<12} {'Time Δ':<12}")
    print("-" * 60)
    for s in summaries[1:]:
        score_delta = ((s['avg_score'] - baseline['avg_score']) / baseline['avg_score']) * 100
        time_delta = ((s['avg_time_ms'] - baseline['avg_time_ms']) / baseline['avg_time_ms']) * 100
        print(f"{s['config_name']:<25} {score_delta:+.1f}%{'':<6} {time_delta:+.1f}%")

    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
