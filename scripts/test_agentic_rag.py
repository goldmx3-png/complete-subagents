"""
Test script for Agentic RAG implementation

Run this to test the agentic RAG workflow with sample banking queries.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.agentic_rag import get_agentic_rag_agent, AgenticRAGAgent
from src.agents.shared.state import AgentState
from src.utils.logger import get_logger
from src.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


# Test queries for banking support
TEST_QUERIES = [
    {
        "query": "What is the interest rate?",
        "expected_type": "simple",
        "description": "Simple factual question"
    },
    {
        "query": "Compare the features and benefits of savings accounts versus checking accounts",
        "expected_type": "complex",
        "description": "Complex comparative question"
    },
    {
        "query": "How do I open a new checking account? What documents do I need?",
        "expected_type": "complex",
        "description": "Multi-part procedural question"
    },
    {
        "query": "When does the bank close?",
        "expected_type": "simple",
        "description": "Simple factual question"
    },
    {
        "query": "What are the steps to apply for a mortgage loan and what are the eligibility criteria?",
        "expected_type": "complex",
        "description": "Complex multi-step question"
    }
]


def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_router():
    """Test the query router"""
    print_section("Testing Query Router")

    agent = get_agentic_rag_agent()

    for test_case in TEST_QUERIES:
        query = test_case["query"]
        expected = test_case["expected_type"]
        description = test_case["description"]

        # Analyze query
        analysis = agent.analyze_query(query)
        should_use = agent.should_use(query)

        actual_type = "complex" if should_use else "simple"
        match = "✓" if actual_type == expected else "✗"

        print(f"{match} Query: {query[:60]}...")
        print(f"   Description: {description}")
        print(f"   Expected: {expected}, Got: {actual_type}")
        print(f"   Complexity: {analysis['complexity_level']}")
        print(f"   Confidence: {analysis['confidence']:.2f}")
        print(f"   Reasons: {', '.join(analysis['reasons'])}")
        print()


async def test_agentic_workflow():
    """Test the full agentic RAG workflow"""
    print_section("Testing Agentic RAG Workflow")

    if not settings.agentic_rag_enabled:
        print("⚠️  Agentic RAG is DISABLED in settings")
        print("   Enable it by setting AGENTIC_RAG_ENABLED=true in .env")
        return

    agent = get_agentic_rag_agent()

    # Test with a complex query
    test_query = "Compare the features and benefits of savings accounts versus checking accounts"
    print(f"Query: {test_query}\n")

    try:
        # Create test state (knowledge base is shared across all users)
        state = AgentState(
            query=test_query,
            user_id="test_user",
            conversation_id="test_conv",
            conversation_history=[],
            route="",
            route_reasoning="",
            rag={},
            api={},
            menu={},
            support={},
            is_button_click=False,
            is_first_message=True,
            in_menu_flow=False,
            metadata={}
        )

        print("⏳ Processing with agentic RAG...\n")

        # Process query
        result = agent.process(state)

        # Display results
        print("✓ Processing complete!\n")
        print(f"Answer:\n{result['answer']}\n")
        print(f"Metadata:")
        print(f"  - Iterations: {result['metadata']['iterations']}")
        print(f"  - Confidence: {result['metadata']['confidence']:.2f}")
        print(f"  - Valid: {result['metadata']['is_valid']}")
        print(f"  - Retrieval Score: {result['metadata']['retrieval_score']:.3f}")
        print(f"  - Relevant Docs: {result['metadata']['relevant_docs_count']}")
        print(f"  - Total Docs: {result['metadata']['total_docs_retrieved']}")

        if result['sources']:
            print(f"\nSources ({len(result['sources'])}):")
            for source in result['sources'][:3]:
                print(f"  - Source {source['index']}: {source['content'][:100]}...")

    except Exception as e:
        print(f"✗ Error during processing: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)


def test_configuration():
    """Test configuration settings"""
    print_section("Configuration Settings")

    print("Agentic RAG Settings:")
    print(f"  - Enabled: {settings.agentic_rag_enabled}")
    print(f"  - Max Iterations: {settings.agentic_rag_max_iterations}")
    print(f"  - Min Relevant Docs: {settings.agentic_rag_min_relevant_docs}")
    print(f"  - Retrieval Top K: {settings.agentic_rag_retrieval_top_k}")
    print(f"  - Timeout: {settings.agentic_rag_timeout}ms")
    print(f"  - Parallel Grading: {settings.enable_parallel_grading}")
    print(f"  - Query Cache: {settings.enable_query_cache}")
    print(f"  - Early Exit: {settings.enable_early_exit}")
    print(f"  - Min Query Length: {settings.agentic_rag_min_query_length}")


def main():
    """Main test runner"""
    print_section("Agentic RAG Implementation Test Suite")

    # Test 1: Configuration
    test_configuration()

    # Test 2: Router
    test_router()

    # Test 3: Full workflow (async)
    print("\n" + "⚠️  NOTE: Full workflow test requires:")
    print("  1. Qdrant running (http://localhost:6333)")
    print("  2. Documents indexed in the 'documents' collection")
    print("  3. OpenRouter API key configured")
    print("\nProceed with full workflow test? (y/n): ", end="")

    response = input().strip().lower()
    if response == 'y':
        asyncio.run(test_agentic_workflow())
    else:
        print("\nSkipping full workflow test")

    print_section("Test Suite Complete")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        print(f"\n✗ Test suite failed: {e}")
        sys.exit(1)
