"""
Test script for agentic RAG integration fix
"""
import asyncio
import sys
from src.agents.orchestrator import OrchestratorAgent

async def test_auth_matrix_query():
    """Test the auth matrix query with agentic RAG"""

    # The complex query that should use agentic RAG
    query = "what is auth matrix and how it works for single and bulk payments?"

    print("=" * 80)
    print("Testing Agentic RAG Integration")
    print("=" * 80)
    print(f"\nQuery: {query}\n")

    # Initialize orchestrator
    print("Initializing orchestrator...")
    orchestrator = OrchestratorAgent()

    # Check routing decision
    if orchestrator.agentic_rag_agent.should_use(query):
        print("✓ Router decision: Use AGENTIC RAG\n")
    else:
        print("✗ Router decision: Use simple RAG (ERROR!)\n")
        return False

    # Run the orchestrator with streaming
    print("Running orchestrator with streaming...")
    print("-" * 80)

    full_response = ""
    async for chunk, state in orchestrator.run_stream(
        query=query,
        user_id="test_user",
        conversation_id="test_conv",
        conversation_history=[],
        is_button_click=False
    ):
        full_response += chunk
        print(chunk, end='', flush=True)

    print("\n" + "-" * 80)

    # Check final state
    route = state.get("route", "")
    active_agent = state.get("active_agent", "")
    metadata = state.get("metadata", {})

    print(f"\nFinal Route: {route}")
    print(f"Active Agent: {active_agent}")

    if metadata:
        print(f"\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    # Verify it used agentic RAG
    success = route == "AGENTIC_RAG" or active_agent == "AgenticRAGAgent"

    if success:
        print("\n✓ SUCCESS: Agentic RAG was used!")
    else:
        print(f"\n✗ FAILURE: Expected agentic RAG, got route={route}, agent={active_agent}")

    print("\nResponse length:", len(full_response))
    print("=" * 80)

    return success

if __name__ == "__main__":
    try:
        result = asyncio.run(test_auth_matrix_query())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
