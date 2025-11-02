"""Test script for agentic RAG workflow"""

import asyncio
import sys
from src.agents.rag.agent import RAGAgent
from src.agents.shared.state import AgentState, RAGState

async def test_agentic():
    """Test the agentic RAG workflow"""

    # Create initial state
    state = AgentState(
        query="What are the transaction processing rules?",
        user_id="test_user",
        conversation_id=None,
        conversation_history=[],
        route="RAG_ONLY",
        route_reasoning="Testing agentic RAG",
        rag=RAGState(
            chunks=[],
            context="",
            is_ambiguous=False,
            disambiguation_options=[],
            reformulated_query=None,
            use_agentic=True  # Enable agentic mode
        ),
        api={},
        menu={},
        support={},
        active_agent="rag",
        final_response="",
        error=None,
        metadata={},
        is_button_click=False,
        is_first_message=True,
        in_menu_flow=False
    )

    # Create RAG agent
    print("Creating RAG agent...")
    rag_agent = RAGAgent()

    # Execute agentic workflow
    print("\n" + "="*60)
    print("TESTING AGENTIC RAG WORKFLOW")
    print("="*60 + "\n")
    print(f"Query: {state['query']}")
    print(f"Use Agentic: {state['rag']['use_agentic']}")
    print("\nExecuting...\n")

    try:
        result_state = await rag_agent.execute(state)

        print("\n" + "="*60)
        print("RESULT")
        print("="*60 + "\n")
        print(f"Success: {not result_state.get('error')}")
        print(f"Error: {result_state.get('error')}")
        print(f"\nFinal Response:\n{result_state.get('final_response')}")

        if result_state.get("rag", {}).get("agentic_metadata"):
            print(f"\nAgentic Metadata:")
            import json
            print(json.dumps(result_state["rag"]["agentic_metadata"], indent=2))

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = asyncio.run(test_agentic())
    sys.exit(0 if success else 1)
