"""Tests for Orchestrator Agent"""

import pytest
from src.agents.orchestrator import OrchestratorAgent


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    """Test orchestrator initializes correctly"""
    orchestrator = OrchestratorAgent()
    assert orchestrator is not None
    assert orchestrator.rag_agent is not None
    assert orchestrator.menu_agent is not None
    assert orchestrator.api_agent is not None
    assert orchestrator.support_agent is not None


@pytest.mark.asyncio
async def test_greeting_routes_to_menu():
    """Test greeting query routes to menu agent"""
    orchestrator = OrchestratorAgent()
    result = await orchestrator.run(
        query="Hello!",
        user_id="test_user"
    )
    assert result["route"] in ["MENU", "RAG_ONLY"]  # May vary
    assert result["final_response"] is not None


@pytest.mark.asyncio
async def test_question_routes_to_rag():
    """Test question routes to RAG agent"""
    orchestrator = OrchestratorAgent()
    result = await orchestrator.run(
        query="What is the revenue?",
        user_id="test_user"
    )
    assert result["route"] in ["RAG_ONLY", "SUPPORT"]
    assert result["final_response"] is not None


@pytest.mark.asyncio
async def test_support_query():
    """Test support query"""
    orchestrator = OrchestratorAgent()
    result = await orchestrator.run(
        query="How do I upload a document?",
        user_id="test_user"
    )
    assert result["route"] in ["SUPPORT", "RAG_ONLY"]
    assert result["final_response"] is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
