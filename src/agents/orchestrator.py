"""
Orchestrator Agent - Main coordinator with unified classification
Routes to domain agents using single LLM call for intent + product
"""

import time
from typing import Optional, List, Dict
from langgraph.graph import StateGraph, END
from src.agents.shared.state import AgentState, RAGState, APIState, MenuState, SupportState
from src.agents.rag.agent import RAGAgent
from src.agents.menu.agent import MenuAgent
from src.agents.api.agent import APIAgent
from src.agents.support.agent import SupportAgent
from src.agents.classifier import UnifiedClassifier
from src.utils.logger import get_logger
from src.config import settings

logger = get_logger(__name__)


class OrchestratorAgent:
    """
    Main orchestrator that coordinates domain agents

    Key Optimization:
    - Uses unified classifier (1 LLM call) for intent + product
    - Button clicks handled instantly (0 LLM calls)
    - Follow-up questions handled via query reformulation
    """

    def __init__(self):
        """Initialize orchestrator with all domain agents"""
        logger.info("Initializing Orchestrator Agent with Unified Classifier")

        # Initialize domain agents
        self.rag_agent = RAGAgent()
        self.menu_agent = MenuAgent()
        self.api_agent = APIAgent()
        self.support_agent = SupportAgent()

        # Unified classifier (replaces separate router + product classifier)
        self.unified_classifier = UnifiedClassifier()

        # Build LangGraph workflow
        self.graph = self._build_graph()

        logger.info("Orchestrator Agent initialized")

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("detect_intent", self._detect_intent_node)
        workflow.add_node("rag", self.rag_agent.execute)
        workflow.add_node("menu", self.menu_agent.execute)
        workflow.add_node("api", self.api_agent.execute)
        workflow.add_node("support", self.support_agent.execute)

        # Set entry point
        workflow.set_entry_point("detect_intent")

        # Add conditional routing from intent detection
        workflow.add_conditional_edges(
            "detect_intent",
            self._route_to_agent,
            {
                "rag": "rag",
                "menu": "menu",
                "api": "api",
                "support": "support"
            }
        )

        # All agents end the workflow
        workflow.add_edge("rag", END)
        workflow.add_edge("menu", END)
        workflow.add_edge("api", END)
        workflow.add_edge("support", END)

        return workflow.compile()

    async def _detect_intent_node(self, state: AgentState) -> AgentState:
        """
        Detect user intent using unified classifier
        Single LLM call returns: intent (RAG/API/Menu) + product (if API)
        """
        logger.info(f"Detecting intent for query: '{state['query'][:100]}...'")

        try:
            # Priority 1: Button clicks - instant response (0 LLM calls)
            if state.get("is_button_click"):
                state["route"] = "MENU"
                state["route_reasoning"] = "button-click"
                logger.info("Button click detected - routing to Menu (instant)")
                return state

            # Priority 2: Menu fast-path (first message, greetings)
            if await self.menu_agent.can_handle(state):
                state["route"] = "MENU"
                state["route_reasoning"] = "menu-quick-classification"
                logger.info("Menu fast-path triggered")
                return state

            # Priority 3: Unified classifier (1 LLM call for intent + product)
            logger.info("Using unified classifier (LLM Call 1)")
            classification = await self.unified_classifier.classify(
                query=state["query"],
                conversation_history=state.get("conversation_history", [])
            )

            intent = classification.get("intent", "rag")
            product = classification.get("product")
            reasoning = classification.get("reasoning", "")
            confidence = classification.get("confidence", 0)

            # Map classifier intents to route strings
            route_map = {
                "rag": "RAG_ONLY",
                "api": "API_ONLY",
                "menu": "MENU"
            }

            state["route"] = route_map.get(intent, "RAG_ONLY")
            state["route_reasoning"] = f"unified-classifier (confidence={confidence:.2f}): {reasoning}"

            # Store product in metadata for API agent
            if intent == "api" and product:
                if "metadata" not in state:
                    state["metadata"] = {}
                state["metadata"]["product"] = product
                logger.info(f"Intent: {intent.upper()}, Product: {product}, Confidence: {confidence:.2f}")
            else:
                logger.info(f"Intent: {intent.upper()}, Confidence: {confidence:.2f}")

        except Exception as e:
            logger.error(f"Intent detection error: {e}, defaulting to RAG")
            state["route"] = "RAG_ONLY"
            state["route_reasoning"] = "error-fallback"

        return state

    def _route_to_agent(self, state: AgentState) -> str:
        """Route to appropriate agent based on detected intent"""
        route = state.get("route", "RAG_ONLY")

        if route == "MENU":
            return "menu"
        elif route == "SUPPORT":
            return "support"
        elif route == "API_ONLY":
            return "api"
        else:  # RAG_ONLY or default
            return "rag"

    async def run(self, query: str, user_id: str, conversation_id: Optional[str] = None,
                  conversation_history: Optional[List[Dict]] = None,
                  is_button_click: bool = False) -> AgentState:
        """
        Run orchestrator with a query

        Args:
            query: User query
            user_id: User ID
            conversation_id: Optional conversation ID
            conversation_history: Optional conversation history
            is_button_click: Whether this is a button click

        Returns:
            Final agent state with response
        """
        start_time = time.time()
        logger.info(f"Orchestrator run: user_id={user_id}, query_len={len(query)}, is_button_click={is_button_click}")

        # Detect if this is the first message (no conversation history)
        is_first_message = not conversation_history or len(conversation_history) == 0

        # Initialize state
        initial_state = AgentState(
            query=query,
            user_id=user_id,
            conversation_id=conversation_id,
            conversation_history=conversation_history or [],
            route="",
            route_reasoning="",
            rag=RAGState(chunks=[], context="", is_ambiguous=False, disambiguation_options=[]),
            api=APIState(product=None, available_apis=[], selected_apis=[], api_responses=[], needs_clarification=False),
            menu=MenuState(intent=None, menu_buttons=[], is_menu=False),
            support=SupportState(faq_result=None, ticket_id=None),
            active_agent=None,
            final_response="",
            error=None,
            metadata={},
            is_button_click=is_button_click,
            is_first_message=is_first_message,
            in_menu_flow=False
        )

        # Run graph
        try:
            result = await self.graph.ainvoke(initial_state)
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Orchestrator complete in {duration_ms:.0f}ms: route={result.get('route')}, agent={result.get('active_agent')}")
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Orchestrator error after {duration_ms:.0f}ms: {e}", exc_info=True)
            initial_state["error"] = str(e)
            initial_state["final_response"] = "I encountered an error processing your request. Please try again."
            return initial_state
