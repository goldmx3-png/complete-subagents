"""Shared state schema for all agents"""

from typing import TypedDict, Optional, List, Dict, Any


class RAGState(TypedDict, total=False):
    """RAG agent state"""
    chunks: List[Dict]
    context: str
    is_ambiguous: bool
    disambiguation_options: List[Dict]
    reformulated_query: Optional[str]


class APIState(TypedDict, total=False):
    """API agent state"""
    product: Optional[str]
    product_confidence: Optional[float]
    available_apis: List[Dict]
    selected_apis: List[Dict]
    api_responses: List[Dict]
    needs_clarification: bool


class MenuState(TypedDict, total=False):
    """Menu agent state"""
    intent: Optional[str]
    intent_confidence: Optional[float]
    menu_buttons: List[Dict]
    menu_type: Optional[str]
    is_menu: bool
    menu_context: Optional[Dict]


class SupportState(TypedDict, total=False):
    """Support agent state"""
    faq_result: Optional[Dict]
    ticket_id: Optional[str]
    cached_response: Optional[str]
    ticket_category: Optional[str]


class AgentState(TypedDict):
    """Global agent state shared across all agents"""
    # Core fields
    query: str
    user_id: str
    conversation_id: Optional[str]
    conversation_history: List[Dict]

    # Routing
    route: str
    route_reasoning: str

    # Agent-specific states
    rag: RAGState
    api: APIState
    menu: MenuState
    support: SupportState

    # Coordination
    active_agent: Optional[str]
    final_response: str
    error: Optional[str]
    metadata: Dict[str, Any]

    # Menu navigation flags
    is_button_click: bool
    is_first_message: bool
    in_menu_flow: bool
