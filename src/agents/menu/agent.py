"""Menu Agent - Intent classification and menu generation"""

from src.agents.base import BaseAgent
from src.agents.shared.state import AgentState
from src.agents.menu.menu_generator import get_menu_generator
from src.agents.menu.intent_classifier import quick_classify, RouteDecision


class MenuAgent(BaseAgent):
    """Menu Agent for fast-path UI navigation"""

    def __init__(self):
        super().__init__()
        self.menu_generator = get_menu_generator()

    async def can_handle(self, state: AgentState) -> bool:
        """Check if menu agent should handle this"""
        # Quick classification
        context = {
            "is_button_click": state.get("is_button_click", False),
            "is_first_message": state.get("is_first_message", False),
            "in_menu_flow": state.get("in_menu_flow", False)
        }

        intent, confidence, route_decision = quick_classify(state["query"], context)

        # Menu agent handles only if route_decision is SHOW_MENU
        return route_decision == RouteDecision.SHOW_MENU

    async def execute(self, state: AgentState) -> AgentState:
        """Execute menu generation"""
        self._log_start("Menu generation")

        try:
            query_lower = state["query"].lower()
            user_id = state["user_id"]

            # Skip FAQ checking for greetings and first messages - always show menu
            is_greeting = any(kw in query_lower for kw in ["hi", "hello", "hey", "start"])
            is_first = state.get("is_first_message", False)

            # Check if it's an FAQ question (but not if it's a greeting or first message)
            if not is_greeting and not is_first:
                faq_answer = self.menu_generator.handle_faq_question(state["query"])
                if faq_answer:
                    state["final_response"] = faq_answer
                    state["menu"]["is_menu"] = False
                    state["route"] = "MENU"
                    self._log_complete("FAQ answer provided")
                    return state

            # Use menu generator to handle menu selection
            menu_data = self.menu_generator.handle_menu_selection(state["query"], user_id)

            # Set menu data in nested menu state structure
            state["menu"]["menu_buttons"] = menu_data["buttons"]
            state["menu"]["is_menu"] = True
            state["menu"]["menu_type"] = menu_data["menu_type"]
            state["final_response"] = menu_data["text"]
            state["route"] = "MENU"

            self._log_complete("Menu generation", menu_type=state["menu"]["menu_type"])

        except Exception as e:
            self._log_error("Menu generation", e)
            state["error"] = str(e)
            # Fallback to main menu
            menu_data = self.menu_generator.get_main_menu(user_id)
            state["final_response"] = menu_data["text"]
            state["menu"]["menu_buttons"] = menu_data["buttons"]
            state["menu"]["is_menu"] = True
            state["menu"]["menu_type"] = "main"

        return state

    def get_name(self) -> str:
        return "MenuAgent"
