"""
Quick Intent Classifier for Menu Navigation Only

Handles ONLY:
- Button clicks (bypass to menu)
- Greetings (hi, hello → show menu)
- Menu navigation (main menu, go back → show menu)

Everything else routes to LLM router for proper classification
"""

import re
from typing import Dict, Optional, Tuple
from enum import Enum
from src.config.menu_config import MenuConfig


class Intent(Enum):
    """Intent categories for quick classification (menu navigation only)"""
    MENU_NAVIGATION = "MENU_NAVIGATION"
    GREETING = "GREETING"
    UNKNOWN = "UNKNOWN"  # Anything else goes to LLM router


class RouteDecision(Enum):
    """Routing decisions"""
    SHOW_MENU = "SHOW_MENU"
    USE_LLM = "USE_LLM"  # Route to orchestrator's LLM-based router


class QuickIntentClassifier:
    """
    Ultra-fast menu navigation classifier

    Bypasses LLM ONLY for:
    - Button clicks
    - Greetings (hi, hello, start)
    - Menu navigation keywords (main menu, go back)

    All other queries route to LLM router for accurate classification
    """

    def __init__(self):
        self.patterns = MenuConfig.INTENT_PATTERNS
        self.confidence_threshold = MenuConfig.CONFIDENCE_THRESHOLD

    def classify(self, query: str, context: Dict = None) -> Tuple[Intent, float, RouteDecision]:
        """
        Quick classify for menu navigation only

        Fast path:
        - Button clicks → SHOW_MENU
        - Greetings (hi, hello) → SHOW_MENU
        - Menu navigation (main menu, go back) → SHOW_MENU

        Everything else → USE_LLM (routes to LLM router)

        Args:
            query: User query
            context: Optional context (menu state, conversation history, is_first_message)

        Returns:
            (intent, confidence, route_decision)
        """
        query_lower = query.lower().strip()

        # Priority 1: Button clicks - always bypass to menu
        if context and context.get("is_button_click"):
            return Intent.MENU_NAVIGATION, 1.0, RouteDecision.SHOW_MENU

        # Priority 2: Check if in menu context (user typed number in menu)
        if context and context.get("in_menu_flow"):
            return Intent.MENU_NAVIGATION, 1.0, RouteDecision.SHOW_MENU

        # Priority 3: Pattern matching for greetings and menu navigation only
        for intent_name, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    intent = Intent[intent_name]
                    confidence = 0.95  # High confidence for exact pattern match
                    route = self._determine_route(intent, query_lower)
                    return intent, confidence, route

        # Everything else: Route to LLM router for proper classification
        # This includes: document searches, questions, API calls, etc.
        return Intent.UNKNOWN, 0.5, RouteDecision.USE_LLM

    def _determine_route(self, intent: Intent, query: str) -> RouteDecision:
        """Determine routing based on intent (simplified for menu navigation only)"""

        # Both MENU_NAVIGATION and GREETING show menu
        if intent in [Intent.MENU_NAVIGATION, Intent.GREETING]:
            return RouteDecision.SHOW_MENU

        # Unknown goes to LLM router
        return RouteDecision.USE_LLM


# Singleton instance
_classifier = None


def get_classifier() -> QuickIntentClassifier:
    """Get singleton classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = QuickIntentClassifier()
    return _classifier


def quick_classify(query: str, context: Dict = None) -> Tuple[Intent, float, RouteDecision]:
    """Quick classification function"""
    classifier = get_classifier()
    return classifier.classify(query, context)
