"""Support Agent - FAQ and tickets (simplified)"""

from src.agents.base import BaseAgent
from src.agents.shared.state import AgentState, SupportState


class SupportAgent(BaseAgent):
    """Support Agent for help and tickets"""

    def __init__(self):
        super().__init__()
        self.faqs = {
            "how do i search": "To search, simply type your question and I'll find relevant information in your documents.",
            "how do i upload": "Use the upload feature to add PDF documents to your knowledge base.",
            "what can you do": "I can search your documents, answer questions, and help you find information quickly."
        }

    async def can_handle(self, state: AgentState) -> bool:
        """Check if support agent should handle this"""
        query_lower = state["query"].lower()
        return any(kw in query_lower for kw in ["how", "what", "why", "support", "ticket"])

    async def execute(self, state: AgentState) -> AgentState:
        """Execute support logic"""
        self._log_start("Support query")

        # Initialize support state
        if "support" not in state:
            state["support"] = SupportState(faq_result=None, ticket_id=None)

        try:
            query_lower = state["query"].lower()

            # Check FAQs
            for question, answer in self.faqs.items():
                if question in query_lower:
                    state["support"]["faq_result"] = {"question": question, "answer": answer}
                    state["final_response"] = f"**{question.title()}**\n\n{answer}"
                    self._log_complete("Support query", type="faq_hit")
                    return state

            # No FAQ match
            state["final_response"] = "I couldn't find an answer to that. Would you like to create a support ticket?"

            self._log_complete("Support query", type="faq_miss")

        except Exception as e:
            self._log_error("Support query", e)
            state["error"] = str(e)
            state["final_response"] = "Support system error. Please try again."

        return state

    def get_name(self) -> str:
        return "SupportAgent"
