"""
API Agent - Complete banking API integration
Handles realtime data queries (balance, transactions, status, etc.)
"""

from typing import Optional, Dict
from src.agents.base import BaseAgent
from src.agents.shared.state import AgentState, APIState
from src.api_tools import APIRegistry, APISelector, APIExecutor, ResponseFormatter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class APIAgent(BaseAgent):
    """
    API Agent for banking API integration

    Flow:
    1. Get product from unified classifier (already in state)
    2. Fetch available APIs from registry for that product
    3. Use LLM to select best API(s) from available options
    4. Execute API call(s) with parameters
    5. Format JSON response into natural language
    """

    def __init__(
        self,
        api_registry: Optional[APIRegistry] = None,
        api_selector: Optional[APISelector] = None,
        api_executor: Optional[APIExecutor] = None,
        response_formatter: Optional[ResponseFormatter] = None
    ):
        super().__init__()
        self.api_registry = api_registry or APIRegistry()
        self.api_selector = api_selector or APISelector()
        self.api_executor = api_executor or APIExecutor()
        self.response_formatter = response_formatter or ResponseFormatter()

    async def can_handle(self, state: AgentState) -> bool:
        """Check if API agent should handle this"""
        return state.get("route") in ["API_ONLY", "RAG_THEN_API"]

    async def execute(self, state: AgentState) -> AgentState:
        """Execute API integration flow"""
        self._log_start("API execution")

        # Initialize API state
        if "api" not in state:
            state["api"] = APIState(
                product=None,
                available_apis=[],
                selected_apis=[],
                api_responses=[],
                needs_clarification=False
            )

        try:
            query = state["query"]
            product = state.get("metadata", {}).get("product")

            if not product:
                logger.error("No product identified for API call")
                state["final_response"] = "I couldn't identify which banking service you're asking about. Could you please be more specific?"
                state["api"]["needs_clarification"] = True
                return state

            logger.info(f"API flow: product={product}, query={query[:100]}")

            # Step 1: Fetch available APIs for product from database
            logger.info(f"Fetching APIs for product: {product}")
            available_apis = await self.api_registry.get_apis_by_product(product)

            if not available_apis:
                logger.warning(f"No APIs found for product: {product}")
                state["final_response"] = f"I don't have access to {product} APIs at the moment. Please try again later or contact support."
                return state

            logger.info(f"Found {len(available_apis)} APIs for product '{product}'")
            state["api"]["product"] = product
            state["api"]["available_apis"] = available_apis

            # Step 2: Use LLM to select best API(s) from available options (LLM Call 2)
            logger.info("Selecting best API using LLM")
            selection_result = await self.api_selector.select_api(
                query=query,
                available_apis=available_apis,
                conversation_history=state.get("conversation_history", [])
            )

            # Check if clarification needed
            if selection_result.get("needs_clarification"):
                clarification_question = selection_result.get("clarification_question")
                logger.info(f"Clarification needed: {clarification_question}")
                state["api"]["needs_clarification"] = True
                state["final_response"] = clarification_question
                return state

            selected_apis = selection_result.get("selected_apis", [])

            if not selected_apis:
                logger.warning("No APIs selected")
                state["final_response"] = "I couldn't find the right API to answer your question. Could you please rephrase?"
                return state

            logger.info(f"Selected {len(selected_apis)} API(s): {[api['api_name'] for api in selected_apis]}")
            state["api"]["selected_apis"] = selected_apis

            # Step 3: Execute API call(s)
            logger.info("Executing API call(s)")

            if len(selected_apis) == 1:
                # Single API call
                api_def = selected_apis[0]["api_definition"]
                parameters = selected_apis[0].get("parameters", {})

                result = await self.api_executor.execute(
                    api_definition=api_def,
                    parameters=parameters,
                    user_context={"user_id": state["user_id"]}
                )
                api_responses = [result]
            else:
                # Multiple API calls in parallel
                api_calls = [
                    {
                        "api_definition": api["api_definition"],
                        "parameters": api.get("parameters", {})
                    }
                    for api in selected_apis
                ]

                api_responses = await self.api_executor.execute_multiple(
                    api_calls=api_calls,
                    user_context={"user_id": state["user_id"]}
                )

            state["api"]["api_responses"] = api_responses

            # Log API results
            for response in api_responses:
                if response.get("success"):
                    logger.info(f"API success: {response['api_name']}, time={response['execution_time_ms']:.0f}ms")
                else:
                    logger.error(f"API failed: {response['api_name']}, error={response.get('error')}")

            # Step 4: Format responses into natural language
            logger.info("Formatting API responses")
            api_definitions = [api["api_definition"] for api in selected_apis]

            formatted_response = await self.response_formatter.format_response(
                query=query,
                api_responses=api_responses,
                api_definitions=api_definitions
            )

            state["final_response"] = formatted_response
            logger.info(f"API flow complete, response length={len(formatted_response)}")

            self._log_complete("API execution")

        except Exception as e:
            self._log_error("API execution", e)
            logger.error(f"API agent error: {str(e)}", exc_info=True)
            state["error"] = str(e)
            state["final_response"] = "I encountered an issue while fetching your information. Please try again or contact support."

        return state

    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.api_registry.close()
            await self.api_executor.close()
        except Exception as e:
            logger.warning(f"Cleanup error: {str(e)}")

    def get_name(self) -> str:
        return "APIAgent"
