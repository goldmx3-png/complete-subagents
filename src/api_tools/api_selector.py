"""
API Selector - LLM Call 2: Select best API from available options
"""

import json
from typing import List, Dict, Optional
from src.llm.openrouter_client import OpenRouterClient
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class APISelector:
    """
    Select best API(s) from available options using LLM
    This is the SECOND LLM call in the API flow:
      1. Unified Classifier (intent + product)
      2. API Selector (which specific API?) ← THIS MODULE
    """

    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        self.llm = llm_client or OpenRouterClient(model=settings.router_model)

    async def select_api(
        self,
        query: str,
        available_apis: List[Dict],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_apis: int = 2
    ) -> Dict:
        """
        Select best API(s) to answer user query

        Args:
            query: User's question
            available_apis: List of API definitions from database
            conversation_history: Previous conversation for context
            max_apis: Maximum number of APIs to select

        Returns:
            {
                "selected_apis": [
                    {
                        "api_name": str,
                        "api_definition": dict,
                        "parameters": dict,
                        "reasoning": str
                    }
                ],
                "confidence": float,
                "needs_clarification": bool,
                "clarification_question": str or None
            }
        """
        logger.info(f"Selecting API from {len(available_apis)} options for query: {query[:100]}")

        # Handle edge cases
        if not available_apis:
            return {
                "selected_apis": [],
                "confidence": 0.0,
                "needs_clarification": True,
                "clarification_question": "No APIs available for this request. Please try rephrasing your question."
            }

        if len(available_apis) == 1:
            # Only one option, select it directly
            return self._select_single_api(available_apis[0], query)

        # Multiple APIs available, use LLM to select
        try:
            messages = self._build_selection_prompt(query, available_apis, conversation_history, max_apis)
            response = await self.llm.chat(
                messages=messages,
                temperature=0.3,  # Low temperature for consistent selection
                max_tokens=800
            )

            result = self._parse_selection_response(response, available_apis)
            logger.info(f"Selected {len(result['selected_apis'])} API(s), confidence={result['confidence']:.2f}")
            return result

        except Exception as e:
            logger.error(f"API selection error: {str(e)}")
            # Fallback to first API
            return self._select_single_api(available_apis[0], query, confidence=0.5)

    def _build_selection_prompt(
        self,
        query: str,
        available_apis: List[Dict],
        conversation_history: Optional[List[Dict[str, str]]],
        max_apis: int
    ) -> List[Dict[str, str]]:
        """Build LLM prompt for API selection"""

        # Format API descriptions
        apis_description = []
        for idx, api in enumerate(available_apis, 1):
            desc = f"""
{idx}. API: {api['api_name']}
   Method: {api.get('http_method', 'POST')}
   Description: {api['api_description']}
   Parameters: {json.dumps(api.get('request_schema', {}), indent=2)}
"""
            apis_description.append(desc)

        apis_text = "\n".join(apis_description)

        # Add conversation context if available
        context_section = ""
        if conversation_history:
            recent_turns = conversation_history[-5:]  # Last 5 turns
            history_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in recent_turns
            ])
            context_section = f"""
Recent Conversation Context:
{history_text}

IMPORTANT: Use this context to understand pronouns, references, and implicit information in the current query.

"""

        system_prompt = """You are an API selector for a banking system.

Your task: Select the BEST API(s) to answer the user's query.

**Critical Rules for Financial Systems:**
- Select APIs based ONLY on what the query explicitly requests
- Do NOT make assumptions about missing information
- If required parameters are unclear, set needs_clarification=true
- Be conservative - only select APIs you're confident will work
- Consider conversation history for context

**Response Format:**
Return ONLY valid JSON (no markdown, no explanation):
{
  "selected_apis": [
    {
      "api_name": "exact_api_name_from_list",
      "parameters": {"param1": "value1"},
      "reasoning": "why this API"
    }
  ],
  "confidence": 0.0-1.0,
  "needs_clarification": true/false,
  "clarification_question": "question to ask user (if needed)" or null
}

**Examples:**

Query: "Show my recent transactions"
Available: [get_recent_transactions, get_transaction_by_id]
→ {"selected_apis": [{"api_name": "get_recent_transactions", "parameters": {}, "reasoning": "Directly retrieves recent transactions"}], "confidence": 0.95, "needs_clarification": false, "clarification_question": null}

Query: "What's the status of my payment?"
Available: [get_payment_status, get_payment_history]
→ {"selected_apis": [], "confidence": 0.0, "needs_clarification": true, "clarification_question": "Which payment are you asking about? Please provide the transaction ID or reference number."}"""

        user_prompt = f"""{context_section}
User Query: "{query}"

Available APIs:
{apis_text}

Select maximum {max_apis} API(s).

Return JSON only:"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _parse_selection_response(
        self,
        response: str,
        available_apis: List[Dict]
    ) -> Dict:
        """Parse LLM response and validate selection"""
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            # Parse JSON
            result = json.loads(response)

            # Validate required fields
            if "selected_apis" not in result:
                raise ValueError("Missing 'selected_apis' field")

            # Enrich selected APIs with full definitions
            enriched_apis = []
            for selected in result.get("selected_apis", []):
                api_name = selected.get("api_name")
                if not api_name:
                    continue

                # Find full API definition
                api_def = next(
                    (api for api in available_apis if api["api_name"] == api_name),
                    None
                )

                if api_def:
                    # Get request schema from API definition
                    request_schema = api_def.get("request_schema", {})
                    if isinstance(request_schema, str):
                        try:
                            request_schema = json.loads(request_schema)
                        except json.JSONDecodeError:
                            request_schema = {}

                    enriched_apis.append({
                        "api_name": api_name,
                        "api_definition": api_def,
                        "parameters": selected.get("parameters", request_schema),
                        "reasoning": selected.get("reasoning", "Selected by LLM")
                    })

            result["selected_apis"] = enriched_apis

            # Ensure required fields
            if "confidence" not in result:
                result["confidence"] = 0.8
            if "needs_clarification" not in result:
                result["needs_clarification"] = False
            if "clarification_question" not in result:
                result["clarification_question"] = None

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {str(e)}, response: {response}")
            # Fallback to first API
            return self._select_single_api(available_apis[0], "fallback", confidence=0.5)
        except Exception as e:
            logger.error(f"Parse error: {str(e)}")
            return self._select_single_api(available_apis[0], "fallback", confidence=0.5)

    def _select_single_api(
        self,
        api_definition: Dict,
        query: str,
        confidence: float = 0.9
    ) -> Dict:
        """Helper to select a single API"""

        # Get request schema
        request_schema = api_definition.get("request_schema", {})
        if isinstance(request_schema, str):
            try:
                request_schema = json.loads(request_schema)
            except json.JSONDecodeError:
                request_schema = {}

        return {
            "selected_apis": [
                {
                    "api_name": api_definition["api_name"],
                    "api_definition": api_definition,
                    "parameters": request_schema,
                    "reasoning": "Only available API" if confidence > 0.8 else "Fallback selection"
                }
            ],
            "confidence": confidence,
            "needs_clarification": False,
            "clarification_question": None
        }

    def validate_parameters(
        self,
        api_definition: Dict,
        provided_parameters: Dict
    ) -> Dict:
        """
        Validate parameters against API request schema

        Args:
            api_definition: API definition with request_schema
            provided_parameters: Parameters to validate

        Returns:
            {
                "valid": bool,
                "missing": list of missing required params,
                "invalid": list of invalid params
            }
        """
        request_schema = api_definition.get("request_schema", {})

        if isinstance(request_schema, str):
            try:
                request_schema = json.loads(request_schema)
            except json.JSONDecodeError:
                request_schema = {}

        if not request_schema:
            return {"valid": True, "missing": [], "invalid": []}

        required_params = request_schema.get("required", [])
        properties = request_schema.get("properties", {})

        missing = [
            param for param in required_params
            if param not in provided_parameters
        ]

        invalid = [
            param for param in provided_parameters
            if param not in properties
        ]

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "invalid": invalid
        }


async def select_best_api(
    query: str,
    available_apis: List[Dict],
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict:
    """
    Convenience function for API selection

    Usage:
        result = await select_best_api("What's my balance?", apis)
        if result["needs_clarification"]:
            print(result["clarification_question"])
        else:
            for api in result["selected_apis"]:
                print(f"Use: {api['api_name']}")
    """
    selector = APISelector()
    return await selector.select_api(query, available_apis, conversation_history)
