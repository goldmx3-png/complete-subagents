"""
Response Formatter - Convert API JSON responses to natural language
"""

import json
from typing import List, Dict, Tuple, Optional
from src.llm.openrouter_client import OpenRouterClient
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResponseFormatter:
    """
    Format API responses into natural, conversational language
    Uses LLM to convert raw JSON data into user-friendly text
    """

    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        # Use main model for quality formatting
        self.llm = llm_client or OpenRouterClient(model=settings.main_model)

    async def format_response(
        self,
        query: str,
        api_responses: List[Dict],
        api_definitions: List[Dict]
    ) -> str:
        """
        Format API responses into natural language

        Args:
            query: Original user query
            api_responses: List of API execution results
            api_definitions: List of API definitions

        Returns:
            Natural language response string
        """
        logger.info(f"Formatting {len(api_responses)} API response(s)")

        # Handle edge cases
        if not api_responses:
            return "I wasn't able to fetch any information. Please try again later."

        # Check if all APIs failed
        if all(not r.get("success") for r in api_responses):
            first_error = api_responses[0].get("error", "Unknown error")
            logger.error(f"All APIs failed: {first_error}")
            return self._format_error(first_error)

        # Build formatting prompt
        try:
            messages = self._build_formatting_prompt(query, api_responses, api_definitions)
            response = await self.llm.chat(
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )

            formatted = response.strip()
            logger.info(f"Response formatted successfully, length={len(formatted)}")
            return formatted

        except Exception as e:
            logger.error(f"Formatting error: {str(e)}")
            return self._fallback_format(api_responses, api_definitions)

    def _build_formatting_prompt(
        self,
        query: str,
        api_responses: List[Dict],
        api_definitions: List[Dict]
    ) -> List[Dict[str, str]]:
        """Build LLM prompt for response formatting"""

        # Format API responses
        responses_text = []
        has_list_formatting = False

        for idx, (response, definition) in enumerate(zip(api_responses, api_definitions), 1):
            if response.get("success"):
                data = response.get("data", {})

                # Check for list formatting template
                formatting_note = ""
                if definition.get("list_formatting_template"):
                    formatting_note = f"\n\n**FORMATTING INSTRUCTION (FOLLOW EXACTLY):**\n{definition['list_formatting_template']}"
                    has_list_formatting = True
                    logger.info(f"Using list formatting template for {definition['api_name']}")

                responses_text.append(f"""
API {idx}: {definition['api_name']}
Description: {definition['api_description']}
Response Data:
```json
{json.dumps(data, indent=2)}
```{formatting_note}
""")
            else:
                # API failed
                error = response.get("error", "Unknown error")
                logger.warning(f"API {definition['api_name']} failed: {error}")
                friendly_msg, suggestion = self._categorize_error(error)
                responses_text.append(f"""
API {idx}: {definition['api_name']}
Status: Failed
Error: {friendly_msg}
Suggestion: {suggestion}
""")

        responses_combined = "\n".join(responses_text)

        # Build instructions
        if has_list_formatting:
            instructions = """Follow any specific formatting instructions provided above. When presenting the information:
- Start with a friendly greeting
- Format tables using proper markdown with headers and separators
- Include totals if available
- Format amounts with currency symbols (e.g., "$1,234.56")
- Add helpful observations about the data
- Keep the tone warm and helpful"""
        else:
            instructions = """Answer naturally and conversationally:
- Use the specific details from the data
- Format numbers, dates, and amounts properly
- Use markdown for readability
- Be clear and friendly"""

        system_prompt = """You are a helpful banking assistant. Answer questions naturally using the data provided.

Important guidelines:
- Use ONLY the data shown in the response - don't make up or assume anything
- If information is missing, clearly say so
- Be accurate first, helpful second
- Keep your tone warm, professional, and conversational
- Avoid technical jargon"""

        user_prompt = f"""User Question: "{query}"

API Responses:
{responses_combined}

{instructions}

Please answer using only the data shown above. If something is missing or incomplete, just let the user know."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _categorize_error(self, error_message: str) -> Tuple[str, str]:
        """
        Categorize error and return user-friendly message

        Args:
            error_message: Raw error from API

        Returns:
            (user_friendly_message, suggestion)
        """
        error_lower = error_message.lower()

        # Authentication errors (401, 403)
        if any(x in error_message for x in ["401", "403"]) or any(x in error_lower for x in ["unauthorized", "forbidden"]):
            return (
                "I'm having trouble accessing your account information.",
                "This might be a temporary authentication issue. Please try again shortly."
            )

        # Not found (404)
        if "404" in error_message or "not found" in error_lower:
            return (
                "The information you're looking for isn't available.",
                "Please verify your request and try again."
            )

        # Server errors (500+)
        if any(x in error_message for x in ["500", "502", "503", "504"]) or "server error" in error_lower:
            return (
                "Our system is experiencing technical difficulties.",
                "We're working to resolve this. Please try again in a few minutes."
            )

        # Timeout
        if "timeout" in error_lower:
            return (
                "The request took longer than expected.",
                "The system might be busy. Please try again in a moment."
            )

        # Connection errors
        if any(x in error_lower for x in ["connection", "connect", "network"]):
            return (
                "I'm having trouble connecting to the banking system.",
                "This could be a temporary network issue. Please try again."
            )

        # Rate limit (429)
        if "429" in error_message or "rate limit" in error_lower or "too many requests" in error_lower:
            return (
                "You've made too many requests in a short time.",
                "Please wait a moment before trying again."
            )

        # Validation errors (400)
        if "400" in error_message or "bad request" in error_lower or "invalid" in error_lower:
            return (
                "There's an issue with the request.",
                "Please check your input and try again."
            )

        # Generic fallback
        logger.warning(f"Uncategorized error: {error_message}")
        return (
            "I encountered an unexpected issue.",
            "Please try again or contact support if the problem continues."
        )

    def _format_error(self, error_message: str) -> str:
        """Format error into user-friendly message"""
        friendly_msg, suggestion = self._categorize_error(error_message)
        return f"{friendly_msg}\n\n{suggestion}"

    def _fallback_format(
        self,
        api_responses: List[Dict],
        api_definitions: List[Dict]
    ) -> str:
        """
        Fallback formatting when LLM fails

        Args:
            api_responses: API responses
            api_definitions: API definitions

        Returns:
            Simple formatted response
        """
        result_parts = []

        for response, definition in zip(api_responses, api_definitions):
            if response.get("success"):
                result_parts.append(f"✓ Retrieved data from {definition['api_name']}")
                data = response.get("data", {})

                # Try to extract key information
                if isinstance(data, dict):
                    # Look for common fields
                    if "balance" in data:
                        result_parts.append(f"  Balance: ${data['balance']}")
                    if "status" in data:
                        result_parts.append(f"  Status: {data['status']}")
                    if "totalRecords" in data:
                        result_parts.append(f"  Total Records: {data['totalRecords']}")
            else:
                error = response.get("error", "Unknown error")
                friendly_msg, suggestion = self._categorize_error(error)
                result_parts.append(f"✗ {friendly_msg}")
                result_parts.append(f"  {suggestion}")

        return "\n".join(result_parts) if result_parts else "I couldn't retrieve the information. Please try again."

    async def format_transaction_list(self, transactions: List[Dict]) -> str:
        """
        Specialized formatter for transaction lists

        Args:
            transactions: List of transaction dicts

        Returns:
            Formatted transaction list
        """
        if not transactions:
            return "No transactions found."

        lines = ["**Your Transactions:**\n"]

        for idx, txn in enumerate(transactions, 1):
            amount = txn.get("amount", 0)
            currency = txn.get("currency", "USD")
            recipient = txn.get("recipient") or txn.get("to") or "Unknown"
            date = txn.get("date") or txn.get("created_at") or "Unknown"
            status = txn.get("status", "Unknown").title()
            ref_no = txn.get("chnRefNo") or txn.get("channelRefNo") or txn.get("id") or "N/A"

            lines.append(f"{idx}. **${amount:,.2f} {currency}** to {recipient}")
            lines.append(f"   Date: {date} | Status: {status} | Ref: {ref_no}")
            lines.append("")

        return "\n".join(lines)

    async def format_account_summary(self, account_data: Dict) -> str:
        """
        Specialized formatter for account summary

        Args:
            account_data: Account information dict

        Returns:
            Formatted account summary
        """
        balance = account_data.get("balance", 0)
        available = account_data.get("available_balance") or account_data.get("available", balance)
        currency = account_data.get("currency", "USD")
        account_number = account_data.get("account_number", "N/A")
        account_type = account_data.get("account_type", "Unknown")

        return f"""**Account Summary**

- Account Number: {account_number}
- Account Type: {account_type}
- Current Balance: **${balance:,.2f} {currency}**
- Available Balance: ${available:,.2f} {currency}"""


async def format_api_response(
    query: str,
    api_responses: List[Dict],
    api_definitions: List[Dict]
) -> str:
    """
    Convenience function for response formatting

    Usage:
        formatted = await format_api_response(
            "What's my balance?",
            [api_result],
            [api_definition]
        )
    """
    formatter = ResponseFormatter()
    return await formatter.format_response(query, api_responses, api_definitions)
