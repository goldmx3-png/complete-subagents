"""
Unified Classifier: Merges Router + Product Classifier into single LLM call
Reduces 3 LLM calls to 2 for API queries
"""

import json
from typing import Dict, List, Optional
from src.llm.openrouter_client import OpenRouterClient
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedClassifier:
    """
    Single LLM call for:
    1. Intent classification (RAG, API, or Menu)
    2. Product identification (if API intent)
    """

    # Product configuration aligned with api_registry database
    PRODUCTS = {
        "single_payment": {
            "keywords": [
                # Transaction history queries
                "transaction", "transactions", "recent transactions", "transaction history",
                "transaction list", "payment history", "transaction summary",
                # Payment/transfer queries
                "payment", "payments", "transfer", "transfers", "pay", "remit", "fund transfer",
                # Status queries - IMPORTANT: These relate to transaction/payment status
                "pending", "pending verification", "pending release", "pending approval",
                "verification records", "pending records", "pending verifier",
                "completed", "completed transactions", "completed payments",
                # Recurring
                "recurring", "recurring transfer", "recurring payment", "standing order",
                # Debit accounts
                "debit account", "debit accounts", "entitled accounts"
            ],
            "description": "Payment transactions, transfers, transaction history, pending approvals/verification/release, completed payments, recurring transfers, and debit accounts"
        },
        "account_service": {
            "keywords": [
                "account", "accounts", "balance", "balances", "account balance",
                "account details", "account info", "account information", "account summary",
                "casa account", "current account", "my accounts"
            ],
            "description": "Account balance information, account details, and account summaries"
        },
        "bulk_payment": {
            "keywords": [
                "bulk", "bulk payment", "bulk transfer", "batch payment", "batch transfer",
                "multiple payments", "mass payment", "file upload"
            ],
            "description": "Bulk or batch payment operations"
        },
        "loans": {
            "keywords": [
                "loan", "loans", "loan application", "loan status", "loan details",
                "borrowing", "credit facility"
            ],
            "description": "Loan applications, status, and details"
        }
    }

    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        self.llm = llm_client or OpenRouterClient(
            model=settings.router_model
        )

    async def classify(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, any]:
        """
        Classify user query and identify product in single LLM call

        Args:
            query: User's question
            conversation_history: Previous conversation turns for context

        Returns:
            {
                "intent": "rag" | "api" | "menu",
                "product": "single_payment" | "account_service" | "cards" | "loans" | null,
                "reasoning": str,
                "confidence": float
            }
        """
        logger.info(f"Classifying query: {query[:100]}...")

        messages = self._build_prompt(query, conversation_history)

        try:
            response = await self.llm.chat(
                messages=messages,
                temperature=settings.router_temperature,
                max_tokens=500
            )

            result = self._parse_response(response)
            logger.info(f"Classification: intent={result['intent']}, product={result.get('product')}, confidence={result.get('confidence', 0):.2f}")
            return result

        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            # Fallback to RAG on error
            return {
                "intent": "rag",
                "product": None,
                "reasoning": f"Error during classification: {str(e)}",
                "confidence": 0.5
            }

    def _build_prompt(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Build the classification prompt"""

        # Build concise product list
        product_list = []
        for product_name, product_info in self.PRODUCTS.items():
            keywords = ", ".join(product_info["keywords"][:3])
            product_list.append(f"- **{product_name}**: {keywords}")
        products_section = "\n".join(product_list)

        system_prompt = f"""Classify banking queries into: CAPABILITY, RAG, API, or MENU.

**INTENT TYPES:**
- **CAPABILITY**: "What can you do?" → Provide direct_answer
- **RAG**: Policies, rates, how-to, general info
- **API**: Live data (balance, transactions, status, payments)
- **MENU**: Greetings, navigation

**PRODUCTS (if API):**
{products_section}
- **bulk_payment**: bulk, batch, file upload, pending bulk verification.
- **single_payment**: pending verification,  pending approval, pending release, transaction status.

**KEY RULES:**
1. "pending verification/records" = single_payment (NOT account_service)
2. "balance/account details" = account_service
3. Uncertain product → use RAG (NOT api)
4. Only API if product is clear

**EXAMPLES:**
- "pending verification records" → {{"intent":"api","product":"single_payment"}}
- "recent transactions" → {{"intent":"api","product":"single_payment"}}
- "account balance" → {{"intent":"api","product":"account_service"}}
- "interest rates" → {{"intent":"rag","product":null}}
- "help me" → {{"intent":"menu","product":null}}

Return JSON only:
{{"intent":"<intent>","product":"<product>|null","reasoning":"<brief>","confidence":0.0-1.0}}
For CAPABILITY, add: "direct_answer":"I'm an AI banking assistant. I help with documents, banking questions, transactions, accounts, and more."
"""

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history for context
        if conversation_history:
            for turn in conversation_history[-5:]:  # Last 5 turns
                messages.append(turn)

        # Add current query
        user_message = f"Classify this query:\n\n\"{query}\"\n\nReturn JSON only."
        messages.append({"role": "user", "content": user_message})

        return messages

    def _parse_response(self, response: str) -> Dict[str, any]:
        """Parse LLM response and extract classification"""
        try:
            # Try to find JSON in response
            response = response.strip()

            # Remove markdown code blocks if present
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
            if "intent" not in result:
                raise ValueError("Missing 'intent' field")

            # Validate intent value
            valid_intents = ["capability", "rag", "api", "menu"]
            if result["intent"] not in valid_intents:
                raise ValueError(f"Invalid intent: {result['intent']}")

            # Validate and handle capability intent
            if result["intent"] == "capability":
                # Ensure direct_answer is present for capability questions
                if "direct_answer" not in result or not result["direct_answer"]:
                    result["direct_answer"] = "I'm an AI banking assistant. I can help you with finding information from documents, answering questions about banking operations, policies, procedures, transactions, accounts, and more. How can I assist you today?"
                result["product"] = None
            # Validate product if API intent
            elif result["intent"] == "api":
                valid_products = list(self.PRODUCTS.keys())
                if result.get("product") not in valid_products:
                    logger.warning(f"Invalid or missing product for API intent: {result.get('product')} - falling back to RAG")
                    result["intent"] = "rag"  # Fallback to RAG instead of guessing
                    result["product"] = None
            else:
                result["product"] = None

            # Ensure confidence is present
            if "confidence" not in result:
                result["confidence"] = 0.8

            # Ensure reasoning is present
            if "reasoning" not in result:
                result["reasoning"] = "Classification successful"

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {str(e)}, response: {response}")
            # Try to extract intent using keywords
            return self._fallback_classification(response)
        except Exception as e:
            logger.error(f"Parse error: {str(e)}")
            return self._fallback_classification(response)

    def _fallback_classification(self, response: str) -> Dict[str, any]:
        """Fallback classification using keyword matching"""
        response_lower = response.lower()

        # Check for explicit intent mentions
        if "api" in response_lower:
            # Try to match product using PRODUCTS keywords
            matched_product = None
            for product_name, product_info in self.PRODUCTS.items():
                # Check if any keywords match
                for keyword in product_info["keywords"][:10]:  # Check first 10 keywords
                    if keyword.lower() in response_lower:
                        matched_product = product_name
                        break
                if matched_product:
                    break

            if matched_product:
                return {
                    "intent": "api",
                    "product": matched_product,
                    "reasoning": "Extracted from fallback classification",
                    "confidence": 0.6
                }
            else:
                # API mentioned but product unclear → fallback to RAG
                logger.info("API intent detected but product unclear - falling back to RAG")
                return {
                    "intent": "rag",
                    "product": None,
                    "reasoning": "API intent unclear, falling back to RAG",
                    "confidence": 0.5
                }

        elif "menu" in response_lower or "help" in response_lower or "greet" in response_lower:
            return {
                "intent": "menu",
                "product": None,
                "reasoning": "Menu intent detected via fallback",
                "confidence": 0.6
            }

        # Default to RAG for all other cases
        return {
            "intent": "rag",
            "product": None,
            "reasoning": "Default fallback to RAG",
            "confidence": 0.5
        }


async def get_unified_classification(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, any]:
    """
    Convenience function for classification

    Usage:
        result = await get_unified_classification("What's my balance?")
        # Returns: {"intent": "api", "product": "accounts", ...}
    """
    classifier = UnifiedClassifier()
    return await classifier.classify(query, conversation_history)
