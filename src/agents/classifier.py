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
                "product": "payments" | "accounts" | "cards" | "loans" | null,
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

        system_prompt = """You are a banking chatbot classifier. Your task is to:
1. Classify user intent into one of three categories: RAG, API, or MENU
2. If intent is API, identify the banking product

**Intent Classification:**

🔹 **CAPABILITY** (Direct Answer - No Agent Needed):
- Questions asking about the AI assistant's capabilities
- Meta-questions about what the system can do
- Examples:
  - "What can you do?"
  - "What are your capabilities?"
  - "How can you help me?"
  - "What can you help me with?"

🔹 **RAG** (Retrieval-Augmented Generation):
- Policy questions (interest rates, terms, conditions, eligibility, features)
- General information about banking products
- "How to" questions about processes
- Educational/informational queries
- Examples:
  - "What is the interest rate for savings accounts?"
  - "What are the eligibility criteria for home loans?"
  - "Tell me about credit card rewards programs"
  - "How do I apply for a personal loan?"

🔹 **API** (Real-time Banking Data):
- Live account data (balance, status, details)
- Transaction history or specific transaction queries
- Payment status checks
- Card activation status
- Loan application status
- Account opening status
- Real-time operations
- Examples:
  - "What is my account balance?"
  - "Show my recent transactions"
  - "Is my payment processed?"
  - "What is the status of my loan application?"
  - "Check if my card is activated"

🔹 **MENU** (Navigation/Help):
- Simple greetings only
- Explicit menu navigation requests
- Examples:
  - "Hello", "Hi", "Hey"
  - "Show main menu", "Main menu", "Go back"

**Product Classification (if intent=API):**
- **payments**: Payments, transfers, transactions
- **accounts**: Account balance, account details, account status
- **cards**: Credit cards, debit cards, card status
- **loans**: Loan applications, loan status, loan details

**Response Format:**
Return ONLY valid JSON (no markdown, no explanation):
{
  "intent": "capability" | "rag" | "api" | "menu",
  "product": "payments" | "accounts" | "cards" | "loans" | null,
  "reasoning": "Brief explanation of classification",
  "confidence": 0.0-1.0,
  "direct_answer": "answer text here" (ONLY if intent is "capability")
}

**Important Rules:**
- If asking about AI assistant capabilities → CAPABILITY + provide direct_answer
- If asking about policies/information → RAG
- If asking about live data/status → API + product
- For CAPABILITY intent, provide a concise, friendly direct_answer explaining what the AI assistant can help with
- Use conversation history for context if provided
- Be confident in your classification
- Product is null unless intent is API

**Direct Answer Template for CAPABILITY:**
"I'm an AI banking assistant. I can help you with: finding information from documents, answering questions about banking operations, policies, procedures, transactions, accounts, and providing general banking assistance. How can I assist you today?"
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
                valid_products = ["payments", "accounts", "cards", "loans"]
                if result.get("product") not in valid_products:
                    logger.warning(f"Invalid or missing product for API intent: {result.get('product')}")
                    result["product"] = "accounts"  # Default fallback
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
        if "api" in response_lower and any(p in response_lower for p in ["payments", "accounts", "cards", "loans"]):
            # Extract product
            if "payments" in response_lower or "transfer" in response_lower:
                product = "payments"
            elif "accounts" in response_lower or "balance" in response_lower:
                product = "accounts"
            elif "cards" in response_lower or "credit" in response_lower or "debit" in response_lower:
                product = "cards"
            elif "loans" in response_lower:
                product = "loans"
            else:
                product = "accounts"

            return {
                "intent": "api",
                "product": product,
                "reasoning": "Extracted from fallback classification",
                "confidence": 0.6
            }

        elif "menu" in response_lower or "help" in response_lower or "greet" in response_lower:
            return {
                "intent": "menu",
                "product": None,
                "reasoning": "Menu intent detected via fallback",
                "confidence": 0.6
            }

        # Default to RAG
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
