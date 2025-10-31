"""
Menu Bot Configuration
Defines menu structure, routing rules, and performance targets
"""

from typing import Dict, List
import os


class MenuConfig:
    """Centralized menu bot configuration"""

    # Feature Flags
    ENABLE_MENU_BOT = os.getenv("ENABLE_MENU_BOT", "true").lower() == "true"
    ENABLE_QUICK_ACTIONS = True
    ENABLE_CACHED_FAQS = True
    ENABLE_LIVE_CHAT = os.getenv("ENABLE_LIVE_CHAT", "false").lower() == "true"

    # Performance Targets (milliseconds)
    TARGET_MENU_RESPONSE_MS = 50
    TARGET_API_RESPONSE_MS = 500
    TARGET_CACHE_RESPONSE_MS = 10
    TARGET_RAG_RESPONSE_MS = 2000

    # Intent Classification
    CONFIDENCE_THRESHOLD = 0.7
    PATTERN_MATCH_SCORE_MIN = 70  # For fuzzy matching

    # Cache Settings (In-Memory)
    CACHE_BACKEND = "memory"  # Use in-memory caching (simple & fast)
    CACHE_FAQ_TTL = 86400 * 7  # 7 days (loaded at startup)
    CACHE_API_TTL = 30  # 30 seconds
    CACHE_MENU_TTL = 86400 * 30  # 30 days (static JSON files)

    # Main Menu Structure
    MAIN_MENU = {
        "title": "VTransact Banking Assistant",
        "greeting": "👋 Hi {user_name}! How can I help you today?",
        "quick_actions": [
            {
                "id": "balance_check",
                "label": "💰 Balance Check",
                "description": "View account balances",
                "route": "API",
                "action": "get_balances"
            },
            {
                "id": "pending_items",
                "label": "📋 Pending Items",
                "description": "Items needing attention",
                "route": "API",
                "action": "get_pending_items"
            },
            {
                "id": "search_docs",
                "label": "🔍 Search Documents",
                "description": "AI-powered document search",
                "route": "RAG",
                "action": "document_search"
            }
        ],
        "options": [
            {
                "id": "accounts",
                "icon": "🏦",
                "label": "Account Inquiries",
                "description": "View balances, statements, transactions",
                "route": "HYBRID",
                "submenu": "accounts_menu",
                "avg_response_time_ms": 200
            },
            {
                "id": "payments",
                "icon": "💸",
                "label": "Payment Information",
                "description": "Check payment status, transaction history",
                "route": "HYBRID",
                "submenu": "payments_menu",
                "avg_response_time_ms": 300
            },
            {
                "id": "documents",
                "icon": "📊",
                "label": "Documents & Reports",
                "description": "Search uploaded docs, generate reports",
                "route": "RAG",
                "submenu": "documents_menu",
                "avg_response_time_ms": 2000
            },
            {
                "id": "support",
                "icon": "👤",
                "label": "Help & Support",
                "description": "Get help, report issues, contact us",
                "route": "CACHE",
                "submenu": "support_menu",
                "avg_response_time_ms": 50
            }
        ],
        "free_text_hint": "💬 Or just ask me anything!"
    }

    # Intent Patterns for Quick Classification
    # NOTE: Only button clicks, greetings, and menu navigation use quick classifier
    # All other queries (balance, payments, documents, etc.) use LLM router for accurate classification
    INTENT_PATTERNS = {
        "MENU_NAVIGATION": [
            r"^[1-9]$",  # Numeric menu choice
            r"\bmain menu\b",
            r"\bgo back\b",
            r"\bshow menu\b",
            r"\bback\b",
            r"^account inquiries$",
            r"^payment information$",
            r"^documents & reports$",
            r"^documents and reports$",
            r"^help & support$",
            r"^help and support$"
        ],
        "GREETING": [
            r"^hi$",
            r"^hello$",
            r"^hey$",
            r"^start$",
            r"\bgood morning\b",
            r"\bgood afternoon\b",
            r"\bgood evening\b"
        ]
    }

    # Support Configuration
    SUPPORT_CONFIG = {
        "live_chat_hours": "09:00-18:00",
        "live_chat_enabled": ENABLE_LIVE_CHAT,
        "avg_response_time": {
            "high": "2-4 hours",
            "medium": "8-12 hours",
            "low": "24-48 hours"
        },
        "contact_numbers": {
            "US": "+1-800-XXX-XXXX",
            "UK": "+44-800-XXX-XXXX",
            "UAE": "+971-4-XXX-XXXX"
        },
        "email": "support@vtransact.com",
        "ticket_categories": [
            "Account Issues",
            "Payment Issues",
            "Document/Report Issues",
            "Technical Issues",
            "Security Issues",
            "Other"
        ],
        "ticket_priorities": {
            "high": {"sla_hours": 4, "label": "🔴 High - Urgent"},
            "medium": {"sla_hours": 12, "label": "🟡 Medium - Important"},
            "low": {"sla_hours": 48, "label": "🟢 Low - General"}
        },
        "faqs": [
            {
                "category": "accounts",
                "question": "How do I check my account balance?",
                "answer": "You can check your account balance by clicking 'Account Inquiries' from the main menu, then select 'Show all account balances'."
            },
            {
                "category": "accounts",
                "question": "How do I view recent transactions?",
                "answer": "Go to 'Account Inquiries' > 'Recent transactions' to see your latest account activity."
            },
            {
                "category": "payments",
                "question": "How do I check payment status?",
                "answer": "Select 'Payment Information' from the main menu, then choose 'Check specific payment' and enter your payment reference."
            },
            {
                "category": "payments",
                "question": "What payments need my approval?",
                "answer": "Click 'Payment Information' > 'Pending my approval' to see all payments waiting for your authorization."
            },
            {
                "category": "documents",
                "question": "How do I search documents?",
                "answer": "Select 'Documents & Reports' and then 'Search in uploaded documents'. You can ask questions about your documents in natural language."
            },
            {
                "category": "documents",
                "question": "How do I upload a document?",
                "answer": "Use the upload button or tell me 'upload document' to start the upload process. We support PDF, DOCX, and TXT files."
            },
            {
                "category": "general",
                "question": "What can this assistant do?",
                "answer": "I can help you with account inquiries, payment information, document search, and general support. You can either use the menu buttons or ask me questions in natural language."
            },
            {
                "category": "general",
                "question": "How do I get help?",
                "answer": "Click 'Help & Support' from the main menu to access FAQs, guides, or contact support."
            }
        ]
    }

    # API Endpoints Mapping
    API_ENDPOINTS = {
        "get_balances": "/accounts/balances",
        "get_transactions": "/accounts/transactions",
        "get_account_details": "/accounts/{account_id}",
        "get_payment_status": "/payments/{reference}",
        "get_pending_approvals": "/payments/pending",
        "get_recent_payments": "/payments/recent",
        "get_documents": "/documents/list",
        "get_statements": "/accounts/statements"
    }

    # Deep Links to Traditional UI
    UI_DEEP_LINKS = {
        "transaction_history": "/transactions",
        "account_details": "/accounts/{account_id}",
        "payment_initiate": "/payments/new",
        "beneficiary_management": "/beneficiaries",
        "bulk_upload": "/payments/bulk",
        "report_builder": "/reports/custom",
        "support_tickets": "/support/tickets"
    }

    @classmethod
    def get_ui_link(cls, link_id: str, **kwargs) -> str:
        """Get UI deep link with parameters"""
        link_template = cls.UI_DEEP_LINKS.get(link_id)
        if not link_template:
            return "/"
        return link_template.format(**kwargs)
