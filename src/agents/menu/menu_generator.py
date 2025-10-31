"""
Menu Generator
Creates dynamic menus based on user context and state
"""

from typing import Dict, List, Optional
from src.config.menu_config import MenuConfig


class MenuGenerator:
    """
    Generates contextual menus for the chatbot
    """

    def __init__(self):
        self.config = MenuConfig

    def get_main_menu(self, user_id: str, user_name: str = None) -> Dict:
        """Generate main menu with buttons"""
        menu = self.config.MAIN_MENU

        greeting = menu["greeting"].format(
            user_name=user_name or "there"
        )

        response = f"{greeting}\n\n"
        response += "**Choose a topic or ask me anything:**\n"

        # Build buttons from menu options
        buttons = []
        for option in menu["options"]:
            buttons.append({
                "label": option["label"],
                "action": option["label"],
                "icon": option["icon"],
                "variant": "primary"
            })

        return {
            "text": response,
            "buttons": buttons,
            "menu_type": "main"
        }

    def get_accounts_menu(self, user_id: str) -> Dict:
        """Generate accounts menu with buttons"""
        response = "🏦 **Account Inquiries**\n\n"
        response += "What account information do you need?\n"

        buttons = [
            {"label": "Show all account balances", "action": "Show all account balances", "icon": "💰", "variant": "primary"},
            {"label": "Recent transactions", "action": "Recent transactions", "icon": "📝", "variant": "primary"},
            {"label": "Account details", "action": "Account details", "icon": "📋", "variant": "primary"},
            {"label": "Back to Main Menu", "action": "main menu", "icon": "🔙", "variant": "secondary"}
        ]

        return {
            "text": response,
            "buttons": buttons,
            "menu_type": "accounts"
        }

    def get_payments_menu(self, user_id: str) -> Dict:
        """Generate payments menu with buttons"""
        response = "💸 **Payment Information**\n\n"
        response += "What payment information do you need?\n"

        buttons = [
            {"label": "Pending my approval", "action": "Pending my approval", "icon": "⏳", "variant": "primary"},
            {"label": "Recent payments", "action": "Recent payments", "icon": "📋", "variant": "primary"},
            {"label": "Check specific payment", "action": "Check specific payment", "icon": "🔍", "variant": "primary"},
            {"label": "Back to Main Menu", "action": "main menu", "icon": "🔙", "variant": "secondary"}
        ]

        return {
            "text": response,
            "buttons": buttons,
            "menu_type": "payments"
        }

    def get_documents_menu(self, user_id: str) -> Dict:
        """Generate documents menu with buttons"""
        response = "📊 **Documents & Reports**\n\n"
        response += "How can I help with documents?\n"

        buttons = [
            {"label": "Search in uploaded documents", "action": "Search in uploaded documents", "icon": "🔍", "variant": "primary"},
            {"label": "Recently uploaded", "action": "Recently uploaded documents", "icon": "📄", "variant": "primary"},
            {"label": "Ask about documents", "action": "Ask questions about my documents", "icon": "💬", "variant": "primary"},
            {"label": "Back to Main Menu", "action": "main menu", "icon": "🔙", "variant": "secondary"}
        ]

        return {
            "text": response,
            "buttons": buttons,
            "menu_type": "documents"
        }

    def get_support_menu(self, user_id: str) -> Dict:
        """Generate support menu with buttons"""
        response = "👤 **Help & Support**\n\n"
        response += "How can I help you?\n"

        buttons = [
            {"label": "FAQs", "action": "Show me FAQs", "icon": "❓", "variant": "primary"},
            {"label": "How-to Guides", "action": "Show me how-to guides", "icon": "📚", "variant": "primary"},
            {"label": "Report a problem", "action": "Report a problem", "icon": "🎫", "variant": "primary"},
            {"label": "Contact Information", "action": "Contact information", "icon": "📞", "variant": "primary"},
            {"label": "Back to Main Menu", "action": "main menu", "icon": "🔙", "variant": "secondary"}
        ]

        return {
            "text": response,
            "buttons": buttons,
            "menu_type": "support"
        }

    def get_faq_menu(self, user_id: str) -> Dict:
        """Show FAQ menu with categories"""
        response = "❓ **Frequently Asked Questions**\n\n"
        response += "Choose a category to see common questions:\n"

        # Get unique categories
        categories = list(set([faq["category"] for faq in self.config.SUPPORT_CONFIG["faqs"]]))

        # Build buttons for each category
        buttons = []
        for cat in categories:
            buttons.append({
                "label": f"{cat.title()} FAQs",
                "action": f"show {cat} faqs",
                "icon": "❓",
                "variant": "primary"
            })

        # Add "View All" and "Back" buttons
        buttons.append({
            "label": "View All FAQs",
            "action": "view all faqs",
            "icon": "📋",
            "variant": "primary"
        })
        buttons.append({
            "label": "Back to Support",
            "action": "Help & Support",
            "icon": "🔙",
            "variant": "secondary"
        })

        return {
            "text": response,
            "buttons": buttons,
            "menu_type": "faq"
        }

    def get_category_faqs(self, category: str, user_id: str) -> Dict:
        """Show FAQs for a specific category"""
        response = f"❓ **{category.title()} FAQs**\n\n"

        faqs = [faq for faq in self.config.SUPPORT_CONFIG["faqs"] if faq["category"] == category]

        buttons = []
        for faq in faqs:
            response += f"• {faq['question']}\n"
            buttons.append({
                "label": faq['question'],
                "action": faq['question'],
                "icon": "❓",
                "variant": "primary"
            })

        response += f"\nClick a question to see the answer.\n"

        buttons.append({
            "label": "Back to FAQs",
            "action": "Show me FAQs",
            "icon": "🔙",
            "variant": "secondary"
        })

        return {
            "text": response,
            "buttons": buttons,
            "menu_type": f"faq_{category}"
        }

    def get_all_faqs(self, user_id: str) -> Dict:
        """Show all FAQs"""
        response = "❓ **All Frequently Asked Questions**\n\n"

        buttons = []
        for faq in self.config.SUPPORT_CONFIG["faqs"]:
            response += f"**{faq['question']}**\n{faq['answer']}\n\n"

        buttons.append({
            "label": "Back to Support",
            "action": "Help & Support",
            "icon": "🔙",
            "variant": "secondary"
        })

        return {
            "text": response,
            "buttons": buttons,
            "menu_type": "all_faqs"
        }

    def get_howto_guides_menu(self, user_id: str) -> Dict:
        """Show how-to guides with buttons"""
        response = "📚 **How-to Guides**\n\n"
        response += "Select a guide to learn step-by-step:\n"

        # Build buttons for each guide
        guides = [
            {"label": "View Account Statements", "action": "how to view account statements", "icon": "📄"},
            {"label": "Check Payment Status", "action": "how to check payment status", "icon": "💸"},
            {"label": "Download Reports", "action": "how to download reports", "icon": "📊"},
            {"label": "Manage Beneficiaries", "action": "how to manage beneficiaries", "icon": "👥"},
            {"label": "Upload Documents", "action": "how to upload documents", "icon": "📤"},
            {"label": "Search Documents", "action": "how to search documents", "icon": "🔍"},
        ]

        buttons = []
        for guide in guides:
            buttons.append({
                "label": guide["label"],
                "action": guide["action"],
                "icon": guide["icon"],
                "variant": "primary"
            })

        # Add "Back" button
        buttons.append({
            "label": "Back to Support",
            "action": "Help & Support",
            "icon": "🔙",
            "variant": "secondary"
        })

        return {
            "text": response,
            "buttons": buttons,
            "menu_type": "howto"
        }

    def get_contact_info(self, user_id: str) -> Dict:
        """Show contact information"""
        contacts = self.config.SUPPORT_CONFIG

        response = "📞 **Contact Information**\n\n"
        response += "**Call Center:**\n"
        for country, number in contacts["contact_numbers"].items():
            response += f"├─ {country}: {number}\n"

        response += f"\n📧 **Email:** {contacts['email']}\n"
        response += f"\n🕐 **Live Chat:** {contacts['live_chat_hours']}\n"

        if contacts["live_chat_enabled"]:
            response += "\n💬 Live chat is available during business hours.\n"

        buttons = [
            {
                "label": "Back to Support",
                "action": "Help & Support",
                "icon": "🔙",
                "variant": "secondary"
            }
        ]

        return {
            "text": response,
            "buttons": buttons,
            "menu_type": "contact"
        }

    def handle_menu_selection(self, selection: str, user_id: str) -> Dict:
        """
        Handle user's menu selection

        Args:
            selection: User's input (button label or text)
            user_id: User ID

        Returns:
            Menu dict with text and buttons
        """
        selection_lower = selection.lower()

        # Main menu options
        if "account inquiries" in selection_lower or selection == "1":
            return self.get_accounts_menu(user_id)
        elif "payment information" in selection_lower or selection == "2":
            return self.get_payments_menu(user_id)
        elif "documents" in selection_lower and "reports" in selection_lower or selection == "3":
            return self.get_documents_menu(user_id)
        elif ("help" in selection_lower and "support" in selection_lower) or selection == "4":
            return self.get_support_menu(user_id)

        # Support submenu options
        elif "show me faqs" in selection_lower or "faqs" in selection_lower:
            return self.get_faq_menu(user_id)
        elif "show me how-to guides" in selection_lower or "how-to guides" in selection_lower:
            return self.get_howto_guides_menu(user_id)
        elif "contact information" in selection_lower:
            return self.get_contact_info(user_id)
        elif "view all faqs" in selection_lower:
            return self.get_all_faqs(user_id)

        # Category FAQs
        elif "show accounts faqs" in selection_lower:
            return self.get_category_faqs("accounts", user_id)
        elif "show payments faqs" in selection_lower:
            return self.get_category_faqs("payments", user_id)
        elif "show documents faqs" in selection_lower:
            return self.get_category_faqs("documents", user_id)
        elif "show general faqs" in selection_lower:
            return self.get_category_faqs("general", user_id)

        # Back to main menu
        elif "main menu" in selection_lower or "back" in selection_lower or selection == "0":
            return self.get_main_menu(user_id)

        # Default: return main menu
        return self.get_main_menu(user_id)

    def handle_faq_question(self, question: str) -> Optional[str]:
        """Handle FAQ question and return answer - exact or near-exact match only"""
        question_lower = question.lower().strip()
        for faq in self.config.SUPPORT_CONFIG["faqs"]:
            faq_question_lower = faq["question"].lower().strip()
            # Only match if questions are very similar (not just substring)
            if faq_question_lower == question_lower or question_lower == faq_question_lower:
                return faq["answer"]
        return None


# Singleton instance
_generator = None


def get_menu_generator() -> MenuGenerator:
    """Get singleton menu generator instance"""
    global _generator
    if _generator is None:
        _generator = MenuGenerator()
    return _generator
