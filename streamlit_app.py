"""
Streamlit UI for RAG System
Simple interface for document upload and chat
"""

import streamlit as st
import requests
from pathlib import Path
import json
from datetime import datetime
import re

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="VTransact Corporate Banking Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean message design with markdown support
st.markdown("""
<style>
    .user-message-container {
        display: flex;
        justify-content: flex-end;
        margin: 12px 0;
    }

    .bot-message-container {
        display: flex;
        justify-content: flex-start;
        margin: 12px 0;
    }

    .user-message {
        background-color: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 12px;
        padding: 12px 16px;
        max-width: 70%;
        text-align: left;
    }

    .bot-message {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 10px 14px;
        max-width: 50%;
        text-align: left;
    }

    /* Compact menu messages */
    .bot-message.compact {
        max-width: 40%;
        padding: 8px 12px;
        font-size: 0.92em;
    }

    .user-message .message-label {
        font-weight: 600;
        margin-bottom: 6px;
        color: #e2e8f0;
    }

    .user-message .message-content {
        color: #f1f5f9;
        line-height: 1.6;
    }

    .bot-message .message-label {
        font-weight: 600;
        margin-bottom: 6px;
        color: #e2e8f0;
    }

    .bot-message .message-content {
        color: #f1f5f9;
        line-height: 1.6;
    }

    /* Markdown element styling */
    .bot-message .message-content h1,
    .bot-message .message-content h2,
    .bot-message .message-content h3,
    .bot-message .message-content h4,
    .bot-message .message-content h5,
    .bot-message .message-content h6 {
        color: #f8fafc;
        margin: 12px 0 8px 0;
        font-weight: 600;
    }

    .bot-message .message-content h1 { font-size: 1.5em; }
    .bot-message .message-content h2 { font-size: 1.3em; }
    .bot-message .message-content h3 { font-size: 1.15em; }

    .bot-message .message-content code {
        background-color: #0f172a;
        color: #93c5fd;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }

    .bot-message .message-content pre {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 12px;
        overflow-x: auto;
        margin: 8px 0;
    }

    .bot-message .message-content pre code {
        background-color: transparent;
        padding: 0;
        color: #e2e8f0;
        display: block;
        white-space: pre;
    }

    .bot-message .message-content ul,
    .bot-message .message-content ol {
        margin: 8px 0;
        padding-left: 24px;
    }

    .bot-message .message-content li {
        margin: 4px 0;
        line-height: 1.6;
    }

    .bot-message .message-content blockquote {
        border-left: 3px solid #475569;
        padding-left: 12px;
        margin: 8px 0;
        color: #cbd5e1;
        font-style: italic;
    }

    .bot-message .message-content a {
        color: #60a5fa;
        text-decoration: none;
        border-bottom: 1px solid #60a5fa;
    }

    .bot-message .message-content a:hover {
        color: #93c5fd;
        border-bottom-color: #93c5fd;
    }

    .bot-message .message-content hr {
        border: none;
        border-top: 1px solid #334155;
        margin: 12px 0;
    }

    .bot-message .message-content strong {
        font-weight: 600;
        color: #f8fafc;
    }

    .bot-message .message-content em {
        font-style: italic;
        color: #e2e8f0;
    }

    .bot-message .message-content del {
        text-decoration: line-through;
        opacity: 0.7;
    }

    /* Table styling */
    .bot-message .message-content table {
        border-collapse: collapse;
        width: 100%;
        margin: 12px 0;
        font-size: 0.9em;
    }

    .bot-message .message-content table th {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 10px;
        text-align: left;
        font-weight: 600;
        color: #f8fafc;
    }

    .bot-message .message-content table td {
        border: 1px solid #334155;
        padding: 8px;
        color: #e2e8f0;
    }

    .bot-message .message-content table tr:nth-child(even) {
        background-color: #1a1f2e;
    }

    .bot-message .message-content table tr:hover {
        background-color: #2d3748;
    }

    /* Spinner styles */
    .spinner-container {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        margin: 12px 0;
        gap: 15px;
    }

    .spinner {
        border: 3px solid #334155;
        border-top: 3px solid #60a5fa;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin-left: 20px;
        flex-shrink: 0;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .loading-text {
        font-size: 14px;
        color: #60a5fa;
        font-weight: 500;
        background: linear-gradient(90deg, #334155 0%, #60a5fa 50%, #334155 100%);
        background-size: 200% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 2s linear infinite;
    }

    @keyframes shimmer {
        0% {
            background-position: -200% 0;
        }
        100% {
            background-position: 200% 0;
        }
    }

    /* Button styling - more compact */
    .stButton > button {
        background-color: #1e40af !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        font-size: 0.9em !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
        max-width: 400px !important;
    }

    .stButton > button:hover {
        background-color: #1e3a8a !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(30, 64, 175, 0.3) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_id' not in st.session_state:
    # Use a consistent user_id that persists across app restarts
    # This is stored in browser's session storage
    st.session_state.user_id = "default"

if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = None

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []

if 'waiting_for_disambiguation' not in st.session_state:
    st.session_state.waiting_for_disambiguation = False

if 'disambiguation_options' not in st.session_state:
    st.session_state.disambiguation_options = []

if 'show_welcome' not in st.session_state:
    st.session_state.show_welcome = True

if 'pending_input' not in st.session_state:
    st.session_state.pending_input = None

if 'is_button_click' not in st.session_state:
    st.session_state.is_button_click = False


def apply_inline_formatting(text):
    """Apply inline markdown formatting to text"""
    # Convert bold first (must come before italic to handle ** before *)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)

    # Convert italic (use word boundaries to avoid matching in middle of words)
    text = re.sub(r'(?<!\*)\*(?!\*)([^\*]+?)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'(?<!_)_(?!_)([^_]+?)_(?!_)', r'<em>\1</em>', text)

    # Convert inline code (`code`)
    text = re.sub(r'`([^`]+?)`', r'<code>\1</code>', text)

    # Convert links [text](url)
    text = re.sub(r'\[([^\]]+?)\]\(([^\)]+?)\)', r'<a href="\2" target="_blank">\1</a>', text)

    # Convert strikethrough (~~text~~)
    text = re.sub(r'~~(.+?)~~', r'<del>\1</del>', text)

    return text


def convert_table_to_html(table_lines):
    """Convert markdown table lines to HTML table"""
    if not table_lines:
        return ""

    html_parts = ['<table style="border-collapse: collapse; width: 100%; margin: 10px 0; background-color: #0f1419;">']

    row_idx = 0
    for idx, line in enumerate(table_lines):
        # Skip separator lines (|---|---|)
        if re.match(r'^\s*\|[\s\-\|:]+\|\s*$', line):
            continue

        # Parse cells
        cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Skip first and last empty splits

        # First row is header
        if idx == 0:
            html_parts.append('<thead><tr>')
            for cell in cells:
                cell_content = apply_inline_formatting(cell)
                html_parts.append(f'<th style="border: 1px solid #334155; padding: 10px; background-color: #1e293b; text-align: left; color: #f8fafc; font-weight: 600;">{cell_content}</th>')
            html_parts.append('</tr></thead>')
            html_parts.append('<tbody>')
        else:
            # Data rows - alternate background colors
            row_bg = "#1a1f2e" if row_idx % 2 == 0 else "#0f1419"
            html_parts.append(f'<tr style="background-color: {row_bg};">')
            for cell in cells:
                cell_content = apply_inline_formatting(cell)
                html_parts.append(f'<td style="border: 1px solid #334155; padding: 8px; color: #e2e8f0;">{cell_content}</td>')
            html_parts.append('</tr>')
            row_idx += 1

    html_parts.append('</tbody></table>')
    return ''.join(html_parts)


def convert_markdown_to_html(text):
    """Convert markdown formatting to HTML with proper escaping"""
    import html

    # First, escape HTML special characters to prevent injection
    text = html.escape(text)

    # Split into lines for processing
    lines = text.split('\n')
    result_lines = []
    in_code_block = False
    code_block_lines = []
    in_list = False
    list_lines = []
    in_table = False
    table_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Handle code blocks (```)
        if line.strip().startswith('```'):
            if in_code_block:
                # End code block
                code_content = '\n'.join(code_block_lines)
                result_lines.append(f'<pre><code>{code_content}</code></pre>')
                code_block_lines = []
                in_code_block = False
            else:
                # Start code block
                in_code_block = True
                # Check if language is specified (e.g., ```python)
                lang = line.strip()[3:].strip()
            i += 1
            continue

        if in_code_block:
            code_block_lines.append(line)
            i += 1
            continue

        # Handle headers (# ## ###)
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            level = len(header_match.group(1))
            content = apply_inline_formatting(header_match.group(2))
            result_lines.append(f'<h{level}>{content}</h{level}>')
            i += 1
            continue

        # Handle blockquotes (>)
        if line.strip().startswith('&gt;'):
            content = apply_inline_formatting(line.strip()[4:].strip())
            result_lines.append(f'<blockquote>{content}</blockquote>')
            i += 1
            continue

        # Handle horizontal rules (--- or ***)
        if re.match(r'^[\-\*]{3,}$', line.strip()):
            result_lines.append('<hr>')
            i += 1
            continue

        # Handle markdown tables (lines with pipes |)
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                table_lines = []
                in_table = True

            table_lines.append(line)

            # Check if next line is still part of the table
            if i + 1 >= len(lines) or '|' not in lines[i + 1] or not lines[i + 1].strip().startswith('|'):
                # End of table - convert to HTML
                table_html = convert_table_to_html(table_lines)
                result_lines.append(table_html)
                in_table = False
                table_lines = []

            i += 1
            continue

        # Handle bullet points and numbered lists
        bullet_match = re.match(r'^[\s]*[\-\*]\s+(.+)$', line)
        number_match = re.match(r'^[\s]*(\d+)\.\s+(.+)$', line)

        if bullet_match or number_match:
            if not in_list:
                list_lines = []
                in_list = True
                list_type = 'ul' if bullet_match else 'ol'

            content = bullet_match.group(1) if bullet_match else number_match.group(2)
            formatted_content = apply_inline_formatting(content)
            list_lines.append(f'<li>{formatted_content}</li>')

            # Check if next line continues the list
            if i + 1 >= len(lines) or (
                not re.match(r'^[\s]*[\-\*]\s+', lines[i + 1]) and
                not re.match(r'^[\s]*\d+\.\s+', lines[i + 1])
            ):
                # End of list
                list_html = ''.join(list_lines)
                result_lines.append(f'<{list_type}>{list_html}</{list_type}>')
                in_list = False
                list_lines = []
        else:
            # Regular line - apply inline formatting
            line = apply_inline_formatting(line)
            result_lines.append(line)

        i += 1

    # Close any unclosed code block
    if in_code_block and code_block_lines:
        code_content = '\n'.join(code_block_lines)
        result_lines.append(f'<pre><code>{code_content}</code></pre>')

    # Close any unclosed list
    if in_list and list_lines:
        list_html = ''.join(list_lines)
        result_lines.append(f'<{list_type}>{list_html}</{list_type}>')

    # Join lines with <br> for line breaks
    return '<br>'.join(result_lines)


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def upload_document_with_progress(file, user_id):
    """Upload document to the API with progress tracking"""
    progress_bar = None
    status_text = None

    try:
        # Create progress placeholder
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Uploading file (20%)
        status_text.text("📤 Uploading file to server...")
        progress_bar.progress(20)

        import time
        time.sleep(0.5)  # Brief pause for visual feedback

        files = {"file": (file.name, file, file.type)}
        data = {"user_id": user_id}

        # Step 2: Send upload request
        status_text.text("📄 Processing document...")
        progress_bar.progress(30)

        # Increase timeout to 5 minutes for large documents
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            data=data,
            timeout=300  # 5 minutes
        )

        if response.status_code == 200:
            result = response.json()

            # Update progress based on actual processing
            chunks_created = result.get('num_chunks', 0)

            if chunks_created > 0:
                # Step 3: Document parsed (50%)
                status_text.text(f"✂️ Chunking complete: {chunks_created} chunks created")
                progress_bar.progress(50)
                time.sleep(0.3)

                # Step 4: Embeddings generated (75%)
                status_text.text("🧠 Embeddings generated")
                progress_bar.progress(75)
                time.sleep(0.3)

                # Step 5: Stored in vector DB (100%)
                status_text.text("💾 Stored in vector database")
                progress_bar.progress(100)
                time.sleep(0.5)
            else:
                # No chunks created - show warning
                status_text.text("⚠️ No chunks were created from the document")
                progress_bar.progress(100)
                time.sleep(0.5)

            # Show success
            status_text.text("✅ Processing complete!")
            time.sleep(1)

            # Clear progress
            progress_bar.empty()
            status_text.empty()

            return result
        else:
            if progress_bar:
                progress_bar.empty()
            if status_text:
                status_text.empty()
            st.error(f"❌ Upload failed: {response.text}")
            return None

    except requests.exceptions.Timeout:
        if progress_bar:
            progress_bar.empty()
        if status_text:
            status_text.empty()
        st.error("⏱️ Upload timed out. The document might be too large or the server is busy.")
        st.info("💡 Tips:\n- Try a smaller document\n- Check if API server is running: `./scripts/rag-cli.sh status`\n- Check API logs: `./scripts/rag-cli.sh logs api`")
        return None
    except requests.exceptions.ConnectionError:
        if progress_bar:
            progress_bar.empty()
        if status_text:
            status_text.empty()
        st.error("🔌 Connection error. API server might not be running.")
        st.info("💡 Start the API server: `./scripts/rag-cli.sh start api`")
        return None
    except Exception as e:
        if progress_bar:
            progress_bar.empty()
        if status_text:
            status_text.empty()
        st.error(f"❌ Error uploading document: {str(e)}")
        st.info("💡 Check API logs for more details: `./scripts/rag-cli.sh logs api`")
        return None


def send_message_stream(message, user_id, conversation_id=None, is_button_click=False):
    """Send message to chat API with streaming"""
    try:
        payload = {
            "user_id": user_id,
            "message": message,
            "is_button_click": is_button_click,
        }

        if conversation_id:
            payload["conversation_id"] = conversation_id

        # Use streaming endpoint
        response = requests.post(
            f"{API_BASE_URL}/chat/stream",
            json=payload,
            timeout=120,
            stream=True
        )

        if response.status_code == 200:
            return response
        else:
            st.error(f"Chat failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")
        return None


def send_message(message, user_id, conversation_id=None, disambiguation_choice=None, is_button_click=False):
    """Send message to chat API (non-streaming fallback)"""
    try:
        payload = {
            "user_id": user_id,
            "message": message,
            "is_button_click": is_button_click,
        }

        if conversation_id:
            payload["conversation_id"] = conversation_id

        if disambiguation_choice is not None:
            payload["disambiguation_choice"] = disambiguation_choice

        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Chat failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")
        return None


def list_documents(user_id):
    """List all documents for user"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/documents/{user_id}",
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        st.error(f"Error listing documents: {str(e)}")
        return []


def delete_document(doc_id, user_id):
    """Delete a document"""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/documents/{doc_id}",
            params={"user_id": user_id},
            timeout=10
        )

        return response.status_code == 200
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False


# Main UI
st.title("🏦 VTransact Corporate Banking Assistant")
st.caption("Your AI-powered banking knowledge companion")

# Check API health
if not check_api_health():
    st.error("⚠️ Assistant service is not available. Please contact IT support.")
    st.stop()

st.success("✅ Assistant ready to help")

# Sidebar
with st.sidebar:
    st.header("📁 Knowledge Base")

    # User ID display
    st.text(f"User ID: {st.session_state.user_id[:12]}...")

    # Document upload
    st.subheader("Upload Policy/Procedure")
    uploaded_file = st.file_uploader(
        "Upload banking document",
        type=["pdf", "txt", "docx"],
        help="Upload banking product guides, service documents, or reference materials"
    )

    if uploaded_file is not None:
        if st.button("📤 Upload & Process", key="upload_btn"):
            result = upload_document_with_progress(uploaded_file, st.session_state.user_id)

            if result:
                # Show detailed success message
                st.success("✅ Document Successfully Uploaded & Processed!")

                # Show processing details in an expander
                with st.expander("📊 Processing Details", expanded=True):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("📄 Pages", result.get('num_pages', 0))

                    with col2:
                        st.metric("🔢 Total Chunks", result.get('num_chunks', 0))

                    with col3:
                        st.metric("📝 Text Chunks", result.get('num_text_chunks', 0))

                    if result.get('num_table_chunks', 0) > 0:
                        st.info(f"📊 Table Chunks: {result['num_table_chunks']}")

                    st.caption(f"📁 File: {result['file_name']}")
                    st.caption(f"💾 Size: {result.get('file_size', 0):,} bytes")
                    st.caption(f"🆔 Document ID: {result['doc_id'][:16]}...")

                # Refresh document list
                st.session_state.uploaded_documents = list_documents(st.session_state.user_id)

                # Auto-refresh after 2 seconds
                import time
                time.sleep(2)
                st.rerun()

    st.divider()

    # Document list
    st.subheader("📚 Available Documents")

    if st.button("🔄 Refresh"):
        st.session_state.uploaded_documents = list_documents(st.session_state.user_id)
        st.rerun()

    documents = list_documents(st.session_state.user_id)

    if documents:
        for doc in documents:
            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.text(f"📄 {doc.get('file_name', 'Unknown')[:20]}...")
                    st.caption(f"Indexed: {doc.get('chunk_count', 0)} sections")

                with col2:
                    if st.button("🗑️", key=f"del_{doc['doc_id']}", help="Remove"):
                        if delete_document(doc['doc_id'], st.session_state.user_id):
                            st.success("Removed!")
                            st.rerun()
    else:
        st.info("No documents in knowledge base")

    st.divider()

    # Clear conversation
    if st.button("🗑️ New Session"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.session_state.waiting_for_disambiguation = False
        st.session_state.disambiguation_options = []
        st.session_state.show_welcome = True
        st.session_state.pending_input = None
        st.rerun()

# Main chat area
st.header("💬 Ask the Assistant")

# Show welcome menu if this is a new session - Fetch from API
if st.session_state.show_welcome and len(st.session_state.messages) == 0:
    # Auto-send "hi" to get welcome menu from backend
    st.session_state.pending_input = "hi"
    st.session_state.show_welcome = False
    st.rerun()

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        # User message - full right alignment
        st.markdown(f"""
        <div class="user-message-container">
            <div class="user-message">
                <div class="message-label">You</div>
                <div class="message-content">{message['content']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Bot message - full left alignment
        content = message["content"]
        sources = message.get("sources", [])

        # Convert markdown to HTML
        formatted_content = convert_markdown_to_html(content)

        # Check if this is a menu message for compact styling
        is_menu = message.get("buttons") is not None
        message_class = "bot-message compact" if is_menu else "bot-message"

        st.markdown(f"""
        <div class="bot-message-container">
            <div class="{message_class}">
                <div class="message-label">VTransact Assistant</div>
                <div class="message-content">{formatted_content}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Check if this is the LAST message and render buttons from API metadata
        is_last_message = (idx == len(st.session_state.messages) - 1)

        if is_last_message and message.get("buttons"):
            st.write("")  # Add spacing
            buttons = message["buttons"]

            # Render buttons dynamically - one after another (vertically)
            for btn_idx, button in enumerate(buttons):
                button_label = f"{button.get('icon', '')} {button['label']}".strip()
                button_key = f"btn_{idx}_{btn_idx}"

                if st.button(button_label, key=button_key, use_container_width=True):
                    st.session_state.pending_input = button["action"]
                    st.session_state.is_button_click = True
                    st.rerun()

# Check if there's a pending input from button click
if st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None
    # is_button_click flag is already set in session state when button was clicked
else:
    # Chat input - always enabled for conversational flow
    user_input = st.chat_input("Ask about banking products, services, transactions, or account information...")
    # Reset button click flag for manual text input
    st.session_state.is_button_click = False

if user_input:
    # Hide welcome menu once user starts chatting
    st.session_state.show_welcome = False

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Display user message immediately
    st.markdown(f"""
    <div class="user-message-container">
        <div class="user-message">
            <div class="message-label">You</div>
            <div class="message-content">{user_input}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Create a placeholder for the streaming response
    response_placeholder = st.empty()
    full_response = ""
    conversation_id = st.session_state.conversation_id

    # Show loading spinner with text
    response_placeholder.markdown("""
    <div class="spinner-container">
        <div class="spinner"></div>
        <div class="loading-text">Searching my knowledge base...</div>
    </div>
    """, unsafe_allow_html=True)

    # Send to API with streaming
    try:
        stream_response = send_message_stream(
            user_input,
            st.session_state.user_id,
            st.session_state.conversation_id,
            st.session_state.is_button_click
        )

        if stream_response:
            first_content_received = False

            # Process SSE stream
            for line in stream_response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')

                    # Parse SSE format
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix

                        try:
                            data = json.loads(data_str)

                            if data.get('type') == 'conversation_id':
                                conversation_id = data.get('conversation_id')

                            elif data.get('type') == 'content':
                                # Append to response
                                content = data.get('content', '')
                                full_response += content

                                # Clear spinner on first content and show response
                                if not first_content_received:
                                    first_content_received = True
                                    response_placeholder.empty()

                                # Update placeholder with streaming content
                                formatted_content = convert_markdown_to_html(full_response)
                                response_placeholder.markdown(f"""
                                <div class="bot-message-container">
                                    <div class="bot-message">
                                        <div class="message-label">VTransact Assistant</div>
                                        <div class="message-content">{formatted_content}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            elif data.get('type') == 'done':
                                # Save metadata from done event
                                metadata = data

                            elif data.get('type') == 'end':
                                # Stream complete
                                break

                            elif data.get('type') == 'error':
                                st.error(f"Error: {data.get('error')}")
                                break

                        except json.JSONDecodeError:
                            continue

            # Update conversation ID
            if conversation_id:
                st.session_state.conversation_id = conversation_id

            # Add assistant response to history with button metadata
            if full_response:
                assistant_message = {
                    "role": "assistant",
                    "content": full_response,
                    "sources": []
                }

                # Add button metadata if this is a menu response
                if metadata and metadata.get("is_menu") and metadata.get("buttons"):
                    assistant_message["buttons"] = metadata["buttons"]
                    assistant_message["menu_type"] = metadata.get("menu_type")

                st.session_state.messages.append(assistant_message)

            st.rerun()
        else:
            st.error("Failed to get response from API")

    except Exception as e:
        st.error(f"Error during streaming: {str(e)}")

        # Show spinner with text for fallback
        response_placeholder.markdown("""
        <div class="spinner-container">
            <div class="spinner"></div>
            <div class="loading-text">Searching my knowledge base...</div>
        </div>
        """, unsafe_allow_html=True)

        # Fallback to non-streaming
        response = send_message(
            user_input,
            st.session_state.user_id,
            st.session_state.conversation_id,
            is_button_click=st.session_state.is_button_click
        )

        if response:
            st.session_state.conversation_id = response['conversation_id']
            st.session_state.messages.append({
                "role": "assistant",
                "content": response['message'],
                "sources": response.get('sources', [])
            })
            st.rerun()

# Footer
st.divider()
st.caption("💡 VTransact Corporate Banking Assistant - Your 24/7 banking knowledge companion")
st.caption("🔒 Secure • 🎯 Accurate • ⚡ Instant")
