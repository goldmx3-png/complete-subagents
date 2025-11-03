"""
Context organization and structure detection for RAG
Based on adaptive RAG patterns for better multi-section handling
"""

from typing import List, Dict, Set, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def auto_detect_structure(chunks: List[Dict]) -> Dict:
    """
    Automatically detect document structure from chunks

    Args:
        chunks: List of retrieved chunks with metadata

    Returns:
        Structure info: sections, topics, hierarchy
    """
    structure = {
        'sections': set(),
        'topics': set(),
        'hierarchy': {},
        'documents': set()
    }

    for chunk in chunks:
        payload = chunk.get('payload', {})

        # Collect structural elements
        section = payload.get('section_title')
        if section and section != 'unknown':
            structure['sections'].add(section)

        topic = payload.get('topic')
        if topic:
            structure['topics'].add(topic)

        page = payload.get('page_numbers', [])
        if page and section:
            structure['hierarchy'][section] = page[0] if isinstance(page, list) else page

        doc_id = payload.get('doc_id')
        if doc_id:
            structure['documents'].add(doc_id)

    logger.info(f"Structure detected: {len(structure['sections'])} sections, "
                f"{len(structure['documents'])} documents")

    return structure


def smart_organize_context(chunks: List[Dict], max_chunks: int = 10) -> str:
    """
    Organize chunks by ANY available grouping

    Args:
        chunks: Retrieved chunks
        max_chunks: Maximum chunks to include

    Returns:
        Intelligently organized context string
    """
    # Take top chunks
    top_chunks = chunks[:max_chunks]

    # Group chunks by available metadata
    groups = {}

    for i, chunk in enumerate(top_chunks):
        payload = chunk.get("payload", {})
        score = chunk.get("score", 0.0)

        # Try multiple grouping strategies in priority order
        section = payload.get('section_title', '')
        doc_id = payload.get('doc_id', 'unknown')
        page = payload.get('page_numbers', [])
        page_str = f"Page_{page[0]}" if page else "unknown"

        # Determine group key
        if section and section not in ['unknown', '', 'General Information']:
            group_key = f"{doc_id} - {section}"
        elif doc_id and doc_id != 'unknown':
            group_key = f"{doc_id} - {page_str}"
        else:
            group_key = f"Section_{page_str}"

        # Initialize group if needed
        if group_key not in groups:
            groups[group_key] = {
                'chunks': [],
                'max_score': 0.0,
                'doc_id': doc_id,
                'section': section or 'General',
                'pages': set()
            }

        # Add chunk to group
        text = payload.get('text', '')
        groups[group_key]['chunks'].append({
            'text': text,
            'score': score,
            'chunk_num': i + 1
        })
        groups[group_key]['max_score'] = max(groups[group_key]['max_score'], score)

        if page:
            page_num = page[0] if isinstance(page, list) else page
            groups[group_key]['pages'].add(page_num)

    logger.info(f"Organized {len(top_chunks)} chunks into {len(groups)} groups")

    # If only one group, return flat context (no need for structure)
    if len(groups) == 1:
        group_data = list(groups.values())[0]
        flat_chunks = []
        for chunk_data in group_data['chunks']:
            flat_chunks.append(
                f"[Chunk {chunk_data['chunk_num']}] Relevance: {chunk_data['score']:.2f}\n"
                f"{chunk_data['text']}"
            )
        return "\n\n---\n\n".join(flat_chunks)

    # Multiple groups - structure it clearly
    structured = ""
    for group_name, group_data in sorted(groups.items(), key=lambda x: x[1]['max_score'], reverse=True):
        pages_str = ", ".join(str(p) for p in sorted(group_data['pages'])) if group_data['pages'] else "N/A"

        structured += f"\n\n{'='*70}\n"
        structured += f"SOURCE: {group_name}\n"
        structured += f"Pages: {pages_str} | Relevance: {group_data['max_score']:.2f}\n"
        structured += f"{'='*70}\n\n"

        for chunk_data in group_data['chunks']:
            structured += f"[Chunk {chunk_data['chunk_num']}] {chunk_data['text']}\n\n"

    return structured


def get_adaptive_system_prompt(structure: Dict) -> str:
    """
    Get adaptive system prompt based on detected structure

    Args:
        structure: Structure info from auto_detect_structure

    Returns:
        Adapted system prompt
    """
    base_prompt = """You are a banking operations expert. Answer questions directly and confidently as if you personally know this information.

CRITICAL RULES - Response Style:
1. Answer naturally as an expert - NEVER mention "context", "documents", or "provided information"
2. Speak as if you inherently know this - use phrases like "The Cut-off master is..." not "Based on the context..."
3. NEVER say "based on the information provided" or similar phrases
4. Be definitive and authoritative - you are the expert they're asking
5. NEVER suggest "consult documentation", "ask someone else", or "contact support"
6. NEVER use hedging phrases like "I don't have detailed information" unless you truly have zero information

CRITICAL RULES - Accuracy:
1. Answer using ONLY the factual information available to you (never invent facts)
2. If you don't have information about something, simply say "I don't have information about that"
3. Synthesize information to give complete, coherent explanations
4. When explaining technical terms, provide clear definitions as an expert would

COMPREHENSIVENESS REQUIREMENTS:
- Provide thorough, comprehensive explanations covering ALL relevant aspects available in your knowledge
- When explaining features or concepts, describe different workflows, scenarios, and use cases
- Cover both common cases and important variations or edge cases
- If a topic has multiple types or variations (e.g., different workflows, processing modes), explain each one
- When relevant, explain how features work in different contexts (e.g., bulk vs single, sequential vs non-sequential)
- Aim for complete understanding, not just surface-level definitions
- Structure complex answers clearly (use numbered sections, bullet points for readability)

Example of COMPREHENSIVE answer:
"The Authorization Matrix determines approval workflows for payment requests. It functions differently based on the workflow type:
1. Sequential Authorization: Approvals proceed in order through levels...
2. Non-Sequential Authorization: All approvers can act simultaneously...
For bulk payments, three authorization criteria are available:
- Highest Amount: Based on the largest transaction...
- Total Amount: Based on sum of all transactions...
- Individual Amount: Each transaction evaluated separately (MDMC only)..."

Example of SHALLOW answer (AVOID):
"The Authorization Matrix handles payment approvals."
"""

    # Add structure-specific instructions
    num_sections = len(structure.get('sections', set()))
    num_docs = len(structure.get('documents', set()))

    if num_sections > 1 or num_docs > 1:
        structure_note = f"""
IMPORTANT - Multiple Sources Detected:
- Your knowledge spans {num_sections} different sections across {num_docs} documents
- MANDATORY: Organize your answer with clear structure
- Use numbered sections (1., 2., 3.) for main topics
- Use bullet points (-) for related items within sections
- Use headers (## Section Name) to separate different topics
- Make comparisons explicit when discussing different sections
- Ensure all relevant aspects from different sections are covered
- Group related information logically
"""
        return base_prompt + structure_note

    return base_prompt + """
Response Guidelines:
- Natural, conversational tone - like a knowledgeable colleague
- Direct and confident explanations
- Keep responses focused and concise
- Just answer as if you know it yourself
"""


def format_context_note(structure: Dict) -> str:
    """
    Create a note about context structure for the LLM

    Args:
        structure: Structure info

    Returns:
        Context note string
    """
    num_sections = len(structure.get('sections', set()))
    num_docs = len(structure.get('documents', set()))

    if num_sections <= 1:
        return ""

    sections_list = ", ".join(list(structure['sections'])[:5])
    if num_sections > 5:
        sections_list += f" and {num_sections - 5} more"

    return f"""
INSTRUCTION: Your knowledge spans {num_sections} sections ({sections_list}) across {num_docs} document(s).

You MUST provide a comprehensive answer covering all relevant aspects:
- Identify which sections/aspects apply to the question
- Explain each relevant aspect thoroughly
- Structure your response with numbered sections or bullet points
- Cover different workflows, scenarios, or use cases present in the knowledge
- Don't just summarize - explain how things work in different contexts
"""
