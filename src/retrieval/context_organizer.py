"""
Context organization and structure detection for RAG
Based on adaptive RAG patterns for better multi-section handling
"""

from typing import List, Dict, Set, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def auto_detect_structure(chunks: List[Dict]) -> Dict:
    """
    Automatically detect document structure from chunks using hierarchical metadata

    Args:
        chunks: List of retrieved chunks with metadata

    Returns:
        Structure info with modules, subsections, hierarchy, and module distribution
    """
    structure = {
        'modules': set(),           # h1 level (root_section)
        'subsections': set(),        # h2 level (parent_section)
        'sections': set(),           # Backward compatibility
        'topics': set(),
        'hierarchy': {},
        'documents': set(),
        'module_distribution': {},   # Count of chunks per module
        'depth_range': {'min': 999, 'max': 0},
        'has_cross_module': False,
        'breadcrumbs': []            # Sample breadcrumb paths
    }

    for chunk in chunks:
        payload = chunk.get('payload', {})
        metadata = payload.get('metadata', {})
        hierarchy = metadata.get('hierarchy', {})

        # Extract hierarchical structure
        if hierarchy:
            # Get module (h1/root_section)
            root_section = hierarchy.get('root_section', '')
            if root_section and root_section != 'unknown':
                structure['modules'].add(root_section)
                structure['module_distribution'][root_section] = \
                    structure['module_distribution'].get(root_section, 0) + 1

            # Get subsection (h2/parent_section)
            parent_section = hierarchy.get('parent_section', '')
            if parent_section and parent_section != root_section:
                structure['subsections'].add(parent_section)

            # Track depth range
            depth = hierarchy.get('depth', 0)
            if depth > 0:
                structure['depth_range']['min'] = min(structure['depth_range']['min'], depth)
                structure['depth_range']['max'] = max(structure['depth_range']['max'], depth)

            # Collect sample breadcrumbs
            full_path = hierarchy.get('full_path', '')
            if full_path and full_path not in structure['breadcrumbs']:
                structure['breadcrumbs'].append(full_path)
                if len(structure['breadcrumbs']) > 5:  # Limit to 5 samples
                    structure['breadcrumbs'].pop(0)

        # Backward compatibility: fallback to old fields
        section = payload.get('section_title')
        if section and section != 'unknown':
            structure['sections'].add(section)

        topic = payload.get('topic')
        if topic:
            structure['topics'].add(topic)

        doc_id = payload.get('doc_id')
        if doc_id:
            structure['documents'].add(doc_id)

    # Detect cross-module queries
    structure['has_cross_module'] = len(structure['modules']) > 1

    # Handle case where no depth was found
    if structure['depth_range']['min'] == 999:
        structure['depth_range']['min'] = 0

    logger.info(f"Structure detected: {len(structure['modules'])} modules, "
                f"{len(structure['subsections'])} subsections, "
                f"{len(structure['documents'])} documents, "
                f"cross-module: {structure['has_cross_module']}")

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
    Get adaptive system prompt based on detected structure with breadcrumb interpretation

    Args:
        structure: Structure info from auto_detect_structure

    Returns:
        Adapted system prompt with module-awareness
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

UNDERSTANDING LOCATION PATHS (Breadcrumbs):
- Each chunk shows its location as: "Module > Section > Subsection"
- The FIRST level = Main module (e.g., "Account Services", "Reports")
- The SECOND level = Feature area (e.g., "External Account Summary")
- The THIRD+ levels = Specific sub-features
- Use these paths to distinguish between similar features in different modules
"""

    # Add structure-specific instructions
    num_modules = len(structure.get('modules', set()))
    num_subsections = len(structure.get('subsections', set()))
    num_docs = len(structure.get('documents', set()))
    has_cross_module = structure.get('has_cross_module', False)
    module_dist = structure.get('module_distribution', {})

    # Cross-module query handling
    if has_cross_module:
        module_list = ", ".join([f'"{m}" ({count} chunks)'
                                for m, count in sorted(module_dist.items(),
                                                      key=lambda x: x[1], reverse=True)])

        structure_note = f"""
🔍 CROSS-MODULE QUERY DETECTED:
- Your knowledge spans {num_modules} different modules: {module_list}
- CRITICAL: Distinguish between modules when answering:
  * Identify which module each fact comes from using the Location path
  * When a feature exists in multiple modules, list each module SEPARATELY
  * Use clear headers like "## In [Module Name]" or "## [Module Name] Module"
  * Example: If asked about "account transfers" that exists in both "Account Services" and "Recurring Transfers", provide separate explanations for each

MODULE COMPARISON GUIDELINES:
1. Check the Location path of each chunk to identify the module
2. Group information by module when presenting your answer
3. Explicitly state the module name when explaining features
4. If comparing modules, use side-by-side comparison format
5. If only one module is relevant to the question, focus on that module only
"""
        return base_prompt + structure_note

    # Multi-section within same module
    elif num_subsections > 1 or num_docs > 1:
        structure_note = f"""
IMPORTANT - Multiple Sections Detected:
- Your knowledge spans {num_subsections} different subsections across {num_docs} documents
- If the question relates to multiple sections, organize your answer with clear headers
- Use headers (## Section Name) to separate different topics
- Make comparisons explicit when discussing different sections
- Group related information logically
- Use the Location paths to understand which section each information comes from
"""
        return base_prompt + structure_note

    # Single focused query
    return base_prompt + """
Response Guidelines:
- Natural, conversational tone - like a knowledgeable colleague
- Direct and confident explanations
- Keep responses focused and concise
- Use the Location path to understand the specific context of the information
"""


def format_context_note(structure: Dict) -> str:
    """
    Create a contextual note about structure for the LLM with module awareness

    Args:
        structure: Structure info from auto_detect_structure

    Returns:
        Context note string with module distribution info
    """
    num_modules = len(structure.get('modules', set()))
    num_subsections = len(structure.get('subsections', set()))
    num_docs = len(structure.get('documents', set()))
    has_cross_module = structure.get('has_cross_module', False)
    module_dist = structure.get('module_distribution', {})

    if num_modules <= 1 and num_subsections <= 1:
        return ""

    # Cross-module scenario
    if has_cross_module:
        module_details = []
        for module, count in sorted(module_dist.items(), key=lambda x: x[1], reverse=True):
            module_details.append(f'"{module}" ({count} chunks)')

        modules_str = ", ".join(module_details[:4])
        if num_modules > 4:
            modules_str += f" and {num_modules - 4} more"

        return f"""
📋 KNOWLEDGE SPAN:
- {num_modules} modules: {modules_str}
- {num_subsections} subsections across {num_docs} document(s)

⚠️  IMPORTANT: If the question involves multiple modules, clearly distinguish between them in your answer.
Use the Location paths to identify which module each information comes from.
"""

    # Multi-section within same module
    sections_list = ", ".join(list(structure.get('sections', set()))[:5])
    if len(structure.get('sections', set())) > 5:
        sections_list += f" and {len(structure.get('sections', set())) - 5} more"

    return f"""
Note: Your knowledge spans {num_subsections} subsections ({sections_list}) across {num_docs} document(s).
Organize your answer using the Location paths if multiple sections are relevant.
"""
