# Generic Structured RAG System

def auto_detect_structure(chunks):
    """Automatically detect document structure from chunks"""
    structure = {
        'sections': set(),
        'topics': set(),
        'hierarchy': {}
    }
    
    for chunk in chunks:
        meta = chunk.get('metadata', {})
        # Collect any structural elements present
        if 'section' in meta:
            structure['sections'].add(meta['section'])
        if 'topic' in meta:
            structure['topics'].add(meta['topic'])
        if 'page' in meta:
            structure['hierarchy'][meta.get('section', 'unknown')] = meta['page']
    
    return structure


def smart_organize_context(chunks):
    """Organize chunks by ANY available grouping"""
    groups = {}
    
    for chunk in chunks:
        meta = chunk.get('metadata', {})
        
        # Try multiple grouping strategies in priority order
        group_key = (
            meta.get('product') or 
            meta.get('section') or 
            meta.get('category') or 
            meta.get('topic') or
            f"Section_{meta.get('page', 'unknown')}"
        )
        
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(chunk['content'])
    
    # If only one group, return flat context
    if len(groups) == 1:
        return "\n\n".join(list(groups.values())[0])
    
    # Multiple groups - structure it
    structured = ""
    for group_name, contents in groups.items():
        structured += f"\n\n{'='*60}\n"
        structured += f"{group_name}\n"
        structured += f"{'='*60}\n"
        structured += "\n\n".join(contents)
    
    return structured


ADAPTIVE_SYSTEM_PROMPT = """You are a documentation expert assistant. When answering:

1. **Identify Structure**: If the context contains multiple sections/topics/categories, organize your answer accordingly

2. **Clear Separation**: When discussing different sections:
   - Use headers (## Section Name)
   - Group related information
   - Make comparisons explicit if relevant

3. **Adapt Format**: 
   - Single topic → Direct detailed answer
   - Multiple topics → Structured sections
   - Comparisons needed → Use tables or bullet comparisons

4. **Be Clear**: If context is mixed or ambiguous, organize it logically before presenting

5. **Cite Context**: Reference which part of documentation you're using
"""


def generic_rag_query(user_query, vector_db, llm):
    """Works with any document structure"""
    
    # 1. Retrieve relevant chunks
    chunks = vector_db.search(
        query=user_query,
        top_k=10
    )
    
    # 2. Detect document structure
    structure = auto_detect_structure(chunks)
    
    # 3. Organize context intelligently
    organized_context = smart_organize_context(chunks)
    
    # 4. Adapt prompt based on structure
    context_note = ""
    if len(structure['sections']) > 1:
        context_note = f"\nNote: Context spans {len(structure['sections'])} sections. Organize your answer accordingly.\n"
    
    prompt = f"""{ADAPTIVE_SYSTEM_PROMPT}

{context_note}

CONTEXT:
{organized_context}

USER QUESTION: {user_query}

Provide a well-structured answer. If multiple sections are relevant, separate them clearly."""

    # 5. Generate response
    response = llm.generate(prompt)
    return response


# ============================================
# SMART CHUNKING WITH AUTO-METADATA
# ============================================

import re

def smart_chunk_with_metadata(text, chunk_size=1000, overlap=200):
    """Automatically extract structure while chunking"""
    
    chunks = []
    lines = text.split('\n')
    
    current_chunk = ""
    current_metadata = {
        'section': None,
        'subsection': None,
        'page': None
    }
    
    for line in lines:
        # Detect headers (common patterns)
        if is_header(line):
            current_metadata['section'] = clean_header(line)
        
        # Detect sub-headers
        if is_subheader(line):
            current_metadata['subsection'] = clean_header(line)
        
        # Detect page numbers
        page_match = re.search(r'Page\s+(\d+)|^\s*(\d+)\s*$', line)
        if page_match:
            current_metadata['page'] = page_match.group(1) or page_match.group(2)
        
        current_chunk += line + "\n"
        
        # Chunk when size reached
        if len(current_chunk) >= chunk_size:
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': current_metadata.copy()
            })
            
            # Overlap
            current_chunk = current_chunk[-overlap:]
    
    # Last chunk
    if current_chunk.strip():
        chunks.append({
            'content': current_chunk.strip(),
            'metadata': current_metadata.copy()
        })
    
    return chunks


def is_header(line):
    """Detect if line is a header"""
    line = line.strip()
    return (
        # Numbered headers: "1.2.3 Title"
        re.match(r'^\d+(\.\d+)*\s+[A-Z]', line) or
        # ALL CAPS headers
        (line.isupper() and len(line) > 3 and len(line) < 100) or
        # Markdown style
        line.startswith('#') or
        # Underlined (next line is ===)
        False  # Would need next line context
    )


def is_subheader(line):
    """Detect sub-headers"""
    line = line.strip()
    return (
        # 1.1.1 style
        re.match(r'^\d+\.\d+\.\d+', line) or
        # Bold indicators (if preserved)
        line.startswith('**') or
        # Indented numbered
        re.match(r'^\s+\d+\.', line)
    )


def clean_header(line):
    """Extract clean header text"""
    line = line.strip()
    # Remove markdown
    line = re.sub(r'^#+\s*', '', line)
    # Remove numbering
    line = re.sub(r'^\d+(\.\d+)*\s*', '', line)
    # Remove bold
    line = re.sub(r'\*\*', '', line)
    return line.strip()


# ============================================
# USAGE
# ============================================

# Index documents
chunks = smart_chunk_with_metadata(pdf_text)
vector_db.add_documents(chunks)

# Query
response = generic_rag_query(
    user_query="What is the authorization matrix?",
    vector_db=vector_db,
    llm=llm
)