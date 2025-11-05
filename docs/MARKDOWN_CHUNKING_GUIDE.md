# Markdown-Based Chunking with Docling - Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [How It Works](#how-it-works)
5. [Integration Steps](#integration-steps)
6. [Testing Guide](#testing-guide)
7. [Comparison with Existing Chunkers](#comparison-with-existing-chunkers)
8. [Troubleshooting](#troubleshooting)
9. [Migration Guide](#migration-guide)

---

## Overview

This implementation adds **docling-based PDF parsing with markdown-aware chunking** to the document processing pipeline. It provides:

- ✅ **Better structure preservation**: Headers and document hierarchy maintained
- ✅ **Smart table handling**: Inline small tables, separate large tables
- ✅ **Two-stage splitting**: Header-based + token-constrained
- ✅ **Rich metadata**: Section hierarchy preserved in chunks
- ✅ **Backward compatible**: Works alongside existing chunkers

### Architecture

```
PDF Document
    ↓
[Docling Parser] → Converts PDF to Markdown
    ↓
Structured Markdown + Table Analysis
    ↓
[MarkdownChunker] → Two-stage splitting
    ├─ Stage 1: Split by headers (h1, h2, h3, h4)
    └─ Stage 2: Apply token size constraints
    ↓
Chunks with Rich Metadata
    ↓
Embeddings → Qdrant Vector Store
```

---

## Installation

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd /Users/kodurimohan/Desktop/AI-Projects/complete-subagents

# Activate virtual environment
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install new dependencies
pip install docling>=2.0.0
pip install langchain-text-splitters>=0.3.0

# Verify installation
python -c "from docling.document_converter import DocumentConverter; print('Docling installed successfully')"
python -c "from langchain_text_splitters import MarkdownHeaderTextSplitter; print('LangChain splitters installed successfully')"
```

### Step 2: Verify Requirements

The following should be in your `requirements.txt`:

```txt
docling>=2.0.0  # Advanced PDF to Markdown conversion
langchain-text-splitters>=0.3.0  # Markdown-aware text splitting
```

**Note**: `unstructured[all-docs]` has been removed as it's no longer used.

---

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# ===== Markdown-Based Chunking Configuration =====

# Enable markdown chunking (default: false)
USE_MARKDOWN_CHUNKING=false

# Chunk size in tokens (recommended: 400-800 for banking docs)
MARKDOWN_CHUNK_SIZE_TOKENS=600

# Overlap percentage (recommended: 10-20%)
MARKDOWN_CHUNK_OVERLAP_PERCENTAGE=15

# Token threshold for table size classification
# Tables below this size stay inline with text
# Tables above this size become separate chunks
MARKDOWN_TABLE_SIZE_THRESHOLD=500

# Preserve header hierarchy in chunk metadata (h1, h2, h3, h4)
MARKDOWN_PRESERVE_HEADERS=true

# Docling parser settings
DOCLING_EXTRACT_TABLES=true
DOCLING_EXTRACT_IMAGES=false  # For future enhancement
```

### Configuration Priority

The system respects configuration flags in this order:

1. **USE_MARKDOWN_CHUNKING** → Uses MarkdownChunker + MarkdownDocumentParser
2. **USE_SEMANTIC_CHUNKING** → Uses SemanticChunker (LLM-based)
3. **USE_TOKEN_BASED_CHUNKING** → Uses TokenBasedChunker
4. **Default** → Uses DocumentChunker (character-based)

This is handled by the `ChunkerFactory` class.

---

## How It Works

### 1. PDF to Markdown Conversion (MarkdownDocumentParser)

```python
from src.document_processing.markdown_parser import MarkdownDocumentParser

parser = MarkdownDocumentParser()
result = parser.parse_pdf("document.pdf")

# Result contains:
# - markdown_content: Full markdown text
# - table_elements: List of tables with metadata
# - text_elements: List of text sections with headers
# - metadata: Document stats (tokens, pages, etc.)
```

**Key Features**:
- Converts PDFs to structured markdown using docling
- Detects and analyzes markdown tables
- Classifies tables as "inline" (< 500 tokens) or "large" (>= 500 tokens)
- Extracts text sections with header hierarchy
- Computes token counts for accurate sizing

### 2. Smart Table Handling

```python
# Small table (< 500 tokens) → Stays inline
## Fee Schedule
| Service | Fee |
|---------|-----|
| ATM     | $2.50 |
```
**Result**: Kept in text chunks, processed with surrounding content

```python
# Large table (>= 500 tokens) → Extracted separately
## Rate Table (100 rows)
[TABLE_0_LARGE]  # Marker in text
```
**Result**:
- Marker replaces table in text
- Table becomes separate chunk with `chunk_type="table_large"`

### 3. Two-Stage Chunking (MarkdownChunker)

**Stage 1: Header-Based Splitting**
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Splits by markdown headers
headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]
```

**Stage 2: Token-Based Size Constraints**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensures chunks fit within token limits
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="cl100k_base",
    chunk_size=600,  # From config
    chunk_overlap=90,  # 15% of 600
)
```

### 4. Metadata Enrichment

Each chunk contains:

```python
{
    "text": "chunk content...",
    "user_id": "user123",
    "doc_id": "doc456",
    "chunk_id": "doc456_chunk_0",
    "chunk_type": "text" | "text_with_table" | "table_large",
    "sequence_number": 0,
    "metadata": {
        "section_hierarchy": {
            "h1": "Banking Services",
            "h2": "Account Types",
            "h3": "Savings Account"
        },
        "header_context": "Banking Services > Account Types > Savings Account",
        "token_count": 450,
        "char_count": 2100,
        "chunking_method": "markdown_header_recursive",
        "has_inline_tables": false,
        "has_table_markers": false
    }
}
```

---

## Integration Steps

### Step 1: Verify Installation

```bash
# Check if docling is installed
python -c "from docling.document_converter import DocumentConverter; print('✓ Docling OK')"

# Check if langchain splitters are installed
python -c "from langchain_text_splitters import MarkdownHeaderTextSplitter; print('✓ LangChain Splitters OK')"
```

### Step 2: Enable Markdown Chunking

Edit `.env`:

```bash
# Disable other chunking methods (optional)
USE_TOKEN_BASED_CHUNKING=false
USE_SEMANTIC_CHUNKING=false

# Enable markdown chunking
USE_MARKDOWN_CHUNKING=true
```

### Step 3: Verify Configuration Loading

```bash
python -c "from src.config import settings; print(f'Markdown chunking: {settings.use_markdown_chunking}')"
```

Expected output: `Markdown chunking: True`

### Step 4: Test Chunker Factory

```bash
python -c "
from src.document_processing.chunker_factory import ChunkerFactory
info = ChunkerFactory.get_chunker_info()
print('Active chunker:', info['chunker_type'])
print('Configuration:', info['configuration'])
"
```

Expected output:
```
Active chunker: MarkdownChunker
Configuration: {'chunk_size_tokens': 600, 'chunk_overlap_percentage': 15, ...}
```

### Step 5: Test Document Upload

```python
# test_markdown_upload.py
import asyncio
from src.document_processing.uploader import DocumentUploader

async def test_upload():
    uploader = DocumentUploader()

    # Check which parser is active
    print(f"Parser: {'Markdown' if uploader.use_markdown else 'Standard'}")
    print(f"Chunker: {uploader.chunker_name}")

    # Upload a test document
    result = await uploader.upload_document(
        file_path="path/to/test.pdf",
        user_id="test_user",
        doc_id="test_doc_001"
    )

    print(f"Status: {result['status']}")
    print(f"Chunks created: {result['chunks_created']}")

asyncio.run(test_upload())
```

---

## Testing Guide

### Manual Testing Steps

#### Test 1: Basic PDF Upload

1. **Prepare test PDF**: Use a banking policy document with headers and tables
2. **Upload via API**:
   ```bash
   curl -X POST "http://localhost:8000/api/upload" \
     -F "file=@test_document.pdf" \
     -F "user_id=test_user"
   ```
3. **Check logs** for:
   - "Using MarkdownDocumentParser with docling"
   - "Chunking document with MarkdownChunker..."
   - Chunk count and statistics

#### Test 2: Verify Chunk Structure

```python
from src.vectorstore.qdrant_store import QdrantStore

store = QdrantStore()
chunks = await store.search("test query", user_id="test_user", limit=5)

for chunk in chunks:
    print("Text:", chunk['text'][:100])
    print("Type:", chunk['chunk_type'])
    print("Metadata:", chunk['metadata'])
    print("---")
```

**Verify**:
- `section_hierarchy` is present
- `header_context` shows the path
- `chunking_method` = "markdown_header_recursive"
- Token counts are accurate

#### Test 3: Compare Chunking Methods

Create a test script to compare different chunking approaches:

```python
# test_compare_chunkers.py
import asyncio
from src.document_processing.markdown_parser import MarkdownDocumentParser
from src.document_processing.parser import DocumentParser
from src.document_processing.markdown_chunker import MarkdownChunker
from src.document_processing.token_chunker import TokenBasedChunker

async def compare_chunkers(pdf_path):
    # Method 1: Markdown chunking
    md_parser = MarkdownDocumentParser()
    md_result = md_parser.parse_pdf(pdf_path)
    md_chunker = MarkdownChunker()
    md_chunks = md_chunker.chunk_document(md_result, "doc1", "user1")

    # Method 2: Token-based chunking
    std_parser = DocumentParser()
    std_result = std_parser.parse_pdf(pdf_path)
    token_chunker = TokenBasedChunker()
    token_chunks = token_chunker.chunk_document(std_result, "doc1")

    # Compare
    print(f"Markdown chunks: {len(md_chunks)}")
    print(f"Token chunks: {len(token_chunks)}")

    # Analyze chunk sizes
    md_tokens = [c['metadata']['token_count'] for c in md_chunks]
    print(f"Markdown - Avg tokens: {sum(md_tokens)/len(md_tokens):.1f}")

    # Check metadata richness
    has_hierarchy = sum(1 for c in md_chunks if c['metadata'].get('section_hierarchy'))
    print(f"Chunks with header hierarchy: {has_hierarchy}/{len(md_chunks)}")

asyncio.run(compare_chunkers("test.pdf"))
```

#### Test 4: Retrieval Quality

```python
# Test retrieval with markdown chunks
from src.agents.rag.agent import RAGAgent

agent = RAGAgent()
response = await agent.arun({
    "query": "What are the savings account fees?",
    "user_id": "test_user",
    "conversation_history": []
})

print("Answer:", response['answer'])
print("Sources:", response['sources'])
```

**Evaluate**:
- Are answers more accurate with section context?
- Do retrieved chunks have relevant headers?
- Is table information properly retrieved?

---

## Comparison with Existing Chunkers

| Feature | Character Chunker | Token Chunker | Semantic Chunker | **Markdown Chunker** |
|---------|-------------------|---------------|------------------|----------------------|
| **Basis** | Character count | Token count | LLM boundaries | Headers + tokens |
| **Structure aware** | ❌ | ❌ | ✅ | ✅✅ |
| **Section context** | ❌ | ❌ | ❌ | ✅ |
| **Table handling** | Basic | Enhanced | Basic | Smart (inline/separate) |
| **Metadata richness** | Low | Medium | Medium | High |
| **Speed** | Fast | Fast | Slow (LLM) | Fast |
| **Token accuracy** | Poor | Excellent | Excellent | Excellent |
| **Best for** | Simple docs | General use | Complex docs | Structured docs |

**When to use Markdown Chunking**:
- ✅ Documents with clear header structure (policies, manuals, guides)
- ✅ Banking documents with sections and subsections
- ✅ Documents with mixed tables and text
- ✅ When section context improves retrieval
- ❌ Unstructured text (use token-based instead)
- ❌ Very short documents (overhead not worth it)

---

## Troubleshooting

### Issue 1: "Failed to import MarkdownDocumentParser"

**Cause**: Docling not installed

**Solution**:
```bash
pip install docling>=2.0.0
```

### Issue 2: "MarkdownHeaderTextSplitter not found"

**Cause**: LangChain text splitters not installed

**Solution**:
```bash
pip install langchain-text-splitters>=0.3.0
```

### Issue 3: Markdown chunking not activating

**Check**:
```python
from src.config import settings
print(settings.use_markdown_chunking)  # Should be True
```

**Verify .env**:
```bash
grep "USE_MARKDOWN_CHUNKING" .env
# Should output: USE_MARKDOWN_CHUNKING=true
```

### Issue 4: Docling conversion fails

**Symptoms**: Exception during PDF parsing

**Solutions**:
- Check PDF is valid and not corrupted
- Ensure PDF is not password-protected
- Try with a simpler PDF first
- Check docling logs for specific errors

**Fallback**: System will catch exception and log error. You can temporarily disable markdown chunking:
```bash
USE_MARKDOWN_CHUNKING=false
```

### Issue 5: Chunks too large or too small

**Adjust configuration**:
```bash
# Increase chunk size
MARKDOWN_CHUNK_SIZE_TOKENS=800

# Decrease overlap
MARKDOWN_CHUNK_OVERLAP_PERCENTAGE=10
```

### Issue 6: Tables not handled correctly

**Adjust table threshold**:
```bash
# Make more tables inline (smaller threshold)
MARKDOWN_TABLE_SIZE_THRESHOLD=300

# Make more tables separate (larger threshold)
MARKDOWN_TABLE_SIZE_THRESHOLD=700
```

---

## Migration Guide

### Migrating from Existing Chunking

**Important**: Markdown chunking changes how documents are parsed and chunked. Existing documents in Qdrant will remain unchanged.

#### Option 1: Gradual Migration (Recommended)

1. **Enable markdown chunking** in `.env`
2. **New uploads** use markdown chunking automatically
3. **Existing documents** remain in Qdrant with old chunking
4. **Re-upload critical documents** when convenient
5. **Monitor retrieval quality** for both old and new chunks

#### Option 2: Full Re-index

1. **Backup current collection**:
   ```python
   # Create new collection for backup
   from src.vectorstore.qdrant_store import QdrantStore
   store = QdrantStore()
   # Implement backup logic
   ```

2. **Delete existing documents**:
   ```python
   # For each document
   await uploader.delete_document(doc_id, user_id)
   ```

3. **Enable markdown chunking**
4. **Re-upload all documents**

#### Option 3: Parallel Collections

Run both systems in parallel:

```bash
# Original collection
QDRANT_COLLECTION=documents

# New markdown collection
QDRANT_COLLECTION=documents_markdown
```

Then compare retrieval quality before switching.

### Migration Checklist

- [ ] Install docling and langchain-text-splitters
- [ ] Update `.env` with markdown chunking config
- [ ] Test with sample documents
- [ ] Verify chunk structure and metadata
- [ ] Compare retrieval quality
- [ ] Decide on migration strategy (gradual/full/parallel)
- [ ] Update monitoring/metrics for new chunk types
- [ ] Document any custom adjustments to configuration

---

## Advanced Configuration

### Fine-Tuning for Banking Documents

```bash
# Larger chunks for detailed policy sections
MARKDOWN_CHUNK_SIZE_TOKENS=800
MARKDOWN_CHUNK_OVERLAP_PERCENTAGE=20

# Smaller threshold to keep more tables inline
MARKDOWN_TABLE_SIZE_THRESHOLD=400

# Always preserve headers for context
MARKDOWN_PRESERVE_HEADERS=true
```

### Optimizing for Performance

```bash
# Smaller chunks for faster retrieval
MARKDOWN_CHUNK_SIZE_TOKENS=400
MARKDOWN_CHUNK_OVERLAP_PERCENTAGE=10

# Larger threshold to minimize separate table chunks
MARKDOWN_TABLE_SIZE_THRESHOLD=700
```

### Custom Parser Settings

```python
from src.document_processing.markdown_parser import MarkdownDocumentParser

# Custom initialization
parser = MarkdownDocumentParser(
    extract_tables=True,
    extract_images=False,  # Not yet supported
    table_size_threshold=600  # Override config
)
```

---

## API Integration

The markdown chunking is transparent to the API. No changes needed:

```bash
# Standard upload endpoint
POST /api/upload
Content-Type: multipart/form-data

{
  "file": <PDF file>,
  "user_id": "user123"
}
```

The system automatically uses markdown chunking if enabled in `.env`.

---

## Monitoring and Metrics

### Check Active Configuration

```python
from src.document_processing.chunker_factory import ChunkerFactory

info = ChunkerFactory.get_chunker_info()
print(info)
# Output:
# {
#   'chunker_type': 'MarkdownChunker',
#   'configuration': {...}
# }
```

### Chunk Statistics

```python
from src.document_processing.markdown_chunker import MarkdownChunker

chunker = MarkdownChunker()
chunks = chunker.chunk_document(parsed_doc, doc_id, user_id)
stats = chunker.get_chunk_stats(chunks)

print(stats)
# Output:
# {
#   'total_chunks': 25,
#   'avg_tokens': 520,
#   'text_chunks': 22,
#   'table_chunks': 3,
#   'chunks_with_tables': 5,
#   'min_tokens': 180,
#   'max_tokens': 600
# }
```

---

## Summary

### What Was Implemented

1. ✅ **MarkdownDocumentParser** - Docling-based PDF to markdown conversion
2. ✅ **MarkdownChunker** - Two-stage header + token splitting
3. ✅ **ChunkerFactory** - Intelligent chunker selection (fixes existing bug)
4. ✅ **Smart table handling** - Inline small tables, separate large tables
5. ✅ **Rich metadata** - Section hierarchy preserved
6. ✅ **DocumentUploader integration** - Automatic parser/chunker selection
7. ✅ **Configuration management** - All settings in .env
8. ✅ **Backward compatibility** - Works alongside existing chunkers

### Quick Start

```bash
# 1. Install dependencies
pip install docling>=2.0.0 langchain-text-splitters>=0.3.0

# 2. Enable in .env
echo "USE_MARKDOWN_CHUNKING=true" >> .env

# 3. Restart server
python -m uvicorn src.api.routes:app --reload

# 4. Upload a document
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@test.pdf" \
  -F "user_id=test_user"
```

### Next Steps

1. Test with real banking documents
2. Compare retrieval quality vs token-based chunking
3. Adjust configuration based on results
4. Monitor chunk sizes and distribution
5. Plan migration strategy for existing documents

---

**For questions or issues**: Check logs in `DocumentUploader` and `ChunkerFactory` for detailed information about which parser and chunker are being used.
