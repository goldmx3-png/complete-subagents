"""
Test script for hierarchical metadata extraction.

This script tests the new hierarchical metadata functionality by:
1. Creating a sample PDF-like markdown document
2. Parsing it with MarkdownDocumentParser
3. Chunking it with MarkdownChunker
4. Displaying the hierarchical metadata
"""

import json
from pathlib import Path
from src.document_processing.markdown_parser import MarkdownDocumentParser
from src.document_processing.markdown_chunker import MarkdownChunker


def create_sample_markdown():
    """Create a sample markdown document with multi-level headings."""
    return """# Corporate Banking Guide

This is the introduction to corporate banking services.

## 1. Account Management

Overview of account management features.

### 1.1 Account Opening

Steps to open a new corporate account.

#### 1.1.1 Required Documents

- Business registration certificate
- Tax identification number
- Board resolution

#### 1.1.2 Verification Process

The verification process takes 3-5 business days.

### 1.2 Account Closure

Process for closing an account.

#### 1.2.1 Closure Requirements

All outstanding balances must be settled.

## 2. Transaction Services

Overview of transaction capabilities.

### 2.1 Domestic Transfers

Information about domestic wire transfers.

#### 2.1.1 Transfer Limits

Daily transfer limits vary by account type.

### 2.2 International Transfers

Guidelines for international payments.

#### 2.2.1 SWIFT Requirements

SWIFT code and beneficiary bank details required.

## 3. Reporting and Analytics

Access to financial reports and analytics tools.

### 3.1 Monthly Reports

Automated monthly statement generation.

### 3.2 Custom Reports

Create custom financial reports on demand.
"""


def test_hierarchical_metadata():
    """Test the hierarchical metadata extraction."""

    print("="*70)
    print("HIERARCHICAL METADATA EXTRACTION TEST")
    print("="*70)

    # Create sample markdown
    sample_markdown = create_sample_markdown()

    # Create a temporary file
    temp_file = Path("/tmp/test_banking_doc.md")
    temp_file.write_text(sample_markdown)

    print("\n1. Testing MarkdownDocumentParser...")
    print("-" * 70)

    # Initialize parser
    parser = MarkdownDocumentParser()

    # Extract text sections
    text_elements = parser._extract_text_sections(sample_markdown)
    print(f"✓ Extracted {len(text_elements)} text sections")

    # Build hierarchical structure
    hierarchy_structure = parser._build_hierarchical_structure(text_elements)
    print(f"✓ Built hierarchical structure with {len(hierarchy_structure)} nodes")

    # Display hierarchy
    print("\n2. Hierarchical Structure:")
    print("-" * 70)
    for idx, hierarchy in hierarchy_structure.items():
        print(f"\nSection {idx}:")
        print(f"  Full Path: {hierarchy['full_path']}")
        print(f"  Depth: {hierarchy['depth']}")
        print(f"  Level: h{hierarchy['level']}")
        print(f"  Parent: {hierarchy.get('parent_section', 'None')}")
        print(f"  Root: {hierarchy.get('root_section', 'None')}")
        print(f"  Position: {hierarchy['position_in_doc']}")
        if hierarchy.get('sibling_indices'):
            print(f"  Siblings: {len(hierarchy['sibling_indices'])} section(s)")
        if hierarchy.get('children_indices'):
            print(f"  Children: {len(hierarchy['children_indices'])} section(s)")

    print("\n3. Testing MarkdownChunker...")
    print("-" * 70)

    # Initialize chunker
    chunker = MarkdownChunker()

    # Create doc_data structure
    doc_data = {
        "markdown_content": sample_markdown,
        "text_elements": text_elements,
        "hierarchy_structure": hierarchy_structure,
        "table_elements": [],
        "metadata": {
            "file_name": "test_banking_doc.md",
            "num_pages": 1,
        }
    }

    # Chunk the document
    chunks = chunker.chunk_document(doc_data, "test_doc_001", "test_user")
    print(f"✓ Created {len(chunks)} chunks")

    # Display chunk metadata
    print("\n4. Sample Chunk Metadata:")
    print("-" * 70)

    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        print(f"\n--- Chunk {i+1} ---")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Type: {chunk['chunk_type']}")

        metadata = chunk.get('metadata', {})

        # Show section hierarchy
        section_hierarchy = metadata.get('section_hierarchy', {})
        if section_hierarchy:
            print(f"Section Hierarchy: {section_hierarchy}")

        # Show header context
        header_context = metadata.get('header_context', '')
        if header_context:
            print(f"Header Context: {header_context}")

        # Show enhanced hierarchy (NEW)
        hierarchy = metadata.get('hierarchy', {})
        if hierarchy:
            print("\nEnhanced Hierarchy Metadata:")
            print(f"  Full Path: {hierarchy.get('full_path', 'N/A')}")
            print(f"  Breadcrumbs: {' > '.join(hierarchy.get('breadcrumbs', []))}")
            print(f"  Depth: {hierarchy.get('depth', 'N/A')}")
            print(f"  Root Section: {hierarchy.get('root_section', 'N/A')}")
            print(f"  Parent Section: {hierarchy.get('parent_section', 'N/A')}")
            print(f"  Has Children: {hierarchy.get('has_children', False)}")
            print(f"  Has Siblings: {hierarchy.get('has_siblings', False)}")
            print(f"  Position in Doc: {hierarchy.get('position_in_doc', 'N/A')}")

            if 'previous_section' in hierarchy:
                print(f"  Previous Section: {hierarchy['previous_section']}")
            if 'next_section' in hierarchy:
                print(f"  Next Section: {hierarchy['next_section']}")

        # Show snippet of text
        text = chunk.get('text', '')
        print(f"\nText Preview: {text[:100]}...")

    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*70)

    # Cleanup
    temp_file.unlink(missing_ok=True)

    return chunks


if __name__ == "__main__":
    try:
        chunks = test_hierarchical_metadata()
        print(f"\n✓ Successfully processed document with hierarchical metadata")
        print(f"✓ Total chunks created: {len(chunks)}")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
