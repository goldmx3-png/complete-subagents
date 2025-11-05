#!/usr/bin/env python3
"""
Verification script for Markdown Chunking implementation.

This script verifies that:
1. All dependencies are installed
2. Configuration is loaded correctly
3. ChunkerFactory selects the right chunker
4. Parsers and chunkers can be instantiated
5. The integration is working as expected

Run this before enabling markdown chunking in production.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """Test that all required packages can be imported"""
    print("=" * 60)
    print("STEP 1: Testing Imports")
    print("=" * 60)

    try:
        from docling.document_converter import DocumentConverter
        print("✓ docling imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import docling: {e}")
        print("  Fix: pip install docling>=2.0.0")
        return False

    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        print("✓ langchain_text_splitters imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import langchain_text_splitters: {e}")
        print("  Fix: pip install langchain-text-splitters>=0.3.0")
        return False

    try:
        import tiktoken
        print("✓ tiktoken imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import tiktoken: {e}")
        print("  Fix: pip install tiktoken>=0.5.2")
        return False

    try:
        from src.config import settings
        print("✓ Project config imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import project config: {e}")
        return False

    print("\n✓ All imports successful!\n")
    return True


def test_configuration():
    """Test that configuration is loaded correctly"""
    print("=" * 60)
    print("STEP 2: Testing Configuration")
    print("=" * 60)

    try:
        from src.config import settings

        print(f"USE_MARKDOWN_CHUNKING: {settings.use_markdown_chunking}")
        print(f"MARKDOWN_CHUNK_SIZE_TOKENS: {settings.markdown_chunk_size_tokens}")
        print(f"MARKDOWN_CHUNK_OVERLAP_PERCENTAGE: {settings.markdown_chunk_overlap_percentage}")
        print(f"MARKDOWN_TABLE_SIZE_THRESHOLD: {settings.markdown_table_size_threshold}")
        print(f"MARKDOWN_PRESERVE_HEADERS: {settings.markdown_preserve_headers}")
        print(f"DOCLING_EXTRACT_TABLES: {settings.docling_extract_tables}")
        print(f"DOCLING_EXTRACT_IMAGES: {settings.docling_extract_images}")

        print("\n✓ Configuration loaded successfully!\n")
        return True
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False


def test_chunker_factory():
    """Test ChunkerFactory selection logic"""
    print("=" * 60)
    print("STEP 3: Testing ChunkerFactory")
    print("=" * 60)

    try:
        from src.document_processing.chunker_factory import ChunkerFactory

        # Get active chunker info
        chunker_name = ChunkerFactory.get_active_chunker_name()
        chunker_info = ChunkerFactory.get_chunker_info()

        print(f"Active Chunker: {chunker_name}")
        print(f"Configuration: {chunker_info['configuration']}")

        # Try to create the chunker
        chunker = ChunkerFactory.create_chunker()
        print(f"✓ Created chunker instance: {type(chunker).__name__}")

        print("\n✓ ChunkerFactory working correctly!\n")
        return True, chunker_name
    except Exception as e:
        print(f"✗ ChunkerFactory failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_markdown_parser():
    """Test MarkdownDocumentParser instantiation"""
    print("=" * 60)
    print("STEP 4: Testing MarkdownDocumentParser")
    print("=" * 60)

    try:
        from src.document_processing.markdown_parser import MarkdownDocumentParser

        parser = MarkdownDocumentParser()
        print(f"✓ MarkdownDocumentParser instantiated")
        print(f"  - extract_tables: {parser.extract_tables}")
        print(f"  - extract_images: {parser.extract_images}")
        print(f"  - table_size_threshold: {parser.table_size_threshold}")
        print(f"  - converter: {type(parser.converter).__name__}")
        print(f"  - tokenizer: {type(parser.tokenizer).__name__}")

        print("\n✓ MarkdownDocumentParser working correctly!\n")
        return True
    except Exception as e:
        print(f"✗ MarkdownDocumentParser failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_markdown_chunker():
    """Test MarkdownChunker instantiation"""
    print("=" * 60)
    print("STEP 5: Testing MarkdownChunker")
    print("=" * 60)

    try:
        from src.document_processing.markdown_chunker import MarkdownChunker

        chunker = MarkdownChunker()
        print(f"✓ MarkdownChunker instantiated")
        print(f"  - chunk_size_tokens: {chunker.chunk_size_tokens}")
        print(f"  - chunk_overlap_percentage: {chunker.chunk_overlap_percentage}")
        print(f"  - chunk_overlap_tokens: {chunker.chunk_overlap_tokens}")
        print(f"  - table_size_threshold: {chunker.table_size_threshold}")
        print(f"  - preserve_headers: {chunker.preserve_headers}")

        print("\n✓ MarkdownChunker working correctly!\n")
        return True
    except Exception as e:
        print(f"✗ MarkdownChunker failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_uploader():
    """Test DocumentUploader with markdown chunking"""
    print("=" * 60)
    print("STEP 6: Testing DocumentUploader Integration")
    print("=" * 60)

    try:
        from src.document_processing.uploader import DocumentUploader

        uploader = DocumentUploader()
        print(f"✓ DocumentUploader instantiated")
        print(f"  - use_markdown: {uploader.use_markdown}")
        print(f"  - parser: {type(uploader.parser).__name__}")
        print(f"  - chunker: {type(uploader.chunker).__name__}")
        print(f"  - chunker_name: {uploader.chunker_name}")

        print("\n✓ DocumentUploader integration working correctly!\n")
        return True
    except Exception as e:
        print(f"✗ DocumentUploader failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_markdown_parsing():
    """Test markdown parsing with sample content"""
    print("=" * 60)
    print("STEP 7: Testing Markdown Parsing (Sample Content)")
    print("=" * 60)

    try:
        from src.document_processing.markdown_parser import MarkdownDocumentParser

        parser = MarkdownDocumentParser()

        # Test table analysis
        sample_markdown = """# Banking Services

## Account Types

Our bank offers various account types.

| Account Type | Min Balance | Fee |
|--------------|-------------|-----|
| Savings | $100 | $0 |
| Checking | $0 | $5 |

## Contact Us

For more information, call us.
"""

        table_elements, text_with_markers = parser._analyze_tables(sample_markdown)
        print(f"✓ Table analysis worked")
        print(f"  - Found {len(table_elements)} table(s)")
        if table_elements:
            print(f"  - Table 0: {table_elements[0]['num_rows']} rows, {table_elements[0]['num_cols']} cols, {table_elements[0]['token_count']} tokens")
            print(f"  - Classification: {table_elements[0]['chunk_type']}")

        # Test section extraction
        text_elements = parser._extract_text_sections(sample_markdown)
        print(f"✓ Section extraction worked")
        print(f"  - Found {len(text_elements)} section(s)")
        for i, elem in enumerate(text_elements[:3]):  # Show first 3
            print(f"  - Section {i}: H{elem['header_level']} - {elem['header_text']}")

        print("\n✓ Markdown parsing working correctly!\n")
        return True
    except Exception as e:
        print(f"✗ Markdown parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_markdown_chunking():
    """Test markdown chunking with sample content"""
    print("=" * 60)
    print("STEP 8: Testing Markdown Chunking (Sample Content)")
    print("=" * 60)

    try:
        from src.document_processing.markdown_chunker import MarkdownChunker

        chunker = MarkdownChunker()

        # Sample parsed document structure
        sample_doc_data = {
            "markdown_content": """# Banking Services

## Savings Account

A savings account helps you save money with competitive interest rates.

## Checking Account

A checking account for your daily transactions.
""",
            "table_elements": [],
            "text_elements": [],
            "metadata": {}
        }

        chunks = chunker.chunk_document(sample_doc_data, "test_doc", "test_user")

        print(f"✓ Chunking worked")
        print(f"  - Created {len(chunks)} chunk(s)")

        if chunks:
            print(f"\n  First chunk details:")
            print(f"    - chunk_id: {chunks[0]['chunk_id']}")
            print(f"    - chunk_type: {chunks[0]['chunk_type']}")
            print(f"    - text_preview: {chunks[0]['text'][:100]}...")
            print(f"    - metadata keys: {list(chunks[0]['metadata'].keys())}")
            if 'section_hierarchy' in chunks[0]['metadata']:
                print(f"    - section_hierarchy: {chunks[0]['metadata']['section_hierarchy']}")

        # Get stats
        stats = chunker.get_chunk_stats(chunks)
        print(f"\n  Chunk statistics:")
        for key, value in stats.items():
            print(f"    - {key}: {value}")

        print("\n✓ Markdown chunking working correctly!\n")
        return True
    except Exception as e:
        print(f"✗ Markdown chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "MARKDOWN CHUNKING VERIFICATION" + " " * 17 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_configuration()))

    factory_result, chunker_name = test_chunker_factory()
    results.append(("ChunkerFactory", factory_result))

    results.append(("MarkdownDocumentParser", test_markdown_parser()))
    results.append(("MarkdownChunker", test_markdown_chunker()))
    results.append(("DocumentUploader", test_document_uploader()))
    results.append(("Markdown Parsing", test_markdown_parsing()))
    results.append(("Markdown Chunking", test_markdown_chunking()))

    # Summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<40} {status}")

    all_passed = all(result for _, result in results)

    print()
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print()
        print("Your markdown chunking implementation is ready to use!")
        print()
        print("Next steps:")
        print("  1. Enable markdown chunking in .env:")
        print("     USE_MARKDOWN_CHUNKING=true")
        print()
        print("  2. Restart your application")
        print()
        print("  3. Upload a test document and verify chunks")
        print()
        print(f"Current active chunker: {chunker_name}")
        print()
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print()
        print("Please fix the failing tests before enabling markdown chunking.")
        print("See the error messages above for details.")
        print()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
