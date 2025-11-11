#!/usr/bin/env python3
"""
Database Initialization Script
Complete Subagents - Banking RAG System

This script:
1. Creates database tables using SQLAlchemy models
2. Verifies table creation
3. Optionally inserts sample data for testing
4. Checks database health

Usage:
    python scripts/init_db.py [--sample-data] [--force]
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError, ProgrammingError
from datetime import datetime
import uuid

# Import models
from src.memory.conversation_store import Base, Conversation, Message, ConversationStore
from src.config import settings


def check_database_connection(database_url: str) -> bool:
    """
    Check if database is accessible

    Args:
        database_url: PostgreSQL connection URL

    Returns:
        True if connection successful, False otherwise
    """
    print("🔍 Checking database connection...")
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        print("✅ Database connection successful")
        return True
    except OperationalError as e:
        print(f"❌ Database connection failed: {e}")
        print(f"\n💡 Make sure PostgreSQL is running:")
        print(f"   docker-compose up -d postgres")
        return False


def create_tables(database_url: str, force: bool = False) -> bool:
    """
    Create database tables

    Args:
        database_url: PostgreSQL connection URL
        force: If True, drop existing tables first

    Returns:
        True if successful, False otherwise
    """
    print("\n📊 Creating database tables...")

    try:
        engine = create_engine(
            database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow
        )

        # Drop tables if force flag is set
        if force:
            print("⚠️  Force flag set - dropping existing tables...")
            Base.metadata.drop_all(engine)
            print("✅ Existing tables dropped")

        # Create all tables
        Base.metadata.create_all(engine)
        print("✅ Tables created successfully")

        # Verify tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"\n📋 Created tables: {', '.join(tables)}")

        # Show table details
        for table_name in tables:
            columns = inspector.get_columns(table_name)
            print(f"\n   {table_name}:")
            for col in columns:
                print(f"      - {col['name']}: {col['type']}")

        return True

    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False


def insert_sample_data(database_url: str) -> bool:
    """
    Insert sample data for testing

    Args:
        database_url: PostgreSQL connection URL

    Returns:
        True if successful, False otherwise
    """
    print("\n🌱 Inserting sample data...")

    try:
        store = ConversationStore(database_url)

        # Sample conversations
        sample_conversations = [
            {
                "conversation_id": f"conv_sample_{uuid.uuid4().hex[:8]}",
                "user_id": "demo_user_001",
                "title": "Banking Account Features",
                "summary": "Discussion about savings account features and benefits",
                "messages": [
                    {"role": "user", "content": "What are the features of your savings account?", "tokens": 12},
                    {"role": "assistant", "content": "Our savings account offers competitive interest rates, no monthly fees, online banking, mobile app access, and ATM withdrawals at 50,000+ locations nationwide.", "tokens": 35},
                    {"role": "user", "content": "What's the minimum balance required?", "tokens": 8},
                    {"role": "assistant", "content": "There is no minimum balance requirement for our standard savings account.", "tokens": 18},
                ]
            },
            {
                "conversation_id": f"conv_sample_{uuid.uuid4().hex[:8]}",
                "user_id": "demo_user_001",
                "title": "Account Balance Inquiry",
                "summary": "API-based account balance check",
                "messages": [
                    {"role": "user", "content": "Check my account balance", "tokens": 6},
                    {"role": "assistant", "content": "Your current account balance is $5,432.50. Your last transaction was a deposit of $500.00 on 2025-11-10.", "tokens": 32},
                ]
            },
            {
                "conversation_id": f"conv_sample_{uuid.uuid4().hex[:8]}",
                "user_id": "demo_user_002",
                "title": "Loan Application Process",
                "summary": "Questions about home loan application",
                "messages": [
                    {"role": "user", "content": "How do I apply for a home loan?", "tokens": 10},
                    {"role": "assistant", "content": "To apply for a home loan, you'll need to: 1) Complete our online application form, 2) Submit required documents (income proof, ID, property details), 3) Wait for credit assessment, 4) Receive approval decision within 5-7 business days.", "tokens": 55},
                    {"role": "user", "content": "What documents are required?", "tokens": 7},
                    {"role": "assistant", "content": "Required documents include: Valid ID proof, last 6 months salary slips, bank statements, property documents, and proof of address.", "tokens": 28},
                ]
            }
        ]

        # Insert conversations and messages
        for conv_data in sample_conversations:
            # Create conversation
            conversation = Conversation(
                conversation_id=conv_data["conversation_id"],
                user_id=conv_data["user_id"],
                title=conv_data["title"],
                summary=conv_data["summary"],
                meta_data={"sample": True, "created_by": "init_script"}
            )
            store.session.add(conversation)

            # Create messages
            for msg_data in conv_data["messages"]:
                message = Message(
                    message_id=f"msg_{uuid.uuid4().hex[:12]}",
                    conversation_id=conv_data["conversation_id"],
                    role=msg_data["role"],
                    content=msg_data["content"],
                    tokens_used=msg_data["tokens"],
                    meta_data={"sample": True}
                )
                store.session.add(message)

            store.session.commit()
            print(f"   ✅ Created conversation: {conv_data['title']}")

        print(f"\n✅ Sample data inserted successfully")
        print(f"   - {len(sample_conversations)} conversations")
        print(f"   - {sum(len(c['messages']) for c in sample_conversations)} messages")

        store.close()
        return True

    except Exception as e:
        print(f"❌ Error inserting sample data: {e}")
        return False


def verify_database(database_url: str):
    """
    Verify database setup and display statistics

    Args:
        database_url: PostgreSQL connection URL
    """
    print("\n🔍 Verifying database setup...")

    try:
        engine = create_engine(database_url)

        with engine.connect() as conn:
            # Check conversations count
            result = conn.execute(text("SELECT COUNT(*) FROM conversations"))
            conv_count = result.fetchone()[0]

            # Check messages count
            result = conn.execute(text("SELECT COUNT(*) FROM messages"))
            msg_count = result.fetchone()[0]

            # Check api_registry count
            result = conn.execute(text("SELECT COUNT(*) FROM api_registry"))
            api_count = result.fetchone()[0]

            # Check database version
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]

            print(f"\n📊 Database Statistics:")
            print(f"   Database: {engine.url.database}")
            print(f"   Host: {engine.url.host}:{engine.url.port}")
            print(f"   Conversations: {conv_count}")
            print(f"   Messages: {msg_count}")
            print(f"   API Registry: {api_count}")
            print(f"   PostgreSQL: {version.split(',')[0]}")

            # Check indexes
            result = conn.execute(text("""
                SELECT tablename, indexname
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
            """))
            indexes = result.fetchall()

            print(f"\n📑 Indexes:")
            current_table = None
            for table, index in indexes:
                if table != current_table:
                    print(f"   {table}:")
                    current_table = table
                print(f"      - {index}")

        print("\n✅ Database verification complete")

    except Exception as e:
        print(f"❌ Error verifying database: {e}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Initialize database for Complete Subagents system"
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Insert sample data for testing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop existing tables before creating (DESTRUCTIVE!)"
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Override database URL from environment"
    )

    args = parser.parse_args()

    # Get database URL
    database_url = args.database_url or settings.database_url

    print("=" * 60)
    print("  Database Initialization - Complete Subagents")
    print("=" * 60)
    print(f"\n📌 Database URL: {database_url.split('@')[1] if '@' in database_url else database_url}")

    if args.force:
        print("\n⚠️  WARNING: Force flag is set. All existing data will be deleted!")
        response = input("   Continue? (yes/no): ")
        if response.lower() != "yes":
            print("❌ Aborted by user")
            return 1

    # Step 1: Check connection
    if not check_database_connection(database_url):
        return 1

    # Step 2: Create tables
    if not create_tables(database_url, force=args.force):
        return 1

    # Step 3: Insert sample data if requested
    if args.sample_data:
        if not insert_sample_data(database_url):
            return 1

    # Step 4: Verify database
    verify_database(database_url)

    print("\n" + "=" * 60)
    print("✅ Database initialization complete!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. Start the API server: python -m uvicorn src.api.routes:app --reload")
    print("   2. Test the API: curl http://localhost:8000/health")
    print("   3. View API docs: http://localhost:8000/docs")

    if args.sample_data:
        print("\n📊 Sample data available for testing:")
        print("   - User: demo_user_001")
        print("   - User: demo_user_002")

    return 0


if __name__ == "__main__":
    sys.exit(main())
