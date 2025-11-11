#!/usr/bin/env python3
"""
Database Health Check Script
Complete Subagents - Banking RAG System

This script performs comprehensive health checks on the database:
1. Connection test
2. Table existence validation
3. Index verification
4. Data integrity checks
5. Performance metrics
6. Recommendations

Usage:
    python scripts/check_db_health.py [--verbose] [--fix]
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError
from src.config import settings

# ANSI color codes
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color


class DatabaseHealthChecker:
    """Database health checker"""

    def __init__(self, database_url: str, verbose: bool = False):
        self.database_url = database_url
        self.verbose = verbose
        self.issues = []
        self.warnings = []
        self.info = []
        self.engine = None

    def log(self, message: str, level: str = "info"):
        """Log message with color coding"""
        if level == "error":
            print(f"{RED}❌ {message}{NC}")
            self.issues.append(message)
        elif level == "warning":
            print(f"{YELLOW}⚠️  {message}{NC}")
            self.warnings.append(message)
        elif level == "success":
            print(f"{GREEN}✅ {message}{NC}")
        elif level == "info":
            print(f"{BLUE}ℹ️  {message}{NC}")
            self.info.append(message)
        else:
            print(message)

    def check_connection(self) -> bool:
        """Test database connection"""
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}1. Database Connection Test{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")

        try:
            self.engine = create_engine(self.database_url)
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]

            self.log(f"Connection successful", "success")
            self.log(f"PostgreSQL version: {version.split(',')[0]}", "info")
            return True

        except OperationalError as e:
            self.log(f"Connection failed: {e}", "error")
            self.log("Make sure PostgreSQL is running: docker-compose up -d postgres", "info")
            return False

    def check_tables(self) -> bool:
        """Verify table existence and structure"""
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}2. Table Structure Verification{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")

        expected_tables = ['conversations', 'messages', 'api_registry']

        try:
            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()

            for table in expected_tables:
                if table in existing_tables:
                    self.log(f"Table '{table}' exists", "success")

                    # Check columns
                    columns = inspector.get_columns(table)
                    if self.verbose:
                        print(f"   Columns:")
                        for col in columns:
                            print(f"      - {col['name']}: {col['type']}")
                else:
                    self.log(f"Table '{table}' is missing!", "error")

            # Check for unexpected tables
            extra_tables = set(existing_tables) - set(expected_tables)
            if extra_tables:
                self.log(f"Unexpected tables found: {', '.join(extra_tables)}", "warning")

            return len(self.issues) == 0

        except Exception as e:
            self.log(f"Error checking tables: {e}", "error")
            return False

    def check_indexes(self) -> bool:
        """Verify indexes for performance"""
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}3. Index Verification{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")

        expected_indexes = {
            'conversations': [
                'idx_conversations_user_id',
                'idx_conversations_created_at',
                'idx_conversations_updated_at',
            ],
            'messages': [
                'idx_messages_conversation_id',
                'idx_messages_created_at',
                'idx_messages_role',
            ]
        }

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT tablename, indexname, indexdef
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                    ORDER BY tablename, indexname
                """))

                indexes = result.fetchall()
                index_dict = {}

                for table, index, definition in indexes:
                    if table not in index_dict:
                        index_dict[table] = []
                    index_dict[table].append(index)

                    if self.verbose:
                        print(f"   {table}.{index}")

                # Check for missing indexes
                for table, expected in expected_indexes.items():
                    existing = index_dict.get(table, [])
                    for expected_idx in expected:
                        if expected_idx in existing:
                            self.log(f"Index '{expected_idx}' exists", "success")
                        else:
                            self.log(f"Index '{expected_idx}' is missing (may affect performance)", "warning")

            return True

        except Exception as e:
            self.log(f"Error checking indexes: {e}", "error")
            return False

    def check_data_integrity(self) -> bool:
        """Check data integrity and foreign key constraints"""
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}4. Data Integrity Checks{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")

        try:
            with self.engine.connect() as conn:
                # Check for orphaned messages
                result = conn.execute(text("""
                    SELECT COUNT(*)
                    FROM messages m
                    LEFT JOIN conversations c ON m.conversation_id = c.conversation_id
                    WHERE c.conversation_id IS NULL
                """))
                orphaned_messages = result.fetchone()[0]

                if orphaned_messages == 0:
                    self.log("No orphaned messages found", "success")
                else:
                    self.log(f"Found {orphaned_messages} orphaned messages (references non-existent conversations)", "error")

                # Check for empty conversations
                result = conn.execute(text("""
                    SELECT COUNT(*)
                    FROM conversations c
                    LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                    WHERE m.message_id IS NULL
                """))
                empty_conversations = result.fetchone()[0]

                if empty_conversations == 0:
                    self.log("No empty conversations found", "success")
                else:
                    self.log(f"Found {empty_conversations} empty conversations", "warning")

                # Check for invalid timestamps
                result = conn.execute(text("""
                    SELECT COUNT(*)
                    FROM messages
                    WHERE created_at > NOW() OR created_at < '2020-01-01'
                """))
                invalid_timestamps = result.fetchone()[0]

                if invalid_timestamps == 0:
                    self.log("All timestamps are valid", "success")
                else:
                    self.log(f"Found {invalid_timestamps} messages with invalid timestamps", "warning")

            return len(self.issues) == 0

        except Exception as e:
            self.log(f"Error checking data integrity: {e}", "error")
            return False

    def check_performance_metrics(self) -> bool:
        """Check performance metrics and statistics"""
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}5. Performance Metrics{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")

        try:
            with self.engine.connect() as conn:
                # Get row counts
                result = conn.execute(text("SELECT COUNT(*) FROM conversations"))
                conv_count = result.fetchone()[0]

                result = conn.execute(text("SELECT COUNT(*) FROM messages"))
                msg_count = result.fetchone()[0]

                result = conn.execute(text("SELECT COUNT(*) FROM api_registry"))
                api_count = result.fetchone()[0]

                self.log(f"Conversations: {conv_count:,}", "info")
                self.log(f"Messages: {msg_count:,}", "info")
                self.log(f"API Registry: {api_count:,}", "info")

                # Get database size
                result = conn.execute(text("SELECT pg_size_pretty(pg_database_size(current_database()))"))
                db_size = result.fetchone()[0]
                self.log(f"Database size: {db_size}", "info")

                # Get table sizes
                result = conn.execute(text("""
                    SELECT
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                """))

                if self.verbose:
                    print("\n   Table sizes:")
                    for table, size in result:
                        print(f"      {table}: {size}")

                # Average messages per conversation
                if conv_count > 0:
                    avg_messages = msg_count / conv_count
                    self.log(f"Average messages per conversation: {avg_messages:.2f}", "info")

                # Check for recent activity
                result = conn.execute(text("""
                    SELECT MAX(created_at) FROM messages
                """))
                last_message = result.fetchone()[0]

                if last_message:
                    time_diff = datetime.now() - last_message
                    self.log(f"Last message: {last_message.strftime('%Y-%m-%d %H:%M:%S')} ({time_diff.days} days ago)", "info")
                else:
                    self.log("No messages in database", "warning")

            return True

        except Exception as e:
            self.log(f"Error checking performance metrics: {e}", "error")
            return False

    def generate_recommendations(self):
        """Generate recommendations based on findings"""
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}6. Recommendations{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")

        if not self.issues and not self.warnings:
            print(f"{GREEN}🎉 Database health is excellent! No issues found.{NC}\n")
            return

        if self.issues:
            print(f"{RED}Critical Issues:{NC}")
            for issue in self.issues:
                print(f"   • {issue}")
            print()

        if self.warnings:
            print(f"{YELLOW}Warnings:{NC}")
            for warning in self.warnings:
                print(f"   • {warning}")
            print()

        print(f"{BLUE}Recommended Actions:{NC}")

        if any("missing" in str(issue).lower() for issue in self.issues):
            print("   • Run: python scripts/init_db.py --force")

        if any("orphaned" in str(issue).lower() for issue in self.issues):
            print("   • Clean orphaned messages: DELETE FROM messages WHERE conversation_id NOT IN (SELECT conversation_id FROM conversations)")

        if any("index" in str(warning).lower() for warning in self.warnings):
            print("   • Recreate missing indexes using: scripts/init_db.sql")

        if any("empty" in str(warning).lower() for warning in self.warnings):
            print("   • Consider cleaning empty conversations: DELETE FROM conversations WHERE conversation_id NOT IN (SELECT conversation_id FROM messages)")

        print()

    def run_health_check(self) -> bool:
        """Run complete health check"""
        print(f"\n{GREEN}{'='*60}{NC}")
        print(f"{GREEN}   Database Health Check - Complete Subagents{NC}")
        print(f"{GREEN}{'='*60}{NC}")

        checks = [
            ("Connection", self.check_connection),
            ("Tables", self.check_tables),
            ("Indexes", self.check_indexes),
            ("Data Integrity", self.check_data_integrity),
            ("Performance", self.check_performance_metrics),
        ]

        all_passed = True
        for check_name, check_func in checks:
            try:
                if not check_func():
                    all_passed = False
            except Exception as e:
                self.log(f"{check_name} check failed: {e}", "error")
                all_passed = False

        self.generate_recommendations()

        return all_passed


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Check database health for Complete Subagents system"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
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

    # Run health check
    checker = DatabaseHealthChecker(database_url, verbose=args.verbose)
    success = checker.run_health_check()

    # Print summary
    print(f"\n{BLUE}{'='*60}{NC}")
    if success:
        print(f"{GREEN}✅ All health checks passed!{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")
        return 0
    else:
        print(f"{RED}❌ Some health checks failed{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
