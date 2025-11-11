#!/bin/bash
# ============================================
# Database Restore Script
# Complete Subagents - Banking RAG System
# ============================================
#
# This script restores a PostgreSQL database from a backup file
#
# Usage:
#   ./scripts/restore_db.sh <backup_file> [--no-confirm]
#
# Options:
#   --no-confirm    Skip confirmation prompt (use with caution!)
#
# Example:
#   ./scripts/restore_db.sh backups/chatbot_20251111_120000.sql
#
# ============================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}❌ Error: Backup file not specified${NC}"
    echo ""
    echo "Usage: $0 <backup_file> [--no-confirm]"
    echo ""
    echo "Example:"
    echo "   $0 backups/chatbot_20251111_120000.sql"
    exit 1
fi

BACKUP_FILE="$1"
NO_CONFIRM=false

# Check for --no-confirm flag
if [ "$2" = "--no-confirm" ]; then
    NO_CONFIRM=true
fi

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${RED}❌ Backup file not found: $BACKUP_FILE${NC}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
else
    echo -e "${RED}❌ .env file not found!${NC}"
    exit 1
fi

# Parse database URL
if [ -z "$DATABASE_URL" ]; then
    echo -e "${RED}❌ DATABASE_URL not set in .env${NC}"
    exit 1
fi

# Extract database connection details
DB_USER=$(echo $DATABASE_URL | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
DB_PASSWORD=$(echo $DATABASE_URL | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')
DB_HOST=$(echo $DATABASE_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
DB_PORT=$(echo $DATABASE_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
DB_NAME=$(echo $DATABASE_URL | sed -n 's/.*\/\([^?]*\).*/\1/p')

# Check if file is compressed
IS_COMPRESSED=false
if [[ "$BACKUP_FILE" == *.gz ]]; then
    IS_COMPRESSED=true
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   Database Restore - Complete Subagents${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${YELLOW}⚠️  WARNING: This will REPLACE all existing data!${NC}"
echo ""
echo "Database: $DB_NAME"
echo "Host: $DB_HOST:$DB_PORT"
echo "Backup file: $(basename $BACKUP_FILE)"
echo "File size: $(du -h "$BACKUP_FILE" | cut -f1)"
echo "Compressed: $([ "$IS_COMPRESSED" = true ] && echo "Yes" || echo "No")"
echo ""

# Get current database statistics
echo -e "${BLUE}📊 Current database statistics:${NC}"
CURRENT_CONV_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM conversations;" 2>/dev/null || echo "0")
CURRENT_MSG_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM messages;" 2>/dev/null || echo "0")

echo "   Conversations: $(echo $CURRENT_CONV_COUNT | xargs)"
echo "   Messages: $(echo $CURRENT_MSG_COUNT | xargs)"
echo ""

# Confirmation prompt
if [ "$NO_CONFIRM" = false ]; then
    read -p "Are you sure you want to restore from this backup? (type 'yes' to confirm): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo -e "${YELLOW}❌ Restore cancelled${NC}"
        exit 0
    fi
fi

echo ""
echo -e "${BLUE}🔄 Starting database restore...${NC}"
echo ""

# Step 1: Check database connection
echo -e "${BLUE}1/4 Checking database connection...${NC}"
if ! PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "SELECT 1" > /dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to PostgreSQL${NC}"
    echo ""
    echo "💡 Make sure PostgreSQL is running:"
    echo "   docker-compose up -d postgres"
    exit 1
fi
echo -e "${GREEN}✅ Database connection successful${NC}"
echo ""

# Step 2: Drop existing tables
echo -e "${BLUE}2/4 Dropping existing tables...${NC}"
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
-- Drop views
DROP VIEW IF EXISTS v_recent_activity CASCADE;
DROP VIEW IF EXISTS v_conversation_message_stats CASCADE;
DROP VIEW IF EXISTS v_user_conversation_stats CASCADE;

-- Drop tables
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS conversations CASCADE;

-- Drop functions
DROP FUNCTION IF EXISTS update_updated_at_column CASCADE;
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Existing tables dropped${NC}"
else
    echo -e "${RED}❌ Error dropping tables${NC}"
    exit 1
fi
echo ""

# Step 3: Restore from backup
echo -e "${BLUE}3/4 Restoring database...${NC}"

if [ "$IS_COMPRESSED" = true ]; then
    gunzip -c "$BACKUP_FILE" | PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME > /dev/null 2>&1
else
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME < "$BACKUP_FILE" > /dev/null 2>&1
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Database restored successfully${NC}"
else
    echo -e "${RED}❌ Error restoring database${NC}"
    exit 1
fi
echo ""

# Step 4: Verify restoration
echo -e "${BLUE}4/4 Verifying restoration...${NC}"

RESTORED_CONV_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM conversations;" 2>/dev/null || echo "0")
RESTORED_MSG_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM messages;" 2>/dev/null || echo "0")
TABLE_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';")

echo "   Tables: $(echo $TABLE_COUNT | xargs)"
echo "   Conversations: $(echo $RESTORED_CONV_COUNT | xargs)"
echo "   Messages: $(echo $RESTORED_MSG_COUNT | xargs)"
echo ""

# Final summary
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}✅ Database restore complete!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "📊 Restoration Summary:"
echo "   Before: $CURRENT_CONV_COUNT conversations, $CURRENT_MSG_COUNT messages"
echo "   After:  $RESTORED_CONV_COUNT conversations, $RESTORED_MSG_COUNT messages"
echo ""
echo "💡 Next steps:"
echo "   1. Verify data: python scripts/check_db_health.py"
echo "   2. Start API: python -m uvicorn src.api.routes:app --reload"
echo "   3. Test endpoints: http://localhost:8000/docs"
