#!/bin/bash
# ============================================
# Database Reset Script
# Complete Subagents - Banking RAG System
# ============================================
#
# This script resets the database by:
# 1. Dropping all tables
# 2. Recreating tables from models
# 3. Optionally inserting sample data
#
# Usage:
#   ./scripts/reset_db.sh [--sample-data] [--no-confirm]
#
# Options:
#   --sample-data    Insert sample data after reset
#   --no-confirm     Skip confirmation prompt (use with caution!)
#
# ============================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
SAMPLE_DATA=false
NO_CONFIRM=false

for arg in "$@"; do
    case $arg in
        --sample-data)
            SAMPLE_DATA=true
            shift
            ;;
        --no-confirm)
            NO_CONFIRM=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            echo "Usage: $0 [--sample-data] [--no-confirm]"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
else
    echo -e "${RED}❌ .env file not found!${NC}"
    echo "Please create .env from .env.example"
    exit 1
fi

# Parse database URL
if [ -z "$DATABASE_URL" ]; then
    echo -e "${RED}❌ DATABASE_URL not set in .env${NC}"
    exit 1
fi

# Extract database connection details
# Format: postgresql://user:password@host:port/database
DB_USER=$(echo $DATABASE_URL | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
DB_PASSWORD=$(echo $DATABASE_URL | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')
DB_HOST=$(echo $DATABASE_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
DB_PORT=$(echo $DATABASE_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
DB_NAME=$(echo $DATABASE_URL | sed -n 's/.*\/\([^?]*\).*/\1/p')

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   Database Reset - Complete Subagents${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${YELLOW}⚠️  WARNING: This will DELETE all data!${NC}"
echo ""
echo "Database: $DB_NAME"
echo "Host: $DB_HOST:$DB_PORT"
echo "User: $DB_USER"
echo ""

# Confirmation prompt
if [ "$NO_CONFIRM" = false ]; then
    read -p "Are you sure you want to reset the database? (type 'yes' to confirm): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo -e "${YELLOW}❌ Reset cancelled${NC}"
        exit 0
    fi
fi

echo ""
echo -e "${BLUE}🔄 Starting database reset...${NC}"
echo ""

# Step 1: Check if PostgreSQL is running
echo -e "${BLUE}1/4 Checking PostgreSQL connection...${NC}"
if ! PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "SELECT 1" > /dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to PostgreSQL${NC}"
    echo ""
    echo "💡 Make sure PostgreSQL is running:"
    echo "   docker-compose up -d postgres"
    exit 1
fi
echo -e "${GREEN}✅ PostgreSQL is running${NC}"
echo ""

# Step 2: Drop all tables
echo -e "${BLUE}2/4 Dropping all tables...${NC}"
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
-- Drop views first
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
    echo -e "${GREEN}✅ Tables dropped successfully${NC}"
else
    echo -e "${RED}❌ Error dropping tables${NC}"
    exit 1
fi
echo ""

# Step 3: Recreate tables using Python script
echo -e "${BLUE}3/4 Recreating tables...${NC}"
cd "$PROJECT_ROOT"

if [ "$SAMPLE_DATA" = true ]; then
    python3 scripts/init_db.py --force --sample-data
else
    python3 scripts/init_db.py --force
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Tables recreated successfully${NC}"
else
    echo -e "${RED}❌ Error recreating tables${NC}"
    exit 1
fi
echo ""

# Step 4: Verify reset
echo -e "${BLUE}4/4 Verifying database...${NC}"
TABLE_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';")
VIEW_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM information_schema.views WHERE table_schema='public';")

echo "   Tables: $TABLE_COUNT"
echo "   Views: $VIEW_COUNT"
echo ""

# Final summary
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}✅ Database reset complete!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "📊 Database Status:"

if [ "$SAMPLE_DATA" = true ]; then
    CONV_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM conversations;")
    MSG_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM messages;")
    echo "   Conversations: $CONV_COUNT"
    echo "   Messages: $MSG_COUNT"
    echo ""
    echo "💡 Sample data inserted for testing"
else
    echo "   Conversations: 0"
    echo "   Messages: 0"
    echo ""
    echo "💡 Database is empty and ready for use"
fi

echo ""
echo "Next steps:"
echo "   1. Start the API: python -m uvicorn src.api.routes:app --reload"
echo "   2. Test endpoints: http://localhost:8000/docs"
