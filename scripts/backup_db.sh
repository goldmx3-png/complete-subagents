#!/bin/bash
# ============================================
# Database Backup Script
# Complete Subagents - Banking RAG System
# ============================================
#
# This script creates a backup of the PostgreSQL database
# Backups are stored in: ./backups/
#
# Usage:
#   ./scripts/backup_db.sh [--output-dir DIR] [--compress]
#
# Options:
#   --output-dir DIR    Specify custom backup directory (default: ./backups)
#   --compress          Compress backup with gzip
#
# ============================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
COMPRESS=false
BACKUP_DIR=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --compress)
            COMPRESS=true
            shift
            ;;
        --output-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        *)
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set default backup directory if not specified
if [ -z "$BACKUP_DIR" ]; then
    BACKUP_DIR="$PROJECT_ROOT/backups"
fi

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

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

# Generate backup filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/${DB_NAME}_${TIMESTAMP}.sql"

if [ "$COMPRESS" = true ]; then
    BACKUP_FILE="${BACKUP_FILE}.gz"
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   Database Backup - Complete Subagents${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Database: $DB_NAME"
echo "Host: $DB_HOST:$DB_PORT"
echo "Backup file: $(basename $BACKUP_FILE)"
echo "Backup directory: $BACKUP_DIR"
echo ""

# Check if PostgreSQL is accessible
echo -e "${BLUE}🔍 Checking database connection...${NC}"
if ! PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1" > /dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to database${NC}"
    echo ""
    echo "💡 Make sure PostgreSQL is running:"
    echo "   docker-compose up -d postgres"
    exit 1
fi
echo -e "${GREEN}✅ Database connection successful${NC}"
echo ""

# Get database statistics before backup
echo -e "${BLUE}📊 Database statistics:${NC}"
CONV_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM conversations;" 2>/dev/null || echo "0")
MSG_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM messages;" 2>/dev/null || echo "0")
DB_SIZE=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT pg_size_pretty(pg_database_size('$DB_NAME'));" 2>/dev/null || echo "Unknown")

echo "   Conversations: $(echo $CONV_COUNT | xargs)"
echo "   Messages: $(echo $MSG_COUNT | xargs)"
echo "   Database size: $(echo $DB_SIZE | xargs)"
echo ""

# Perform backup
echo -e "${BLUE}💾 Creating backup...${NC}"

if [ "$COMPRESS" = true ]; then
    PGPASSWORD=$DB_PASSWORD pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME \
        --format=plain \
        --no-owner \
        --no-acl \
        --verbose \
        2>&1 | gzip > "$BACKUP_FILE"
else
    PGPASSWORD=$DB_PASSWORD pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME \
        --format=plain \
        --no-owner \
        --no-acl \
        --verbose \
        > "$BACKUP_FILE" 2>&1
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Backup created successfully${NC}"
else
    echo -e "${RED}❌ Backup failed${NC}"
    exit 1
fi

# Get backup file size
BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}✅ Backup complete!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "📁 Backup details:"
echo "   File: $(basename $BACKUP_FILE)"
echo "   Size: $BACKUP_SIZE"
echo "   Location: $BACKUP_FILE"
echo ""

# List recent backups
echo "📋 Recent backups in $BACKUP_DIR:"
ls -lht "$BACKUP_DIR" | head -6

echo ""
echo "💡 To restore this backup:"
if [ "$COMPRESS" = true ]; then
    echo "   gunzip -c $BACKUP_FILE | PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"
else
    echo "   ./scripts/restore_db.sh $BACKUP_FILE"
fi

# Cleanup old backups (keep last 10)
echo ""
echo -e "${BLUE}🧹 Cleaning up old backups...${NC}"
BACKUP_COUNT=$(ls -1 "$BACKUP_DIR"/${DB_NAME}_*.sql* 2>/dev/null | wc -l)

if [ $BACKUP_COUNT -gt 10 ]; then
    OLD_BACKUPS=$(ls -1t "$BACKUP_DIR"/${DB_NAME}_*.sql* | tail -n +11)
    echo "   Found $BACKUP_COUNT backups, removing $(echo "$OLD_BACKUPS" | wc -l) old backups..."
    echo "$OLD_BACKUPS" | xargs rm -f
    echo -e "${GREEN}✅ Old backups cleaned up${NC}"
else
    echo "   Total backups: $BACKUP_COUNT (no cleanup needed)"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
