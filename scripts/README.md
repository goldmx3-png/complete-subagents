# Database Scripts

This directory contains comprehensive database management scripts for the Complete Subagents system.

## 📋 Available Scripts

### 1. Database Initialization

#### `init_db.sql`
Raw SQL script for database and user setup. Run this as PostgreSQL superuser.

```bash
# Run directly with psql
psql -U postgres -f scripts/init_db.sql

# Or via Docker
docker exec -i subagents-postgres psql -U postgres < scripts/init_db.sql
```

**Features:**
- Creates database and user
- Creates tables with proper indexes
- Sets up views for analytics
- Creates triggers for automatic timestamp updates
- Optional sample data insertion

#### `init_db.py`
Python script for table creation using SQLAlchemy models.

```bash
# Basic initialization
python scripts/init_db.py

# With sample data
python scripts/init_db.py --sample-data

# Force recreate (drops existing tables)
python scripts/init_db.py --force

# Combine flags
python scripts/init_db.py --force --sample-data
```

**Features:**
- Creates tables from SQLAlchemy models
- Verifies table structure
- Inserts realistic sample data
- Shows database statistics
- Works with existing `.env` configuration

---

### 2. Database Reset

#### `reset_db.sh`
Complete database reset with data deletion.

```bash
# Interactive reset (with confirmation)
./scripts/reset_db.sh

# Reset with sample data
./scripts/reset_db.sh --sample-data

# Skip confirmation (use with caution!)
./scripts/reset_db.sh --no-confirm --sample-data
```

**Use Cases:**
- Development environment cleanup
- Testing fresh database state
- Removing corrupted data
- Starting over with clean slate

**⚠️ WARNING:** This is destructive! All data will be permanently deleted.

---

### 3. Database Backup

#### `backup_db.sh`
Creates timestamped backups of the database.

```bash
# Standard backup
./scripts/backup_db.sh

# Compressed backup (smaller file size)
./scripts/backup_db.sh --compress

# Custom backup directory
./scripts/backup_db.sh --output-dir /path/to/backups
```

**Features:**
- Timestamped backup files
- Optional gzip compression
- Database statistics in output
- Automatic cleanup (keeps last 10 backups)
- Backup verification

**Backup Location:** `./backups/` (default)

**Backup Naming:** `chatbot_YYYYMMDD_HHMMSS.sql[.gz]`

---

### 4. Database Restore

#### `restore_db.sh`
Restores database from a backup file.

```bash
# Interactive restore (with confirmation)
./scripts/restore_db.sh backups/chatbot_20251111_120000.sql

# Skip confirmation
./scripts/restore_db.sh backups/chatbot_20251111_120000.sql --no-confirm

# Restore from compressed backup
./scripts/restore_db.sh backups/chatbot_20251111_120000.sql.gz
```

**Features:**
- Shows before/after statistics
- Handles compressed backups automatically
- Drops existing tables before restore
- Verification after restoration

**⚠️ WARNING:** This replaces all existing data!

---

### 5. Database Health Check

#### `check_db_health.py`
Comprehensive database health diagnostics.

```bash
# Basic health check
python scripts/check_db_health.py

# Verbose output (shows detailed information)
python scripts/check_db_health.py --verbose

# Custom database URL
python scripts/check_db_health.py --database-url postgresql://user:pass@host:5432/db
```

**Health Checks:**
1. **Connection Test** - Verifies database accessibility
2. **Table Structure** - Validates all required tables exist
3. **Index Verification** - Checks performance-critical indexes
4. **Data Integrity** - Finds orphaned records and inconsistencies
5. **Performance Metrics** - Database size, row counts, activity
6. **Recommendations** - Actionable suggestions for issues found

**Exit Codes:**
- `0` - All checks passed
- `1` - Some checks failed

---

## 🔄 Common Workflows

### Initial Setup
```bash
# 1. Start PostgreSQL
docker-compose up -d postgres

# 2. Initialize database
python scripts/init_db.py --sample-data

# 3. Verify setup
python scripts/check_db_health.py
```

### Daily Development
```bash
# Reset with fresh sample data
./scripts/reset_db.sh --sample-data
```

### Before Making Changes
```bash
# Create backup
./scripts/backup_db.sh --compress
```

### After Deployment
```bash
# Health check
python scripts/check_db_health.py --verbose

# Backup if healthy
./scripts/backup_db.sh --compress
```

### Disaster Recovery
```bash
# 1. Find latest backup
ls -lh backups/

# 2. Restore from backup
./scripts/restore_db.sh backups/chatbot_20251111_120000.sql.gz

# 3. Verify restoration
python scripts/check_db_health.py
```

---

## 📊 Database Schema

### Tables

#### `conversations`
Stores conversation metadata.

| Column | Type | Description |
|--------|------|-------------|
| conversation_id | VARCHAR(255) | Primary key |
| user_id | VARCHAR(255) | User identifier |
| title | VARCHAR(500) | Conversation title |
| created_at | TIMESTAMP | Creation time |
| updated_at | TIMESTAMP | Last update time |
| summary | TEXT | Conversation summary |
| metadata | JSONB | Additional metadata |

**Indexes:**
- `idx_conversations_user_id` - Fast user lookup
- `idx_conversations_created_at` - Chronological ordering
- `idx_conversations_updated_at` - Recent activity
- `idx_conversations_metadata` - Metadata search (GIN)

#### `messages`
Stores individual messages within conversations.

| Column | Type | Description |
|--------|------|-------------|
| message_id | VARCHAR(255) | Primary key |
| conversation_id | VARCHAR(255) | Foreign key to conversations |
| role | VARCHAR(50) | user/assistant/system |
| content | TEXT | Message content |
| created_at | TIMESTAMP | Message timestamp |
| metadata | JSONB | Message metadata |
| tokens_used | INTEGER | Token count |

**Indexes:**
- `idx_messages_conversation_id` - Fast conversation lookup
- `idx_messages_created_at` - Chronological ordering
- `idx_messages_role` - Filter by role
- `idx_messages_metadata` - Metadata search (GIN)

### Views

#### `v_user_conversation_stats`
Per-user conversation statistics.

```sql
SELECT * FROM v_user_conversation_stats;
```

#### `v_conversation_message_stats`
Message statistics per conversation.

```sql
SELECT * FROM v_conversation_message_stats;
```

#### `v_recent_activity`
Last 100 messages across all conversations.

```sql
SELECT * FROM v_recent_activity;
```

---

## 🔧 Configuration

All scripts use environment variables from `.env`:

```bash
# Database Configuration
DATABASE_URL=postgresql://chatbot_user:changeme@localhost:5432/chatbot
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
```

### Docker Compose Settings

From `docker-compose.yml`:
```yaml
postgres:
  image: postgres:15-alpine
  environment:
    POSTGRES_USER: chatbot_user
    POSTGRES_PASSWORD: changeme
    POSTGRES_DB: chatbot
  ports:
    - "5432:5432"
```

---

## 🐛 Troubleshooting

### Cannot connect to database
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Start PostgreSQL
docker-compose up -d postgres

# Check logs
docker logs subagents-postgres
```

### Permission denied errors
```bash
# Make scripts executable
chmod +x scripts/*.sh scripts/*.py
```

### Tables already exist
```bash
# Use --force flag to drop and recreate
python scripts/init_db.py --force
```

### Orphaned messages after deletion
```bash
# Run health check to identify issues
python scripts/check_db_health.py

# Clean orphaned messages
psql -d chatbot -c "DELETE FROM messages WHERE conversation_id NOT IN (SELECT conversation_id FROM conversations)"
```

### Backup/Restore fails
```bash
# Check PostgreSQL tools are installed
which pg_dump
which psql

# Install if missing (Ubuntu/Debian)
sudo apt-get install postgresql-client

# Or use Docker
docker exec subagents-postgres pg_dump -U chatbot_user chatbot > backup.sql
```

---

## 📚 Additional Resources

### PostgreSQL Commands
```bash
# Connect to database
psql -h localhost -p 5432 -U chatbot_user -d chatbot

# List tables
\dt

# Describe table
\d conversations

# Show indexes
\di

# View data
SELECT * FROM conversations LIMIT 10;

# Exit
\q
```

### SQL Queries for Debugging
```sql
-- Count conversations per user
SELECT user_id, COUNT(*) as conv_count
FROM conversations
GROUP BY user_id
ORDER BY conv_count DESC;

-- Average messages per conversation
SELECT
    AVG(msg_count) as avg_messages
FROM (
    SELECT conversation_id, COUNT(*) as msg_count
    FROM messages
    GROUP BY conversation_id
) subquery;

-- Recent activity (last 24 hours)
SELECT
    c.title,
    m.role,
    m.content,
    m.created_at
FROM messages m
JOIN conversations c ON m.conversation_id = c.conversation_id
WHERE m.created_at > NOW() - INTERVAL '24 hours'
ORDER BY m.created_at DESC;

-- Database size
SELECT pg_size_pretty(pg_database_size('chatbot'));

-- Table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

## 🔐 Security Best Practices

1. **Never commit backups** to version control
   ```bash
   # Add to .gitignore
   echo "backups/" >> .gitignore
   ```

2. **Change default password** in production
   ```bash
   # Update .env
   DATABASE_URL=postgresql://chatbot_user:STRONG_PASSWORD@localhost:5432/chatbot
   ```

3. **Restrict database access**
   - Use firewall rules
   - Bind to localhost only (for local deployment)
   - Use SSL/TLS for remote connections

4. **Regular backups**
   ```bash
   # Set up cron job for daily backups
   0 2 * * * /path/to/scripts/backup_db.sh --compress
   ```

5. **Monitor database health**
   ```bash
   # Run health checks regularly
   python scripts/check_db_health.py --verbose
   ```

---

## 📝 Notes

- All scripts support `--help` flag for detailed usage information
- Scripts are idempotent where possible (safe to run multiple times)
- Backups are automatically cleaned up (keeps last 10)
- Sample data uses realistic banking scenarios
- All timestamps are in UTC

---

**Need Help?** Check the main project README or documentation in `docs/`
