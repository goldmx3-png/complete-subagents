-- ============================================
-- Database Initialization Script
-- Complete Subagents - Banking RAG System
-- ============================================

-- This script initializes the PostgreSQL database
-- Run this as postgres superuser or database admin

-- Drop database if exists (use with caution in production!)
-- DROP DATABASE IF EXISTS chatbot;
-- DROP USER IF EXISTS chatbot_user;

-- Create user
CREATE USER chatbot_user WITH PASSWORD 'changeme';

-- Create database
CREATE DATABASE chatbot
    WITH
    OWNER = chatbot_user
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE chatbot TO chatbot_user;

-- Connect to the database
\c chatbot

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO chatbot_user;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search improvements

-- Grant extension usage
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO chatbot_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO chatbot_user;

-- ============================================
-- Create Tables
-- ============================================

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    title VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    summary TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for conversations
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_metadata ON conversations USING gin(metadata);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    message_id VARCHAR(255) PRIMARY KEY,
    conversation_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    tokens_used INTEGER
);

-- Create indexes for messages
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_messages_metadata ON messages USING gin(metadata);

-- Foreign key constraint
ALTER TABLE messages
    DROP CONSTRAINT IF EXISTS fk_messages_conversation;

ALTER TABLE messages
    ADD CONSTRAINT fk_messages_conversation
    FOREIGN KEY (conversation_id)
    REFERENCES conversations(conversation_id)
    ON DELETE CASCADE;

-- API Registry table (for banking API definitions)
CREATE TABLE IF NOT EXISTS api_registry (
    id SERIAL PRIMARY KEY,
    product VARCHAR(100) NOT NULL,  -- Banking product (payments, accounts, cards, loans)
    api_name VARCHAR(255) NOT NULL UNIQUE,  -- Unique API identifier
    api_description TEXT NOT NULL,  -- What the API does
    http_method VARCHAR(10) NOT NULL DEFAULT 'POST',  -- GET, POST, PUT, DELETE, PATCH
    endpoint_url TEXT NOT NULL,  -- API endpoint URL
    request_schema JSONB,  -- JSON schema for request validation
    response_schema JSONB,  -- JSON schema for response validation
    example_request JSONB,  -- Example request payload
    example_response JSONB,  -- Example response payload
    list_formatting_template TEXT,  -- Template for formatting list responses
    is_active BOOLEAN DEFAULT true,  -- Enable/disable API
    requires_auth BOOLEAN DEFAULT true,  -- Authentication requirement
    rate_limit VARCHAR(50),  -- Rate limiting info (e.g., "100/minute")
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for api_registry
CREATE INDEX IF NOT EXISTS idx_api_registry_product ON api_registry(product);
CREATE INDEX IF NOT EXISTS idx_api_registry_api_name ON api_registry(api_name);
CREATE INDEX IF NOT EXISTS idx_api_registry_is_active ON api_registry(is_active);
CREATE INDEX IF NOT EXISTS idx_api_registry_product_active ON api_registry(product, is_active);
CREATE INDEX IF NOT EXISTS idx_api_registry_description ON api_registry USING gin(to_tsvector('english', api_description));  -- Full-text search

-- Unique constraint for product + api_name combination
CREATE UNIQUE INDEX IF NOT EXISTS idx_api_registry_product_api_name
    ON api_registry(product, api_name);

-- ============================================
-- Create Functions and Triggers
-- ============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for conversations table
DROP TRIGGER IF EXISTS update_conversations_updated_at ON conversations;
CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for api_registry table
DROP TRIGGER IF EXISTS update_api_registry_updated_at ON api_registry;
CREATE TRIGGER update_api_registry_updated_at
    BEFORE UPDATE ON api_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- Create Views for Analytics
-- ============================================

-- View: Conversation statistics per user
CREATE OR REPLACE VIEW v_user_conversation_stats AS
SELECT
    user_id,
    COUNT(DISTINCT conversation_id) as total_conversations,
    MAX(updated_at) as last_activity,
    MIN(created_at) as first_activity
FROM conversations
GROUP BY user_id;

-- View: Message statistics per conversation
CREATE OR REPLACE VIEW v_conversation_message_stats AS
SELECT
    c.conversation_id,
    c.user_id,
    c.title,
    c.created_at,
    c.updated_at,
    COUNT(m.message_id) as total_messages,
    SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_messages,
    SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages,
    SUM(COALESCE(m.tokens_used, 0)) as total_tokens
FROM conversations c
LEFT JOIN messages m ON c.conversation_id = m.conversation_id
GROUP BY c.conversation_id, c.user_id, c.title, c.created_at, c.updated_at;

-- View: Recent activity
CREATE OR REPLACE VIEW v_recent_activity AS
SELECT
    c.conversation_id,
    c.user_id,
    c.title,
    m.role,
    m.content,
    m.created_at,
    m.tokens_used
FROM conversations c
JOIN messages m ON c.conversation_id = m.conversation_id
ORDER BY m.created_at DESC
LIMIT 100;

-- ============================================
-- Grant Permissions
-- ============================================

-- Grant table permissions to chatbot_user
GRANT SELECT, INSERT, UPDATE, DELETE ON conversations TO chatbot_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON messages TO chatbot_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON api_registry TO chatbot_user;
GRANT USAGE, SELECT ON SEQUENCE api_registry_id_seq TO chatbot_user;

-- Grant view permissions
GRANT SELECT ON v_user_conversation_stats TO chatbot_user;
GRANT SELECT ON v_conversation_message_stats TO chatbot_user;
GRANT SELECT ON v_recent_activity TO chatbot_user;

-- ============================================
-- Insert Sample Data (Optional - for testing)
-- ============================================

-- Uncomment to insert sample data
/*
INSERT INTO conversations (conversation_id, user_id, title, summary, metadata) VALUES
    ('conv_test_001', 'user_demo', 'Sample Banking Query', 'Discussion about account features', '{"agent": "rag", "domain": "banking"}'),
    ('conv_test_002', 'user_demo', 'API Integration Test', 'Testing banking API endpoints', '{"agent": "api", "domain": "banking"}');

INSERT INTO messages (message_id, conversation_id, role, content, tokens_used, metadata) VALUES
    ('msg_001', 'conv_test_001', 'user', 'What are the features of a savings account?', 15, '{"intent": "query"}'),
    ('msg_002', 'conv_test_001', 'assistant', 'A savings account offers interest on deposits, online banking, and withdrawal flexibility.', 25, '{"agent": "rag", "confidence": 0.95}'),
    ('msg_003', 'conv_test_002', 'user', 'Check my account balance', 10, '{"intent": "api_call"}'),
    ('msg_004', 'conv_test_002', 'assistant', 'Your current account balance is $5,432.50', 15, '{"agent": "api", "api_endpoint": "/accounts/balance"}');

-- Sample API Registry entries
INSERT INTO api_registry (product, api_name, api_description, http_method, endpoint_url, request_schema, response_schema, example_request, example_response, is_active, requires_auth, rate_limit) VALUES
    -- Accounts APIs
    ('accounts', 'get_account_balance', 'Retrieve current account balance for a specific account', 'GET', 'https://api.bank.example/v1/accounts/{account_id}/balance',
     '{"type": "object", "properties": {"account_id": {"type": "string"}}, "required": ["account_id"]}'::jsonb,
     '{"type": "object", "properties": {"balance": {"type": "number"}, "currency": {"type": "string"}}}'::jsonb,
     '{"account_id": "ACC123456"}'::jsonb,
     '{"balance": 5432.50, "currency": "USD", "available_balance": 5432.50}'::jsonb,
     true, true, '100/minute'),

    ('accounts', 'get_account_details', 'Get detailed information about a bank account', 'GET', 'https://api.bank.example/v1/accounts/{account_id}',
     '{"type": "object", "properties": {"account_id": {"type": "string"}}, "required": ["account_id"]}'::jsonb,
     '{"type": "object", "properties": {"account_number": {"type": "string"}, "account_type": {"type": "string"}}}'::jsonb,
     '{"account_id": "ACC123456"}'::jsonb,
     '{"account_number": "ACC123456", "account_type": "savings", "branch": "Main Branch", "status": "active"}'::jsonb,
     true, true, '100/minute'),

    ('accounts', 'list_accounts', 'List all accounts for the authenticated user', 'GET', 'https://api.bank.example/v1/accounts',
     '{"type": "object", "properties": {}}'::jsonb,
     '{"type": "array", "items": {"type": "object"}}'::jsonb,
     '{}'::jsonb,
     '[{"account_id": "ACC123", "type": "savings", "balance": 5000}, {"account_id": "ACC456", "type": "checking", "balance": 2500}]'::jsonb,
     true, true, '50/minute'),

    -- Payments APIs
    ('payments', 'initiate_payment', 'Initiate a new payment transaction', 'POST', 'https://api.bank.example/v1/payments',
     '{"type": "object", "properties": {"from_account": {"type": "string"}, "to_account": {"type": "string"}, "amount": {"type": "number"}}, "required": ["from_account", "to_account", "amount"]}'::jsonb,
     '{"type": "object", "properties": {"transaction_id": {"type": "string"}, "status": {"type": "string"}}}'::jsonb,
     '{"from_account": "ACC123", "to_account": "ACC456", "amount": 100.00, "currency": "USD"}'::jsonb,
     '{"transaction_id": "TXN789", "status": "pending", "timestamp": "2025-11-11T10:00:00Z"}'::jsonb,
     true, true, '20/minute'),

    ('payments', 'get_payment_status', 'Check the status of a payment transaction', 'GET', 'https://api.bank.example/v1/payments/{transaction_id}',
     '{"type": "object", "properties": {"transaction_id": {"type": "string"}}, "required": ["transaction_id"]}'::jsonb,
     '{"type": "object", "properties": {"status": {"type": "string"}, "amount": {"type": "number"}}}'::jsonb,
     '{"transaction_id": "TXN789"}'::jsonb,
     '{"transaction_id": "TXN789", "status": "completed", "amount": 100.00}'::jsonb,
     true, true, '100/minute'),

    -- Cards APIs
    ('cards', 'list_cards', 'List all cards associated with the user', 'GET', 'https://api.bank.example/v1/cards',
     '{"type": "object", "properties": {}}'::jsonb,
     '{"type": "array", "items": {"type": "object"}}'::jsonb,
     '{}'::jsonb,
     '[{"card_id": "CARD001", "type": "credit", "last4": "1234", "status": "active"}]'::jsonb,
     true, true, '50/minute'),

    ('cards', 'get_card_transactions', 'Get recent transactions for a specific card', 'GET', 'https://api.bank.example/v1/cards/{card_id}/transactions',
     '{"type": "object", "properties": {"card_id": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["card_id"]}'::jsonb,
     '{"type": "array", "items": {"type": "object"}}'::jsonb,
     '{"card_id": "CARD001", "limit": 10}'::jsonb,
     '[{"date": "2025-11-10", "merchant": "Coffee Shop", "amount": 5.50}]'::jsonb,
     true, true, '100/minute'),

    -- Loans APIs
    ('loans', 'list_loans', 'List all active loans for the user', 'GET', 'https://api.bank.example/v1/loans',
     '{"type": "object", "properties": {}}'::jsonb,
     '{"type": "array", "items": {"type": "object"}}'::jsonb,
     '{}'::jsonb,
     '[{"loan_id": "LOAN001", "type": "home", "outstanding": 250000, "due_date": "2025-12-01"}]'::jsonb,
     true, true, '50/minute'),

    ('loans', 'get_loan_details', 'Get detailed information about a specific loan', 'GET', 'https://api.bank.example/v1/loans/{loan_id}',
     '{"type": "object", "properties": {"loan_id": {"type": "string"}}, "required": ["loan_id"]}'::jsonb,
     '{"type": "object", "properties": {"loan_id": {"type": "string"}, "amount": {"type": "number"}}}'::jsonb,
     '{"loan_id": "LOAN001"}'::jsonb,
     '{"loan_id": "LOAN001", "type": "home", "amount": 250000, "interest_rate": 3.5, "term_months": 360}'::jsonb,
     true, true, '100/minute');
*/

-- ============================================
-- Database Information
-- ============================================

-- Display table information
SELECT
    'Database initialized successfully!' as status,
    current_database() as database,
    current_user as user,
    version() as postgres_version;

-- Display table counts
SELECT
    'conversations' as table_name,
    COUNT(*) as row_count
FROM conversations
UNION ALL
SELECT
    'messages' as table_name,
    COUNT(*) as row_count
FROM messages
UNION ALL
SELECT
    'api_registry' as table_name,
    COUNT(*) as row_count
FROM api_registry;
