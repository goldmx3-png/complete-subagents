"""
Configuration management for the subagent RAG system
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings"""

    # OpenRouter Configuration
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # Models
    main_model: str = os.getenv("MAIN_MODEL", "mistralai/magistral-small-2506")
    router_model: str = os.getenv("ROUTER_MODEL", "mistralai/magistral-small-2506")

    # LLM Configuration
    max_tokens: int = int(os.getenv("MAX_TOKENS", "4096"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    router_temperature: float = float(os.getenv("ROUTER_TEMPERATURE", "0.3"))
    use_rule_based: bool = os.getenv("USE_RULE_BASED", "true").lower() == "true"

    # Vector Store
    vector_store: str = os.getenv("VECTOR_STORE", "qdrant")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "documents")

    # Embeddings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    # Reranker
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranker_device: str = os.getenv("RERANKER_DEVICE", "cpu")
    use_reranker: bool = os.getenv("USE_RERANKER", "false").lower() == "true"

    # Database
    database_url: str = os.getenv("DATABASE_URL", "postgresql://chatbot_user:changeme@localhost:5432/chatbot")
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "20"))
    db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "40"))

    # Application Settings
    max_context_length: int = int(os.getenv("MAX_CONTEXT_LENGTH", "32768"))
    max_chunks_per_query: int = int(os.getenv("MAX_CHUNKS_PER_QUERY", "10"))
    conversation_history_limit: int = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "10"))
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1024"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # RAG Settings
    min_similarity_score: float = float(os.getenv("MIN_SIMILARITY_SCORE", "0.3"))
    ambiguity_threshold: float = float(os.getenv("AMBIGUITY_THRESHOLD", "0.15"))
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "20"))
    top_k_rerank: int = int(os.getenv("TOP_K_RERANK", "10"))

    # Query Enhancement Settings
    use_query_rewriting: bool = os.getenv("USE_QUERY_REWRITING", "true").lower() == "true"
    query_rewrite_cache_ttl: int = int(os.getenv("QUERY_REWRITE_CACHE_TTL", "86400"))  # 24 hours

    # API Settings
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_workers: int = int(os.getenv("API_WORKERS", "4"))
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")

    # Monitoring
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug_mode: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"

    # Security
    api_key_required: bool = os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
    allowed_upload_extensions: str = os.getenv("ALLOWED_UPLOAD_EXTENSIONS", "pdf,docx,txt")
    upload_directory: str = os.getenv("UPLOAD_DIRECTORY", "uploads")
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    max_requests_per_minute: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))

    # Banking API Integration
    banking_api_jwt_token: Optional[str] = os.getenv("BANKING_API_JWT_TOKEN")
    banking_api_timeout: int = int(os.getenv("BANKING_API_TIMEOUT", "30"))
    banking_api_max_retries: int = int(os.getenv("BANKING_API_MAX_RETRIES", "3"))
    banking_api_verify_ssl: bool = os.getenv("BANKING_API_VERIFY_SSL", "true").lower() == "true"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()
