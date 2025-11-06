"""
Cross-Encoder Reranker for improving retrieval precision
Uses BAAI/bge-reranker-large for accurate relevance scoring
"""

from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
from src.config import settings
from src.utils.logger import get_logger
import os
import time
from pathlib import Path
from tqdm import tqdm
import threading
import signal

logger = get_logger(__name__)


class Reranker:
    """
    Cross-encoder reranker for final relevance scoring

    Features:
    - Batch processing for efficiency
    - Caching of loaded model
    - Configurable top-k selection
    """

    def __init__(self):
        self.model = None
        self.model_name = settings.reranker_model_v2
        self.device = settings.reranker_device_v2
        self.batch_size = settings.reranker_batch_size

    def _is_model_cached(self) -> bool:
        """
        Check if the model is already downloaded in the cache

        Returns:
            True if model is cached, False otherwise
        """
        try:
            from huggingface_hub import cached_assets_path
            from transformers.utils import TRANSFORMERS_CACHE

            # Check common cache locations
            cache_dir = os.environ.get('TRANSFORMERS_CACHE') or os.environ.get('HF_HOME')
            if not cache_dir:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

            # Construct expected model path (simplified check)
            model_cache_name = self.model_name.replace("/", "--")
            model_path = Path(cache_dir) / f"models--{model_cache_name}"

            return model_path.exists()
        except Exception:
            return False

    def _load_model_with_timeout(self, timeout_seconds: int):
        """
        Load model with timeout protection

        Args:
            timeout_seconds: Maximum time to wait for model loading

        Raises:
            TimeoutError: If loading exceeds timeout
        """
        result = {"model": None, "error": None}

        def load_target():
            try:
                result["model"] = CrossEncoder(
                    self.model_name,
                    max_length=512,
                    device=self.device
                )
            except Exception as e:
                result["error"] = e

        # Start loading in a separate thread
        load_thread = threading.Thread(target=load_target, daemon=True)
        load_thread.start()

        # Wait with timeout
        load_thread.join(timeout=timeout_seconds)

        if load_thread.is_alive():
            # Thread is still running - timeout occurred
            raise TimeoutError(
                f"Model loading timed out after {timeout_seconds} seconds. "
                f"This usually indicates a network issue or very slow download. "
                f"Try: 1) Check internet connection, 2) Increase RERANKER_LOAD_TIMEOUT, "
                f"3) Manually download model, or 4) Set ENABLE_RERANKING=false"
            )

        if result["error"]:
            raise result["error"]

        return result["model"]

    def _load_model(self):
        """
        Lazy load the reranker model with progress tracking and timeout

        Features:
        - Shows download progress if model not cached
        - Provides diagnostic information on failure
        - Validates model after loading
        - Times out if loading takes too long
        """
        if self.model is None:
            try:
                is_cached = self._is_model_cached()
                timeout = settings.reranker_load_timeout

                if is_cached:
                    logger.info(f"Loading reranker model from cache: {self.model_name}")
                else:
                    logger.info(f"Downloading reranker model: {self.model_name}")

                    # Provide size estimates based on model
                    size_info = {
                        "cross-encoder/ms-marco-MiniLM-L-6-v2": "~80MB",
                        "BAAI/bge-reranker-v2-m3": "~140MB",
                        "BAAI/bge-reranker-base": "~278MB",
                        "BAAI/bge-reranker-large": "~560MB"
                    }
                    model_size = size_info.get(self.model_name, "~100-500MB")
                    logger.info(f"Model size: {model_size}")
                    logger.info("Download location: HuggingFace cache (~/.cache/huggingface/)")
                    logger.info(f"Timeout: {timeout} seconds")

                start_time = time.time()

                # Load model with timeout protection
                self.model = self._load_model_with_timeout(timeout)

                elapsed = time.time() - start_time
                logger.info(f"✓ Reranker model loaded successfully on {self.device} ({elapsed:.1f}s)")

            except TimeoutError as e:
                logger.error(f"Model loading timed out: {str(e)}")
                self._log_diagnostics()
                raise

            except Exception as e:
                logger.error(f"Failed to load reranker model: {str(e)}")
                logger.error(f"Model name: {self.model_name}")
                logger.error(f"Device: {self.device}")

                # Provide diagnostic information
                self._log_diagnostics()
                raise

    def _log_diagnostics(self):
        """Log diagnostic information for troubleshooting"""
        logger.info("=== Reranker Loading Diagnostics ===")

        # Check internet connectivity
        try:
            import socket
            socket.create_connection(("huggingface.co", 443), timeout=5)
            logger.info("✓ Internet connectivity: OK")
        except Exception:
            logger.error("✗ Internet connectivity: FAILED - Cannot reach huggingface.co")

        # Check cache directory
        cache_dir = os.environ.get('TRANSFORMERS_CACHE') or os.environ.get('HF_HOME')
        if not cache_dir:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

        logger.info(f"Cache directory: {cache_dir}")

        try:
            cache_path = Path(cache_dir)
            if cache_path.exists():
                logger.info("✓ Cache directory exists")
                # Check write permissions
                test_file = cache_path / ".write_test"
                try:
                    test_file.touch()
                    test_file.unlink()
                    logger.info("✓ Cache directory is writable")
                except Exception:
                    logger.error("✗ Cache directory is NOT writable")
            else:
                logger.warning("⚠ Cache directory does not exist (will be created)")
        except Exception as e:
            logger.error(f"✗ Cache directory check failed: {str(e)}")

        # Check disk space
        try:
            import shutil
            cache_path = Path(cache_dir)
            if cache_path.exists():
                total, used, free = shutil.disk_usage(cache_path)
                free_gb = free / (1024 ** 3)
                logger.info(f"Free disk space: {free_gb:.2f} GB")
                if free_gb < 1:
                    logger.error("✗ Low disk space (< 1GB free)")
        except Exception:
            pass

        logger.info("=== End Diagnostics ===")

    def preload(self):
        """
        Preload the reranker model (for startup initialization)

        Call this during service startup to avoid first-request latency

        Features:
        - Shows download progress bars for model files
        - Provides estimated download time
        - Logs cache status
        """
        try:
            # Enable HuggingFace download progress bars
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

            # Check if model is cached
            is_cached = self._is_model_cached()

            if not is_cached:
                # Provide size estimates based on model
                size_info = {
                    "cross-encoder/ms-marco-MiniLM-L-6-v2": "~80MB",
                    "BAAI/bge-reranker-v2-m3": "~140MB",
                    "BAAI/bge-reranker-base": "~278MB",
                    "BAAI/bge-reranker-large": "~560MB"
                }
                model_size = size_info.get(self.model_name, "~100-500MB")

                logger.info("=" * 80)
                logger.info("MODEL DOWNLOAD REQUIRED")
                logger.info("=" * 80)
                logger.info(f"Model: {self.model_name}")
                logger.info(f"Size: {model_size}")
                logger.info("This is a ONE-TIME download. Subsequent starts will be instant.")
                logger.info("=" * 80)

            self._load_model()
            logger.info("✓ Reranker model preloaded successfully")

        except Exception as e:
            logger.error(f"Failed to preload reranker model: {str(e)}")
            logger.error("Consider setting ENABLE_RERANKING=false in .env if issues persist")
            raise

    async def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank documents using cross-encoder

        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of documents to return (uses settings if None)

        Returns:
            Reranked documents with updated scores
        """
        if not documents:
            return []

        # Load model if needed
        self._load_model()

        top_k = top_k or settings.reranker_return_top_k

        try:
            # Prepare query-document pairs
            pairs = []
            for doc in documents:
                text = doc.get("payload", {}).get("text", "")
                pairs.append([query, text])

            logger.info(f"Reranking {len(pairs)} documents")

            # Get scores from cross-encoder
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )

            # Update documents with new scores
            reranked_docs = []
            for doc, score in zip(documents, scores):
                reranked_doc = doc.copy()
                reranked_doc["score"] = float(score)
                reranked_doc["original_score"] = doc.get("score", 0.0)
                reranked_doc["search_type"] = "reranked"
                reranked_docs.append(reranked_doc)

            # Sort by new score and return top-k
            reranked_docs.sort(key=lambda x: x["score"], reverse=True)

            logger.info(f"Reranking complete, returning top {top_k} documents")
            return reranked_docs[:top_k]

        except Exception as e:
            logger.error(f"Reranking error: {str(e)}")
            # Fallback: return original documents
            return documents[:top_k]

    async def rerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[Dict]],
        top_k: Optional[int] = None
    ) -> List[List[Dict]]:
        """
        Rerank multiple query-document sets

        Args:
            queries: List of queries
            documents_list: List of document lists
            top_k: Number of documents to return per query

        Returns:
            List of reranked document lists
        """
        results = []
        for query, docs in zip(queries, documents_list):
            reranked = await self.rerank(query, docs, top_k)
            results.append(reranked)
        return results


# Global reranker instance (lazy loaded)
_reranker_instance = None


def get_reranker() -> Reranker:
    """
    Get or create global reranker instance

    Returns:
        Reranker instance
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance


def preload_reranker():
    """
    Preload the global reranker instance and model

    Call this during service startup to load the model into memory

    Note: First-time model download may take 2-5 minutes depending on
    internet speed. Progress bars will be displayed during download.
    """
    logger.info("Initializing reranker...")
    reranker = get_reranker()
    reranker.preload()
    logger.info("✓ Reranker ready for requests")
