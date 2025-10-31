"""Performance metrics and timing utilities"""

import time
import functools
from typing import Optional, Any, Dict, Callable
from contextlib import contextmanager
import logging


def log_performance(logger: Optional[logging.Logger] = None, operation: Optional[str] = None):
    """Decorator to log function execution time"""
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__
        func_logger = logger or logging.getLogger(func.__module__)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_logger.info(f"{op_name} started")
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                func_logger.info(f"{op_name} completed ({duration_ms:.0f}ms)")
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                func_logger.error(f"{op_name} failed after {duration_ms:.0f}ms: {str(e)}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_logger.info(f"{op_name} started")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                func_logger.info(f"{op_name} completed ({duration_ms:.0f}ms)")
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                func_logger.error(f"{op_name} failed after {duration_ms:.0f}ms: {str(e)}")
                raise

        return async_wrapper if functools.iscoroutinefunction(func) else sync_wrapper
    return decorator


@contextmanager
def timed_operation(logger: logging.Logger, operation: str, **extra_fields):
    """Context manager for timing operations"""
    metrics: Dict[str, Any] = {}
    extra_str = ", ".join(f"{k}={v}" for k, v in extra_fields.items())
    start_msg = f"{operation} started"
    if extra_str:
        start_msg += f" ({extra_str})"
    logger.info(start_msg)

    start_time = time.time()
    try:
        yield metrics
    finally:
        duration_ms = (time.time() - start_time) * 1000
        metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        all_fields = f"{duration_ms:.0f}ms"
        if metrics_str:
            all_fields += f", {metrics_str}"
        logger.info(f"{operation} completed ({all_fields})")


def log_llm_metrics(logger: logging.Logger, model: str, duration_ms: float,
                    prompt_tokens: Optional[int] = None, completion_tokens: Optional[int] = None,
                    **extra_metrics):
    """Log LLM inference metrics"""
    metrics = [f"{duration_ms:.0f}ms"]
    if prompt_tokens is not None:
        metrics.append(f"prompt_tokens={prompt_tokens}")
    if completion_tokens is not None:
        metrics.append(f"completion_tokens={completion_tokens}")
    if prompt_tokens is not None and completion_tokens is not None:
        metrics.append(f"total_tokens={prompt_tokens + completion_tokens}")
    metrics.extend(f"{k}={v}" for k, v in extra_metrics.items())
    logger.info(f"LLM inference: {model} ({', '.join(metrics)})")
