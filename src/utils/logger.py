"""Production-grade logging utilities"""

import logging
import sys
from typing import Optional
from .correlation import get_correlation_id


class CorrelationFormatter(logging.Formatter):
    """Custom formatter that includes correlation ID"""

    def format(self, record: logging.LogRecord) -> str:
        correlation_id = get_correlation_id()
        record.correlation_id = correlation_id if correlation_id else "N/A"
        record.location = f"{record.module}:{record.lineno}"
        return super().format(record)


def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Create and configure a logger"""
    logger = logging.getLogger(name)

    if level:
        log_level = getattr(logging, level.upper(), logging.INFO)
    else:
        try:
            from src.config import settings
            log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
        except:
            log_level = logging.INFO

    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    formatter = CorrelationFormatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(location)s] [request_id=%(correlation_id)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger"""
    return setup_logger(name)
