"""Request correlation ID management for distributed tracing"""

import uuid
from contextvars import ContextVar
from typing import Optional
from contextlib import contextmanager

_correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


def generate_correlation_id() -> str:
    """Generate a new unique correlation ID"""
    return str(uuid.uuid4())


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set the correlation ID for the current context"""
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    _correlation_id.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """Get the correlation ID for the current context"""
    return _correlation_id.get()


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context"""
    _correlation_id.set(None)


@contextmanager
def correlation_context(correlation_id: Optional[str] = None):
    """Context manager for setting correlation ID within a specific scope"""
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    previous_id = get_correlation_id()
    try:
        set_correlation_id(correlation_id)
        yield correlation_id
    finally:
        if previous_id is not None:
            set_correlation_id(previous_id)
        else:
            clear_correlation_id()


def get_or_create_correlation_id() -> str:
    """Get current correlation ID, or create one if it doesn't exist"""
    correlation_id = get_correlation_id()
    if correlation_id is None:
        correlation_id = set_correlation_id()
    return correlation_id
