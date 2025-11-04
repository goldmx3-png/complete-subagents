"""
Async helpers for running async code in sync contexts

This module provides utilities to safely run async functions from sync code,
especially when dealing with existing event loops (like uvloop).
"""

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Coroutine, TypeVar

T = TypeVar('T')


def run_async_in_new_loop(coro_or_func: Any, *args, **kwargs) -> T:
    """
    Run an async coroutine or function in a completely new event loop in a separate thread.

    This is safe to use even when there's an existing running event loop
    (like uvloop) because it creates a brand new loop in a new thread.

    Args:
        coro_or_func: Either a coroutine or an async function
        *args: Arguments to pass if coro_or_func is a function
        **kwargs: Keyword arguments to pass if coro_or_func is a function

    Returns:
        The result of the coroutine

    Examples:
        # With a pre-created coroutine
        result = run_async_in_new_loop(some_async_function(arg1, arg2))

        # With a function (creates coroutine in new thread - better for httpx clients)
        result = run_async_in_new_loop(some_async_function, arg1, arg2)
    """
    def run_in_new_loop():
        """Helper function to run in thread pool"""
        # Create a completely new event loop for this thread
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            # If it's a coroutine, run it directly
            if asyncio.iscoroutine(coro_or_func):
                return new_loop.run_until_complete(coro_or_func)
            # If it's a callable (function), call it to create the coroutine in this thread
            elif callable(coro_or_func):
                coro = coro_or_func(*args, **kwargs)
                return new_loop.run_until_complete(coro)
            else:
                raise TypeError(f"Expected coroutine or callable, got {type(coro_or_func)}")
        finally:
            # Properly clean up tasks and close the loop
            try:
                # Cancel any pending tasks
                pending = asyncio.all_tasks(new_loop)
                for task in pending:
                    task.cancel()
                # Run the loop one more time to handle cancellations
                if pending:
                    new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)

    # Run in a thread pool to avoid conflicts with existing event loop
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_in_new_loop)
        return future.result()


def make_sync(async_func):
    """
    Decorator to create a sync wrapper for an async function.

    The wrapped function will safely call the async function even when
    there's an existing event loop running.

    Args:
        async_func: The async function to wrap

    Returns:
        A synchronous version of the function

    Example:
        @make_sync
        async def my_async_func(x, y):
            return x + y

        # Can now call it synchronously
        result = my_async_func(1, 2)
    """
    @functools.wraps(async_func)
    def wrapper(*args, **kwargs):
        coro = async_func(*args, **kwargs)
        return run_async_in_new_loop(coro)
    return wrapper
