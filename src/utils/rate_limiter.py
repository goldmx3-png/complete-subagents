"""
Rate limiter for API calls with exponential backoff and retry logic
Handles both per-minute and daily rate limits
"""

import asyncio
import time
from typing import Optional, Callable, Any
from collections import deque
from datetime import datetime, timedelta
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter with support for:
    - Per-minute rate limits
    - Daily rate limits
    - Exponential backoff on rate limit errors
    - Automatic retry with delays
    """

    def __init__(
        self,
        requests_per_minute: int = 20,
        requests_per_day: Optional[int] = 50,
        burst_size: Optional[int] = None
    ):
        """
        Initialize rate limiter

        Args:
            requests_per_minute: Maximum requests per minute (e.g., 20 for OpenRouter free)
            requests_per_day: Maximum requests per day (e.g., 50 or 1000)
            burst_size: Maximum burst size (defaults to requests_per_minute)
        """
        self.rpm = requests_per_minute
        self.rpd = requests_per_day
        self.burst_size = burst_size or requests_per_minute

        # Token bucket for per-minute limiting
        self.tokens = float(self.burst_size)
        self.last_update = time.time()

        # Sliding window for daily limiting
        self.daily_requests = deque()
        self.daily_limit_start = datetime.now()

        # Lock for thread safety
        self.lock = asyncio.Lock()

        logger.info(
            f"Rate limiter initialized: {self.rpm} RPM, "
            f"{self.rpd or 'unlimited'} RPD, burst={self.burst_size}"
        )

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the rate limiter

        Args:
            tokens: Number of tokens to acquire (default 1)

        Returns:
            True if tokens acquired, waits if necessary
        """
        async with self.lock:
            while True:
                # Refill tokens based on time passed
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(
                    self.burst_size,
                    self.tokens + (time_passed * self.rpm / 60.0)
                )
                self.last_update = now

                # Check daily limit
                if self.rpd:
                    self._cleanup_daily_requests()
                    if len(self.daily_requests) >= self.rpd:
                        # Hit daily limit
                        wait_time = self._time_until_daily_reset()
                        logger.warning(
                            f"Daily rate limit reached ({self.rpd} requests). "
                            f"Waiting {wait_time:.0f}s until reset"
                        )
                        await asyncio.sleep(min(wait_time, 60))  # Wait max 1 min at a time
                        continue

                # Check if we have enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    if self.rpd:
                        self.daily_requests.append(datetime.now())
                    return True

                # Calculate wait time for token refill
                tokens_needed = tokens - self.tokens
                wait_time = (tokens_needed * 60.0) / self.rpm

                logger.debug(
                    f"Rate limit: waiting {wait_time:.2f}s "
                    f"(tokens: {self.tokens:.2f}/{self.burst_size})"
                )

                await asyncio.sleep(wait_time)

    def _cleanup_daily_requests(self):
        """Remove requests older than 24 hours from daily counter"""
        cutoff = datetime.now() - timedelta(days=1)
        while self.daily_requests and self.daily_requests[0] < cutoff:
            self.daily_requests.popleft()

    def _time_until_daily_reset(self) -> float:
        """Calculate seconds until daily limit resets"""
        if not self.daily_requests:
            return 0

        oldest_request = self.daily_requests[0]
        reset_time = oldest_request + timedelta(days=1)
        return (reset_time - datetime.now()).total_seconds()

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        max_retries: int = 3,
        base_delay: float = 1.0,
        **kwargs
    ) -> Any:
        """
        Execute a function with rate limiting and retry logic

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            max_retries: Maximum number of retries on rate limit errors
            base_delay: Base delay for exponential backoff (seconds)
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            Exception: If all retries exhausted
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Acquire token before making request
                await self.acquire()

                # Execute the function
                result = await func(*args, **kwargs)
                return result

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if it's a rate limit error
                is_rate_limit = any(
                    term in error_str
                    for term in ['rate limit', 'too many requests', '429', 'quota']
                )

                if is_rate_limit and attempt < max_retries:
                    # Exponential backoff
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {delay:.1f}s: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                else:
                    # Not a rate limit error, or out of retries
                    if attempt == max_retries:
                        logger.error(f"Max retries exhausted: {str(e)}")
                    raise

        raise last_error

    def get_stats(self) -> dict:
        """Get current rate limiter statistics"""
        self._cleanup_daily_requests()
        return {
            "tokens_available": self.tokens,
            "burst_size": self.burst_size,
            "rpm": self.rpm,
            "rpd": self.rpd,
            "daily_requests_used": len(self.daily_requests),
            "daily_requests_remaining": self.rpd - len(self.daily_requests) if self.rpd else None,
            "time_until_daily_reset": self._time_until_daily_reset()
        }


class OpenRouterRateLimiter(RateLimiter):
    """
    Pre-configured rate limiter for OpenRouter API

    Configurations:
    - Free tier: 20 RPM, 50 RPD
    - Paid tier (< $10): 20 RPM, 50 RPD
    - Paid tier (≥ $10): 20 RPM, 1000 RPD
    """

    def __init__(self, tier: str = "free"):
        """
        Initialize OpenRouter rate limiter

        Args:
            tier: "free", "paid_basic", or "paid_premium"
        """
        if tier == "free":
            super().__init__(
                requests_per_minute=20,
                requests_per_day=50,
                burst_size=20
            )
        elif tier == "paid_basic":
            super().__init__(
                requests_per_minute=20,
                requests_per_day=50,
                burst_size=20
            )
        elif tier == "paid_premium":
            super().__init__(
                requests_per_minute=20,
                requests_per_day=1000,
                burst_size=20
            )
        else:
            raise ValueError(f"Invalid tier: {tier}. Use 'free', 'paid_basic', or 'paid_premium'")

        logger.info(f"OpenRouter rate limiter initialized: tier={tier}")


# Global rate limiter instance
_global_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance"""
    global _global_limiter

    if _global_limiter is None:
        from src.config import settings

        # Determine tier from config
        tier = getattr(settings, 'openrouter_tier', 'free')
        _global_limiter = OpenRouterRateLimiter(tier=tier)

    return _global_limiter


def reset_rate_limiter():
    """Reset global rate limiter (useful for testing)"""
    global _global_limiter
    _global_limiter = None
