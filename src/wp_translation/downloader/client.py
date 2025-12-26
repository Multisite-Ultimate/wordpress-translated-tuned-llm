"""Rate-limited HTTP client for translate.wordpress.org."""

import asyncio
import time
from typing import Optional

import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..utils.logging import get_logger

logger = get_logger(__name__)


class RateLimitedClient:
    """HTTP client with polite rate limiting for translate.wordpress.org.

    Implements rate limiting and automatic retries for robust downloading.
    """

    def __init__(
        self,
        requests_per_minute: int = 30,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: str = "WordPressTranslationBot/1.0 (research; polite)",
    ):
        """Initialize the rate-limited client.

        Args:
            requests_per_minute: Maximum requests per minute (default: 30)
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum retry attempts for failed requests
            user_agent: User-Agent header for requests
        """
        self.requests_per_minute = requests_per_minute
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.user_agent = user_agent

        self._min_interval = 60.0 / requests_per_minute
        self._last_request_time: float = 0
        self._lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "RateLimitedClient":
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure HTTP session is created."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
            )

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting by waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                wait_time = self._min_interval - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            self._last_request_time = time.monotonic()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def get(self, url: str) -> bytes:
        """Fetch URL content with rate limiting and retries.

        Args:
            url: URL to fetch

        Returns:
            Response content as bytes

        Raises:
            aiohttp.ClientError: If request fails after retries
            asyncio.TimeoutError: If request times out
        """
        await self._ensure_session()
        await self._apply_rate_limit()

        logger.debug(f"Fetching: {url}")

        async with self._session.get(url) as response:
            response.raise_for_status()
            content = await response.read()
            logger.debug(f"Fetched {len(content)} bytes from {url}")
            return content

    async def get_text(self, url: str) -> str:
        """Fetch URL content as text.

        Args:
            url: URL to fetch

        Returns:
            Response content as string
        """
        content = await self.get(url)
        return content.decode("utf-8")

    async def head(self, url: str) -> dict:
        """Perform HEAD request to get headers.

        Args:
            url: URL to check

        Returns:
            Response headers as dict
        """
        await self._ensure_session()
        await self._apply_rate_limit()

        async with self._session.head(url) as response:
            return dict(response.headers)
