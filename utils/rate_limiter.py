"""
Rate limiter for API calls
"""

import asyncio
import time
from typing import Dict, Optional
from asyncio import Semaphore

from utils.logger import logger


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_calls_per_second: float = 5.0):
        """
        Initialize rate limiter
        
        Args:
            max_calls_per_second: Maximum API calls per second
        """
        self.max_calls_per_second = max_calls_per_second
        self.interval = 1.0 / max_calls_per_second
        self.last_call_time = 0.0
        # Don't create lock and semaphore in __init__
        self._lock = None
        self._semaphore = None
        self._max_concurrent = int(max_calls_per_second)
        
    async def acquire(self):
        """
        Acquire permission to make an API call
        Ensures rate limiting is enforced
        """
        # Create lock and semaphore lazily
        if self._lock is None:
            self._lock = asyncio.Lock()
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        
        async with self._semaphore:
            async with self._lock:
                current_time = time.time()
                time_since_last = current_time - self.last_call_time
                
                if time_since_last < self.interval:
                    sleep_time = self.interval - time_since_last
                    await asyncio.sleep(sleep_time)
                
                self.last_call_time = time.time()
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass


class APIRateLimiters:
    """Manage rate limiters for different APIs"""
    
    def __init__(self):
        """Initialize rate limiters for different APIs"""
        # Anthropic Claude: 5 requests per second
        self.anthropic = RateLimiter(max_calls_per_second=5.0)
        
        # OpenAI: 10 requests per second
        self.openai = RateLimiter(max_calls_per_second=10.0)
        
        # General rate limiter for other APIs
        self.general = RateLimiter(max_calls_per_second=3.0)
        
        logger.info("API rate limiters initialized")
    
    def get_limiter(self, provider: str) -> RateLimiter:
        """
        Get rate limiter for specific provider
        
        Args:
            provider: API provider name ('anthropic', 'openai', etc.)
            
        Returns:
            Appropriate rate limiter
        """
        provider_lower = provider.lower()
        
        if 'anthropic' in provider_lower or 'claude' in provider_lower:
            return self.anthropic
        elif 'openai' in provider_lower or 'gpt' in provider_lower:
            return self.openai
        else:
            return self.general


