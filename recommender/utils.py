"""Thread-safe caching and retry utilities for the recommender pipeline.

Provides:
- SafeCache: Thread-safe LRU-like cache with TTL
- retry: Decorator/function for retrying transient failures
- CircuitBreaker: Prevents cascading failures
"""

from __future__ import annotations

import functools
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class SafeCache:
    """Thread-safe cache with optional TTL and LRU eviction."""

    def __init__(self, maxsize: int = 128, ttl: float = 0):
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._cache:
                return None
            timestamp, value = self._cache[key]
            if self._ttl > 0 and time.time() - timestamp > self._ttl:
                del self._cache[key]
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.time(), value)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Decorator: retry a function on transient failure with exponential backoff."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error = None
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            raise last_error  # type: ignore
        return wrapper  # type: ignore
    return decorator


class CircuitBreaker:
    """Prevents cascading failures by stopping calls after consecutive errors."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self._lock = threading.Lock()
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._failures = 0
        self._last_failure_time = 0.0
        self._state = "closed"  # closed, open, half-open

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._state == "open":
                if time.time() - self._last_failure_time > self._reset_timeout:
                    self._state = "half-open"
                    return False
                return True
            return False

    def success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "closed"

    def failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()
            if self._failures >= self._failure_threshold:
                self._state = "open"


def safe_globals() -> dict:
    """Create a thread-local storage dict for module-level globals."""
    return threading.local().__dict__