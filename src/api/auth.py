"""API-key authentication and per-client request budgets.

Scoring endpoints cost real compute, so they are gated and budgeted. ``/health``
and ``/ready`` stay open: liveness probes must not need credentials.

Configuration is by environment variable:

``API_KEYS``
    Comma-separated list of accepted keys. When it is set, every scoring request
    must present one in the ``X-API-Key`` header. When it is unset the service
    runs in **local-open** mode: requests are accepted without a key and the
    active mode is reported by ``/`` and ``/model-info`` so it can never be an
    invisible default. Open mode is only safe because the default bind address
    is loopback; set ``API_KEYS`` before exposing the service elsewhere.

``RATE_LIMITER_BACKEND``
    ``memory`` (default) or ``redis``. When ``redis``, requires ``REDIS_URL``.

Rate limiting is always on, keyed by API key when authenticated and by client
address otherwise. Budgets are per process with ``memory`` backend; with
``redis`` they are shared across workers.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

from src.config import (
    RATE_LIMIT_REQUESTS_PER_MINUTE,
    RATE_LIMIT_ROWS_PER_MINUTE,
    RATE_LIMIT_WINDOW_SECONDS,
)

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"

# Declared as a real security scheme so /docs renders an Authorize button and
# the requirement is visible in the OpenAPI document. auto_error=False keeps the
# 401 message ours instead of FastAPI's default.
api_key_scheme = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    description="Required on scoring endpoints when the service sets API_KEYS.",
)

AUTH_MODE_API_KEY = "api-key"
AUTH_MODE_OPEN = "local-open"


def configured_api_keys() -> frozenset[str]:
    """Accepted API keys from ``API_KEYS`` (comma-separated, empty entries ignored)."""
    raw = os.environ.get("API_KEYS", "")
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


def auth_mode() -> str:
    """Which mode the service is actually running in."""
    return AUTH_MODE_API_KEY if configured_api_keys() else AUTH_MODE_OPEN


def _key_matches(candidate: str, keys: frozenset[str]) -> bool:
    """Constant-time comparison against every configured key."""
    matched = False
    for key in keys:
        matched |= hmac.compare_digest(candidate, key)
    return matched


@dataclass
class _Window:
    started: float
    used: int


class BaseRateLimiter(ABC):
    """Abstract rate limiter interface."""

    @abstractmethod
    def check(self, identity: str, cost: int = 1) -> float | None:
        """Consume `cost` units. Returns None when allowed, else seconds to wait."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Drop all counters (used by tests)."""
        ...


class MemoryRateLimiter(BaseRateLimiter):
    """Fixed-window counter per identity (in-memory, per-process)."""

    def __init__(
        self,
        limit: int,
        window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
        max_identities: int = 4096,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_identities = max_identities
        self._windows: dict[str, _Window] = {}
        self._lock = threading.Lock()

    def check(self, identity: str, cost: int = 1) -> float | None:
        now = time.monotonic()
        with self._lock:
            window = self._windows.get(identity)
            if window is None or now - window.started >= self.window_seconds:
                if window is None and len(self._windows) >= self.max_identities:
                    self._sweep(now)
                window = _Window(started=now, used=0)
                self._windows[identity] = window
            if window.used + cost > self.limit:
                return max(0.0, self.window_seconds - (now - window.started))
            window.used += cost
            return None

    def _sweep(self, now: float) -> None:
        stale = [
            k for k, w in self._windows.items() if now - w.started >= self.window_seconds
        ]
        for key in stale:
            self._windows.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()


class RedisRateLimiter(BaseRateLimiter):
    """Fixed-window counter per identity backed by Redis (shared across workers)."""

    def __init__(
        self,
        limit: int,
        window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
        redis_url: str | None = None,
        key_prefix: str = "ratelimit:",
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix
        self._redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._client: redis.Redis | None = None

    def _get_client(self) -> redis.Redis:
        if self._client is None:
            import redis
            self._client = redis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def check(self, identity: str, cost: int = 1) -> float | None:
        import time as _time
        client = self._get_client()
        now = _time.time()
        window_start = int(now // self.window_seconds) * self.window_seconds
        key = f"{self.key_prefix}{identity}:{int(window_start)}"

        # Use Lua script for atomic check-and-increment
        lua_script = """
        local current = redis.call('GET', KEYS[1])
        if current == false then
            current = 0
        else
            current = tonumber(current)
        end
        if current + tonumber(ARGV[1]) > tonumber(ARGV[2]) then
            return -1
        end
        local new_val = redis.call('INCRBY', KEYS[1], ARGV[1])
        if new_val == tonumber(ARGV[1]) then
            redis.call('EXPIRE', KEYS[1], ARGV[3])
        end
        return 0
        """
        try:
            script = client.register_script(lua_script)
            result = script(keys=[key], args=[cost, self.limit, int(self.window_seconds) + 1])
        except Exception as exc:
            logger.warning(f"Redis rate limiter error, failing open: {exc}")
            return None  # Fail open on Redis errors
        else:
            if result == -1:
                ttl = client.ttl(key)
                return max(0.0, float(ttl if ttl > 0 else self.window_seconds))
            return None

    def reset(self) -> None:
        # Not typically used with Redis; would require SCAN + DEL
        pass


def _create_limiter(limit: int) -> BaseRateLimiter:
    """Factory for rate limiter based on RATE_LIMITER_BACKEND env var."""
    backend = os.environ.get("RATE_LIMITER_BACKEND", "memory").lower()
    if backend == "redis":
        try:
            return RedisRateLimiter(limit)
        except ImportError:
            logger.warning("redis-py not installed, falling back to memory rate limiter")
            return MemoryRateLimiter(limit)
    return MemoryRateLimiter(limit)


# Backward compatibility alias for tests and existing code
RateLimiter = MemoryRateLimiter

request_limiter = _create_limiter(RATE_LIMIT_REQUESTS_PER_MINUTE)
row_limiter = _create_limiter(RATE_LIMIT_ROWS_PER_MINUTE)


def identity_for(request: Request, api_key: str | None) -> tuple[str, bool]:
    """Return the budget identity and whether a key authenticated this request.

    The key itself is never used as the identity: only a short digest of it, so
    logs and counters cannot be turned back into a credential.
    """
    if api_key is not None and _key_matches(api_key, configured_api_keys()):
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
        return f"key:{digest}", True
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}", False


def require_api_key(
    request: Request,
    api_key: str | None = Depends(api_key_scheme),
) -> str:
    """FastAPI dependency: authenticate and return the budget identity."""
    keys = configured_api_keys()
    if not keys:
        identity, _ = identity_for(request, None)
        return identity
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail=f"missing API key: send it in the {API_KEY_HEADER} header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if not _key_matches(api_key, keys):
        raise HTTPException(
            status_code=401,
            detail="invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    identity, _ = identity_for(request, api_key)
    return identity


def enforce_budget(identity: str, rows: int = 1) -> None:
    """Charge one request and `rows` rows against this identity's budget."""
    retry = row_limiter.check(identity, rows)
    if retry is not None:
        raise HTTPException(
            status_code=429,
            detail=(
                f"row budget exceeded ({RATE_LIMIT_ROWS_PER_MINUTE} rows per "
                f"{int(RATE_LIMIT_WINDOW_SECONDS)}s); retry in {retry:.0f}s"
            ),
            headers={"Retry-After": str(max(1, int(retry + 0.5)))},
        )
    retry = request_limiter.check(identity, 1)
    if retry is not None:
        raise HTTPException(
            status_code=429,
            detail=(
                f"request budget exceeded ({RATE_LIMIT_REQUESTS_PER_MINUTE} requests "
                f"per {int(RATE_LIMIT_WINDOW_SECONDS)}s); retry in {retry:.0f}s"
            ),
            headers={"Retry-After": str(max(1, int(retry + 0.5)))},
        )
