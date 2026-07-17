"""Short-lived cache for broker read calls.

Trading 212 rate-limits each endpoint aggressively — the portfolio endpoint
returns 429 after roughly one call every few seconds. The dashboard reads
account and positions on every load, so without a cache a couple of refreshes
trip the limit and the UI breaks. (It did exactly that.)

Two behaviours matter here:

  * **Fresh-within-TTL reads skip the broker entirely.** Repeated loads and
    multiple tabs collapse to one broker call per window.

  * **A rate-limit or outage serves the last known value rather than erroring.**
    A stale account balance from 20 seconds ago is far more useful on a
    dashboard than a red banner, and it is honest as long as the UI can say the
    data is stale.

This is deliberately an in-process cache, sized for Phase 1. It is not shared
across worker processes, which is acceptable because the goal is to stop
*hammering*, not to be a distributed source of truth — that is Phase 3's
persisted snapshots. A per-key lock prevents a thundering herd when several
requests miss the same key at once (the dashboard fires account and positions in
parallel, and browsers open several tabs).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.broker.types import BrokerRateLimitError, BrokerUnavailableError


@dataclass(slots=True)
class CachedRead[T]:
    """A value plus whether it came from cache and how old it is."""

    value: T
    from_cache: bool
    age_seconds: float
    #: True when the underlying fetch failed transiently and a stale value was
    #: served instead. The caller should signal staleness to the user.
    is_stale: bool = False


@dataclass(slots=True)
class _Entry:
    value: object
    fetched_at: float


class BrokerReadCache:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def get_or_fetch[T](
        self,
        key: str,
        ttl_seconds: float,
        fetch: Callable[[], Awaitable[T]],
    ) -> CachedRead[T]:
        """Return a cached value if fresh, otherwise fetch and cache.

        On a transient broker failure (rate limit or unavailability) with a
        previously cached value present, the stale value is returned rather than
        raising. Any other error propagates — a rejected credential is not
        something to paper over with old data.
        """
        now = time.monotonic()
        entry = self._entries.get(key)

        if entry is not None and (now - entry.fetched_at) < ttl_seconds:
            return CachedRead(
                value=entry.value,  # type: ignore[arg-type]
                from_cache=True,
                age_seconds=now - entry.fetched_at,
            )

        async with self._lock_for(key):
            # Re-check after acquiring the lock: another coroutine may have
            # refreshed this key while we waited, and re-fetching would waste a
            # scarce broker call.
            now = time.monotonic()
            entry = self._entries.get(key)
            if entry is not None and (now - entry.fetched_at) < ttl_seconds:
                return CachedRead(
                    value=entry.value,  # type: ignore[arg-type]
                    from_cache=True,
                    age_seconds=now - entry.fetched_at,
                )

            try:
                value = await fetch()
            except (BrokerRateLimitError, BrokerUnavailableError):
                if entry is not None:
                    # Serve the stale value; the dashboard survives a 429.
                    return CachedRead(
                        value=entry.value,  # type: ignore[arg-type]
                        from_cache=True,
                        age_seconds=time.monotonic() - entry.fetched_at,
                        is_stale=True,
                    )
                raise

            self._entries[key] = _Entry(value=value, fetched_at=time.monotonic())
            return CachedRead(value=value, from_cache=False, age_seconds=0.0)

    def invalidate(self, key: str) -> None:
        """Drop a cached entry.

        Called after an action that changes broker state — placing or cancelling
        an order — so the next read reflects it immediately rather than after the
        TTL.
        """
        self._entries.pop(key, None)

    def clear(self) -> None:
        self._entries.clear()


#: Process-wide instance. A module singleton so the cache actually persists
#: across requests — the whole point is that a new request reuses a prior
#: request's fetch.
broker_read_cache = BrokerReadCache()
