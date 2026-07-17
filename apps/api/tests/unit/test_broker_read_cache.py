"""Broker read cache behaviour.

The cache exists to stop the dashboard hammering Trading 212's tight rate
limits, so its important properties are: fresh reads skip the broker, transient
failures serve stale data instead of erroring, and a hard failure still
propagates. Each is pinned here.
"""

from __future__ import annotations

import asyncio

import pytest

from app.broker.read_cache import BrokerReadCache
from app.broker.types import BrokerAuthError, BrokerRateLimitError, BrokerUnavailableError


class _Counter:
    """A fetch function that records how many times it actually ran."""

    def __init__(self, value: object = "v", raises: Exception | None = None) -> None:
        self.calls = 0
        self._value = value
        self._raises = raises

    async def __call__(self) -> object:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._value


class TestFreshness:
    async def test_first_call_fetches(self) -> None:
        cache = BrokerReadCache()
        fetch = _Counter(value="account")
        result = await cache.get_or_fetch("k", ttl_seconds=60, fetch=fetch)
        assert result.value == "account"
        assert result.from_cache is False
        assert fetch.calls == 1

    async def test_second_call_within_ttl_is_served_from_cache(self) -> None:
        cache = BrokerReadCache()
        fetch = _Counter()
        await cache.get_or_fetch("k", ttl_seconds=60, fetch=fetch)
        result = await cache.get_or_fetch("k", ttl_seconds=60, fetch=fetch)
        assert result.from_cache is True
        # The broker was called exactly once, not twice.
        assert fetch.calls == 1

    async def test_expired_entry_refetches(self) -> None:
        cache = BrokerReadCache()
        fetch = _Counter()
        await cache.get_or_fetch("k", ttl_seconds=0, fetch=fetch)
        await asyncio.sleep(0.01)
        await cache.get_or_fetch("k", ttl_seconds=0, fetch=fetch)
        assert fetch.calls == 2

    async def test_distinct_keys_do_not_share(self) -> None:
        cache = BrokerReadCache()
        account = _Counter(value="account")
        positions = _Counter(value="positions")
        a = await cache.get_or_fetch("account", ttl_seconds=60, fetch=account)
        p = await cache.get_or_fetch("positions", ttl_seconds=60, fetch=positions)
        assert a.value == "account"
        assert p.value == "positions"


class TestStaleOnFailure:
    @pytest.mark.parametrize(
        "error",
        [BrokerRateLimitError("429"), BrokerUnavailableError("down")],
    )
    async def test_transient_failure_serves_stale_value(self, error: Exception) -> None:
        """A 429 or outage after a good read must not break the dashboard."""
        cache = BrokerReadCache()
        good = _Counter(value="last-good")
        await cache.get_or_fetch("k", ttl_seconds=0, fetch=good)
        await asyncio.sleep(0.01)

        failing = _Counter(raises=error)
        result = await cache.get_or_fetch("k", ttl_seconds=0, fetch=failing)

        assert result.value == "last-good"
        assert result.is_stale is True
        assert result.from_cache is True

    async def test_transient_failure_with_no_prior_value_propagates(self) -> None:
        """There is nothing to serve, so the error must surface."""
        cache = BrokerReadCache()
        failing = _Counter(raises=BrokerRateLimitError("429"))
        with pytest.raises(BrokerRateLimitError):
            await cache.get_or_fetch("k", ttl_seconds=60, fetch=failing)

    async def test_non_transient_failure_always_propagates(self) -> None:
        """A rejected credential must never be papered over with stale data."""
        cache = BrokerReadCache()
        await cache.get_or_fetch("k", ttl_seconds=0, fetch=_Counter(value="old"))
        await asyncio.sleep(0.01)

        failing = _Counter(raises=BrokerAuthError("401"))
        with pytest.raises(BrokerAuthError):
            await cache.get_or_fetch("k", ttl_seconds=0, fetch=failing)


class TestConcurrency:
    async def test_concurrent_misses_collapse_to_one_fetch(self) -> None:
        """The dashboard fires account and positions in parallel, and browsers
        open several tabs. A herd of simultaneous misses on one key must not
        multiply into a burst of broker calls that trips the very limit the
        cache exists to avoid."""
        cache = BrokerReadCache()

        slow_calls = 0

        async def slow_fetch() -> str:
            nonlocal slow_calls
            slow_calls += 1
            await asyncio.sleep(0.05)
            return "value"

        results = await asyncio.gather(
            *(cache.get_or_fetch("k", ttl_seconds=60, fetch=slow_fetch) for _ in range(10))
        )

        assert all(r.value == "value" for r in results)
        # One real fetch; the other nine waited on the lock and got the cached
        # result.
        assert slow_calls == 1


class TestInvalidation:
    async def test_invalidate_forces_a_refetch(self) -> None:
        cache = BrokerReadCache()
        fetch = _Counter()
        await cache.get_or_fetch("k", ttl_seconds=60, fetch=fetch)
        cache.invalidate("k")
        await cache.get_or_fetch("k", ttl_seconds=60, fetch=fetch)
        assert fetch.calls == 2
