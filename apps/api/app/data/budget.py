"""Provider credit budgets (§4).

Free tiers are the binding constraint on this system, so spend is a
first-class, persisted resource rather than a hope.

Two limits, two stores, for different reasons:

  * The **daily** budget lives in PostgreSQL, incremented in the same
    transaction that authorises the request. A process crash between "decided
    to call" and "called" therefore over-counts by at most one, which is the
    safe direction. Keeping it in Redis would risk an eviction silently
    restoring hundreds of credits we have already spent.

  * The **per-minute** limit lives in Redis with a 60s expiry, because it is
    high-frequency, disposable, and needs to be shared across API and worker
    processes.

Reserves are not decoration. When the operational limit is reached we stop
issuing non-critical requests but retain the reserve, so an open live position
can still have its exit price checked. A budget that can strand a position is
worse than no budget.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import IntEnum

import redis.asyncio as aioredis
import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.types import ProviderQuotaExceededError
from app.models.enums import ProviderKind
from app.models.market_data import ProviderUsage

log = structlog.get_logger(__name__)


class RequestPriority(IntEnum):
    """Spend priority (§4). Lower value is more important.

    The ordering is not preference — it is the order in which the system stops
    doing things as the budget depletes. Everything at or above
    `CRITICAL_THRESHOLD` may draw on the emergency reserve.
    """

    OPEN_LIVE_POSITION = 1
    PROTECTIVE_STOP = 2
    CORE_STRATEGY_SIGNAL = 3
    APPROVAL_REVALIDATION = 4
    SCANNER_VERIFICATION = 5
    BACKGROUND_BACKFILL = 6


#: Priorities strictly above this may not touch the reserve.
CRITICAL_THRESHOLD = RequestPriority.PROTECTIVE_STOP


class BudgetDecision:
    __slots__ = ("allowed", "reason", "remaining", "used_reserve")

    def __init__(
        self, *, allowed: bool, reason: str, remaining: int, used_reserve: bool = False
    ) -> None:
        self.allowed = allowed
        self.reason = reason
        self.remaining = remaining
        self.used_reserve = used_reserve


class ProviderBudget:
    """Enforces one provider's daily and per-minute limits."""

    def __init__(
        self,
        session: AsyncSession,
        redis: aioredis.Redis | None,
        *,
        provider: ProviderKind,
        daily_operational_limit: int,
        emergency_reserve: int = 0,
        per_minute_limit: int | None = None,
    ) -> None:
        self._session = session
        self._redis = redis
        self._provider = provider
        self._daily_limit = daily_operational_limit
        self._reserve = emergency_reserve
        self._per_minute_limit = per_minute_limit
        self._log = log.bind(provider=str(provider))

    async def _get_or_create_usage(self, usage_date: date) -> ProviderUsage:
        """Fetch today's row, creating it atomically if absent.

        ON CONFLICT DO NOTHING rather than a check-then-insert: two workers
        starting at midnight would otherwise race and one would crash on the
        unique constraint.
        """
        stmt = (
            pg_insert(ProviderUsage)
            .values(
                provider=self._provider,
                usage_date=usage_date,
                requests_used=0,
                requests_failed=0,
                operational_limit=self._daily_limit,
                emergency_reserve=self._reserve,
            )
            .on_conflict_do_nothing(index_elements=["provider", "usage_date"])
        )
        await self._session.execute(stmt)

        result = await self._session.execute(
            select(ProviderUsage).where(
                ProviderUsage.provider == self._provider,
                ProviderUsage.usage_date == usage_date,
            )
        )
        usage = result.scalar_one()
        return usage

    async def check(self, priority: RequestPriority) -> BudgetDecision:
        """Would a request at `priority` be permitted right now? No spend."""
        usage = await self._get_or_create_usage(datetime.now(UTC).date())
        remaining = usage.operational_limit - usage.requests_used

        if remaining > 0:
            return BudgetDecision(allowed=True, reason="within budget", remaining=remaining)

        if priority <= CRITICAL_THRESHOLD:
            reserve_left = (usage.operational_limit + usage.emergency_reserve) - usage.requests_used
            if reserve_left > 0:
                return BudgetDecision(
                    allowed=True,
                    reason=f"drawing on emergency reserve for priority {priority.name}",
                    remaining=reserve_left,
                    used_reserve=True,
                )
            return BudgetDecision(
                allowed=False,
                reason="emergency reserve exhausted",
                remaining=0,
            )

        return BudgetDecision(
            allowed=False,
            reason=(
                f"daily operational limit of {usage.operational_limit} reached; "
                f"priority {priority.name} may not use the reserve"
            ),
            remaining=0,
        )

    async def _check_per_minute(self) -> bool:
        """Per-minute gate. Fails *closed* when Redis is unavailable.

        An unreachable Redis means we cannot know the recent rate. Guessing
        "probably fine" is how a free tier gets suspended, so we decline.
        """
        if self._per_minute_limit is None:
            return True
        if self._redis is None:
            return True  # Not configured for per-minute limiting at all.

        bucket = datetime.now(UTC).strftime("%Y%m%d%H%M")
        key = f"provider_budget:{self._provider}:minute:{bucket}"
        try:
            count = await self._redis.incr(key)
            if count == 1:
                # Expire slightly beyond the window so a slow round-trip near
                # the boundary cannot drop the counter early.
                await self._redis.expire(key, 90)
            return int(count) <= self._per_minute_limit
        except Exception as exc:
            self._log.error("budget.redis_unavailable_failing_closed", error=str(exc))
            return False

    async def consume(self, priority: RequestPriority, *, count: int = 1) -> BudgetDecision:
        """Authorise and record `count` requests, or raise.

        Call this *before* the request. The spend is committed by the caller's
        transaction; over-counting on a subsequent failure is deliberate, since
        the provider counted the call whether or not we liked the answer.
        """
        decision = await self.check(priority)
        if not decision.allowed:
            self._log.warning("budget.denied", priority=priority.name, reason=decision.reason)
            raise ProviderQuotaExceededError(f"{self._provider}: {decision.reason}")

        if not await self._check_per_minute():
            raise ProviderQuotaExceededError(
                f"{self._provider}: per-minute limit of {self._per_minute_limit} reached"
            )

        usage = await self._get_or_create_usage(datetime.now(UTC).date())
        usage.requests_used += count
        usage.last_request_at = datetime.now(UTC)
        await self._session.flush()

        if decision.used_reserve:
            self._log.warning(
                "budget.reserve_drawn", priority=priority.name, remaining=decision.remaining
            )
        return decision

    async def record_failure(self) -> None:
        """Note a failed call. The credit is still spent."""
        usage = await self._get_or_create_usage(datetime.now(UTC).date())
        usage.requests_failed += 1
        await self._session.flush()

    async def remaining(self) -> int:
        usage = await self._get_or_create_usage(datetime.now(UTC).date())
        return usage.remaining
