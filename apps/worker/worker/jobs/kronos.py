"""Nightly Kronos forecasts over the strategy universe (§8).

**This job is deliberately absent from the beat schedule.** Every other job in
this package is scheduled in `worker.app`; this one is registered as a task and
never fired automatically, because running it needs torch — ~250-300MB resident
on import alone, against a 448MB worker on the deployment box. Importing it
there would make the whole worker unbootable whether or not anything used it.

So the arrangement is: this runs on a machine with the memory, on demand or from
a local scheduler, and writes `kronos_predictions`. The deployment box reads that
table and never imports torch. The strategy treats an absent forecast as missing
information rather than as an error, so a night this does not run costs precision
and not correctness.

    celery -A worker.app call worker.jobs.kronos.forecast_universe

Gated on `KRONOS_REPO_PATH` being set, exactly as the sentiment sweep is gated on
a Finnhub key: unset means the sweep simply does not run.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import redis
import structlog
from app.db import session_scope
from app.models.strategy import StrategyConfiguration
from app.services.kronos import KronosService
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from worker.app import app
from worker.locks import LockNotAcquiredError, distributed_lock
from worker.runner import run_job

log = structlog.get_logger(__name__)

#: Generous: a 32-path forecast is ~12s, so a 40-name universe is ~8 minutes,
#: and a cold start also downloads weights.
_LOCK_TTL_SECONDS = 3_600


def _redis() -> redis.Redis:
    return redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6380/0"))


async def _universe_ids(session: AsyncSession) -> list[uuid.UUID]:
    """Every instrument any enabled strategy is currently watching.

    The union across configurations rather than one strategy's list, because a
    forecast is a property of the instrument and not of whoever asked for it —
    two strategies sharing a name should share the one row, not generate it
    twice at twelve seconds a go.
    """
    configs = (
        (
            await session.execute(
                select(StrategyConfiguration).where(StrategyConfiguration.is_active.is_(True))
            )
        )
        .scalars()
        .all()
    )
    ids: set[uuid.UUID] = set()
    for config in configs:
        universe = config.universe or {}
        for raw in list(universe.get("instrument_ids") or []) + list(
            (universe.get("weights") or {}).keys()
        ):
            try:
                ids.add(uuid.UUID(str(raw)))
            except (ValueError, TypeError):
                continue
    return sorted(ids)


async def _forecast(limit: int | None) -> dict[str, Any]:
    async with session_scope() as session:
        instrument_ids = await _universe_ids(session)
        if not instrument_ids:
            return {"attempted": 0, "recorded": 0, "reason": "no active universe"}
        return await KronosService(session).forecast_universe(instrument_ids, limit=limit)


@app.task(bind=True, name="worker.jobs.kronos.forecast_universe", max_retries=1)
def forecast_universe(self, limit: int | None = None) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Forecast every instrument in the active strategy universe."""
    from app.config import get_settings

    if not get_settings().kronos_repo_path:
        return {"skipped": True, "reason": "KRONOS_REPO_PATH is not set"}
    try:
        with distributed_lock(_redis(), "kronos_forecast", ttl_seconds=_LOCK_TTL_SECONDS):
            result = run_job(_forecast(limit))
            log.info("job.kronos.completed", **result)
            return result
    except LockNotAcquiredError:
        return {"skipped": True, "reason": "another worker holds the lock"}
    except Exception as exc:
        log.exception("job.kronos.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=1_800) from exc
