"""Nightly crash-overlay measurement (§9).

Refreshes the feature history, refits the detector from it and records the
exposure the overlay implies. Advisory only — nothing here places an order.

The whole span is refetched and rescored every night rather than only the tail,
which sounds wasteful and is the cheap option: the series are small, several are
revised after publication, and every day's probability depends only on its own
past, so a full replay is both idempotent and self-healing. A night the worker
did not run costs nothing but the gap it then fills.

Locked and long-ttl'd because the refit walks two decades of history. It runs
after the daily candle refresh so the index close it reads is the settled one.
"""

from __future__ import annotations

import os
from typing import Any

import redis
import structlog
from app.db import session_scope
from app.services.crash_overlay import CrashOverlayService

from worker.app import app
from worker.locks import LockNotAcquiredError, distributed_lock
from worker.runner import run_job

log = structlog.get_logger(__name__)


def _redis() -> redis.Redis:
    return redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6380/0"))


async def _measure() -> dict[str, Any]:
    async with session_scope() as session:
        service = CrashOverlayService(session)
        days = await service.refresh()
        reading = await service.evaluate()
    if reading is None:
        return {"days": days, "evaluated": False, "reason": "not enough history yet"}
    return {
        "days": days,
        "evaluated": True,
        "as_of": str(reading.as_of),
        "probability": reading.probability,
        "trigger": reading.trigger,
        "warning": reading.is_warning,
        "target_exposure": reading.target_exposure,
        "reason": reading.reason,
    }


@app.task(bind=True, name="worker.jobs.crash_overlay.measure_crash_overlay", max_retries=2)
def measure_crash_overlay(self) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Refresh features, refit the detector, record the target exposure."""
    try:
        with distributed_lock(_redis(), "measure_crash_overlay", ttl_seconds=3600):
            result = run_job(_measure())
            log.info("job.crash_overlay.completed", **result)
            return result
    except LockNotAcquiredError:
        return {"skipped": True, "reason": "another worker holds the lock"}
    except Exception as exc:
        log.exception("job.crash_overlay.failed", error=str(exc))
        raise
