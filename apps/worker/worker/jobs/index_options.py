"""Daily index-options reading (§9).

Polls one S&P option chain and stores what it was pricing: dealer gamma, the
25-delta skew, and at-the-money implied volatility.

The reason this is a job and not something the strategy computes on demand is
that an option chain has no history. A provider will say what is priced right
now and nothing at all about last month, so the only way to ever have a series
is to record one, daily, starting from the day this begins running. Anything
wanting a trend in these numbers has to wait for it to accumulate.

Scheduled just before the regime measurement, so a day's index picture is
complete before anything sizes a position against it.
"""

from __future__ import annotations

import os
from typing import Any

import redis
import structlog
from app.db import session_scope
from app.services.index_options import IndexOptionsService

from worker.app import app
from worker.locks import LockNotAcquiredError, distributed_lock
from worker.runner import run_job

log = structlog.get_logger(__name__)


def _redis() -> redis.Redis:
    return redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6380/0"))


async def _measure() -> dict[str, Any]:
    async with session_scope() as session:
        snapshot = await IndexOptionsService(session).measure_and_record()
        if snapshot is None:
            # No usable chain today. Not an error — the strategy treats a
            # missing reading as "no opinion" and simply does not trade on it.
            return {"recorded": False}
        return {
            "recorded": True,
            "symbol": snapshot.symbol,
            "gamma_exposure": float(snapshot.gamma_exposure)
            if snapshot.gamma_exposure is not None
            else None,
            "skew_25delta": float(snapshot.skew_25delta)
            if snapshot.skew_25delta is not None
            else None,
            "contracts_used": snapshot.contracts_used,
        }


@app.task(bind=True, name="worker.jobs.index_options.measure_index_options", max_retries=2)
def measure_index_options(self) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Read and store today's index option-chain snapshot."""
    try:
        with distributed_lock(_redis(), "measure_index_options", ttl_seconds=900):
            result = run_job(_measure())
            log.info("job.index_options.completed", **result)
            return result
    except LockNotAcquiredError:
        return {"skipped": True, "reason": "another worker holds the lock"}
    except Exception as exc:
        log.exception("job.index_options.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=600 * (2**self.request.retries)) from exc
