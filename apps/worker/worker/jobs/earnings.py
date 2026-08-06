"""Weekly earnings-calendar sync, for the post-earnings drift signal (§6).

Weekly rather than daily, deliberately. Report dates move on a quarterly cycle,
so re-asking every day would spend a provider call per instrument to learn
nothing — and the *scoring* half of PEAD is pure-local anyway, computed from
candles already stored, so the daily scan stays store-only.

Scoped to the instruments the scanner actually ranks rather than the whole
catalogue: one call per instrument does not scale to a 15,000-name universe on
a free tier, and a name nobody is considering does not need a calendar.
"""

from __future__ import annotations

import os
from typing import Any

import redis
import structlog
from app.db import session_scope
from app.models.enums import ProviderKind
from app.models.instrument import Instrument, MarketDataMapping
from app.services.pead import PeadService
from sqlalchemy import select

from worker.app import app
from worker.locks import LockNotAcquiredError, distributed_lock
from worker.runner import run_job

log = structlog.get_logger(__name__)

#: How many instruments one pass covers. One provider call each, so this is the
#: knob that keeps the job inside a sensible rate budget.
BATCH_LIMIT = 300


def _redis() -> redis.Redis:
    return redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6380/0"))


async def _sync(limit: int) -> dict[str, Any]:
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(Instrument.id, MarketDataMapping.provider_symbol)
                .join(MarketDataMapping, MarketDataMapping.instrument_id == Instrument.id)
                .where(
                    MarketDataMapping.provider == ProviderKind.YFINANCE,
                    MarketDataMapping.is_signal_source.is_(True),
                    MarketDataMapping.is_active.is_(True),
                    Instrument.is_scanner_eligible.is_(True),
                    Instrument.suspended_at.is_(None),
                )
                .order_by(Instrument.last_scanned_at.desc().nullslast())
                .limit(limit)
            )
        ).all()

        service = PeadService(session)
        covered = 0
        events = 0
        for instrument_id, symbol in rows:
            try:
                written = await service.ingest_dates(symbol, instrument_id)
            except Exception as exc:
                # Coverage is thin outside the US; one symbol without a
                # calendar must not end the sweep.
                log.warning("job.earnings.symbol_failed", symbol=symbol, error=str(exc))
                continue
            if written:
                covered += 1
                events += written
        return {"instruments": len(rows), "with_events": covered, "events": events}


@app.task(bind=True, name="worker.jobs.earnings.sync_earnings_dates", max_retries=2)
def sync_earnings_dates(self, limit: int = BATCH_LIMIT) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Refresh recent earnings dates for the scanner-eligible universe."""
    try:
        with distributed_lock(_redis(), "sync_earnings_dates", ttl_seconds=3600):
            result = run_job(_sync(limit))
            log.info("job.earnings.completed", **result)
            return result
    except LockNotAcquiredError:
        return {"skipped": True, "reason": "another worker holds the lock"}
    except Exception as exc:
        log.exception("job.earnings.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=900 * (2**self.request.retries)) from exc
