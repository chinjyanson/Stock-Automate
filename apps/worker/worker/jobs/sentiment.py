"""Daily news-sentiment sweep (§4, §6).

Reads a week of headlines per instrument, scores their tone with the vendored
Loughran-McDonald lexicon, and stores one row per instrument per day. The
scanner and the risk engine then read the table, so neither ever waits on a news
feed — the same store-only discipline the regime and index-options readings keep.

Two things make this job different from the others:

  * **It is the first metered caller.** `ProviderBudget` has existed since Phase
    1 with a daily Postgres ledger, a Redis per-minute limiter and a priority
    reserve, and nothing has ever used it. Finnhub's free tier is 60 calls a
    minute, one call per instrument, which is exactly the shape it was written
    for. A credit is spent *before* the call, so a crash over-counts rather than
    under-counts.

  * **A spent budget ends the pass cleanly.** Running out of credit is a normal
    outcome, not a failure: the sweep stops, reports how far it got, and the
    instruments it did not reach simply keep yesterday's reading until it ages
    out. Retrying would spend a budget that is already gone.

Ordered by least-recently-scanned so successive passes rotate through the
universe rather than re-reading the same head of the list every night.
"""

from __future__ import annotations

import os
from typing import Any

import redis
import structlog
from app.config import get_settings
from app.data.budget import ProviderBudget, RequestPriority
from app.data.finnhub import FinnhubClient, FinnhubPremiumRequiredError
from app.data.types import ProviderQuotaExceededError
from app.db import session_scope
from app.models.enums import ProviderKind
from app.models.instrument import Instrument, MarketDataMapping
from app.services.sentiment import SentimentService
from sqlalchemy import select

from worker.app import app
from worker.locks import LockNotAcquiredError, distributed_lock
from worker.runner import run_job

log = structlog.get_logger(__name__)

#: Instruments per pass. One or two provider calls each, so this is the knob
#: that keeps the sweep inside the daily budget.
BATCH_LIMIT = 400


def _redis() -> redis.Redis:
    return redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6380/0"))


async def _sweep(limit: int) -> dict[str, Any]:
    settings = get_settings()
    key = settings.finnhub_api_key
    if key is None or not key.get_secret_value():
        # Not an error. Without a key the signal is simply unavailable, and both
        # consumers already treat that as "no information" rather than bad news.
        return {"skipped": True, "reason": "FINNHUB_API_KEY is not configured"}

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

        budget = ProviderBudget(
            session,
            _redis(),
            provider=ProviderKind.FINNHUB,
            daily_operational_limit=settings.finnhub_daily_operational_limit,
            emergency_reserve=settings.finnhub_daily_emergency_reserve,
            per_minute_limit=settings.finnhub_per_minute_limit,
        )
        service = SentimentService(session)

        scored = 0
        empty = 0
        budget_exhausted = False
        client = FinnhubClient(api_key=key.get_secret_value(), base_url=settings.finnhub_base_url)
        try:
            for instrument_id, symbol in rows:
                try:
                    # Background priority: a news reading must never draw on the
                    # reserve that exists to let an open live position be
                    # checked. Sentiment going stale costs a signal; a stranded
                    # position costs money.
                    await budget.consume(RequestPriority.BACKGROUND_BACKFILL)
                except ProviderQuotaExceededError as exc:
                    log.info("job.sentiment.budget_spent", reason=str(exc))
                    budget_exhausted = True
                    break

                try:
                    snapshot = await service.ingest_symbol(client, symbol, instrument_id)
                except FinnhubPremiumRequiredError as exc:
                    # The key is invalid or unentitled — every subsequent symbol
                    # will fail identically, so stop rather than burning the
                    # whole budget discovering it 400 more times.
                    log.error("job.sentiment.not_entitled", error=str(exc))
                    await budget.record_failure()
                    break
                except Exception as exc:
                    # Coverage is thin outside the US; one symbol without news
                    # must not end the sweep.
                    log.warning("job.sentiment.symbol_failed", symbol=symbol, error=str(exc))
                    await budget.record_failure()
                    continue

                if snapshot is None:
                    empty += 1
                else:
                    scored += 1
        finally:
            await client.close()

        return {
            "instruments": len(rows),
            "scored": scored,
            "no_headlines": empty,
            "budget_exhausted": budget_exhausted,
            "budget_remaining": await budget.remaining(),
        }


@app.task(bind=True, name="worker.jobs.sentiment.sweep_sentiment", max_retries=1)
def sweep_sentiment(self, limit: int = BATCH_LIMIT) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Refresh stored news tone for the scanner-eligible universe."""
    try:
        with distributed_lock(_redis(), "sweep_sentiment", ttl_seconds=3600):
            result = run_job(_sweep(limit))
            log.info("job.sentiment.completed", **result)
            return result
    except LockNotAcquiredError:
        return {"skipped": True, "reason": "another worker holds the lock"}
    except Exception as exc:
        log.exception("job.sentiment.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=900 * (2**self.request.retries)) from exc
