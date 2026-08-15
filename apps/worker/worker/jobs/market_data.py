"""Daily candle refresh job (§16).

Refreshes only instruments that are actually needed — Bot Universe members and
scanner-eligible instruments — rather than the whole catalogue. Refreshing
everything nightly is the fastest way to exhaust a free tier and is the reason
§4 specifies a rotation.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import redis
import structlog
from app.config import get_settings
from app.data.edgar import EDGARClient
from app.data.factory import resolve_provider
from app.data.types import ProviderQuotaExceededError
from app.db import session_scope
from app.models.enums import ProviderKind
from app.models.instrument import Instrument, MarketDataMapping
from app.services.ingestion import IngestionService
from app.services.insider import InsiderIngestionService
from sqlalchemy import or_, select

from worker.app import app
from worker.locks import LockNotAcquiredError, distributed_lock
from worker.runner import run_job

log = structlog.get_logger(__name__)

#: Ceiling per run. Bounds both runtime and provider spend.
#:
#: Sized against the batched fetch, which requests ~50 symbols per call: 5,000
#: instruments is ~100 provider requests, the same spend that previously bought
#: 100 instruments one at a time. Against a ~12.8k catalogue that is a complete
#: lap every three nights instead of every four months — and the scanner ranks
#: on stored candles, so the lap time *is* the age of the prices it ranks on.
#:
#: Not simply "all of them": a ceiling keeps one run's failure bounded, and
#: leaves headroom under yfinance's unpublished rate limits. Raise it if the
#: nightly job comfortably finishes.
DEFAULT_MAX_INSTRUMENTS = 5000


def _redis() -> redis.Redis:
    return redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6380/0"))


async def _refresh(provider_kind: ProviderKind, limit: int) -> dict[str, Any]:
    settings = get_settings()
    provider = resolve_provider(provider_kind, settings)

    try:
        async with session_scope() as session:
            # Only instruments with an active mapping for this provider can be
            # ingested at all; the join avoids waking up rows that would only be
            # skipped.
            result = await session.execute(
                select(Instrument)
                .join(MarketDataMapping, MarketDataMapping.instrument_id == Instrument.id)
                .where(
                    MarketDataMapping.provider == provider_kind,
                    MarketDataMapping.is_active.is_(True),
                    Instrument.suspended_at.is_(None),
                    or_(
                        Instrument.is_bot_universe.is_(True),
                        Instrument.is_scanner_eligible.is_(True),
                    ),
                    # Symbols resting after repeated empty responses. Roughly a
                    # fifth of this catalogue is delisted or absent from the
                    # provider and always will be; without this they take a slot
                    # in every rotation and the sweep spends that share of its
                    # budget re-learning it. They are retried, just not nightly.
                    or_(
                        MarketDataMapping.retry_after.is_(None),
                        MarketDataMapping.retry_after <= datetime.now(UTC),
                    ),
                )
                # Bot Universe first, unconditionally: those are the names the
                # risk engine prices open positions against, and sizing or
                # stopping out on a stale bar is the failure this ordering
                # exists to prevent. They are few (tens), so they cost little of
                # the batch and are refreshed every run rather than once per
                # sweep.
                #
                # Then this job's own cursor — never the scanner's. Ordering by a
                # column this job does not write means the queue never advances;
                # see `Instrument.last_refresh_attempt_at`.
                .order_by(
                    Instrument.is_bot_universe.desc(),
                    Instrument.last_refresh_attempt_at.asc().nulls_first(),
                )
                .limit(limit)
            )
            instruments = list(result.scalars().unique().all())

            if not instruments:
                return {"instruments": 0, "candles_written": 0, "note": "nothing mapped yet"}

            # Batched: one request per ~50 symbols rather than one per symbol.
            # The per-instrument path is still correct and still used elsewhere;
            # it is simply the wrong tool for a nightly sweep of thousands.
            results = await IngestionService(session).refresh_many_batched(instruments, provider)

            # Stamp what was *attempted*, so a symbol that returns nothing still
            # moves to the back of the queue rather than being retried forever.
            #
            # Driven off `results` rather than `instruments`: `ingest_many` stops
            # early when the provider budget is exhausted, and the instruments it
            # never reached must stay unstamped or they would be skipped for a
            # whole cycle on the strength of a call that never happened.
            attempted = {r.instrument_id for r in results}
            now = datetime.now(UTC)
            for instrument in instruments:
                if instrument.id in attempted:
                    instrument.last_refresh_attempt_at = now

            written = sum(r.candles_written for r in results)
            failed = [r for r in results if r.errors]
            skipped = [r for r in results if r.skipped_reason]

            return {
                "instruments": len(instruments),
                "processed": len(results),
                "candles_written": written,
                "failed": len(failed),
                "skipped": len(skipped),
                # Below `processed` when the budget ran out mid-batch; those
                # instruments keep their place at the front of the queue.
                "attempted": len(attempted),
            }
    finally:
        await provider.close()


@app.task(bind=True, name="worker.jobs.market_data.refresh_daily_candles", max_retries=2)
def refresh_daily_candles(  # type: ignore[no-untyped-def]
    self, provider: str = "yfinance", limit: int = DEFAULT_MAX_INSTRUMENTS
) -> dict[str, Any]:
    """Incrementally refresh daily candles.

    Each instrument re-requests only a short overlapping tail (§4), so this is
    cheap after the first backfill and safe to repeat.
    """
    try:
        provider_kind = ProviderKind(provider)
    except ValueError:
        log.error("job.refresh_daily_candles.unknown_provider", provider=provider)
        raise

    try:
        with distributed_lock(_redis(), "refresh_daily_candles", ttl_seconds=1800):
            result = run_job(_refresh(provider_kind, limit))
            log.info("job.refresh_daily_candles.completed", **result)
            return result
    except LockNotAcquiredError:
        log.info("job.refresh_daily_candles.skipped", reason="already running")
        return {"skipped": True, "reason": "another worker holds the lock"}
    except ProviderQuotaExceededError as exc:
        # Budget exhaustion is an expected end-state, not a fault. Retrying
        # would spend the reserve that exists for open positions (§4).
        log.warning("job.refresh_daily_candles.quota_exhausted", error=str(exc))
        return {"skipped": True, "reason": "provider budget exhausted"}
    except Exception as exc:
        log.exception("job.refresh_daily_candles.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=300 * (2**self.request.retries)) from exc


@app.task(bind=True, name="worker.jobs.market_data.ingest_insider_filings", max_retries=2)
def ingest_insider_filings(self, limit: int = 100) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Pull the newest SEC Form 4 filings into the local store (§4, §6).

    Feeds the scanner's insider factor. EDGAR is free and unauthenticated, so
    there is no budget to exhaust — the only limit is the SEC's request rate,
    which the client paces itself against.

    A missing contact address is a configuration gap, not a failure: EDGAR
    refuses anonymous automated readers, so the job skips rather than hammering
    the endpoint into a block.
    """
    settings = get_settings()
    contact = settings.edgar_contact_email
    if not contact:
        log.info("job.ingest_insider_filings.skipped", reason="EDGAR_CONTACT_EMAIL not set")
        return {"skipped": True, "reason": "EDGAR_CONTACT_EMAIL not configured"}

    async def _run() -> dict[str, Any]:
        client = EDGARClient(contact_email=str(contact))
        try:
            async with session_scope() as session:
                return await InsiderIngestionService(session).ingest_recent(client, limit=limit)
        finally:
            await client.close()

    try:
        with distributed_lock(_redis(), "ingest_insider_filings", ttl_seconds=1800):
            result = run_job(_run())
            log.info("job.ingest_insider_filings.completed", **result)
            return result
    except LockNotAcquiredError:
        return {"skipped": True, "reason": "another worker holds the lock"}
    except Exception as exc:
        log.exception("job.ingest_insider_filings.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=300 * (2**self.request.retries)) from exc
