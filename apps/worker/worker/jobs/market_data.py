"""Daily candle refresh job (§16).

Refreshes only instruments that are actually needed — Bot Universe members and
scanner-eligible instruments — rather than the whole catalogue. Refreshing
everything nightly is the fastest way to exhaust a free tier and is the reason
§4 specifies a rotation.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import redis
import structlog
from app.config import get_settings
from app.data.factory import resolve_provider
from app.data.types import ProviderQuotaExceededError
from app.db import session_scope
from app.models.enums import ProviderKind
from app.models.instrument import Instrument, MarketDataMapping
from app.services.ingestion import IngestionService
from sqlalchemy import or_, select

from worker.app import app
from worker.locks import LockNotAcquiredError, distributed_lock

log = structlog.get_logger(__name__)

#: Ceiling per run. Bounds both runtime and provider spend; the rotation in
#: Phase 2 will make the selection smarter than "first N".
DEFAULT_MAX_INSTRUMENTS = 100


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
                )
                .order_by(Instrument.last_scanned_at.asc().nulls_first())
                .limit(limit)
            )
            instruments = list(result.scalars().unique().all())

            if not instruments:
                return {"instruments": 0, "candles_written": 0, "note": "nothing mapped yet"}

            results = await IngestionService(session).ingest_many(instruments, provider)

            written = sum(r.candles_written for r in results)
            failed = [r for r in results if r.errors]
            skipped = [r for r in results if r.skipped_reason]

            return {
                "instruments": len(instruments),
                "processed": len(results),
                "candles_written": written,
                "failed": len(failed),
                "skipped": len(skipped),
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
            result = asyncio.run(_refresh(provider_kind, limit))
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
