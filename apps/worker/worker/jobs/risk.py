"""Risk, stop, reconciliation and EOD jobs (§9, §10, §16).

These are the scheduled half of Phase 3. Stop monitoring and reconciliation run
often and cheaply; the EOD summary runs once after the close. All three operate
on the internal paper venue, which is DB-backed, so they take a session and
resolve the broker against it.

Idempotency: stop management skips closed intents and only ratchets stops upward;
reconciliation transitions are convergent; the EOD summary upserts per (broker,
day). None of these can reach a live broker, so no distributed lock is needed —
the paper venue is local.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from app.broker.factory import resolve_broker
from app.db import session_scope
from app.models.enums import BrokerKind
from app.risk.config import load_active_risk_config
from app.risk.stops import StopService
from app.services.eod import EODSummaryService
from app.services.reconciliation import ReconciliationService

from worker.app import app

log = structlog.get_logger(__name__)


async def _monitor_stops() -> dict[str, int]:
    async with session_scope() as session:
        broker = resolve_broker(BrokerKind.INTERNAL_PAPER, session=session)
        config = await load_active_risk_config(session)
        result = await StopService(session).manage(broker, config)
        return result


@app.task(bind=True, name="worker.jobs.risk.monitor_stops", max_retries=2)
def monitor_stops(self) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Trigger, trail and time-stop open paper positions (§9)."""
    try:
        result = asyncio.run(_monitor_stops())
        if any(result.values()):
            log.info("job.monitor_stops.completed", **result)
        return result
    except Exception as exc:
        log.exception("job.monitor_stops.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=120 * (2**self.request.retries)) from exc


async def _reconcile() -> dict[str, Any]:
    async with session_scope() as session:
        broker = resolve_broker(BrokerKind.INTERNAL_PAPER, session=session)
        result = await ReconciliationService(session).run(broker)
        return {
            "is_clean": result.is_clean,
            "discrepancies": len(result.discrepancies),
            "positions_checked": result.positions_checked,
        }


@app.task(bind=True, name="worker.jobs.risk.reconcile_broker", max_retries=2)
def reconcile_broker(self) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Reconcile the paper venue; halt on divergence, resolve on clean (§10)."""
    try:
        result = asyncio.run(_reconcile())
        log.info("job.reconcile_broker.completed", **result)
        return result
    except Exception as exc:
        log.exception("job.reconcile_broker.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=120 * (2**self.request.retries)) from exc


async def _generate_eod() -> dict[str, Any]:
    async with session_scope() as session:
        broker = resolve_broker(BrokerKind.INTERNAL_PAPER, session=session)
        summary = await EODSummaryService(session).generate(broker)
        return {
            "date": summary.summary_date.isoformat(),
            "equity": str(summary.equity),
            "open_positions": summary.open_positions,
        }


@app.task(bind=True, name="worker.jobs.risk.generate_eod_summary", max_retries=2)
def generate_eod_summary(self) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Persist the end-of-day paper account summary (§16)."""
    try:
        result = asyncio.run(_generate_eod())
        log.info("job.generate_eod_summary.completed", **result)
        return result
    except Exception as exc:
        log.exception("job.generate_eod_summary.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=300 * (2**self.request.retries)) from exc
