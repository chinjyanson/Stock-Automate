"""Risk, stop, reconciliation and EOD jobs (§9, §10, §16).

Stop monitoring and reconciliation run often; the EOD summary runs once after the
close. All three operate on **the active venue** — whichever the paper/live toggle
selects — so they resolve the broker per run rather than assuming one.

Because the venue is now a real broker, each job closes its client and treats
missing credentials as "skip", not "fail": a deployment without keys should stay
quiet rather than retry-loop.

Idempotency: stop management skips closed intents and only ratchets stops upward;
reconciliation transitions are convergent; the EOD summary upserts per (broker, day).
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from app.broker.factory import (
    BrokerNotConfiguredError,
    LiveTradingDisabledError,
    resolve_broker,
)
from app.db import session_scope
from app.models.enums import BrokerKind
from app.risk.config import load_active_risk_config
from app.risk.live_guard import LiveGuardService
from app.risk.stops import StopService
from app.services.eod import EODSummaryService
from app.services.reconciliation import ReconciliationService
from app.services.system_settings import active_broker_kind

from worker.app import app

log = structlog.get_logger(__name__)

#: Raised when the active venue has no usable credentials — a skip, not a fault.
_NOT_CONFIGURED = (LiveTradingDisabledError, BrokerNotConfiguredError)


async def _monitor_stops() -> dict[str, Any]:
    async with session_scope() as session:
        broker = resolve_broker(await active_broker_kind(session), session=session)
        try:
            config = await load_active_risk_config(session)
            return dict(await StopService(session).manage(broker, config))
        finally:
            await broker.close()


@app.task(bind=True, name="worker.jobs.risk.monitor_stops", max_retries=2)
def monitor_stops(self) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Trigger, trail and time-stop open positions on the active venue (§9)."""
    try:
        result = asyncio.run(_monitor_stops())
        if any(v for v in result.values() if isinstance(v, int)):
            log.info("job.monitor_stops.completed", **result)
        return result
    except _NOT_CONFIGURED as exc:
        log.info("job.monitor_stops.skipped", reason=str(exc))
        return {"skipped": True, "reason": "venue not configured"}
    except Exception as exc:
        log.exception("job.monitor_stops.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=120 * (2**self.request.retries)) from exc


async def _reconcile() -> dict[str, Any]:
    async with session_scope() as session:
        broker = resolve_broker(await active_broker_kind(session), session=session)
        try:
            result = await ReconciliationService(session).run(broker)
        finally:
            await broker.close()
        return {
            "is_clean": result.is_clean,
            "discrepancies": len(result.discrepancies),
            "positions_checked": result.positions_checked,
        }


@app.task(bind=True, name="worker.jobs.risk.reconcile_broker", max_retries=2)
def reconcile_broker(self) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Reconcile the active venue; halt on divergence, resolve on clean (§10)."""
    try:
        result = asyncio.run(_reconcile())
        log.info("job.reconcile_broker.completed", **result)
        return result
    except _NOT_CONFIGURED as exc:
        log.info("job.reconcile_broker.skipped", reason=str(exc))
        return {"skipped": True, "reason": "venue not configured"}
    except Exception as exc:
        log.exception("job.reconcile_broker.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=120 * (2**self.request.retries)) from exc


async def _generate_eod() -> dict[str, Any]:
    async with session_scope() as session:
        broker = resolve_broker(await active_broker_kind(session), session=session)
        try:
            summary = await EODSummaryService(session).generate(broker)
        finally:
            await broker.close()
        return {
            "date": summary.summary_date.isoformat(),
            "equity": str(summary.equity),
            "open_positions": summary.open_positions,
        }


@app.task(bind=True, name="worker.jobs.risk.generate_eod_summary", max_retries=2)
def generate_eod_summary(self) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Persist the end-of-day account summary for the active venue (§16)."""
    try:
        result = asyncio.run(_generate_eod())
        log.info("job.generate_eod_summary.completed", **result)
        return result
    except _NOT_CONFIGURED as exc:
        log.info("job.generate_eod_summary.skipped", reason=str(exc))
        return {"skipped": True, "reason": "venue not configured"}
    except Exception as exc:
        log.exception("job.generate_eod_summary.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=300 * (2**self.request.retries)) from exc


# -- Live safety (§14) -----------------------------------------------------


async def _live_guard() -> dict[str, Any]:
    async with session_scope() as session:
        return await LiveGuardService(session).enforce()


@app.task(bind=True, name="worker.jobs.risk.live_guard", max_retries=2)
def live_guard(self) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Auto-disarm live and halt if the day's realised loss breaches the limit (§14).

    Needs no broker or credentials — it reads local intents — so it runs
    everywhere and is a no-op when nothing is armed.
    """
    try:
        result = asyncio.run(_live_guard())
        if result.get("breached"):
            log.warning("job.live_guard.breached", **result)
        return result
    except Exception as exc:
        log.exception("job.live_guard.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=120 * (2**self.request.retries)) from exc


async def _reconcile_live() -> dict[str, Any]:
    async with session_scope() as session:
        broker = resolve_broker(BrokerKind.TRADING212_LIVE)
        try:
            result = await ReconciliationService(session).run(broker)
        finally:
            await broker.close()
        return {"is_clean": result.is_clean, "discrepancies": len(result.discrepancies)}


@app.task(bind=True, name="worker.jobs.risk.reconcile_live", max_retries=2)
def reconcile_live(self) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Reconcile the live broker; halt on divergence (§10). No-op without live creds."""
    try:
        result = asyncio.run(_reconcile_live())
        log.info("job.reconcile_live.completed", **result)
        return result
    except (LiveTradingDisabledError, BrokerNotConfiguredError) as exc:
        log.info("job.reconcile_live.skipped", reason=str(exc))
        return {"skipped": True, "reason": "live broker not configured"}
    except Exception as exc:
        log.exception("job.reconcile_live.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=180 * (2**self.request.retries)) from exc
