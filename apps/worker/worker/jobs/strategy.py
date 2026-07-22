"""Strategy evaluation jobs (§8, §16).

Evaluates the active strategies on a schedule. Each run reads the local candle
store, produces signals, and routes them through the risk engine — the worker
never bypasses it. Idempotent by construction: a strategy that already holds a
position does not re-enter it, a trailing/target adjustment converges, and the
paper venue is local so no distributed lock is required.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from app.db import session_scope
from app.models.enums import Interval
from app.models.strategy import StrategyConfiguration
from app.strategies.engine import StrategyEngine
from sqlalchemy import select

from worker.app import app

log = structlog.get_logger(__name__)


async def _evaluate(interval: Interval | None) -> dict[str, Any]:
    async with session_scope() as session:
        stmt = select(StrategyConfiguration).where(StrategyConfiguration.is_active.is_(True))
        if interval is not None:
            stmt = stmt.where(StrategyConfiguration.interval == interval)
        configs = (await session.execute(stmt)).scalars().all()

        engine = StrategyEngine(session)
        totals = {"strategies": 0, "signals": 0, "executed": 0, "rejected": 0}
        for config in configs:
            summary = await engine.run(config, selection_reason="scheduled")
            totals["strategies"] += 1
            totals["signals"] += summary.signals
            totals["executed"] += summary.executed
            totals["rejected"] += summary.rejected
        return totals


@app.task(bind=True, name="worker.jobs.strategy.evaluate_strategies", max_retries=2)
def evaluate_strategies(self, interval: str | None = None) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Evaluate active strategies (optionally only those at one interval)."""
    parsed: Interval | None = None
    if interval is not None:
        try:
            parsed = Interval(interval)
        except ValueError:
            log.error("job.evaluate_strategies.unknown_interval", interval=interval)
            raise
    try:
        result = asyncio.run(_evaluate(parsed))
        if result["strategies"]:
            log.info("job.evaluate_strategies.completed", **result)
        return result
    except Exception as exc:
        log.exception("job.evaluate_strategies.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=180 * (2**self.request.retries)) from exc
