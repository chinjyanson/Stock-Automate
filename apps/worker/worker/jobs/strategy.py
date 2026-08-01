"""Strategy evaluation jobs (§8, §16).

Evaluates the active strategies on a schedule. Each run reads the local candle
store, produces signals, and routes them through the risk engine — the worker
never bypasses it. Idempotent by construction: a strategy that already holds a
position does not re-enter it, a trailing/target adjustment converges, and the
paper venue is local so no distributed lock is required.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from app.broker.internal_paper import InternalPaperBroker
from app.db import session_scope
from app.models.enums import Interval, StrategyKind
from app.models.instrument import Instrument
from app.models.scanner import ScannerResult, ScannerRun
from app.models.strategy import StrategyConfiguration
from app.strategies.engine import StrategyEngine
from sqlalchemy import select, update

from worker.app import app
from worker.runner import run_job

log = structlog.get_logger(__name__)


async def _evaluate(interval: Interval | None) -> dict[str, Any]:
    async with session_scope() as session:
        stmt = select(StrategyConfiguration).where(StrategyConfiguration.is_active.is_(True))
        if interval is not None:
            stmt = stmt.where(StrategyConfiguration.interval == interval)
        configs = (await session.execute(stmt)).scalars().all()

        engine = StrategyEngine(session)
        totals = {
            "strategies": 0,
            "signals": 0,
            "executed": 0,
            "rejected": 0,
            "skipped": 0,
            "stale": 0,
        }
        for config in configs:
            summary = await engine.run(config, selection_reason="scheduled")
            totals["strategies"] += 1
            totals["signals"] += summary.signals
            totals["executed"] += summary.executed
            totals["rejected"] += summary.rejected
            # Instruments with too little history to evaluate. Surfaced in the
            # job result because "0 signals" alone cannot distinguish a healthy
            # quiet run from one that had no data to look at.
            totals["skipped"] += summary.skipped
            # Instruments evaluated on bars past their freshness threshold —
            # usually this worker not having refreshed candles.
            totals["stale"] += summary.stale
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
        result = run_job(_evaluate(parsed))
        if result["strategies"]:
            log.info("job.evaluate_strategies.completed", **result)
        return result
    except Exception as exc:
        log.exception("job.evaluate_strategies.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=180 * (2**self.request.retries)) from exc


#: How many scanner-ranked names the mean-reversion strategy watches. The risk
#: engine caps how many are *held* at once; this is the pool it chooses from.
DEFAULT_UNIVERSE_SIZE = 20


async def _sync_universe(size: int) -> dict[str, Any]:
    """Point the mean-reversion strategy at the scanner's current top names (§6, §8).

    This is the seam between the two layers: the scanner decides *what* is worth
    owning, the strategy decides *when*. Keeping the selection here rather than
    inside the strategy preserves the `Strategy` contract — it reads candles and
    nothing else, so it stays testable against fixtures with no scanner in sight.

    Only broker-tradable results are eligible: ranking an instrument the venue
    will not sell us produces signals that can only ever be refused downstream.

    Instruments already held are kept in the universe even if they have dropped
    out of the top `size`. Otherwise the strategy would stop seeing a position it
    owns and could never emit its exit signal — the stock would sit there until
    the ATR stop or the holding cap removed it.
    """
    async with session_scope() as session:
        config = (
            await session.execute(
                select(StrategyConfiguration).where(
                    StrategyConfiguration.kind == StrategyKind.MEAN_REVERSION
                )
            )
        ).scalar_one_or_none()
        if config is None:
            return {"skipped": True, "reason": "no mean-reversion configuration"}

        # Ad-hoc runs are excluded: rescanning one instrument on demand records
        # a score, it does not re-rank the universe. Without this filter the
        # most recent such run would be "the latest", and the strategy would be
        # left watching whichever single name someone last looked at.
        latest_run = (
            await session.execute(
                select(ScannerRun.id)
                .where(ScannerRun.is_ad_hoc.is_(False))
                .order_by(ScannerRun.started_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if latest_run is None:
            return {"skipped": True, "reason": "no scanner run to rank from"}

        ranked = (
            (
                await session.execute(
                    select(ScannerResult.instrument_id)
                    .where(
                        ScannerResult.run_id == latest_run,
                        ScannerResult.is_trading212_tradable.is_(True),
                    )
                    .order_by(ScannerResult.primary_score.desc())
                    .limit(size)
                )
            )
            .scalars()
            .all()
        )
        chosen = [str(i) for i in ranked]

        # Anything currently held stays in view so its exit can still fire.
        broker = InternalPaperBroker(session)
        held = {p.broker_ticker for p in await broker.get_positions() if p.quantity > 0}
        retained = sorted(held - set(chosen))
        universe = chosen + retained

        config.universe = {"instrument_ids": universe}

        # The daily refresh prioritises the Bot Universe, so these are the names
        # whose candles must be current for the strategy to read them tomorrow.
        await session.execute(
            update(Instrument)
            .where(Instrument.is_bot_universe.is_(True))
            .values(is_bot_universe=False)
        )
        if universe:
            await session.execute(
                update(Instrument)
                .where(Instrument.id.in_([uuid.UUID(i) for i in universe]))
                .values(is_bot_universe=True)
            )

        return {"ranked": len(chosen), "retained_held": len(retained), "total": len(universe)}


@app.task(bind=True, name="worker.jobs.strategy.sync_strategy_universe", max_retries=2)
def sync_strategy_universe(self, size: int = DEFAULT_UNIVERSE_SIZE) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Refresh the mean-reversion universe from the latest scanner ranking."""
    try:
        result = run_job(_sync_universe(size))
        log.info("job.sync_strategy_universe.completed", **result)
        return result
    except Exception as exc:
        log.exception("job.sync_strategy_universe.failed", error=str(exc))
        raise self.retry(exc=exc, countdown=300 * (2**self.request.retries)) from exc
