"""Load stored candles and replay the strategy over them (§8).

The thin I/O layer around `engine.replay`, kept separate for the same reason
`scanner/scoring.py` is separate from `scanner/engine.py`: the arithmetic stays
pure and testable, and only this file knows about a database.

Reads the candle store and nothing else — no provider calls — so a backtest can
be re-run any number of times without spending a single API request, and two runs
over the same stored history give byte-identical answers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.backtest.engine import (
    BacktestResult,
    PortfolioResult,
    ReplayConfig,
    is_continuous,
    replay,
)
from app.data.store import CandleStore
from app.indicators.series import candles_to_series
from app.models.enums import Interval
from app.models.instrument import Instrument
from app.models.scanner import ScannerResult, ScannerRun, ScannerRunStatus
from app.strategies.mean_reversion import EntryRules

log = structlog.get_logger(__name__)

#: Bars pulled per instrument. Deep enough for the ~1,700-bar names the store now
#: holds; asking for more costs nothing when they do not exist.
DEFAULT_HISTORY_BARS = 2000


@dataclass(frozen=True, slots=True)
class InstrumentRun:
    instrument_id: uuid.UUID
    name: str
    bars: int
    result: BacktestResult


@dataclass(frozen=True, slots=True)
class Skipped:
    """Why an instrument was left out, so exclusions are visible not silent."""

    too_short: int = 0
    discontinuous: int = 0
    failed: int = 0


class BacktestService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._store = CandleStore(session)

    async def run(
        self,
        instruments: list[Instrument],
        rules: EntryRules,
        config: ReplayConfig | None = None,
        *,
        history_bars: int = DEFAULT_HISTORY_BARS,
        min_bars: int | None = None,
        require_continuous: bool = True,
    ) -> tuple[PortfolioResult, list[InstrumentRun], Skipped]:
        """Replay `rules` over every instrument with usable stored history.

        An instrument with too few bars is skipped rather than counted as a run
        that found nothing — the two are different facts, and conflating them
        would let a thin sample masquerade as a strategy that does not trade.

        **`min_bars` is what makes a sweep apples-to-apples.** Eligibility
        otherwise derives from `rules`, so two configurations being compared
        could silently be measured over different instruments — and the one that
        happened to admit a few more thinly-covered names would look different
        for a reason that has nothing to do with the rule being tested. Pass the
        same `min_bars` to every configuration in a comparison and the samples
        are identical by construction.

        It also fixes a subtler mismatch: eligibility used to be checked against
        `required_bars` (~21) while `replay` starts at the warmup (~260), so
        instruments between the two were admitted and then contributed nothing
        at all, padding the denominator with rows that never traded.

        `require_continuous` drops series carrying an unadjusted split. The raw
        OHLC the replay needs is uncorrected — only `adjusted_close` is fixed —
        so a reverse split appears as a genuine price move and the ATR-based
        stop sits an absurd distance away. One such instrument once contributed
        +2,489R against roughly -10R from 928 others.
        """
        config = config or ReplayConfig()
        warmup = config.warmup_bars if config.warmup_bars is not None else rules.preferred_bars
        threshold = min_bars if min_bars is not None else max(warmup, rules.required_bars) + 1

        per_instrument: dict[str, BacktestResult] = {}
        runs: list[InstrumentRun] = []
        too_short = discontinuous = failed = 0

        for instrument in instruments:
            candles = await self._store.get_candles(
                instrument.id, Interval.D1, limit=history_bars, closed_only=True
            )
            if len(candles) < threshold:
                too_short += 1
                continue
            series = candles_to_series(candles)
            if require_continuous and not is_continuous(series):
                discontinuous += 1
                continue
            try:
                result = replay(series, rules, config)
            except Exception as exc:  # one bad series must not end the sweep
                failed += 1
                log.warning(
                    "backtest.instrument_failed",
                    instrument_id=str(instrument.id),
                    error=str(exc),
                )
                continue
            key = str(instrument.id)
            per_instrument[key] = result
            runs.append(
                InstrumentRun(
                    instrument_id=instrument.id,
                    name=instrument.name or instrument.exchange_ticker or key,
                    bars=series.length,
                    result=result,
                )
            )

        return (
            PortfolioResult(per_instrument),
            runs,
            Skipped(too_short=too_short, discontinuous=discontinuous, failed=failed),
        )

    async def top_ranked_instruments(self, limit: int) -> list[Instrument]:
        """The universe the strategy would actually have been given.

        Backtesting over the whole catalogue would measure a different system:
        the live strategy only ever sees the scanner's top names, and its edge —
        if it has one — is partly the scanner's. Reading the same ranking keeps
        the two questions from being quietly merged.
        """
        latest_run = (
            await self._session.execute(
                select(ScannerRun.id)
                .where(ScannerRun.status == ScannerRunStatus.COMPLETED)
                .where(ScannerRun.is_ad_hoc.is_(False))
                .order_by(ScannerRun.started_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if latest_run is None:
            return []

        ids = (
            (
                await self._session.execute(
                    select(ScannerResult.instrument_id)
                    .where(ScannerResult.run_id == latest_run)
                    .order_by(ScannerResult.primary_score.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        if not ids:
            return []
        rows = (
            (await self._session.execute(select(Instrument).where(Instrument.id.in_(ids))))
            .scalars()
            .all()
        )
        order = {instrument_id: i for i, instrument_id in enumerate(ids)}
        return sorted(rows, key=lambda r: order.get(r.id, 1 << 30))

    async def instruments_with_history(self, limit: int) -> list[Instrument]:
        """Fallback universe: anything the store holds daily candles for.

        Used when no scan has run yet. Measures the entry rule against a broader,
        unranked sample, which is a different — and weaker — question than the
        one `top_ranked_instruments` asks.
        """
        rows = (
            (
                await self._session.execute(
                    select(Instrument)
                    .where(Instrument.is_scanner_eligible.is_(True))
                    .order_by(Instrument.last_scanned_at.desc().nullslast())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return list(rows)
