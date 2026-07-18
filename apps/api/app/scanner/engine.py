"""Scanner engine (§6).

Orchestrates a scan: pick a slice of the universe, load each instrument's stored
candles, score them, and persist a `ScannerResult` per instrument under one
`ScannerRun`. It reads the local candle store, never a provider — a scan reflects
data we have already fetched, versioned and quality-checked (§4).

A scan is idempotent-friendly: it creates a new run each time (runs are the audit
trail), but scoring the same instrument over the same candles yields the same
numbers, so results are reproducible.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import AuditService
from app.data.store import CandleStore
from app.indicators.series import PriceSeries, candles_to_series
from app.models.enums import ActorKind, AuditEventKind, Interval
from app.models.instrument import Instrument, MarketDataMapping
from app.models.market_data import FundamentalSnapshot
from app.models.scanner import (
    Classification,
    ScannerConfiguration,
    ScannerResult,
    ScannerRun,
    ScannerRunStatus,
)
from app.scanner import scoring

log = structlog.get_logger(__name__)

#: Below this many closed daily bars, scoring still runs but the result is
#: flagged low-confidence rather than skipped (§6: do not imply an unscanned or
#: thinly-covered instrument "failed").
MIN_BARS_TO_SCORE = 30


@dataclass
class ScanSummary:
    run_id: uuid.UUID
    considered: int = 0
    scored: int = 0
    skipped: int = 0
    screening_candidates: int = 0
    watchlist_candidates: int = 0
    errors: list[str] = field(default_factory=list)


class ScannerEngine:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._store = CandleStore(session)
        self._audit = AuditService(session)

    async def run(
        self,
        instruments: list[Instrument],
        *,
        configuration: ScannerConfiguration | None = None,
        selection_reason: str = "manual",
        actor_user_id: uuid.UUID | None = None,
    ) -> ScanSummary:
        """Score a slice of instruments under a single run."""
        run_config = _config_values(configuration)

        run = ScannerRun(
            configuration_id=configuration.id if configuration else None,
            status=ScannerRunStatus.RUNNING,
            started_at=datetime.now(UTC),
            instruments_considered=len(instruments),
            selection_reason=selection_reason,
        )
        self._session.add(run)
        await self._session.flush()

        await self._audit.record(
            kind=AuditEventKind.SCANNER_RUN_STARTED,
            summary=f"Scanner run started over {len(instruments)} instruments ({selection_reason})",
            actor_kind=ActorKind.USER if actor_user_id else ActorKind.SCHEDULER,
            actor_user_id=actor_user_id,
            subject_type="scanner_run",
            subject_id=str(run.id),
        )

        summary = ScanSummary(run_id=run.id, considered=len(instruments))
        benchmark_series = (
            await self._load_benchmark(run_config.benchmark_symbol)
            if run_config.benchmark_symbol
            else None
        )

        for instrument in instruments:
            try:
                scored = await self._score_one(run, instrument, run_config, benchmark_series)
            except Exception as exc:
                log.exception("scanner.instrument_failed", instrument_id=str(instrument.id))
                summary.errors.append(f"{instrument.name}: {exc}")
                summary.skipped += 1
                continue

            if scored is None:
                summary.skipped += 1
                continue

            summary.scored += 1
            if scored is Classification.SCREENING_CANDIDATE:
                summary.screening_candidates += 1
            elif scored is Classification.WATCHLIST_CANDIDATE:
                summary.watchlist_candidates += 1

            instrument.last_scanned_at = datetime.now(UTC)

        run.status = ScannerRunStatus.COMPLETED
        run.completed_at = datetime.now(UTC)
        run.instruments_scored = summary.scored
        run.instruments_skipped = summary.skipped
        run.screening_candidates = summary.screening_candidates
        run.watchlist_candidates = summary.watchlist_candidates
        await self._session.flush()

        await self._audit.record(
            kind=AuditEventKind.SCANNER_RUN_COMPLETED,
            summary=(
                f"Scanner run scored {summary.scored}, skipped {summary.skipped}: "
                f"{summary.screening_candidates} screening, "
                f"{summary.watchlist_candidates} watchlist candidates"
            ),
            actor_kind=ActorKind.USER if actor_user_id else ActorKind.SCHEDULER,
            actor_user_id=actor_user_id,
            subject_type="scanner_run",
            subject_id=str(run.id),
            payload={
                "scored": summary.scored,
                "skipped": summary.skipped,
                "screening": summary.screening_candidates,
                "watchlist": summary.watchlist_candidates,
            },
        )

        log.info(
            "scanner.run_completed",
            run_id=str(run.id),
            scored=summary.scored,
            screening=summary.screening_candidates,
        )
        return summary

    async def _score_one(
        self,
        run: ScannerRun,
        instrument: Instrument,
        run_config: RunConfig,
        benchmark: PriceSeries | None,
    ) -> Classification | None:
        candles = await self._store.get_candles(
            instrument.id, Interval.D1, limit=ind_year_plus(), closed_only=True
        )
        if len(candles) < MIN_BARS_TO_SCORE:
            # Not scored, not "failed". The UI shows it as unscanned/insufficient
            # so an absence is never mistaken for a low score (§6).
            return None

        series = candles_to_series(candles)
        fundamentals = await self._load_fundamentals(instrument.id)

        result = scoring.score_series(
            series,
            weights=run_config.weights,
            thresholds=run_config.thresholds,
            benchmark=benchmark,
            fundamentals=fundamentals,
        )

        # The primary score (and the classification derived from it) blends
        # momentum and value per the run's configuration. A value-primary run
        # classifies on cheapness; a momentum-primary run on strength.
        value_score = result.value.value_score if result.value else None
        primary_score = scoring.combine_primary_score(
            result.core_score,
            value_score,
            momentum_weight=run_config.momentum_weight,
            value_weight=run_config.value_weight,
        )
        primary_classification = scoring.classify(primary_score, run_config.thresholds)

        freshness_days = None
        if candles:
            age = datetime.now(UTC) - candles[-1].timestamp
            freshness_days = Decimal(str(round(age.total_seconds() / 86400, 2)))

        self._session.add(
            ScannerResult(
                run_id=run.id,
                instrument_id=instrument.id,
                primary_score=Decimal(str(round(primary_score, 2))),
                core_score=Decimal(str(result.core_score)),
                trend_score=Decimal(str(round(result.categories["trend"].points, 2))),
                momentum_score=Decimal(str(round(result.categories["momentum"].points, 2))),
                risk_score=Decimal(str(round(result.categories["risk"].points, 2))),
                liquidity_score=Decimal(str(round(result.categories["liquidity"].points, 2))),
                positioning_score=Decimal(str(round(result.categories["positioning"].points, 2))),
                fundamental_score=(
                    Decimal(str(result.fundamental_score))
                    if result.fundamental_score is not None
                    else None
                ),
                value_score=(Decimal(str(result.value.value_score)) if result.value else None),
                price_value_score=(
                    Decimal(str(result.value.price_value_score)) if result.value else None
                ),
                fundamental_value_score=(
                    Decimal(str(result.value.fundamental_value_score))
                    if result.value and result.value.fundamental_value_score is not None
                    else None
                ),
                value_signals=(
                    {
                        "positive": result.value.positive_signals,
                        "negative": result.value.negative_signals,
                    }
                    if result.value
                    else None
                ),
                classification=primary_classification,
                data_completeness=Decimal(str(result.data_completeness)),
                data_freshness_days=freshness_days,
                confidence=Decimal(str(result.confidence)),
                candles_used=result.candles_used,
                provider=str(candles[-1].provider) if candles else None,
                positive_signals={"items": result.positive_signals},
                negative_signals={"items": result.negative_signals},
                missing_information={"items": result.missing_information},
                metrics=_json_safe(result.metrics),
                is_trading212_tradable=await self._is_trading212_tradable(instrument),
            )
        )
        await self._session.flush()
        return primary_classification

    async def _load_benchmark(self, symbol: str) -> PriceSeries | None:
        """Load a benchmark series by its provider symbol, for relative momentum.

        Resolves the mapping to an instrument, then loads that instrument's
        candles. Returns None if the benchmark is not in the store yet — relative
        momentum then simply drops out of scoring rather than erroring.
        """
        result = await self._session.execute(
            select(MarketDataMapping).where(MarketDataMapping.provider_symbol == symbol).limit(1)
        )
        mapping = result.scalar_one_or_none()
        if mapping is None:
            return None
        candles = await self._store.get_candles(
            mapping.instrument_id, Interval.D1, limit=ind_year_plus(), closed_only=True
        )
        if len(candles) < MIN_BARS_TO_SCORE:
            return None
        return candles_to_series(candles)

    async def _load_fundamentals(
        self, instrument_id: uuid.UUID
    ) -> dict[str, Decimal | None] | None:
        result = await self._session.execute(
            select(FundamentalSnapshot)
            .where(FundamentalSnapshot.instrument_id == instrument_id)
            .order_by(FundamentalSnapshot.as_of.desc())
            .limit(1)
        )
        snap = result.scalar_one_or_none()
        if snap is None:
            return None
        return {
            "trailing_pe": snap.trailing_pe,
            "price_to_book": snap.price_to_book,
            "profit_margin": snap.profit_margin,
            "revenue_growth": snap.revenue_growth,
            "debt_to_equity": snap.debt_to_equity,
            "dividend_yield": snap.dividend_yield,
        }

    async def _is_trading212_tradable(self, instrument: Instrument) -> bool:
        """Whether this instrument is currently available through the broker (§6).

        Scanner-only instruments are labelled non-tradable so the UI never offers
        a proposal it cannot route.
        """
        from app.models.enums import BrokerKind
        from app.models.instrument import BrokerInstrument

        result = await self._session.execute(
            select(BrokerInstrument).where(
                BrokerInstrument.instrument_id == instrument.id,
                BrokerInstrument.broker.in_(
                    [BrokerKind.TRADING212_DEMO, BrokerKind.TRADING212_LIVE]
                ),
                BrokerInstrument.is_currently_available.is_(True),
            )
        )
        return result.first() is not None


def ind_year_plus() -> int:
    """One year of trading history plus warm-up for the 200-day average."""
    return 300


@dataclass(frozen=True)
class RunConfig:
    weights: dict[str, float]
    thresholds: dict[str, float]
    benchmark_symbol: str | None
    momentum_weight: float
    value_weight: float

    @property
    def is_value_primary(self) -> bool:
        return self.value_weight > self.momentum_weight


def _config_values(config: ScannerConfiguration | None) -> RunConfig:
    if config is None:
        return RunConfig(
            weights=scoring.DEFAULT_WEIGHTS,
            thresholds=scoring.DEFAULT_THRESHOLDS,
            benchmark_symbol="SPY",
            momentum_weight=1.0,
            value_weight=0.0,
        )
    return RunConfig(
        weights=_floatify(config.weights) or scoring.DEFAULT_WEIGHTS,
        thresholds=_floatify(config.thresholds) or scoring.DEFAULT_THRESHOLDS,
        benchmark_symbol=config.benchmark_symbol,
        momentum_weight=float(config.momentum_weight),
        value_weight=float(config.value_weight),
    )


def _floatify(raw: dict[str, object] | None) -> dict[str, float] | None:
    if not raw:
        return None
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def _json_safe(metrics: dict[str, object]) -> dict[str, object]:
    """Round floats and drop NaN/inf so the metrics survive JSON serialisation."""
    import math

    out: dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                out[key] = None
            else:
                out[key] = round(value, 6)
        else:
            out[key] = value
    return out
