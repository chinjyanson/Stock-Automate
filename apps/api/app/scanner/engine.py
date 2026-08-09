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
from app.services.insider import InsiderIngestionService
from app.services.sentiment import SentimentService

log = structlog.get_logger(__name__)

#: Below this many closed daily bars, scoring still runs but the result is
#: flagged low-confidence rather than skipped (§6: do not imply an unscanned or
#: thinly-covered instrument "failed").
MIN_BARS_TO_SCORE = 30

#: Yahoo `.info` sector name → SPDR sector-ETF symbol, used as a global GICS
#: proxy for that sector's performance. A stock whose sector is absent or not in
#: this map simply gets no sector signal (it drops out, never penalised).
_SECTOR_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

#: Rate-sensitivity proxy: a 7-10 year Treasury *price* series. Deliberately an
#: ETF and not a yield index like ^TNX — yields move inversely to bond prices, so
#: a yield series would silently flip the sign of every correlation without
#: erroring. Loaded once per run and shared by every instrument, like SPY.
RATES_PROXY_SYMBOL = "IEF"


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
        is_ad_hoc: bool = False,
    ) -> ScanSummary:
        """Score a slice of instruments under a single run.

        Set `is_ad_hoc` when the slice was named explicitly rather than chosen
        by the rotation. The scores are recorded identically either way; the
        flag only stops a one-off rescan from being mistaken for the nightly
        ranking when the strategy universe is synced (see `ScannerRun`).
        """
        run_config = _config_values(configuration)

        run = ScannerRun(
            configuration_id=configuration.id if configuration else None,
            status=ScannerRunStatus.RUNNING,
            started_at=datetime.now(UTC),
            instruments_considered=len(instruments),
            selection_reason=selection_reason,
            is_ad_hoc=is_ad_hoc,
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
        # One load per run, shared by every instrument — the rate-sensitivity
        # signal is a correlation against a series that is the same for all of them.
        rates_series = await self._load_benchmark(RATES_PROXY_SYMBOL)
        # Each sector ETF is loaded at most once per run and reused across every
        # instrument in that sector (≤11 loads, not one per stock).
        sector_cache: dict[str, PriceSeries | None] = {}

        for instrument in instruments:
            try:
                scored = await self._score_one(
                    run, instrument, run_config, benchmark_series, rates_series, sector_cache
                )
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
        rates: PriceSeries | None,
        sector_cache: dict[str, PriceSeries | None],
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
        sector_series = await self._resolve_sector_series(instrument, sector_cache)

        # Insider activity is looked up rather than derived from the series, so
        # it is resolved here and passed in. None for anything without recent
        # Form 4 filings, which is most of the catalogue — the blend drops a
        # scoreless group along with its weight, so absence is neutral.
        insider, insider_sell_penalty = await self._insider_factor(instrument.id)

        result = scoring.score_series(
            series,
            weights=run_config.weights,
            thresholds=run_config.thresholds,
            benchmark=benchmark,
            sector=sector_series,
            rates=rates,
            sentiment=await self._sentiment_score(instrument.id),
            fundamentals=fundamentals,
            insider=insider,
            insider_sell_penalty=insider_sell_penalty,
            fundamentals_penalty=run_config.fundamentals_penalty,
        )

        freshness_days = None
        if candles:
            age = datetime.now(UTC) - candles[-1].timestamp
            freshness_days = Decimal(str(round(age.total_seconds() / 86400, 2)))

        # The five group scores are persisted alongside the blend they produce,
        # all on the same 0-100 scale, so a reader can see *why* a stock ranked
        # where it did without re-running anything.
        self._session.add(
            ScannerResult(
                run_id=run.id,
                instrument_id=instrument.id,
                primary_score=Decimal(str(result.score)),
                fundamental_value_score=_group_decimal(result, "value"),
                price_value_score=_group_decimal(result, "cheapness"),
                insider_score=_group_decimal(result, "insider"),
                quality_score=_group_decimal(result, "quality"),
                sector_score=_group_decimal(result, "sector"),
                insider_sell_penalty=Decimal(str(round(result.insider_sell_penalty, 4))),
                value_signals=_value_signals(result),
                classification=result.classification,
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
        return result.classification

    async def _resolve_sector_series(
        self, instrument: Instrument, cache: dict[str, PriceSeries | None]
    ) -> PriceSeries | None:
        """The sector-ETF proxy series for `instrument`, cached per run.

        Maps the instrument's `sector` tag to its SPDR ETF symbol and loads that
        ETF's candles (once per run). None when the instrument is untagged, its
        sector has no proxy, or the ETF has not been ingested yet — the sector
        signals then drop out with no penalty.
        """
        symbol = _SECTOR_ETF.get((instrument.sector or "").strip())
        if symbol is None:
            return None
        if symbol not in cache:
            cache[symbol] = await self._load_benchmark(symbol)
        return cache[symbol]

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

    async def _insider_factor(self, instrument_id: uuid.UUID) -> tuple[float | None, float]:
        """Insider buying/selling as a 0-100 factor, or None if there is none.

        Wrapped rather than called inline so a fault in the insider path can
        never fail a scan: this is a small, optional factor over a minority of
        the catalogue, and an exception here would take down the scoring of an
        instrument whose other five factors are perfectly good.
        """
        try:
            score = await InsiderIngestionService(self._session).score_instrument(instrument_id)
        except Exception as exc:
            log.warning(
                "scanner.insider_factor_failed",
                instrument_id=str(instrument_id),
                error=str(exc),
            )
            return None, 0.0
        if score is None:
            return None, 0.0
        return score.score, score.sell_penalty

    async def _sentiment_score(self, instrument_id: uuid.UUID) -> float | None:
        """News tone as a polarity in [-1, +1], or None when there is none.

        Reads the stored snapshot only — the sweep that calls Finnhub is a
        separate job, so a scan never depends on a news feed being reachable.
        Wrapped for the same reason as `_insider_factor` and `_pead_score`: an
        optional signal covering a minority of the catalogue must never take
        down the scoring of an instrument whose other signals are fine.
        """
        try:
            score = await SentimentService(self._session).score_instrument(instrument_id)
        except Exception as exc:
            log.warning(
                "scanner.sentiment_failed", instrument_id=str(instrument_id), error=str(exc)
            )
            return None
        return score.polarity if score is not None else None

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
            "earnings_growth": snap.earnings_growth,
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
    #: Weights of the five scoring groups (see scoring.DEFAULT_WEIGHTS).
    weights: dict[str, float]
    thresholds: dict[str, float]
    benchmark_symbol: str | None
    fundamentals_penalty: float


def _config_values(config: ScannerConfiguration | None) -> RunConfig:
    if config is None:
        return RunConfig(
            weights=scoring.DEFAULT_WEIGHTS,
            thresholds=scoring.DEFAULT_THRESHOLDS,
            benchmark_symbol="SPY",
            fundamentals_penalty=scoring.DEFAULT_FUNDAMENTALS_PENALTY,
        )
    penalty = getattr(config, "fundamentals_penalty", None)
    return RunConfig(
        weights=_floatify(config.weights) or scoring.DEFAULT_WEIGHTS,
        thresholds=_floatify(config.thresholds) or scoring.DEFAULT_THRESHOLDS,
        benchmark_symbol=config.benchmark_symbol,
        fundamentals_penalty=float(penalty)
        if penalty is not None
        else scoring.DEFAULT_FUNDAMENTALS_PENALTY,
    )


def _group_decimal(result: scoring.ScoreResult, name: str) -> Decimal | None:
    """One group's 0-100 score as a Decimal, or None when it did not score.

    None is the honest answer for a group nothing could be measured for, and it
    is what `combine_score` acted on — persisting a 0 or a 50 instead would make
    the stored breakdown disagree with the score it is supposed to explain.
    """
    score = result.group_score(name)
    return Decimal(str(round(score, 2))) if score is not None else None


def _value_signals(result: scoring.ScoreResult) -> dict[str, list[str]] | None:
    """The explanations behind the two valuation groups, for the detail panel."""
    signals = [*result.groups["value"].signals, *result.groups["cheapness"].signals]
    available = [s for s in signals if s.available and s.explanation]
    if not available:
        return None
    return {
        "positive": [s.explanation for s in available if s.positive],
        "negative": [s.explanation for s in available if not s.positive],
    }


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
