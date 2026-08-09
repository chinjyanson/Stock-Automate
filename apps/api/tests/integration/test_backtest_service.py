"""Backtest service against real PostgreSQL.

The engine's arithmetic is covered offline in `tests/unit/test_backtest_engine.py`.
What is worth testing here is the boundary: that it reads the candle store and
nothing else, that it replays the universe the strategy would actually have been
given, and that a thin instrument is skipped rather than counted as a run which
found no trades — those are different facts and conflating them would let a
sparse sample masquerade as a strategy that does not fire.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.backtest.engine import ReplayConfig
from app.backtest.service import BacktestService
from app.data.types import Candle as CandleDTO
from app.models.enums import InstrumentKind, PriceUnit, ProviderKind
from app.models.instrument import Exchange, Instrument, MarketDataMapping
from app.models.scanner import Classification, ScannerResult, ScannerRun, ScannerRunStatus
from app.strategies.mean_reversion import EntryRules

pytestmark = pytest.mark.asyncio

_SHORT_WARMUP = ReplayConfig(warmup_bars=40)


def _cyclical(cycles: int = 8) -> list[float]:
    closes: list[float] = []
    for _ in range(cycles):
        closes.extend(100 + (3 if i % 2 else -3) for i in range(30))
        closes.extend([94.0, 88.0, 84.0, 88.0, 94.0, 99.0, 101.0])
    return closes


async def _instrument(db: AsyncSession, ticker: str, closes: list[float]) -> Instrument:
    from app.data.store import CandleStore
    from app.models.enums import Interval

    exchange = (
        await db.execute(__import__("sqlalchemy").select(Exchange).where(Exchange.mic == "XNAS"))
    ).scalar_one_or_none()
    if exchange is None:
        exchange = Exchange(mic="XNAS", name="Nasdaq", country="US", timezone="America/New_York")
        db.add(exchange)
        await db.flush()

    instrument = Instrument(
        id=uuid.uuid4(),
        exchange_id=exchange.id,
        exchange_ticker=ticker,
        name=f"{ticker} Co.",
        kind=InstrumentKind.STOCK,
        currency="USD",
        price_unit=PriceUnit.USD,
        is_scanner_eligible=True,
    )
    db.add(instrument)
    await db.flush()
    db.add(
        MarketDataMapping(
            instrument_id=instrument.id,
            provider=ProviderKind.MOCK,
            provider_symbol=ticker,
            is_signal_source=True,
            confirmed_by_user=True,
        )
    )
    await db.flush()

    now = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    n = len(closes)
    await CandleStore(db).upsert_candles(
        instrument.id,
        [
            CandleDTO(
                symbol=ticker,
                interval=Interval.D1,
                timestamp=now - timedelta(days=n - 1 - i),
                open=Decimal(str(c)),
                high=Decimal(str(c)) * Decimal("1.01"),
                low=Decimal(str(c)) * Decimal("0.99"),
                close=Decimal(str(c)),
                volume=Decimal("100000"),
                currency="USD",
                price_unit=PriceUnit.USD,
                provider=ProviderKind.MOCK,
                is_closed=True,
            )
            for i, c in enumerate(closes)
        ],
    )
    await db.flush()
    return instrument


class TestReplayFromTheStore:
    async def test_it_produces_trades_from_stored_candles(self, db: AsyncSession) -> None:
        instrument = await _instrument(db, "CYCLE", _cyclical())
        await db.commit()

        pooled, runs = await BacktestService(db).run([instrument], EntryRules(), _SHORT_WARMUP)
        assert len(runs) == 1
        assert runs[0].bars > 200
        assert pooled.combined.trade_count > 0

    async def test_two_runs_over_the_same_history_agree_exactly(self, db: AsyncSession) -> None:
        """No provider calls and no randomness, so a re-run is free and identical.

        Worth pinning: a backtest whose answer drifts between runs cannot be used
        to compare two configurations, which is the only thing it is for.
        """
        instrument = await _instrument(db, "STABLE", _cyclical())
        await db.commit()
        service = BacktestService(db)

        first, _ = await service.run([instrument], EntryRules(), _SHORT_WARMUP)
        second, _ = await service.run([instrument], EntryRules(), _SHORT_WARMUP)
        assert first.combined.trade_count == second.combined.trade_count
        assert first.combined.total_r == pytest.approx(second.combined.total_r)

    async def test_an_instrument_with_too_little_history_is_skipped_not_zeroed(
        self, db: AsyncSession
    ) -> None:
        thin = await _instrument(db, "THIN", [100.0 + i for i in range(10)])
        await db.commit()

        pooled, runs = await BacktestService(db).run([thin], EntryRules(), _SHORT_WARMUP)
        assert runs == []
        assert pooled.per_instrument == {}

    async def test_a_changed_threshold_changes_the_answer(self, db: AsyncSession) -> None:
        instrument = await _instrument(db, "SWEEP", _cyclical())
        await db.commit()
        service = BacktestService(db)

        loose, _ = await service.run([instrument], EntryRules(entry_threshold=0.50), _SHORT_WARMUP)
        strict, _ = await service.run([instrument], EntryRules(entry_threshold=0.95), _SHORT_WARMUP)
        assert loose.combined.trade_count > strict.combined.trade_count


class TestEligibilityIsApplesToApples:
    """Two configurations in a sweep must be measured over the same sample.

    Without this, a bucket could differ from its neighbour because it happened
    to admit a couple of extra thinly-covered instruments — a difference that has
    nothing whatever to do with the rule being tested, and which reads exactly
    like a finding.
    """

    async def test_eligibility_defaults_to_the_warmup_not_the_indicator_minimum(
        self, db: AsyncSession
    ) -> None:
        """An instrument that clears the indicators but not the warmup trades nothing.

        It used to be admitted anyway — the check was against `required_bars`
        (~21) while the replay starts at the warmup — so it padded the "N
        instruments replayed" denominator with rows that could never trade.
        """
        middling = await _instrument(db, "MIDDLING", _cyclical(1))  # ~37 bars
        await db.commit()

        rules = EntryRules()
        assert middling is not None
        # Comfortably past the indicator minimum...
        assert rules.required_bars < 37
        # ...but nowhere near a 100-bar warmup, so it must not be counted.
        pooled, runs = await BacktestService(db).run(
            [middling], rules, ReplayConfig(warmup_bars=100)
        )
        assert runs == []
        assert pooled.per_instrument == {}

    async def test_the_same_min_bars_admits_the_same_sample_for_every_config(
        self, db: AsyncSession
    ) -> None:
        """The guarantee a sweep depends on: identical instruments, every bucket."""
        deep = await _instrument(db, "DEEP", _cyclical(8))
        shallow = await _instrument(db, "SHALLOW", _cyclical(2))
        await db.commit()
        service = BacktestService(db)
        universe = [deep, shallow]

        samples = []
        for threshold in (0.45, 0.60, 0.95):
            _, runs = await service.run(
                universe, EntryRules(entry_threshold=threshold), _SHORT_WARMUP, min_bars=120
            )
            samples.append({r.instrument_id for r in runs})

        assert all(s == samples[0] for s in samples)
        assert samples[0] == {deep.id}  # shallow is below 120 bars and excluded throughout

    async def test_an_explicit_min_bars_overrides_the_rules(self, db: AsyncSession) -> None:
        """So a caller can hold the sample fixed while sweeping a rule that would
        otherwise move the eligibility bar underneath it."""
        instrument = await _instrument(db, "FIXED", _cyclical(3))
        await db.commit()
        service = BacktestService(db)

        _, admitted = await service.run([instrument], EntryRules(), _SHORT_WARMUP, min_bars=50)
        _, excluded = await service.run([instrument], EntryRules(), _SHORT_WARMUP, min_bars=100_000)
        assert len(admitted) == 1
        assert excluded == []


class TestUniverseSelection:
    async def test_it_replays_the_scanner_ranking_not_the_whole_catalogue(
        self, db: AsyncSession
    ) -> None:
        """The live strategy only ever sees the top names, so a backtest over
        everything would measure a different system."""
        best = await _instrument(db, "BEST", _cyclical(2))
        worst = await _instrument(db, "WORST", _cyclical(2))
        run = ScannerRun(status=ScannerRunStatus.COMPLETED, started_at=datetime.now(UTC))
        db.add(run)
        await db.flush()
        for instrument, score in ((best, 90), (worst, 10)):
            db.add(
                ScannerResult(
                    run_id=run.id,
                    instrument_id=instrument.id,
                    primary_score=Decimal(score),
                    classification=Classification.DOES_NOT_PASS,
                    data_completeness=Decimal("1"),
                    confidence=Decimal("1"),
                    candles_used=200,
                )
            )
        await db.commit()

        top = await BacktestService(db).top_ranked_instruments(1)
        assert [i.id for i in top] == [best.id]

    async def test_no_scan_yet_falls_back_rather_than_returning_nothing(
        self, db: AsyncSession
    ) -> None:
        instrument = await _instrument(db, "NOSCAN", _cyclical(2))
        await db.commit()

        service = BacktestService(db)
        assert await service.top_ranked_instruments(10) == []
        fallback = await service.instruments_with_history(10)
        assert instrument.id in {i.id for i in fallback}
