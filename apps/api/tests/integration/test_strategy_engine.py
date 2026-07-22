"""Strategy evaluation end to end against real PostgreSQL.

Each strategy is exercised on hand-crafted candles (deterministic, not the random
mock walk) so the signal is guaranteed, then the engine is checked for the whole
chain: signal → proposal / targeted order → risk engine → paper fill → decision.
The risk engine still gates everything, so an active halt turns an entry into a
recorded refusal rather than a trade.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.broker.internal_paper import InternalPaperBroker
from app.data.store import CandleStore
from app.data.types import Candle as CandleDTO
from app.models.enums import (
    HaltKind,
    HaltScope,
    InstrumentKind,
    Interval,
    PriceUnit,
    ProviderKind,
    StrategyDecisionOutcome,
    StrategyKind,
)
from app.models.instrument import Exchange, Instrument
from app.models.risk import RiskConfiguration
from app.models.strategy import StrategyConfiguration, StrategyDecision
from app.risk.halts import HaltService
from app.strategies.engine import StrategyEngine

pytestmark = pytest.mark.asyncio

_STEP = {Interval.D1: timedelta(days=1), Interval.M15: timedelta(minutes=15)}


async def _instrument(db: object, ticker: str) -> Instrument:
    exchange = (
        await db.execute(select(Exchange).where(Exchange.mic == "XNAS"))  # type: ignore[attr-defined]
    ).scalar_one_or_none()
    if exchange is None:
        exchange = Exchange(mic="XNAS", name="Nasdaq", country="US", timezone="America/New_York")
        db.add(exchange)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]
    instrument = Instrument(
        id=uuid.uuid4(),
        isin=None,
        exchange_id=exchange.id,
        exchange_ticker=ticker,
        name=f"{ticker} Inc.",
        kind=InstrumentKind.STOCK,
        currency="USD",
        price_unit=PriceUnit.USD,
    )
    db.add(instrument)  # type: ignore[attr-defined]
    await db.flush()  # type: ignore[attr-defined]
    return instrument


async def _upsert(
    db: object, instrument: Instrument, interval: Interval, closes: list[float]
) -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    step = _STEP[interval]
    n = len(closes)
    candles = [
        CandleDTO(
            symbol=instrument.exchange_ticker or "X",
            interval=interval,
            timestamp=now - step * (n - 1 - i),
            open=Decimal(str(close)),
            high=Decimal(str(close)) * Decimal("1.01"),
            low=Decimal(str(close)) * Decimal("0.99"),
            close=Decimal(str(close)),
            volume=Decimal("100000"),
            currency="USD",
            price_unit=PriceUnit.USD,
            provider=ProviderKind.MOCK,
            is_closed=True,
        )
        for i, close in enumerate(closes)
    ]
    await CandleStore(db).upsert_candles(instrument.id, candles)  # type: ignore[arg-type]


async def _risk_config(db: object) -> None:
    db.add(RiskConfiguration(name="default", is_active=True))  # type: ignore[attr-defined]
    await db.flush()  # type: ignore[attr-defined]


class TestTrend:
    async def test_uptrend_is_entered(self, db: object) -> None:
        await _risk_config(db)
        instrument = await _instrument(db, "GOLD")
        # A clean 130-day uptrend: price above its SMA100, rising, positive return.
        await _upsert(db, instrument, Interval.D1, [80 + i * 0.5 for i in range(130)])

        config = StrategyConfiguration(
            kind=StrategyKind.TREND_FOLLOWING,
            name="trend",
            is_active=True,
            interval=Interval.D1,
            auto_execute=True,
            params={"sma_period": 100, "slope_window": 21, "return_lookback": 60},
            universe={"instrument_ids": [str(instrument.id)]},
        )
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]

        assert summary.signals == 1
        assert summary.executed == 1
        positions = await InternalPaperBroker(db).get_positions()  # type: ignore[arg-type]
        assert any(p.broker_ticker == str(instrument.id) for p in positions)

    async def test_no_signal_without_a_trend(self, db: object) -> None:
        await _risk_config(db)
        instrument = await _instrument(db, "FLAT")
        await _upsert(db, instrument, Interval.D1, [100.0] * 130)
        config = StrategyConfiguration(
            kind=StrategyKind.TREND_FOLLOWING,
            name="trend",
            is_active=True,
            interval=Interval.D1,
            params={"sma_period": 100},
            universe={"instrument_ids": [str(instrument.id)]},
        )
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]
        assert summary.signals == 0


class TestMeanReversion:
    async def test_oversold_below_mean_is_entered(self, db: object) -> None:
        await _risk_config(db)
        instrument = await _instrument(db, "SPY")
        # Daily bars for sizing the stop, plus an intraday series that ends
        # stretched below its mean and oversold.
        await _upsert(db, instrument, Interval.D1, [100.0] * 60)
        intraday = [100.0] * 185 + [98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70]
        await _upsert(db, instrument, Interval.M15, intraday)

        config = StrategyConfiguration(
            kind=StrategyKind.SP500_MEAN_REVERSION,
            name="meanrev",
            is_active=True,
            interval=Interval.M15,
            auto_execute=True,
            params={
                "sma_period": 20,
                "zscore_entry": -1.0,
                "rsi_period": 14,
                "rsi_oversold": 45.0,
                "zscore_exit": 0.0,
            },
            universe={"instrument_ids": [str(instrument.id)]},
        )
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]

        assert summary.signals == 1
        assert summary.executed == 1


class TestPie:
    async def test_rebalance_buys_toward_target_weights(self, db: object) -> None:
        await _risk_config(db)
        a = await _instrument(db, "AAA")
        b = await _instrument(db, "BBB")
        await _upsert(db, a, Interval.D1, [100.0] * 30)
        await _upsert(db, b, Interval.D1, [50.0] * 30)

        config = StrategyConfiguration(
            kind=StrategyKind.PIE_REBALANCE,
            name="pie",
            is_active=True,
            interval=Interval.D1,
            auto_execute=True,
            params={"drift_band": 0.05},
            universe={"weights": {str(a.id): 0.5, str(b.id): 0.5}},
            account_equity=Decimal("10000"),
        )
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]

        assert summary.executed == 2
        positions = {
            p.broker_ticker: p.quantity
            for p in await InternalPaperBroker(db).get_positions()  # type: ignore[arg-type]
        }
        # 50% of 10k = 5k each: 50 units of AAA @100, 100 units of BBB @50.
        assert positions[str(a.id)] == Decimal("50")
        assert positions[str(b.id)] == Decimal("100")


class TestRiskGate:
    async def test_a_halt_turns_an_entry_into_a_recorded_refusal(self, db: object) -> None:
        await _risk_config(db)
        instrument = await _instrument(db, "GOLD")
        await _upsert(db, instrument, Interval.D1, [80 + i * 0.5 for i in range(130)])
        await HaltService(db).activate(  # type: ignore[arg-type]
            HaltKind.KILL_SWITCH, "halted", scope=HaltScope.GLOBAL
        )
        config = StrategyConfiguration(
            kind=StrategyKind.TREND_FOLLOWING,
            name="trend",
            is_active=True,
            interval=Interval.D1,
            params={"sma_period": 100},
            universe={"instrument_ids": [str(instrument.id)]},
        )
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]

        assert summary.executed == 0
        assert summary.rejected == 1
        decision = (
            await db.execute(select(StrategyDecision))  # type: ignore[attr-defined]
        ).scalars().one()
        assert decision.outcome is StrategyDecisionOutcome.REJECTED_BY_RISK
        assert await InternalPaperBroker(db).get_positions() == []  # type: ignore[arg-type]
