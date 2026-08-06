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
from app.broker.types import BrokerOrderRequest
from app.data.store import CandleStore
from app.data.types import Candle as CandleDTO
from app.models.enums import (
    HaltKind,
    HaltScope,
    InstrumentKind,
    Interval,
    OrderSide,
    OrderType,
    PriceUnit,
    ProviderKind,
    StrategyDecisionOutcome,
    StrategyKind,
)
from app.models.instrument import Exchange, Instrument
from app.models.risk import RiskConfiguration
from app.models.strategy import StrategyConfiguration, StrategyDecision
from app.risk.halts import HaltService
from app.strategies.base import InsiderPressure
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
    db: object,
    instrument: Instrument,
    interval: Interval,
    closes: list[float],
    *,
    age: timedelta = timedelta(0),
) -> None:
    """Seed `closes` as candles, the last one `age` old (fresh by default)."""
    now = datetime.now(UTC).replace(second=0, microsecond=0) - age
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


#: A stable, oscillating base then a sharp sell-off — the dislocation this
#: strategy exists to catch. A *gradual* decline does not work as a fixture and
#: that is not an accident: the bands follow a trend down, so price never breaks
#: its own lower band. Only a sudden move outruns them.
_STABLE_BASE = [100 + (2 if i % 2 else -2) for i in range(45)]
_SELLOFF = [*_STABLE_BASE, 95.0, 90.0, 86.0]
_RECOVERED = [*_STABLE_BASE, 95.0, 90.0, 86.0, 92.0, 97.0, 100.0]

#: A sell-off that has already bounced off its low. All three of band, RSI and
#: ATR still say "enter" — but the last close (69) sits above the average price
#: paid since the trough (67), which is precisely the case anchored VWAP exists
#: to decline. Margins are deliberately wide (band +5.9, RSI +7.9) so the
#: fixture proves the AVWAP gate rather than accidentally tripping another one.
_BOUNCED = [*_STABLE_BASE, 95.0, 84.0, 65.0, 69.0]

#: The sell-off above lands RSI at ~38.7. Pinned in the fixture rather than
#: relying on the configured default, so retuning the strategy cannot silently
#: change what these tests prove.
_ENTRY_PARAMS = {
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "rsi_oversold": 40.0,
    "atr_period": 14,
    "min_atr_pct": 0.02,
}


def _config(instrument: Instrument, name: str, **overrides: object) -> StrategyConfiguration:
    params = {**_ENTRY_PARAMS, **overrides}
    return StrategyConfiguration(
        kind=StrategyKind.MEAN_REVERSION,
        name=name,
        is_active=True,
        interval=Interval.D1,
        auto_execute=True,
        params=params,
        universe={"instrument_ids": [str(instrument.id)]},
    )


class TestCapitalAllocation:
    """A sleeve sizes against its own share, not the whole account.

    This is what stops the two strategies competing for the same capital: every
    percentage limit below the split — risk per trade, position size, total open
    risk — becomes a percentage of the sleeve.
    """

    async def test_a_smaller_sleeve_takes_a_smaller_position(self, db: object) -> None:
        await _risk_config(db)
        full = await _instrument(db, "FULL")
        await _upsert(db, full, Interval.D1, _SELLOFF)
        quarter = await _instrument(db, "QUARTER")
        await _upsert(db, quarter, Interval.D1, _SELLOFF)

        whole = _config(full, "sleeve-whole")
        db.add(whole)  # type: ignore[attr-defined]
        part = _config(quarter, "sleeve-quarter")
        part.capital_allocation_pct = Decimal("0.25")
        db.add(part)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        broker = InternalPaperBroker(db)  # type: ignore[arg-type]
        await StrategyEngine(db, broker=broker).run(whole)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]
        positions = {p.broker_ticker: p.quantity for p in await broker.get_positions()}
        whole_qty = positions[str(full.id)]

        await StrategyEngine(db, broker=InternalPaperBroker(db)).run(part)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]
        positions = {
            p.broker_ticker: p.quantity
            for p in await InternalPaperBroker(db).get_positions()  # type: ignore[arg-type]
        }
        quarter_qty = positions[str(quarter.id)]

        # A quarter of the capital cannot buy as much as all of it. The exact
        # ratio depends on which cap binds, so the property tested is the
        # direction and that the sleeve genuinely bounds it.
        assert quarter_qty < whole_qty

    async def test_no_allocation_means_the_whole_account(self, db: object) -> None:
        """Existing configurations must behave exactly as they did before."""
        await _risk_config(db)
        instrument = await _instrument(db, "UNSPLIT")
        await _upsert(db, instrument, Interval.D1, _SELLOFF)
        config = _config(instrument, "sleeve-none")
        assert config.capital_allocation_pct is None
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]
        assert summary.executed == 1


class TestMeanReversion:
    """Entry needs all three indicators to agree; each is tested for its own veto."""

    async def test_band_break_with_oversold_rsi_is_entered(self, db: object) -> None:
        await _risk_config(db)
        instrument = await _instrument(db, "RISKY")
        await _upsert(db, instrument, Interval.D1, _SELLOFF)
        config = _config(instrument, "meanrev")
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

    async def test_anchored_vwap_is_off_by_default(self, db: object) -> None:
        """The default path must be exactly what it was before the gate existed.

        A bounced sell-off still enters, because with the gate off nothing looks
        at where the price sits relative to what buyers since the low have paid.
        """
        await _risk_config(db)
        instrument = await _instrument(db, "BOUNCED")
        await _upsert(db, instrument, Interval.D1, _BOUNCED)
        config = _config(instrument, "meanrev-avwap-default")
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]
        assert summary.signals == 1

    async def test_anchored_vwap_can_veto_a_band_break(self, db: object) -> None:
        """Enabled, it declines a dip that has already been bought.

        Band, RSI and ATR all still agree here. The only thing that changed is
        that price is now above the average paid since the trough — buying at
        the top of the recovering crowd's range rather than below it.
        """
        await _risk_config(db)
        instrument = await _instrument(db, "BOUNCEDVW")
        await _upsert(db, instrument, Interval.D1, _BOUNCED)
        config = _config(instrument, "meanrev-avwap-on", avwap_enabled=True)
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]
        assert summary.signals == 0

    async def test_anchored_vwap_still_admits_a_stock_making_new_lows(self, db: object) -> None:
        """The boundary case, stated so it is not mistaken for a bug.

        A stock whose latest close *is* its low is by definition at or below the
        average paid since that low, so the gate never blocks the freshest
        dislocations — which are the ones this strategy most wants.
        """
        await _risk_config(db)
        instrument = await _instrument(db, "NEWLOW")
        await _upsert(db, instrument, Interval.D1, _SELLOFF)
        config = _config(instrument, "meanrev-avwap-newlow", avwap_enabled=True)
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]
        assert summary.signals == 1

    async def test_rsi_can_veto_a_band_break(self, db: object) -> None:
        """Price below the band is not enough — momentum must be washed out too.

        This is the "cheap and still falling" case a band break alone cannot
        distinguish from a snapback candidate.
        """
        await _risk_config(db)
        instrument = await _instrument(db, "FALLING")
        await _upsert(db, instrument, Interval.D1, _SELLOFF)
        # Same bars, but demanding a far more oversold RSI than this move produced.
        config = _config(instrument, "meanrev-rsi-veto", rsi_oversold=20.0)
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]
        assert summary.signals == 0

    async def test_atr_can_veto_a_band_break(self, db: object) -> None:
        """A stock too quiet to be worth trading is filtered out by ATR.

        The band and RSI both agree here; only the volatility floor refuses, which
        is what keeps the strategy on names with a snapback worth capturing.
        """
        await _risk_config(db)
        instrument = await _instrument(db, "QUIET")
        await _upsert(db, instrument, Interval.D1, _SELLOFF)
        config = _config(instrument, "meanrev-atr-veto", min_atr_pct=0.50)
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]
        assert summary.signals == 0

    async def test_recovery_to_the_middle_band_exits(self, db: object) -> None:
        """The other half of the round trip: reverted to the mean, so take it."""
        await _risk_config(db)
        instrument = await _instrument(db, "RECOVER")
        await _upsert(db, instrument, Interval.D1, _RECOVERED)
        broker = InternalPaperBroker(db)  # type: ignore[arg-type]
        await broker.place_order(
            BrokerOrderRequest(
                broker_ticker=str(instrument.id),
                side=OrderSide.BUY,
                quantity=Decimal("10"),
                order_type=OrderType.MARKET,
            )
        )
        config = _config(instrument, "meanrev-exit")
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await StrategyEngine(db, broker=broker).run(config)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]

        assert summary.signals == 1
        assert summary.executed == 1
        decision = (
            (await db.execute(select(StrategyDecision)))  # type: ignore[attr-defined]
            .scalars()
            .one()
        )
        assert decision.side is OrderSide.SELL
        assert "recovered to the middle band" in decision.reason
        assert await InternalPaperBroker(db).get_positions() == []  # type: ignore[arg-type]

    async def test_stale_bars_block_the_entry(self, db: object) -> None:
        """A valid setup on old bars is signalled, then refused at the gate."""
        await _risk_config(db)
        instrument = await _instrument(db, "STALE")
        await _upsert(db, instrument, Interval.D1, _SELLOFF, age=timedelta(days=10))
        config = _config(instrument, "meanrev-stale")
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]

        assert summary.signals == 1
        assert summary.executed == 0
        assert summary.rejected == 1
        decision = (
            (await db.execute(select(StrategyDecision)))  # type: ignore[attr-defined]
            .scalars()
            .one()
        )
        assert decision.outcome is StrategyDecisionOutcome.REJECTED_BY_RISK
        assert "stale 1d data" in decision.reason
        assert await InternalPaperBroker(db).get_positions() == []  # type: ignore[arg-type]


class TestStaleReporting:
    """A quiet run on old bars must not look like a quiet run on fresh ones."""

    def _config(self, instrument: Instrument, name: str) -> StrategyConfiguration:
        return StrategyConfiguration(
            kind=StrategyKind.MEAN_REVERSION,
            name=name,
            is_active=True,
            interval=Interval.M15,
            auto_execute=True,
            params={"sma_period": 20},
            universe={"instrument_ids": [str(instrument.id)]},
        )

    async def test_stale_bars_are_counted_even_with_no_signal(self, db: object) -> None:
        """The freshness gate only fires on an entry, so a no-signal run bypasses it.

        This is the case that reads as "nothing happened": the strategy evaluated
        fine, found no setup, and reported zero of everything — while looking at
        prices three days old.
        """
        await _risk_config(db)
        instrument = await _instrument(db, "OLD")
        # Flat and plentiful, so no signal, and deliberately days out of date.
        await _upsert(
            db,
            instrument,
            Interval.M15,
            [100.0 + (i % 3) * 0.1 for i in range(120)],
            age=timedelta(days=3),
        )
        config = self._config(instrument, "meanrev-stale-count")
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]

        assert summary.signals == 0
        assert summary.skipped == 0  # it had plenty of history — just old history
        assert summary.stale == 1

    async def test_fresh_bars_report_no_staleness(self, db: object) -> None:
        """The counterpart, so `stale` cannot be a constant that happens to pass."""
        await _risk_config(db)
        instrument = await _instrument(db, "FRESH")
        await _upsert(db, instrument, Interval.M15, [100.0 + (i % 3) * 0.1 for i in range(120)])
        config = self._config(instrument, "meanrev-fresh-count")
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]

        assert summary.signals == 0
        assert summary.stale == 0


class TestInsufficientHistory:
    async def test_too_few_bars_is_recorded_not_silent(self, db: object) -> None:
        """A strategy that could not look must not resemble one that found nothing.

        Both report zero signals; only the SKIPPED decision distinguishes them,
        and its absence is why a fortnight of empty intraday runs went unnoticed.
        """
        await _risk_config(db)
        instrument = await _instrument(db, "THIN")
        # 10 bars against a 20-period SMA: nowhere near enough to evaluate.
        await _upsert(db, instrument, Interval.M15, [100.0] * 10)

        config = StrategyConfiguration(
            kind=StrategyKind.MEAN_REVERSION,
            name="meanrev-thin",
            is_active=True,
            interval=Interval.M15,
            auto_execute=True,
            params={"sma_period": 20},
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
        assert summary.skipped == 1
        decision = (
            (await db.execute(select(StrategyDecision)))  # type: ignore[attr-defined]
            .scalars()
            .one()
        )
        assert decision.outcome is StrategyDecisionOutcome.SKIPPED
        # Nothing was decided, so there is no side to record.
        assert decision.side is None
        assert decision.instrument_id == instrument.id
        assert "10 closed 15m bars available, needs 20" in decision.reason

    async def test_sufficient_history_records_no_skip(self, db: object) -> None:
        """The counterpart: a real evaluation that declines leaves no skip behind."""
        await _risk_config(db)
        instrument = await _instrument(db, "CALM")
        # Flat and plentiful: evaluated in full, and legitimately uninteresting.
        await _upsert(db, instrument, Interval.M15, [100.0 + (i % 3) * 0.1 for i in range(120)])

        config = StrategyConfiguration(
            kind=StrategyKind.MEAN_REVERSION,
            name="meanrev-calm",
            is_active=True,
            interval=Interval.M15,
            auto_execute=True,
            params={"sma_period": 20},
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
        assert summary.skipped == 0
        assert (
            (await db.execute(select(StrategyDecision)))  # type: ignore[attr-defined]
            .scalars()
            .all()
            == []
        )


class TestRiskGate:
    async def test_a_halt_turns_an_entry_into_a_recorded_refusal(self, db: object) -> None:
        await _risk_config(db)
        instrument = await _instrument(db, "RISKY")
        # The same sell-off that `TestMeanReversion` proves is entered, so a
        # refusal here can only be the halt and not an absent signal.
        await _upsert(db, instrument, Interval.D1, _SELLOFF)
        await HaltService(db).activate(  # type: ignore[arg-type]
            HaltKind.KILL_SWITCH, "halted", scope=HaltScope.GLOBAL
        )
        config = _config(instrument, "halted-entry")
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
            (
                await db.execute(select(StrategyDecision))  # type: ignore[attr-defined]
            )
            .scalars()
            .one()
        )
        assert decision.outcome is StrategyDecisionOutcome.REJECTED_BY_RISK
        assert await InternalPaperBroker(db).get_positions() == []  # type: ignore[arg-type]


class TestInsiderExit:
    """Insider selling closes a position — but only before the market reacts."""

    def _config(self, instrument: Instrument, name: str) -> StrategyConfiguration:
        return StrategyConfiguration(
            kind=StrategyKind.MEAN_REVERSION,
            name=name,
            is_active=True,
            interval=Interval.D1,
            auto_execute=True,
            params={
                **_ENTRY_PARAMS,
                "insider_sell_veto": 0.10,
                "insider_exit_max_drop_atr": 1.0,
            },
            universe={"instrument_ids": [str(instrument.id)]},
        )

    async def _hold(self, db: object, instrument: Instrument) -> InternalPaperBroker:
        broker = InternalPaperBroker(db)  # type: ignore[arg-type]
        await broker.place_order(
            BrokerOrderRequest(
                broker_ticker=str(instrument.id),
                side=OrderSide.BUY,
                quantity=Decimal("10"),
                order_type=OrderType.MARKET,
            )
        )
        return broker

    async def _run(
        self,
        db: object,
        config: StrategyConfiguration,
        broker: InternalPaperBroker,
        penalty: float,
        move_atr: float | None,
    ) -> object:
        engine = StrategyEngine(db, broker=broker)  # type: ignore[arg-type]
        engine._insider_pressure = _fixed_pressure(penalty, move_atr)  # type: ignore[method-assign]
        summary = await engine.run(config)
        await db.commit()  # type: ignore[attr-defined]
        return summary

    async def test_exits_when_the_drop_has_not_happened_yet(self, db: object) -> None:
        """Flat since the filing — the whole point, get out before the fall."""
        await _risk_config(db)
        instrument = await _instrument(db, "LEAVING")
        await _upsert(db, instrument, Interval.D1, _SELLOFF)
        broker = await self._hold(db, instrument)
        config = self._config(instrument, "insider-exit-flat")
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await self._run(db, config, broker, penalty=0.30, move_atr=0.0)
        assert summary.signals == 1  # type: ignore[attr-defined]
        assert summary.executed == 1  # type: ignore[attr-defined]
        decision = (
            (await db.execute(select(StrategyDecision)))  # type: ignore[attr-defined]
            .scalars()
            .one()
        )
        assert decision.side is OrderSide.SELL
        assert "Insider exit" in decision.reason
        assert await InternalPaperBroker(db).get_positions() == []  # type: ignore[arg-type]

    async def test_holds_when_the_stock_has_already_fallen(self, db: object) -> None:
        """Priced in. Selling here realises the loss at the bottom, which is the
        one outcome this rule exists to avoid."""
        await _risk_config(db)
        instrument = await _instrument(db, "ALREADYDOWN")
        await _upsert(db, instrument, Interval.D1, _SELLOFF)
        broker = await self._hold(db, instrument)
        config = self._config(instrument, "insider-exit-priced-in")
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        # Same selling pressure, but the stock is already 2 ATR down.
        summary = await self._run(db, config, broker, penalty=0.30, move_atr=-2.0)
        assert summary.signals == 0  # type: ignore[attr-defined]
        assert await InternalPaperBroker(db).get_positions() != []  # type: ignore[arg-type]

    async def test_exits_when_the_stock_has_risen_since_the_filing(self, db: object) -> None:
        """A rise is the best moment to leave, not a reason to damp the signal.

        Guards against using the magnitude of the move rather than its sign —
        an easy mistake, since the *ranking* damping is deliberately symmetric.
        """
        await _risk_config(db)
        instrument = await _instrument(db, "ROSE")
        await _upsert(db, instrument, Interval.D1, _SELLOFF)
        broker = await self._hold(db, instrument)
        config = self._config(instrument, "insider-exit-risen")
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        summary = await self._run(db, config, broker, penalty=0.30, move_atr=+2.0)
        assert summary.signals == 1  # type: ignore[attr-defined]
        assert summary.executed == 1  # type: ignore[attr-defined]

    async def test_entries_are_not_vetoed(self, db: object) -> None:
        """No entry check: the scanner's 40% penalty already drops such a stock
        out of the ranked universe, so a second gate would duplicate it."""
        await _risk_config(db)
        instrument = await _instrument(db, "STILLBUYS")
        await _upsert(db, instrument, Interval.D1, _SELLOFF)
        config = self._config(instrument, "insider-no-entry-veto")
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        broker = InternalPaperBroker(db)  # type: ignore[arg-type]
        summary = await self._run(db, config, broker, penalty=0.30, move_atr=0.0)
        assert summary.signals == 1  # type: ignore[attr-defined]
        assert summary.executed == 1  # type: ignore[attr-defined]


def _fixed_pressure(penalty: float, move_atr: float | None):  # type: ignore[no-untyped-def]
    """Stub for the engine's insider lookup, so tests need no EDGAR data."""

    async def _pressure(instruments: list[Instrument]) -> dict[uuid.UUID, InsiderPressure]:
        if penalty <= 0:
            return {}
        return {i.id: InsiderPressure(sell_penalty=penalty, move_atr=move_atr) for i in instruments}

    return _pressure
