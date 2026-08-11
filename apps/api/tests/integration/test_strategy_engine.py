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
#:
#: 245 bars rather than the 45 this used to be. `backtest.features.compute`
#: returns nothing at all below 220 bars, so a shorter fixture leaves the model
#: with no input and the strategy correctly declines to have an opinion — which
#: would make every test here pass or fail for the wrong reason.
_STABLE_BASE = [100 + (2 if i % 2 else -2) for i in range(245)]
_SELLOFF = [*_STABLE_BASE, 95.0, 90.0, 86.0]
_RECOVERED = [*_STABLE_BASE, 95.0, 90.0, 86.0, 92.0, 97.0, 100.0]

#: A sell-off that has already bounced off its low. All three of band, RSI and
#: ATR still say "enter" — but the last close (69) sits above the average price
#: paid since the trough (67), which is precisely the case anchored VWAP exists
#: to decline. Margins are deliberately wide (band +5.9, RSI +7.9) so the
#: fixture proves the AVWAP gate rather than accidentally tripping another one.
_BOUNCED = [*_STABLE_BASE, 95.0, 84.0, 65.0, 69.0]


def _ramp(start: float, end: float, n: int) -> list[float]:
    return [start + (end - start) * i / (n - 1) for i in range(n)]


#: The same sell-off, reached from two opposite long-run trends. Both are 268
#: bars, because `sma_slope(closes, 200, 21)` needs 200 bars plus its fit window
#: and returns None below that — which is exactly why `_SELLOFF` alone cannot
#: test the trend gate.
#:
#: The last 20 bars are identical in both, so the Bollinger bands, RSI and ATR
#: are identical too (band 90.19, RSI ~38, close 86). The *only* thing that
#: differs is the direction of the 200-day average underneath: +0.17%/day in one
#: and -0.19%/day in the other. That is what makes these a test of the trend
#: filter rather than of anything else.
_UPTREND_SELLOFF = [*_ramp(60.0, 98.0, 220), *_SELLOFF]
_DOWNTREND_SELLOFF = [*_ramp(160.0, 102.0, 220), *_SELLOFF]

#: Pinned in the fixture rather than read from the configured defaults, so
#: retuning the strategy cannot silently change what these tests prove — they
#: are about the engine and the gates, not about the tuning.
_ENTRY_PARAMS = {
    "bb_period": 20,
    "bb_std": 2.0,
    "atr_period": 14,
    "min_atr_pct": 0.02,
    "entry_probability": 0.55,
}

#: The features the served model expects, in order.
_MODEL_FEATURES = ("discount_sma200", "rsi_14", "sma200_slope", "atr_pct")


async def _seed_model(
    db: object, *, intercept: float = 3.0, coefficients: tuple[float, ...] | None = None
) -> None:
    """Install an active model, because the strategy serves one or emits nothing.

    Default is deliberately permissive — `sigmoid(3.0)` is 0.95, comfortably over
    the 0.55 entry probability whatever the features say — so a test about the
    *engine* is not also a test of whether some fitted coefficient happened to
    like the fixture. Tests about the probability gate pass their own intercept.
    """
    from app.models_ml.logistic import FittedModel
    from app.services.strategy_model import StrategyModelService

    n = len(_MODEL_FEATURES)
    model = FittedModel(
        feature_names=_MODEL_FEATURES,
        coefficients=coefficients or (0.0,) * n,
        intercept=intercept,
        means=(0.0,) * n,
        sds=(1.0,) * n,
        scale_known=(True,) * n,
        prior_means=(0.0,) * n,
        prior_taus=(1.0,) * n,
        shrinkage=(1.0,) * n,
        standard_errors=(0.1,) * n,
        n_observations=1_000,
        positive_rate=0.44,
        auc=0.55,
        brier=0.24,
        log_loss=0.68,
        label_definition="test fixture",
    )
    await StrategyModelService(db).save(StrategyKind.LOGISTIC_STOCK, model)  # type: ignore[arg-type]


def _config(instrument: Instrument, name: str, **overrides: object) -> StrategyConfiguration:
    params = {**_ENTRY_PARAMS, **overrides}
    return StrategyConfiguration(
        kind=StrategyKind.LOGISTIC_STOCK,
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
        await _seed_model(db)
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
        await _seed_model(db)
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


class TestLogisticEntry:
    """The model decides entry; the gates decide admissibility.

    The split is the design. A probability answers "is this likely to work" and
    a gate answers "should this ever be bought" — and blending them would let an
    attractive enough setup buy its way past a safety rule. So a gate can refuse
    a confident model and no probability can talk a gate round.
    """

    async def _run(self, db: object, ticker: str, closes: list[float], **params: object) -> int:
        await _risk_config(db)
        await _seed_model(db, intercept=float(params.pop("intercept", 3.0)))
        instrument = await _instrument(db, ticker)
        await _upsert(db, instrument, Interval.D1, closes)
        config = _config(instrument, f"logistic-{ticker.lower()}", **params)
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]
        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]
        return summary.signals

    async def test_a_confident_model_enters(self, db: object) -> None:
        assert await self._run(db, "RISKY", _UPTREND_SELLOFF) == 1

    async def test_a_doubtful_model_does_not(self, db: object) -> None:
        """sigmoid(-3) is 0.047, far under the 0.55 entry probability."""
        assert await self._run(db, "DOUBTED", _UPTREND_SELLOFF, intercept=-3.0) == 0

    async def test_a_probability_over_the_threshold_enters(self, db: object) -> None:
        """sigmoid(1.0) is 0.73, comfortably over a 0.55 bar."""
        assert await self._run(db, "MIDLOW", _SELLOFF, intercept=1.0, entry_probability=0.55) == 1

    async def test_the_same_probability_under_a_higher_threshold_does_not(self, db: object) -> None:
        """Same model and same bars as the test above — only the acted-on
        probability differs, which is what makes the pair a test of the
        threshold rather than of the fixture."""
        assert await self._run(db, "MIDHIGH", _SELLOFF, intercept=1.0, entry_probability=0.90) == 0

    async def test_no_model_means_no_signal(self, db: object) -> None:
        """A strategy with nothing fitted emits nothing.

        Deliberately not a fallback to some default weighting, which would be a
        different strategy trading under this one's name — and would do it
        silently, at whatever moment a fit failed to load.
        """
        await _risk_config(db)
        instrument = await _instrument(db, "NOMODEL")
        await _upsert(db, instrument, Interval.D1, _SELLOFF)
        config = _config(instrument, "logistic-nomodel")
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]
        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]
        assert summary.signals == 0

    async def test_the_probability_is_recorded_and_becomes_the_conviction(self, db: object) -> None:
        """`conviction` is documented as 0..1 for ranking, so a calibrated
        probability is exactly what belongs in it — and the features that
        produced it are recorded beside it, because a probability with no
        account of where it came from is unreviewable."""
        await self._run(db, "SCORED", _UPTREND_SELLOFF)
        decision = (
            (await db.execute(select(StrategyDecision)))  # type: ignore[attr-defined]
            .scalars()
            .one()
        )
        assert decision.metrics is not None
        assert "probability" in decision.metrics
        probability = float(decision.metrics["probability"])
        assert 0.0 <= probability <= 1.0
        assert float(decision.conviction) == pytest.approx(probability, abs=1e-6)
        # The per-feature log-odds breakdown, for explaining a trade afterwards.
        assert any(k.startswith("logodds_") for k in decision.metrics)
        assert any(k.startswith("feature_") for k in decision.metrics)

    async def test_a_short_history_still_trades(self, db: object) -> None:
        """A feature that cannot be computed imputes to its training mean.

        `sma200_slope` is unmeasurable below ~221 bars, which covers every recent
        listing. Treating "cannot tell" as "no" would quietly stop the strategy
        trading anything without a year of history — a silent, growing
        restriction nobody asked for. It contributes zero instead.
        """
        assert await self._run(db, "SHORTHIST", _SELLOFF) == 1

    async def test_a_long_history_supplies_the_slope_feature(self, db: object) -> None:
        """And with enough bars it is a real number, recorded on the decision."""
        await self._run(db, "UPTREND", _UPTREND_SELLOFF)
        decision = (
            (await db.execute(select(StrategyDecision)))  # type: ignore[attr-defined]
            .scalars()
            .one()
        )
        assert decision.metrics is not None
        # Present and finite is the claim. Its *sign* depends on the fixture's
        # shape rather than on anything this test is about, and asserting one
        # would make the test fail the next time the fixture is lengthened.
        assert "feature_sma200_slope" in decision.metrics
        assert float(decision.metrics["feature_sma200_slope"]) == float(
            decision.metrics["feature_sma200_slope"]
        )

    async def test_atr_can_veto_a_confident_model(self, db: object) -> None:
        """The gate a probability must never overrule.

        A stock whose true range is a rounding error has no move worth trading
        and would get a meaningless stop from the risk engine. That is
        tradeability, not prediction, so it stays absolute.
        """
        assert await self._run(db, "QUIET", _UPTREND_SELLOFF, min_atr_pct=0.50) == 0

    async def test_recovery_to_the_middle_band_exits(self, db: object) -> None:
        """The other half of the round trip: reverted to the mean, so take it.

        Unchanged by the model — only the entry moved.
        """
        await _risk_config(db)
        await _seed_model(db)
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
        config = _config(instrument, "logistic-exit")
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
        await _seed_model(db)
        instrument = await _instrument(db, "STALE")
        await _upsert(db, instrument, Interval.D1, _SELLOFF, age=timedelta(days=10))
        config = _config(instrument, "logistic-stale")
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


class TestPeadVeto:
    """Do not buy a dip the market is still repricing.

    PEAD says a bad earnings reaction keeps drifting for weeks, so buying that
    dip now is buying in front of the rest of it. The drift decays over 60 days
    and the veto decays with it.
    """

    async def _seed_report(self, db: object, instrument: Instrument, days_ago: int) -> None:
        from app.models.earnings import EarningsEvent

        db.add(  # type: ignore[attr-defined]
            EarningsEvent(
                instrument_id=instrument.id,
                report_date=(datetime.now(UTC) - timedelta(days=days_ago)).date(),
            )
        )
        await db.flush()  # type: ignore[attr-defined]

    async def _run(
        self, db: object, ticker: str, *, report_days_ago: int | None, **params: object
    ) -> int:
        await _risk_config(db)
        await _seed_model(db)
        instrument = await _instrument(db, ticker)
        await _upsert(db, instrument, Interval.D1, _SELLOFF)
        if report_days_ago is not None:
            await self._seed_report(db, instrument, report_days_ago)
        config = _config(instrument, f"meanrev-{ticker.lower()}", **params)
        db.add(config)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]
        summary = await StrategyEngine(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).run(config)
        await db.commit()  # type: ignore[attr-defined]
        return summary.signals

    async def test_no_earnings_event_does_not_block(self, db: object) -> None:
        """The common case, especially outside the US where the calendar is thin.

        A missing measurement must read as "no objection". Treating a thin
        earnings calendar as bad news would veto most UK listings permanently.
        """
        assert await self._run(db, "NOREPORT", report_days_ago=None) == 1

    async def test_a_fresh_bad_reaction_vetoes_the_entry(self, db: object) -> None:
        """The sell-off *is* the reaction: -12% over the three bars after the report."""
        assert await self._run(db, "BADNEWS", report_days_ago=3) == 0

    async def test_an_old_reaction_no_longer_vetoes(self, db: object) -> None:
        """Drift decays linearly over 60 days, and so does the veto.

        At 57 days the same -12% reaction carries 5% of its original weight,
        which lifts the score back above the threshold. A cliff at day 60 would
        make the gate's behaviour depend on the calendar rather than on how much
        drift is plausibly left.
        """
        assert await self._run(db, "OLDNEWS", report_days_ago=57) == 1

    async def test_the_threshold_is_configurable(self, db: object) -> None:
        """Set below anything achievable and the veto stands down entirely."""
        assert await self._run(db, "PEADOFF", report_days_ago=3, pead_veto_below=-1.0) == 1

    async def test_the_score_is_recorded_on_the_decision(self, db: object) -> None:
        await self._run(db, "PEADREC", report_days_ago=3, pead_veto_below=-1.0)
        decision = (
            (await db.execute(select(StrategyDecision)))  # type: ignore[attr-defined]
            .scalars()
            .one()
        )
        assert decision.metrics is not None
        assert float(decision.metrics["pead_score"]) < 40.0


class TestStaleReporting:
    """A quiet run on old bars must not look like a quiet run on fresh ones."""

    def _config(self, instrument: Instrument, name: str) -> StrategyConfiguration:
        return StrategyConfiguration(
            kind=StrategyKind.LOGISTIC_STOCK,
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
        await _seed_model(db)
        instrument = await _instrument(db, "OLD")
        # Flat and plentiful, so no signal, and deliberately days out of date.
        await _upsert(
            db,
            instrument,
            Interval.M15,
            [100.0 + (i % 3) * 0.1 for i in range(260)],
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
        await _seed_model(db)
        instrument = await _instrument(db, "FRESH")
        await _upsert(db, instrument, Interval.M15, [100.0 + (i % 3) * 0.1 for i in range(260)])
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
        await _seed_model(db)
        instrument = await _instrument(db, "THIN")
        # 10 bars against a 20-period SMA: nowhere near enough to evaluate.
        await _upsert(db, instrument, Interval.M15, [100.0] * 10)

        config = StrategyConfiguration(
            kind=StrategyKind.LOGISTIC_STOCK,
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
        assert "10 closed 15m bars available, needs" in decision.reason

    async def test_sufficient_history_records_no_skip(self, db: object) -> None:
        """The counterpart: a real evaluation that declines leaves no skip behind."""
        await _risk_config(db)
        await _seed_model(db)
        instrument = await _instrument(db, "CALM")
        # Flat and plentiful: evaluated in full, and legitimately uninteresting.
        await _upsert(db, instrument, Interval.M15, [100.0 + (i % 3) * 0.1 for i in range(260)])

        config = StrategyConfiguration(
            kind=StrategyKind.LOGISTIC_STOCK,
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
        await _seed_model(db)
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
            kind=StrategyKind.LOGISTIC_STOCK,
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
        await _seed_model(db)
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
        await _seed_model(db)
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
        await _seed_model(db)
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
        await _seed_model(db)
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
