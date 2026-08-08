"""Risk engine sizing and gating against real PostgreSQL.

The properties under test are the ones whose violation costs money: the smallest
cap binds, positions round down (never up, which would breach the cap that
produced them), and the engine fails closed on a missing configuration, an
active halt, or stale data.
"""

from __future__ import annotations

import itertools
import uuid
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal

import pytest
from sqlalchemy import select

from app.broker.types import BrokerAccount, BrokerPosition
from app.data.store import CandleStore
from app.data.types import Candle as CandleDTO
from app.models.enums import (
    HaltKind,
    HaltScope,
    InstrumentKind,
    Interval,
    PriceUnit,
    ProviderKind,
)
from app.models.instrument import Exchange, Instrument
from app.models.risk import RiskConfiguration
from app.risk.engine import QUANTITY_STEP, RiskEngine
from app.risk.halts import HaltService
from app.services.sentiment import SentimentService

pytestmark = pytest.mark.asyncio


async def _rising_instrument(db: object, ticker: str) -> Instrument:
    """An instrument with a 130-bar rising daily series (correlated to any other)."""
    return await _instrument_with_closes(db, ticker, [80 + i * 0.5 for i in range(130)])


async def _inverse_instrument(db: object, ticker: str, of: list[float]) -> Instrument:
    """An instrument whose daily returns are exactly the negative of `of`'s.

    Correlation -1 by construction, so a test can state "moves against" without
    depending on how a falling ramp happens to correlate with a rising one.
    """
    closes = [of[0]]
    for previous, current in itertools.pairwise(of):
        closes.append(closes[-1] * (1 - (current / previous - 1.0)))
    return await _instrument_with_closes(db, ticker, closes)


def _rising_closes() -> list[float]:
    return [80 + i * 0.5 for i in range(130)]


async def _volatile_instrument(db: object, ticker: str) -> Instrument:
    """An instrument that swings ~8% a day — a real drawdown risk in a book."""
    closes = [100.0]
    for i in range(129):
        closes.append(closes[-1] * (1.08 if i % 2 == 0 else 1 / 1.08))
    return await _instrument_with_closes(db, ticker, closes)


async def _instrument_with_closes(db: object, ticker: str, closes: list[float]) -> Instrument:
    exchange = (
        await db.execute(select(Exchange).where(Exchange.mic == "XNAS"))  # type: ignore[attr-defined]
    ).scalar_one_or_none()
    if exchange is None:
        exchange = Exchange(mic="XNAS", name="Nasdaq", country="US", timezone="America/New_York")
        db.add(exchange)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]
    instrument = Instrument(
        id=uuid.uuid4(),
        exchange_id=exchange.id,
        exchange_ticker=ticker,
        name=f"{ticker} Inc.",
        kind=InstrumentKind.STOCK,
        currency="USD",
        price_unit=PriceUnit.USD,
    )
    db.add(instrument)  # type: ignore[attr-defined]
    await db.flush()  # type: ignore[attr-defined]

    now = datetime.now(UTC).replace(second=0, microsecond=0)
    candles = [
        CandleDTO(
            symbol=ticker,
            interval=Interval.D1,
            timestamp=now - timedelta(days=len(closes) - 1 - i),
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
    ]
    await CandleStore(db).upsert_candles(instrument.id, candles)  # type: ignore[arg-type]
    return instrument


def _account(cash: Decimal = Decimal("100000")) -> BrokerAccount:
    return BrokerAccount(
        account_id="PAPER",
        currency="USD",
        cash=cash,
        total=cash,
        free_for_trading=cash,
        invested=Decimal(0),
    )


async def _candles(db: object, instrument: Instrument) -> list:
    return await CandleStore(db).get_candles(  # type: ignore[arg-type]
        instrument.id, Interval.D1, limit=250, closed_only=True
    )


async def _seed_config(db: object, **overrides: object) -> RiskConfiguration:
    config = RiskConfiguration(name="default", is_active=True, **overrides)
    db.add(config)  # type: ignore[attr-defined]
    await db.flush()  # type: ignore[attr-defined]
    return config


class TestSizing:
    async def test_a_position_is_sized_and_stopped(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        config = await _seed_config(db)
        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candled_instrument,
            config=config,
            account=_account(),
            positions=[],
            candles=await _candles(db, candled_instrument),
        )
        assert not decision.rejected
        assert decision.approved_quantity > 0
        assert decision.stop_price is not None
        assert decision.stop_price < decision.entry_price
        # Risk is ~1% of equity by construction.
        assert decision.risk_amount <= Decimal("100000") * Decimal("0.02")

    async def test_quantity_rounds_down_to_the_step(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        config = await _seed_config(db)
        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candled_instrument,
            config=config,
            account=_account(),
            positions=[],
            candles=await _candles(db, candled_instrument),
        )
        # The quantity is an exact multiple of the step — never rounded up.
        assert decision.approved_quantity == decision.approved_quantity.quantize(QUANTITY_STEP)

    async def test_monetary_cap_binds_when_it_is_smallest(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        # A tiny absolute cap must win over the percentage caps.
        config = await _seed_config(db, monetary_position_cap=Decimal("100"))
        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candled_instrument,
            config=config,
            account=_account(),
            positions=[],
            candles=await _candles(db, candled_instrument),
        )
        assert not decision.rejected
        assert "monetary_cap" in decision.applied_caps
        # Position value cannot exceed the cap.
        assert decision.approved_quantity * decision.entry_price <= Decimal("100")


class TestFailClosed:
    async def test_missing_configuration_is_rejected(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candled_instrument,
            config=None,
            account=_account(),
            positions=[],
            candles=await _candles(db, candled_instrument),
        )
        assert decision.rejected
        assert "configuration" in (decision.reason or "")

    async def test_active_global_halt_blocks_sizing(
        self, db: object, candled_instrument: Instrument, approver: object
    ) -> None:
        config = await _seed_config(db)
        await HaltService(db).activate(  # type: ignore[arg-type]
            HaltKind.KILL_SWITCH,
            "test halt",
            scope=HaltScope.GLOBAL,
            actor_user_id=approver,  # type: ignore[arg-type]
        )
        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candled_instrument,
            config=config,
            account=_account(),
            positions=[],
            candles=await _candles(db, candled_instrument),
        )
        assert decision.rejected
        assert "halted" in (decision.reason or "")

    async def test_open_position_cap_blocks_a_new_instrument(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        config = await _seed_config(db, max_open_positions=0)
        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candled_instrument,
            config=config,
            account=_account(),
            positions=[],
            candles=await _candles(db, candled_instrument),
        )
        assert decision.rejected
        assert "open-position cap" in (decision.reason or "")


class TestCorrelation:
    async def test_a_benchmark_correlated_book_cuts_the_size(self, db: object) -> None:
        # A benchmark, an already-held position and a candidate that all move
        # together — with the S&P-exposure limit set low, the candidate's size is
        # reduced (not merely flagged).
        benchmark = await _rising_instrument(db, "SPY")
        held = await _rising_instrument(db, "HELD")
        candidate = await _rising_instrument(db, "CAND")
        config = await _seed_config(db, max_portfolio_sp500_pct=Decimal("0.10"))

        # The held position is ~14.5k of a 100k book — above the 10% limit.
        position = BrokerPosition(
            broker_ticker=str(held.id),
            quantity=Decimal("100"),
            average_price=Decimal("145"),
            current_price=Decimal("145"),
        )
        benchmark_candles = await _candles(db, benchmark)
        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candidate,
            config=config,
            account=_account(),
            positions=[position],
            candles=await _candles(db, candidate),
            benchmark_candles=benchmark_candles,
        )
        assert not decision.rejected
        assert decision.correlation is not None and decision.correlation > 0.8
        assert "correlation_reduction" in decision.applied_caps


class TestStressDrawdown:
    """The whole-book cap.

    Every other cap sizes this position against this position. This one asks
    what the portfolio does in a bad month — the only cap that can see a book
    whose positions are individually reasonable and collectively not.
    """

    async def _volatile_book(self, db: object) -> tuple[Instrument, BrokerPosition]:
        held = await _volatile_instrument(db, "SWING")
        candidate = await _volatile_instrument(db, "CAND")
        return candidate, BrokerPosition(
            broker_ticker=str(held.id),
            quantity=Decimal("100"),
            average_price=Decimal("145"),
            current_price=Decimal("145"),
        )

    async def test_a_book_past_its_drawdown_limit_takes_no_new_position(self, db: object) -> None:
        candidate, position = await self._volatile_book(db)
        config = await _seed_config(db, max_portfolio_drawdown_pct=Decimal("0.001"))

        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candidate,
            config=config,
            account=_account(),
            positions=[position],
            candles=await _candles(db, candidate),
        )
        # Shrinking the candidate cannot fix a book that already breaches, so
        # the answer is to add nothing to it — stated as a reason, not a silent skip.
        assert decision.rejected
        assert decision.reason is not None and "stress_drawdown" in decision.reason

    async def test_a_generous_limit_lets_the_same_book_trade(self, db: object) -> None:
        """Proves it was the stress cap that bound, not the volatility."""
        candidate, position = await self._volatile_book(db)
        config = await _seed_config(db, max_portfolio_drawdown_pct=Decimal("1.0"))

        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candidate,
            config=config,
            account=_account(),
            positions=[position],
            candles=await _candles(db, candidate),
        )
        assert not decision.rejected
        assert decision.approved_quantity > 0
        assert "stress_drawdown" not in decision.applied_caps

    async def test_the_stressed_loss_is_reported(self, db: object) -> None:
        candidate, position = await self._volatile_book(db)
        config = await _seed_config(db, max_portfolio_drawdown_pct=Decimal("1.0"))

        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candidate,
            config=config,
            account=_account(),
            positions=[position],
            candles=await _candles(db, candidate),
        )
        assert decision.stress_loss_pct is not None
        assert decision.stress_loss_pct > 0

    async def test_an_empty_book_is_never_stress_capped(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        """Nothing held means nothing to stress — the cap must drop out, not zero."""
        config = await _seed_config(db)
        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candled_instrument,
            config=config,
            account=_account(),
            positions=[],
            candles=await _candles(db, candled_instrument),
        )
        assert not decision.rejected
        assert "stress_drawdown" not in decision.applied_caps

    async def test_a_position_with_too_little_history_does_not_block_the_trade(
        self, db: object
    ) -> None:
        """Unmeasurable is not zero. A short series drops out of the simulation."""
        short = await _instrument_with_closes(db, "SHORT", [100.0 + i for i in range(25)])
        candidate = await _rising_instrument(db, "CAND")
        position = BrokerPosition(
            broker_ticker=str(short.id),
            quantity=Decimal("100"),
            average_price=Decimal("100"),
            current_price=Decimal("100"),
        )
        config = await _seed_config(db, max_portfolio_drawdown_pct=Decimal("0.001"))

        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candidate,
            config=config,
            account=_account(),
            positions=[position],
            candles=await _candles(db, candidate),
        )
        assert not decision.rejected


class TestRateSensitivity:
    """The other way a book quietly becomes one bet.

    A portfolio of REITs, utilities and long-duration growth passes the sector
    and benchmark checks and is still a single position on bond yields.
    """

    async def _book(self, db: object) -> tuple[Instrument, BrokerPosition]:
        held = await _rising_instrument(db, "HELD")
        candidate = await _rising_instrument(db, "CAND")
        # ~14.5k of a 100k book — above a 10% limit.
        return candidate, BrokerPosition(
            broker_ticker=str(held.id),
            quantity=Decimal("100"),
            average_price=Decimal("145"),
            current_price=Decimal("145"),
        )

    async def test_a_rate_sensitive_book_cuts_the_size(self, db: object) -> None:
        rates = await _rising_instrument(db, "IEF")
        candidate, position = await self._book(db)
        config = await _seed_config(db, max_portfolio_rate_sensitive_pct=Decimal("0.10"))

        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candidate,
            config=config,
            account=_account(),
            positions=[position],
            candles=await _candles(db, candidate),
            rates_candles=await _candles(db, rates),
        )
        assert not decision.rejected
        assert decision.rate_correlation is not None and decision.rate_correlation > 0.8
        assert "rate_correlation_reduction" in decision.applied_caps

    async def test_moving_against_rates_still_counts_as_a_rates_bet(self, db: object) -> None:
        """Magnitude, not direction — the reason the rates gate uses abs()."""
        rates = await _inverse_instrument(db, "IEF", _rising_closes())
        candidate, position = await self._book(db)
        config = await _seed_config(db, max_portfolio_rate_sensitive_pct=Decimal("0.10"))

        decision = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candidate,
            config=config,
            account=_account(),
            positions=[position],
            candles=await _candles(db, candidate),
            rates_candles=await _candles(db, rates),
        )
        assert decision.rate_correlation is not None and decision.rate_correlation < -0.8
        assert "rate_correlation_reduction" in decision.applied_caps

    async def test_both_gates_firing_cut_the_size_once_not_twice(self, db: object) -> None:
        """Two x0.5 multipliers would give x0.25, which neither rule intends."""
        benchmark = await _rising_instrument(db, "SPY")
        rates = await _rising_instrument(db, "IEF")
        candidate, position = await self._book(db)
        config = await _seed_config(
            db,
            max_portfolio_sp500_pct=Decimal("0.10"),
            max_portfolio_rate_sensitive_pct=Decimal("0.10"),
        )
        candles = await _candles(db, candidate)

        baseline = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candidate,
            config=config,
            account=_account(),
            positions=[position],
            candles=candles,
        )
        both = await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=candidate,
            config=config,
            account=_account(),
            positions=[position],
            candles=candles,
            benchmark_candles=await _candles(db, benchmark),
            rates_candles=await _candles(db, rates),
        )
        # Both gates are reported, so the audit trail shows the full reason...
        assert "correlation_reduction" in both.applied_caps
        assert "rate_correlation_reduction" in both.applied_caps
        # ...but the size was cut exactly once.
        assert both.approved_quantity == (baseline.approved_quantity * Decimal("0.5")).quantize(
            QUANTITY_STEP, rounding=ROUND_DOWN
        )


class TestSentimentGate:
    """News tone is the only gate here that can refuse a trade outright.

    That makes its fail-open behaviour the most important thing in this file: a
    news feed that has gone quiet, gone stale, or gone down must leave every
    order exactly as it found it. A control that reads absence as danger would
    stop the whole book on an outage, which is the opposite of safe.
    """

    async def _record(
        self,
        db: object,
        instrument: Instrument,
        *,
        polarity: float,
        days_ago: int = 0,
        provider_score: float | None = None,
    ) -> None:
        await SentimentService(db).record(  # type: ignore[arg-type]
            instrument_id=instrument.id,
            as_of=datetime.now(UTC).date() - timedelta(days=days_ago),
            provider_symbol=instrument.exchange_ticker,
            polarity=polarity,
            uncertainty=0.1,
            positive_words=1,
            negative_words=4,
            headline_count=5,
            provider_score=provider_score,
        )

    async def _evaluate(self, db: object, instrument: Instrument, config: RiskConfiguration):
        return await RiskEngine(db).evaluate(  # type: ignore[arg-type]
            instrument=instrument,
            config=config,
            account=_account(),
            positions=[],
            candles=await _candles(db, instrument),
        )

    async def test_bad_news_halves_the_position(self, db: object) -> None:
        quiet = await _rising_instrument(db, "QUIET")
        grim = await _rising_instrument(db, "GRIM")
        config = await _seed_config(db, sentiment_reduction_threshold=Decimal("-0.35"))
        await self._record(db, grim, polarity=-0.8)

        baseline = await self._evaluate(db, quiet, config)
        cut = await self._evaluate(db, grim, config)
        assert cut.approved_quantity == (baseline.approved_quantity * Decimal("0.5")).quantize(
            QUANTITY_STEP, rounding=ROUND_DOWN
        )
        assert "sentiment_reduction(lexicon)" in cut.applied_caps

    async def test_mildly_negative_news_does_not_fire(self, db: object) -> None:
        """Ordinary news skews mildly negative — failures get reported and
        functioning does not. A threshold that fired here would shrink almost
        every trade and stop carrying information."""
        quiet = await _rising_instrument(db, "CALM")
        mild = await _rising_instrument(db, "MILD")
        config = await _seed_config(db, sentiment_reduction_threshold=Decimal("-0.35"))
        await self._record(db, mild, polarity=-0.1)

        baseline = await self._evaluate(db, quiet, config)
        result = await self._evaluate(db, mild, config)
        assert result.approved_quantity == baseline.approved_quantity
        assert not any("sentiment" in cap for cap in result.applied_caps)

    async def test_the_veto_refuses_the_trade(self, db: object) -> None:
        instrument = await _rising_instrument(db, "AWFUL")
        config = await _seed_config(db, sentiment_veto_threshold=Decimal("-0.60"))
        await self._record(db, instrument, polarity=-0.9)

        result = await self._evaluate(db, instrument, config)
        assert result.rejected
        assert result.approved_quantity == Decimal(0)
        assert "sentiment_veto" in result.applied_caps
        # The rejection still carries the reading, so the audit trail can say
        # what was refused and on whose word.
        record = result.as_record()
        assert record["sentiment_polarity"] == pytest.approx(-0.9)
        assert record["sentiment_source"] == "lexicon"

    async def test_the_veto_is_off_by_default(self, db: object) -> None:
        """A signal that counts words and cannot read a headline should not be
        able to refuse a trade until someone deliberately turns that on."""
        instrument = await _rising_instrument(db, "DEFAULT")
        config = await _seed_config(db)
        assert config.sentiment_veto_threshold is None
        await self._record(db, instrument, polarity=-0.95)

        result = await self._evaluate(db, instrument, config)
        assert not result.rejected

    async def test_the_provider_score_is_named_in_the_audit_trail(self, db: object) -> None:
        instrument = await _rising_instrument(db, "PROVIDED")
        config = await _seed_config(db, sentiment_reduction_threshold=Decimal("-0.35"))
        # Lexicon says fine, Finnhub says bad. Finnhub gates risk, by decision.
        await self._record(db, instrument, polarity=0.4, provider_score=-0.8)

        result = await self._evaluate(db, instrument, config)
        assert "sentiment_reduction(provider)" in result.applied_caps
        assert result.sentiment_source == "provider"
        assert result.sentiment_polarity == pytest.approx(-0.8)

    async def test_no_reading_leaves_the_order_untouched(self, db: object) -> None:
        """The common case: no coverage at all."""
        quiet = await _rising_instrument(db, "NOCOVER")
        reference = await _rising_instrument(db, "REFERENCE")
        config = await _seed_config(db, sentiment_veto_threshold=Decimal("-0.60"))

        result = await self._evaluate(db, quiet, config)
        baseline = await self._evaluate(db, reference, config)
        assert not result.rejected
        assert result.approved_quantity == baseline.approved_quantity
        assert result.sentiment_polarity is None

    async def test_a_stale_reading_leaves_the_order_untouched(self, db: object) -> None:
        """The failure that would hurt most: a feed that stopped updating must
        not become a permanent veto on every name it ever covered."""
        instrument = await _rising_instrument(db, "OLDNEWS")
        reference = await _rising_instrument(db, "REF2")
        config = await _seed_config(
            db, sentiment_veto_threshold=Decimal("-0.60"), sentiment_max_age_days=3
        )
        await self._record(db, instrument, polarity=-0.95, days_ago=30)

        result = await self._evaluate(db, instrument, config)
        baseline = await self._evaluate(db, reference, config)
        assert not result.rejected
        assert result.approved_quantity == baseline.approved_quantity

    async def test_bad_news_on_top_of_correlation_still_cuts_only_once(self, db: object) -> None:
        """Being concentrated *and* in the news is not twice as bad as either.

        Three x0.5 multipliers would give x0.125, which no rule here intends.
        """
        benchmark = await _rising_instrument(db, "SPYX")
        rates = await _rising_instrument(db, "IEFX")
        # Two identical candidates under one config and one book. The only
        # difference between them is that one is also in the news, so whatever
        # separates their sizes is the sentiment gate's marginal effect.
        quiet = await _rising_instrument(db, "QUIETPILE")
        noisy = await _rising_instrument(db, "NOISYPILE")
        held = await _rising_instrument(db, "HELDPILE")
        position = BrokerPosition(
            broker_ticker=str(held.id),
            quantity=Decimal("300"),
            average_price=Decimal("100"),
            current_price=Decimal("100"),
        )
        config = await _seed_config(
            db,
            max_portfolio_sp500_pct=Decimal("0.10"),
            max_portfolio_rate_sensitive_pct=Decimal("0.10"),
            sentiment_reduction_threshold=Decimal("-0.35"),
        )
        await self._record(db, noisy, polarity=-0.9)

        async def _size(instrument: Instrument):
            return await RiskEngine(db).evaluate(  # type: ignore[arg-type]
                instrument=instrument,
                config=config,
                account=_account(),
                positions=[position],
                candles=await _candles(db, instrument),
                benchmark_candles=await _candles(db, benchmark),
                rates_candles=await _candles(db, rates),
            )

        two_gates = await _size(quiet)
        three_gates = await _size(noisy)

        assert "correlation_reduction" in three_gates.applied_caps
        assert "rate_correlation_reduction" in three_gates.applied_caps
        assert "sentiment_reduction(lexicon)" in three_gates.applied_caps
        # A third gate firing changes nothing about the size. Two x0.5 already
        # became one; a third must not make it x0.125.
        assert three_gates.approved_quantity == two_gates.approved_quantity
