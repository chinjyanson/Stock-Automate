"""The index-timing strategy's decision rules.

`StrategyContext` hands a strategy everything it needs, so this exercises the
real `evaluate` against hand-built conditions with no database behind it — the
same discipline the mean-reversion unit path follows.

The case that matters most is the one where the options reading is missing.
This strategy's entire thesis is the options surface, so losing it must mean
"no opinion" and never "sell": a provider outage that liquidated a position
would be a data failure expressing itself as a trade.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from app.broker.types import BrokerPosition
from app.models.enums import OrderSide, StrategyKind
from app.models.strategy import StrategyConfiguration
from app.strategies.base import IndexConditions, StrategyContext
from app.strategies.index_timing import IndexTimingStrategy

pytestmark = pytest.mark.asyncio

_INSTRUMENT_ID = uuid.uuid4()


class _FakeInstrument:
    id = _INSTRUMENT_ID
    name = "Vanguard S&P 500 UCITS ETF"


def _config(**params: object) -> StrategyConfiguration:
    return StrategyConfiguration(
        kind=StrategyKind.INDEX_TIMING, name="index", is_active=True, params=params or None
    )


def _ctx(
    conditions: IndexConditions, *, held: Decimal = Decimal(0), **params: object
) -> StrategyContext:
    positions = (
        [
            BrokerPosition(
                broker_ticker=str(_INSTRUMENT_ID),
                quantity=held,
                average_price=Decimal("100"),
                current_price=Decimal("100"),
            )
        ]
        if held > 0
        else []
    )
    return StrategyContext(
        config=_config(**params),
        store=None,  # type: ignore[arg-type]  — never touched; no series is read
        instruments=[_FakeInstrument()],  # type: ignore[list-item]
        positions=positions,
        index_conditions=conditions,
    )


def _good() -> IndexConditions:
    """Dealers long gamma, calm skew, healthy regime."""
    return IndexConditions(
        regime_factor=1.0,
        gamma_exposure=2.5,
        skew_25delta=0.02,
        atm_iv=0.15,
        contracts_used=400,
        options_available=True,
    )


async def _run(ctx: StrategyContext) -> list:
    return await IndexTimingStrategy(ctx.config).evaluate(ctx)


class TestEntry:
    async def test_positive_dealer_gamma_in_a_calm_regime_buys(self) -> None:
        signals = await _run(_ctx(_good()))
        assert len(signals) == 1
        assert signals[0].side is OrderSide.BUY

    async def test_negative_dealer_gamma_does_not_buy(self) -> None:
        """Dealers short gamma amplify moves — the opposite of the condition
        this strategy wants to be long into."""
        conditions = IndexConditions(
            regime_factor=1.0, gamma_exposure=-1.5, skew_25delta=0.02, options_available=True
        )
        assert await _run(_ctx(conditions)) == []

    async def test_a_risk_off_regime_vetoes_an_otherwise_good_entry(self) -> None:
        conditions = IndexConditions(
            regime_factor=0.60, gamma_exposure=2.5, skew_25delta=0.02, options_available=True
        )
        assert await _run(_ctx(conditions)) == []

    async def test_steep_skew_vetoes_an_otherwise_good_entry(self) -> None:
        """Cheap protection is not a reason to buy; dear protection is a reason
        not to. Skew is a veto, never a trigger."""
        conditions = IndexConditions(
            regime_factor=1.0, gamma_exposure=2.5, skew_25delta=0.15, options_available=True
        )
        assert await _run(_ctx(conditions)) == []

    async def test_it_does_not_add_to_an_existing_position(self) -> None:
        signals = await _run(_ctx(_good(), held=Decimal("10")))
        assert signals == []


class TestExit:
    async def test_dealers_turning_short_gamma_exits(self) -> None:
        conditions = IndexConditions(
            regime_factor=1.0, gamma_exposure=-2.0, skew_25delta=0.02, options_available=True
        )
        signals = await _run(_ctx(conditions, held=Decimal("10")))
        assert len(signals) == 1
        assert signals[0].side is OrderSide.SELL
        assert signals[0].target_quantity == Decimal("10")

    async def test_a_risk_off_regime_exits(self) -> None:
        conditions = IndexConditions(
            regime_factor=0.60, gamma_exposure=2.5, skew_25delta=0.02, options_available=True
        )
        signals = await _run(_ctx(conditions, held=Decimal("10")))
        assert len(signals) == 1
        assert signals[0].side is OrderSide.SELL

    async def test_steep_skew_exits(self) -> None:
        conditions = IndexConditions(
            regime_factor=1.0, gamma_exposure=2.5, skew_25delta=0.20, options_available=True
        )
        signals = await _run(_ctx(conditions, held=Decimal("10")))
        assert len(signals) == 1
        assert signals[0].side is OrderSide.SELL

    async def test_good_conditions_hold_the_position(self) -> None:
        assert await _run(_ctx(_good(), held=Decimal("10"))) == []


class TestMissingData:
    async def test_no_options_reading_means_no_opinion_not_a_sell(self) -> None:
        """The case worth being careful about.

        A provider outage must not express itself as a liquidation. Without the
        surface there is nothing to act on, in either direction.
        """
        blind = IndexConditions(regime_factor=1.0, options_available=False)
        assert await _run(_ctx(blind)) == []
        assert await _run(_ctx(blind, held=Decimal("10"))) == []

    async def test_a_reading_without_gamma_does_not_buy(self) -> None:
        partial = IndexConditions(
            regime_factor=1.0, gamma_exposure=None, skew_25delta=0.02, options_available=True
        )
        assert await _run(_ctx(partial)) == []


class TestConfigurability:
    async def test_the_entry_threshold_is_a_parameter(self) -> None:
        """A reading of +2.5 is an entry by default and not against a raised bar."""
        conditions = _good()
        assert await _run(_ctx(conditions)) != []
        assert await _run(_ctx(conditions, gex_enter=5.0)) == []
