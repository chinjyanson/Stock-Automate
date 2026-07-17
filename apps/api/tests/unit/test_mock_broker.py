"""Mock broker behaviour.

The mock underpins every credential-free test, so its own accounting has to be
right — a simulator that silently lets you spend cash you do not have would
validate strategies that cannot exist.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.broker import BrokerOrderRejectedError, BrokerOrderRequest, MockBroker
from app.models.enums import BrokerKind, OrderSide, OrderStatus, OrderType


@pytest.fixture
def broker() -> MockBroker:
    return MockBroker(starting_cash=Decimal("10000.00"))


class TestCatalogue:
    async def test_sync_returns_a_stable_catalogue(self, broker: MockBroker) -> None:
        first = await broker.sync_instruments()
        second = await broker.sync_instruments()
        assert [i.broker_ticker for i in first] == [i.broker_ticker for i in second]
        assert len(first) > 0

    async def test_fixtures_carry_real_isins(self, broker: MockBroker) -> None:
        # Identity is what the mapping layer resolves, so the ISINs must be real
        # even though the prices are invented.
        instruments = {i.broker_ticker: i for i in await broker.sync_instruments()}
        assert instruments["VUAGl_EQ"].isin == "IE00BFMXXD54"
        assert instruments["SPY_US_EQ"].isin == "US78462F1030"

    async def test_is_never_live(self, broker: MockBroker) -> None:
        assert broker.kind is BrokerKind.MOCK
        assert not broker.is_live


class TestCashAccounting:
    async def test_buy_decreases_cash_and_opens_a_position(self, broker: MockBroker) -> None:
        await broker.place_order(
            BrokerOrderRequest(
                broker_ticker="AAPL_US_EQ", side=OrderSide.BUY, quantity=Decimal("10")
            )
        )
        account = await broker.get_account()
        positions = await broker.get_positions()

        # 10 @ 232.15 = 2321.50
        assert account.cash == Decimal("10000.00") - Decimal("2321.50")
        assert len(positions) == 1
        assert positions[0].quantity == Decimal("10")
        assert positions[0].average_price == Decimal("232.15")

    async def test_buy_beyond_cash_is_rejected(self, broker: MockBroker) -> None:
        with pytest.raises(BrokerOrderRejectedError, match="Insufficient cash"):
            await broker.place_order(
                BrokerOrderRequest(
                    broker_ticker="AAPL_US_EQ",
                    side=OrderSide.BUY,
                    quantity=Decimal("1000"),
                )
            )
        # The rejection must not have moved anything.
        account = await broker.get_account()
        assert account.cash == Decimal("10000.00")
        assert await broker.get_positions() == []

    async def test_selling_more_than_held_is_rejected(self, broker: MockBroker) -> None:
        with pytest.raises(BrokerOrderRejectedError, match="Cannot sell"):
            await broker.place_order(
                BrokerOrderRequest(
                    broker_ticker="AAPL_US_EQ", side=OrderSide.SELL, quantity=Decimal("1")
                )
            )

    async def test_averaging_up_recomputes_weighted_cost(self, broker: MockBroker) -> None:
        for _ in range(2):
            await broker.place_order(
                BrokerOrderRequest(
                    broker_ticker="AAPL_US_EQ", side=OrderSide.BUY, quantity=Decimal("5")
                )
            )
        positions = await broker.get_positions()
        assert positions[0].quantity == Decimal("10")
        # Both fills at the same fixture price, so the average is that price.
        assert positions[0].average_price == Decimal("232.15")

    async def test_round_trip_restores_cash(self, broker: MockBroker) -> None:
        request = BrokerOrderRequest(
            broker_ticker="AAPL_US_EQ", side=OrderSide.BUY, quantity=Decimal("10")
        )
        await broker.place_order(request)
        await broker.place_order(
            BrokerOrderRequest(
                broker_ticker="AAPL_US_EQ", side=OrderSide.SELL, quantity=Decimal("10")
            )
        )
        account = await broker.get_account()
        # No commission or slippage modelled in the mock; the simulator in
        # Phase 3 is where those assumptions live.
        assert account.cash == Decimal("10000.00")

    async def test_unknown_instrument_is_rejected(self, broker: MockBroker) -> None:
        with pytest.raises(BrokerOrderRejectedError, match="Unknown instrument"):
            await broker.place_order(
                BrokerOrderRequest(
                    broker_ticker="NOPE_EQ", side=OrderSide.BUY, quantity=Decimal("1")
                )
            )


class TestOrderValidation:
    async def test_zero_quantity_is_rejected(self, broker: MockBroker) -> None:
        with pytest.raises(ValueError, match="quantity must be positive"):
            await broker.place_order(
                BrokerOrderRequest(
                    broker_ticker="AAPL_US_EQ", side=OrderSide.BUY, quantity=Decimal("0")
                )
            )

    async def test_negative_quantity_is_rejected(self, broker: MockBroker) -> None:
        # Side carries direction; a negative quantity is a caller bug.
        with pytest.raises(ValueError, match="quantity must be positive"):
            await broker.place_order(
                BrokerOrderRequest(
                    broker_ticker="AAPL_US_EQ", side=OrderSide.BUY, quantity=Decimal("-5")
                )
            )

    async def test_limit_order_without_a_limit_price_is_rejected(self, broker: MockBroker) -> None:
        with pytest.raises(ValueError, match="requires limit_price"):
            await broker.place_order(
                BrokerOrderRequest(
                    broker_ticker="AAPL_US_EQ",
                    side=OrderSide.BUY,
                    quantity=Decimal("1"),
                    order_type=OrderType.LIMIT,
                )
            )

    async def test_stop_order_without_a_stop_price_is_rejected(self, broker: MockBroker) -> None:
        with pytest.raises(ValueError, match="requires stop_price"):
            await broker.place_order(
                BrokerOrderRequest(
                    broker_ticker="AAPL_US_EQ",
                    side=OrderSide.BUY,
                    quantity=Decimal("1"),
                    order_type=OrderType.STOP,
                )
            )


class TestInjectedFailures:
    async def test_the_next_order_can_be_forced_to_fail(self, broker: MockBroker) -> None:
        """Lets tests exercise the ambiguous-response path deterministically (§11)."""
        broker.fail_next_order_with = BrokerOrderRejectedError("simulated venue refusal")

        with pytest.raises(BrokerOrderRejectedError, match="simulated venue refusal"):
            await broker.place_order(
                BrokerOrderRequest(
                    broker_ticker="AAPL_US_EQ", side=OrderSide.BUY, quantity=Decimal("1")
                )
            )

        # The injection is one-shot: the next order succeeds.
        order = await broker.place_order(
            BrokerOrderRequest(
                broker_ticker="AAPL_US_EQ", side=OrderSide.BUY, quantity=Decimal("1")
            )
        )
        assert order.status is OrderStatus.FILLED


class TestCancellation:
    async def test_cancelling_a_filled_order_is_a_noop(self, broker: MockBroker) -> None:
        order = await broker.place_order(
            BrokerOrderRequest(
                broker_ticker="AAPL_US_EQ", side=OrderSide.BUY, quantity=Decimal("1")
            )
        )
        # Must not raise: cancelling something already terminal is not an error.
        await broker.cancel_order(order.broker_order_id)
        history = await broker.get_order_history()
        assert history[0].status is OrderStatus.FILLED

    async def test_cancelling_an_unknown_order_is_a_noop(self, broker: MockBroker) -> None:
        await broker.cancel_order("does-not-exist")


class TestReconciliation:
    async def test_clean_when_nothing_diverges(self, broker: MockBroker) -> None:
        result = await broker.reconcile()
        assert result.is_clean
        assert result.broker is BrokerKind.MOCK
