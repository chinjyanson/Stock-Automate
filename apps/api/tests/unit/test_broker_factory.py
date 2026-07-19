"""Broker selection safety (§17).

The rules under test are the ones whose violation loses real money:
live trading is off unless deliberately enabled, a live request never silently
becomes a simulator, and the default is never live.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from app.broker import (
    BrokerNotConfiguredError,
    InternalPaperBroker,
    LiveTradingDisabledError,
    MockBroker,
    Trading212DemoBroker,
    Trading212LiveBroker,
    default_paper_broker_kind,
    resolve_broker,
)
from app.config import Settings
from app.models.enums import BrokerKind


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "test",
        "live_trading_enabled": False,
        "trading212_demo_api_key": None,
        "trading212_demo_api_secret": None,
        "trading212_live_api_key": None,
        "trading212_live_api_secret": None,
    }
    base.update(overrides)

    # Trading 212 uses key + secret. Most tests care only that a *credential* is
    # present, so a key without an explicit secret gets a matching one here —
    # keeping those tests about the behaviour they name. Tests that exercise a
    # half-configured credential pass the secret (or its absence) explicitly.
    for env in ("demo", "live"):
        if (
            base.get(f"trading212_{env}_api_key")
            and f"trading212_{env}_api_secret" not in overrides
        ):
            base[f"trading212_{env}_api_secret"] = SecretStr(f"{env}-secret")

    return Settings(**base)  # type: ignore[arg-type]


class TestLiveTradingGate:
    def test_live_is_refused_when_the_server_flag_is_off(self) -> None:
        """LIVE_TRADING_ENABLED=false must make live unreachable.

        Even with a valid live key present — the flag is the outer gate.
        """
        settings = _settings(
            live_trading_enabled=False,
            trading212_live_api_key=SecretStr("a-real-looking-key"),
        )
        with pytest.raises(LiveTradingDisabledError):
            resolve_broker(BrokerKind.TRADING212_LIVE, settings)

    def test_live_without_credentials_raises_rather_than_falling_back(self) -> None:
        """The single most dangerous possible fallback.

        A caller that asked for live and received a mock would believe it had
        traded. It must raise instead.
        """
        settings = _settings(live_trading_enabled=True, trading212_live_api_key=None)
        with pytest.raises(BrokerNotConfiguredError):
            resolve_broker(BrokerKind.TRADING212_LIVE, settings)

    def test_live_with_empty_string_key_is_treated_as_absent(self) -> None:
        settings = _settings(live_trading_enabled=True, trading212_live_api_key=SecretStr(""))
        with pytest.raises(BrokerNotConfiguredError):
            resolve_broker(BrokerKind.TRADING212_LIVE, settings)

    def test_live_is_constructed_only_when_flag_and_key_are_both_present(self) -> None:
        settings = _settings(
            live_trading_enabled=True, trading212_live_api_key=SecretStr("live-key")
        )
        broker = resolve_broker(BrokerKind.TRADING212_LIVE, settings)
        assert isinstance(broker, Trading212LiveBroker)
        assert broker.is_live

    def test_is_live_is_derived_from_the_class_not_configuration(self) -> None:
        """`is_live` must not be spoofable by configuration."""
        settings = _settings(
            live_trading_enabled=True,
            trading212_live_api_key=SecretStr("live-key"),
            trading212_demo_api_key=SecretStr("demo-key"),
        )
        live = resolve_broker(BrokerKind.TRADING212_LIVE, settings)
        demo = resolve_broker(BrokerKind.TRADING212_DEMO, settings)

        assert live.is_live is True
        assert demo.is_live is False
        assert MockBroker().is_live is False


class TestNoSubstitution:
    """A requested venue is honoured or refused — never swapped."""

    def test_demo_without_credentials_raises_rather_than_mocking(self) -> None:
        """The old behaviour silently returned a mock here.

        That was worse than an error: the caller would see filled orders and
        moving cash against invented fixture prices, while believing it was
        exercising Trading 212's real tickers, rate limits and order semantics.
        """
        settings = _settings(trading212_demo_api_key=None)
        with pytest.raises(BrokerNotConfiguredError, match="TRADING212_DEMO_API_KEY"):
            resolve_broker(BrokerKind.TRADING212_DEMO, settings)

    def test_demo_with_empty_string_key_is_treated_as_absent(self) -> None:
        settings = _settings(trading212_demo_api_key=SecretStr(""))
        with pytest.raises(BrokerNotConfiguredError):
            resolve_broker(BrokerKind.TRADING212_DEMO, settings)

    def test_demo_with_key_but_no_secret_is_refused(self) -> None:
        """Trading 212 needs both halves; a key alone cannot authenticate.

        This is the exact shape of the real-world 401 that prompted the switch
        to Basic auth — a saved key with the secret never recorded.
        """
        settings = _settings(
            trading212_demo_api_key=SecretStr("demo-key"),
            trading212_demo_api_secret=None,
        )
        with pytest.raises(BrokerNotConfiguredError, match="TRADING212_DEMO_API_SECRET"):
            resolve_broker(BrokerKind.TRADING212_DEMO, settings)

    def test_demo_with_secret_but_no_key_is_refused(self) -> None:
        settings = _settings(
            trading212_demo_api_key=None,
            trading212_demo_api_secret=SecretStr("demo-secret"),
        )
        with pytest.raises(BrokerNotConfiguredError, match="TRADING212_DEMO_API_KEY"):
            resolve_broker(BrokerKind.TRADING212_DEMO, settings)

    def test_demo_with_both_halves_uses_the_real_demo_adapter(self) -> None:
        settings = _settings(
            trading212_demo_api_key=SecretStr("demo-key"),
            trading212_demo_api_secret=SecretStr("demo-secret"),
        )
        broker = resolve_broker(BrokerKind.TRADING212_DEMO, settings)
        assert isinstance(broker, Trading212DemoBroker)
        assert broker.kind is BrokerKind.TRADING212_DEMO

    def test_mock_is_reachable_only_by_asking_for_it(self) -> None:
        """The mock is not gone — it is opt-in.

        Nothing falls back to it, but naming it explicitly still works, which is
        what keeps offline tests and CI runnable without credentials.
        """
        broker = resolve_broker(BrokerKind.MOCK, _settings())
        assert isinstance(broker, MockBroker)
        assert not broker.is_live

    def test_no_configuration_can_make_a_request_return_a_different_venue(self) -> None:
        settings = _settings(
            trading212_demo_api_key=SecretStr("demo-key"),
            trading212_live_api_key=SecretStr("live-key"),
            live_trading_enabled=True,
        )
        assert resolve_broker(BrokerKind.MOCK, settings).kind is BrokerKind.MOCK
        assert (
            resolve_broker(BrokerKind.TRADING212_DEMO, settings).kind is BrokerKind.TRADING212_DEMO
        )
        assert (
            resolve_broker(BrokerKind.TRADING212_LIVE, settings).kind is BrokerKind.TRADING212_LIVE
        )


class TestInternalPaper:
    """The internal paper venue is DB-backed, so it needs a session — and it is
    never silently swapped for something that does not."""

    def test_internal_paper_without_a_session_is_refused(self) -> None:
        # Without a session there is no venue to back it. Raising keeps the gap
        # visible instead of handing back an in-memory stand-in.
        with pytest.raises(BrokerNotConfiguredError, match="session"):
            resolve_broker(BrokerKind.INTERNAL_PAPER, _settings())

    def test_internal_paper_with_a_session_constructs_that_venue(self) -> None:
        # A sentinel stands in for the AsyncSession; construction stores it and
        # touches no database, so this stays a pure unit test.
        broker = resolve_broker(BrokerKind.INTERNAL_PAPER, _settings(), session=object())  # type: ignore[arg-type]
        assert isinstance(broker, InternalPaperBroker)
        assert broker.kind is BrokerKind.INTERNAL_PAPER
        assert not broker.is_live


class TestDefaultBroker:
    def test_default_is_demo(self) -> None:
        settings = _settings(trading212_demo_api_key=SecretStr("demo-key"))
        assert default_paper_broker_kind(settings) is BrokerKind.TRADING212_DEMO

    def test_default_is_demo_even_with_no_credentials(self) -> None:
        """The default does not change shape based on what is configured.

        With no demo key this yields a kind that `resolve_broker` will refuse —
        an error, not a mock. Choosing a different venue because a key is
        missing is precisely the substitution we removed.
        """
        settings = _settings(trading212_demo_api_key=None)
        assert default_paper_broker_kind(settings) is BrokerKind.TRADING212_DEMO
        with pytest.raises(BrokerNotConfiguredError):
            resolve_broker(default_paper_broker_kind(settings), settings)

    def test_default_is_never_live_even_when_live_is_fully_configured(self) -> None:
        """§7: the default mode must be demo or internal paper. Never live."""
        settings = _settings(
            live_trading_enabled=True,
            trading212_live_api_key=SecretStr("live-key"),
            trading212_demo_api_key=SecretStr("demo-key"),
        )
        assert default_paper_broker_kind(settings) is not BrokerKind.TRADING212_LIVE
        assert not resolve_broker(default_paper_broker_kind(settings), settings).is_live

    def test_default_is_not_live_when_only_live_is_configured(self) -> None:
        """Configuring live must not make it the default by omission."""
        settings = _settings(
            live_trading_enabled=True,
            trading212_live_api_key=SecretStr("live-key"),
            trading212_demo_api_key=None,
        )
        assert default_paper_broker_kind(settings) is not BrokerKind.TRADING212_LIVE
