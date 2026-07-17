"""Broker selection safety (§17).

The rules under test are the ones whose violation loses real money:
live trading is off unless deliberately enabled, a live request never silently
becomes a simulator, and the default is never live.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from app.broker import (
    BrokerAuthError,
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
        "trading212_live_api_key": None,
    }
    base.update(overrides)
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
        with pytest.raises(BrokerAuthError):
            resolve_broker(BrokerKind.TRADING212_LIVE, settings)

    def test_live_with_empty_string_key_is_treated_as_absent(self) -> None:
        settings = _settings(live_trading_enabled=True, trading212_live_api_key=SecretStr(""))
        with pytest.raises(BrokerAuthError):
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


class TestDemoFallback:
    def test_demo_without_credentials_degrades_to_mock(self) -> None:
        """Safe direction: mock is strictly less privileged than demo."""
        settings = _settings(trading212_demo_api_key=None)
        broker = resolve_broker(BrokerKind.TRADING212_DEMO, settings)
        assert isinstance(broker, MockBroker)
        assert not broker.is_live

    def test_demo_with_credentials_uses_the_real_demo_adapter(self) -> None:
        settings = _settings(trading212_demo_api_key=SecretStr("demo-key"))
        broker = resolve_broker(BrokerKind.TRADING212_DEMO, settings)
        assert isinstance(broker, Trading212DemoBroker)
        assert broker.kind is BrokerKind.TRADING212_DEMO


class TestDefaultBroker:
    def test_default_is_mock_when_nothing_is_configured(self) -> None:
        assert default_paper_broker_kind(_settings()) is BrokerKind.MOCK

    def test_default_is_demo_when_demo_credentials_exist(self) -> None:
        settings = _settings(trading212_demo_api_key=SecretStr("demo-key"))
        assert default_paper_broker_kind(settings) is BrokerKind.TRADING212_DEMO

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
        assert default_paper_broker_kind(settings) is BrokerKind.MOCK
