"""Strategy construction from a configuration.

One place maps a `StrategyKind` to its implementation, so the engine never grows
a chain of `if kind == ...`. A configuration whose kind has no implementation is
an error, not a silent no-op.
"""

from __future__ import annotations

from app.models.enums import StrategyKind
from app.models.strategy import StrategyConfiguration
from app.strategies.base import Strategy
from app.strategies.mean_reversion import MeanReversionStrategy
from app.strategies.pie import PieRebalanceStrategy
from app.strategies.trend import TrendFollowingStrategy

_REGISTRY: dict[StrategyKind, type[Strategy]] = {
    StrategyKind.SP500_MEAN_REVERSION: MeanReversionStrategy,
    StrategyKind.TREND_FOLLOWING: TrendFollowingStrategy,
    StrategyKind.PIE_REBALANCE: PieRebalanceStrategy,
}


def build_strategy(config: StrategyConfiguration) -> Strategy:
    """Instantiate the strategy for `config`, or raise if the kind is unknown."""
    try:
        strategy_cls = _REGISTRY[config.kind]
    except KeyError as exc:
        raise ValueError(f"No strategy implementation for kind {config.kind}") from exc
    return strategy_cls(config)
