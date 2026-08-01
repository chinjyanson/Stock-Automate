"""Strategy construction from a configuration.

One place maps a `StrategyKind` to its implementation, so the engine never grows
a chain of `if kind == ...`. A configuration whose kind has no implementation is
an error, not a silent no-op.

The registry currently holds a single strategy. It stays a registry rather than
collapsing into a direct call because the seam is what keeps the engine ignorant
of which strategy it is running — the engine's job is gating, sizing and
recording, and it should not have to change to accommodate a second one.
"""

from __future__ import annotations

from app.models.enums import StrategyKind
from app.models.strategy import StrategyConfiguration
from app.strategies.base import Strategy
from app.strategies.mean_reversion import MeanReversionStrategy

_REGISTRY: dict[StrategyKind, type[Strategy]] = {
    StrategyKind.MEAN_REVERSION: MeanReversionStrategy,
}


def build_strategy(config: StrategyConfiguration) -> Strategy:
    """Instantiate the strategy for `config`, or raise if the kind is unknown."""
    try:
        strategy_cls = _REGISTRY[config.kind]
    except KeyError as exc:
        raise ValueError(f"No strategy implementation for kind {config.kind}") from exc
    return strategy_cls(config)
