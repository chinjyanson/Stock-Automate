"""Trading strategies (§8).

A strategy is a pure opinion: given candles it has read from the local store, it
returns signals. It never sizes an order, never talks to a broker, and never
submits anything — the `StrategyEngine` turns a signal into a proposal and the
risk engine disposes. Long-only in this phase.
"""

from app.strategies.base import Strategy, StrategyContext, StrategySignal
from app.strategies.registry import build_strategy

__all__ = ["Strategy", "StrategyContext", "StrategySignal", "build_strategy"]
