"""Technical indicators.

Pure numeric functions (`functions`) plus the Decimal→float boundary
(`series`). No database access, no I/O — everything here is deterministic and
unit-testable without fixtures.
"""

from app.indicators.functions import (
    TRADING_DAYS_PER_MONTH,
    TRADING_DAYS_PER_YEAR,
    FloatArray,
    annualised_volatility,
    average_traded_value,
    average_true_range,
    average_volume,
    daily_returns,
    distance_from_high,
    distance_from_low,
    downside_deviation,
    largest_daily_loss,
    max_drawdown,
    position_in_range,
    relative_momentum,
    relative_strength_index,
    rolling_correlation,
    simple_moving_average,
    sma_series,
    sma_slope,
    stale_price_days,
    trailing_return,
    zero_volume_days,
)
from app.indicators.series import PriceSeries, candles_to_series

__all__ = [
    "TRADING_DAYS_PER_MONTH",
    "TRADING_DAYS_PER_YEAR",
    "FloatArray",
    "PriceSeries",
    "annualised_volatility",
    "average_traded_value",
    "average_true_range",
    "average_volume",
    "candles_to_series",
    "daily_returns",
    "distance_from_high",
    "distance_from_low",
    "downside_deviation",
    "largest_daily_loss",
    "max_drawdown",
    "position_in_range",
    "relative_momentum",
    "relative_strength_index",
    "rolling_correlation",
    "simple_moving_average",
    "sma_series",
    "sma_slope",
    "stale_price_days",
    "trailing_return",
    "zero_volume_days",
]
