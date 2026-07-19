"""Loading the active risk configuration (§9).

No risk limit is a code constant — they live in `RiskConfiguration`, versioned
and audited. These module-level values are *only* the seed for a first install
and the documented fallback used when no row is active; a live system reads the
active row on every evaluation.
"""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.risk import RiskConfiguration

#: The default limits a fresh install is seeded with. Mirrors the column
#: defaults on `RiskConfiguration` and the control table in docs/risk-model.md.
DEFAULT_RISK_PER_TRADE = Decimal("0.01")
DEFAULT_ATR_STOP_MULTIPLIER = Decimal("2.0")
DEFAULT_MAX_POSITION_PCT = Decimal("0.10")
DEFAULT_MAX_INSTRUMENT_PCT = Decimal("0.20")
DEFAULT_MAX_TOTAL_OPEN_RISK_PCT = Decimal("0.06")
DEFAULT_MAX_OPEN_POSITIONS = 10
DEFAULT_MAX_TRADES_PER_DAY = 10


async def load_active_risk_config(session: AsyncSession) -> RiskConfiguration | None:
    """The single active configuration, or None if none has been seeded.

    A missing configuration is a fail-closed condition for the engine, not a cue
    to invent limits: the caller rejects rather than sizing against guesses.
    """
    return (
        await session.execute(
            select(RiskConfiguration).where(RiskConfiguration.is_active.is_(True)).limit(1)
        )
    ).scalar_one_or_none()


def default_configuration(name: str = "default") -> RiskConfiguration:
    """A `RiskConfiguration` populated with the documented defaults, for seeding."""
    return RiskConfiguration(name=name, is_active=True)
