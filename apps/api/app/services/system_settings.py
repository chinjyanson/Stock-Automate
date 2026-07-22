"""Runtime settings overlay (§4).

`app.config.Settings` supplies defaults from the environment at boot. A
`SystemSetting` row overlays one of them so it can be changed without a redeploy;
once a row exists, it wins. This module is the single place that resolves that
overlay for the settings we expose to operators, so "what is the effective value"
has one answer.

The autonomous-live flag is the sensitive one: it decides whether a strategy may
place real-money orders with no per-trade human. Its default is still off (the
env default), and changing it is an admin, re-authenticated, audited action —
moving it from env-only to editable does not remove those guards, it moves them
from "needs shell access" to "needs admin + a fresh password".
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.enums import BrokerKind
from app.models.system import SystemSetting

#: Runtime key for the autonomous-live opt-in.
AUTONOMOUS_LIVE_KEY = "live.autonomous_enabled"

#: Runtime key for the scheduled-scan toggle.
SCANNER_AUTORUN_KEY = "scanner.auto_run_enabled"

#: Runtime key for the paper/live venue toggle. False = paper, True = live.
TRADING_LIVE_MODE_KEY = "trading.live_mode"


async def get_bool_setting(session: AsyncSession, key: str, *, env_default: bool) -> bool:
    """The effective value of a boolean setting: DB overlay, else the env default."""
    row = (
        await session.execute(select(SystemSetting).where(SystemSetting.key == key))
    ).scalar_one_or_none()
    if row is not None and isinstance(row.value, dict) and "value" in row.value:
        return bool(row.value["value"])
    return env_default


async def set_bool_setting(
    session: AsyncSession,
    key: str,
    value: bool,
    *,
    description: str,
    is_sensitive: bool,
    user_id: uuid.UUID | None,
) -> SystemSetting:
    """Upsert a boolean setting. The caller owns the commit and the audit record."""
    row = (
        await session.execute(select(SystemSetting).where(SystemSetting.key == key))
    ).scalar_one_or_none()
    if row is None:
        row = SystemSetting(key=key, value={"value": bool(value)}, description=description)
        session.add(row)
    else:
        row.value = {"value": bool(value)}
    row.is_sensitive = is_sensitive
    row.updated_by_user_id = user_id
    await session.flush()
    return row


async def autonomous_live_enabled(session: AsyncSession) -> bool:
    """Whether autonomous live trading is permitted — DB overlay over the env default."""
    return await get_bool_setting(
        session, AUTONOMOUS_LIVE_KEY, env_default=get_settings().live_autonomous_enabled
    )


async def scanner_auto_run_enabled(session: AsyncSession) -> bool:
    """Whether the scheduled rotating scan runs — DB overlay over the env default."""
    return await get_bool_setting(
        session, SCANNER_AUTORUN_KEY, env_default=get_settings().scanner_auto_run_enabled
    )


async def live_mode_enabled(session: AsyncSession) -> bool:
    """Whether the product is pointed at the live venue rather than paper.

    True does not by itself permit a real order: `LIVE_TRADING_ENABLED` and live
    credentials are still required, and the execution preflight re-checks both.
    """
    return await get_bool_setting(
        session, TRADING_LIVE_MODE_KEY, env_default=get_settings().trading_live_mode
    )


async def active_broker_kind(session: AsyncSession) -> BrokerKind:
    """The single venue the product is currently trading and reporting on.

    Paper is the Trading 212 *demo* account; live is the real one. One resolver so
    the portfolio, the dashboard, execution, strategies and the scheduled jobs can
    never disagree about which account is "the" account.
    """
    if await live_mode_enabled(session):
        return BrokerKind.TRADING212_LIVE
    return BrokerKind.TRADING212_DEMO
