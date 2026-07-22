"""Runtime settings overlay against real PostgreSQL.

The autonomous-live flag defaults off (the env default) and is turned on/off by a
DB overlay — the mechanism behind the admin toggle. Once a row exists, it wins.
"""

from __future__ import annotations

import pytest

from app.services.system_settings import (
    AUTONOMOUS_LIVE_KEY,
    SCANNER_AUTORUN_KEY,
    autonomous_live_enabled,
    scanner_auto_run_enabled,
    set_bool_setting,
)

pytestmark = pytest.mark.asyncio


class TestScannerAutoRun:
    async def test_defaults_on(self, db: object) -> None:
        # Scanning is read-only and safe, so it defaults to on (the env default).
        assert await scanner_auto_run_enabled(db) is True  # type: ignore[arg-type]

    async def test_overlay_can_turn_it_off(self, db: object) -> None:
        await set_bool_setting(
            db,  # type: ignore[arg-type]
            SCANNER_AUTORUN_KEY,
            False,
            description="x",
            is_sensitive=False,
            user_id=None,
        )
        assert await scanner_auto_run_enabled(db) is False  # type: ignore[arg-type]


class TestAutonomousOverlay:
    async def test_defaults_off_without_a_row(self, db: object) -> None:
        assert await autonomous_live_enabled(db) is False  # type: ignore[arg-type]

    async def test_overlay_enables_it(self, db: object) -> None:
        await set_bool_setting(
            db,  # type: ignore[arg-type]
            AUTONOMOUS_LIVE_KEY,
            True,
            description="x",
            is_sensitive=True,
            user_id=None,
        )
        assert await autonomous_live_enabled(db) is True  # type: ignore[arg-type]

    async def test_overlay_can_disable_again(self, db: object) -> None:
        for value in (True, False):
            await set_bool_setting(
                db,  # type: ignore[arg-type]
                AUTONOMOUS_LIVE_KEY,
                value,
                description="x",
                is_sensitive=True,
                user_id=None,
            )
        assert await autonomous_live_enabled(db) is False  # type: ignore[arg-type]
