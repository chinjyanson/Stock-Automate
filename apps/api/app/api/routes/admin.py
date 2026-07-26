"""Admin-only operational endpoints (§19).

Currently just visibility into the catalogue backfill — how far the tradable
universe has progressed from "known to the broker" to "scannable". The backfill
itself is *run* out-of-band (the CLI `python -m app.scripts.backfill_catalogue`,
or the Celery task `worker.jobs.backfill.backfill_catalogue`); this endpoint lets
an operator watch it fill.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import AuthContext, require_admin
from app.db import get_db
from app.services.backfill import BackfillService

router = APIRouter(prefix="/admin", tags=["admin"])
log = structlog.get_logger(__name__)


class BackfillStatusResponse(BaseModel):
    #: Instruments tradable on Trading 212 — the backfill's target universe.
    tradable: int
    #: …of those, with an active yfinance signal-source mapping.
    mapped: int
    #: …with at least one stored daily candle.
    candled: int
    #: …with enough history to actually be scored by the scanner.
    scannable: int


@router.get("/backfill/status", response_model=BackfillStatusResponse)
async def backfill_status(
    context: AuthContext = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> BackfillStatusResponse:
    """Funnel counts for the catalogue backfill (admin only)."""
    counts = await BackfillService(db).funnel_counts(trading212_only=True)
    return BackfillStatusResponse(
        tradable=counts.tradable,
        mapped=counts.mapped,
        candled=counts.candled,
        scannable=counts.scannable,
    )
