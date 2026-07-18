"""Stock scanner (§6).

Product #1: screens Trading 212-supported instruments with transparent, basic
heuristics. Nothing here asserts an instrument is a good investment — results
are framed as screening/watchlist candidates against a configured screen.
"""

from app.scanner.engine import ScannerEngine, ScanSummary
from app.scanner.proposals import ProposalError, ProposalInputs, ProposalService
from app.scanner.rotation import select_instruments
from app.scanner.scoring import (
    DEFAULT_THRESHOLDS,
    DEFAULT_WEIGHTS,
    ScoreResult,
    ValueResult,
    classify,
    score_series,
    score_value,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "DEFAULT_WEIGHTS",
    "ProposalError",
    "ProposalInputs",
    "ProposalService",
    "ScanSummary",
    "ScannerEngine",
    "ScoreResult",
    "ValueResult",
    "classify",
    "score_series",
    "score_value",
    "select_instruments",
]
