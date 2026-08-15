"""Financial-text sentiment scoring.

Pure and offline: the lexicon is vendored, so scoring a headline costs
microseconds and no network. Fetching the headlines is somebody else's job.
"""

from app.sentiment.loughran_mcdonald import (
    NEGATION_WINDOW,
    NEGATORS,
    SentimentReading,
    score_headlines,
    score_text,
    tokenise,
)

__all__ = [
    "NEGATION_WINDOW",
    "NEGATORS",
    "SentimentReading",
    "score_headlines",
    "score_text",
    "tokenise",
]
