"""Score financial text with the Loughran-McDonald lexicon (§6).

Pure — no I/O, no model weights, no network. A headline is tokenised, matched
against three vendored word lists, and reduced to a polarity in [-1, +1] plus an
uncertainty fraction. Cost is a few microseconds and ~2MB of resident set for
the frozensets, which is the entire reason this is here rather than a
transformer: the worker runs in 448MB on a free-tier box, and a BERT-class model
costs more than that before it reads a single word.

Two properties are load-bearing:

  * **The lexicon is financial, not general.** "Liability", "cost", "tax" and
    "capital" are ordinary business vocabulary, and a general-purpose lexicon
    scores them as negative. Loughran and McDonald built these lists precisely
    because roughly three-quarters of Harvard-IV's negative hits in 10-Ks are
    words like those. Headlines about companies have the same problem.

  * **Negation is handled.** "Not profitable" is not a positive headline, and a
    bag-of-words counter says it is. A tone word preceded within
    `NEGATION_WINDOW` tokens by a negator is flipped, which is the standard LM
    refinement. It is deliberately asymmetric in effect rather than in rule: the
    same flip applies both ways, but negated positives ("no growth") are far
    more common in practice than negated negatives.

What this is not: it is not a claim to understand the text. It counts words. A
headline whose meaning depends on sarcasm, on a number, or on what it omits will
be scored wrongly, and that is why the reading is one bounded sub-signal inside
a category rather than anything that decides a trade on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.sentiment.lexicon import NEGATIVE, POSITIVE, UNCERTAINTY

#: Tokens that flip the tone word following them. Kept small and unambiguous —
#: "without" and "hardly" shade meaning rather than reversing it, and adding
#: them costs more false flips than it fixes.
NEGATORS = frozenset(
    ("NO", "NOT", "NONE", "NEITHER", "NEVER", "NOBODY", "NOTHING", "NOWHERE", "CANNOT")
)

#: How many tokens after a negator stay flipped. Three is the LM convention and
#: matches how headlines are written ("not a strong quarter"). Widening it makes
#: a negator at the start of a sentence poison the whole line.
NEGATION_WINDOW = 3

#: Words are letters only. Tickers, numbers and punctuation carry no tone and
#: would otherwise dilute the denominator.
_WORD = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True, slots=True)
class SentimentReading:
    """Tone of a body of text, or of a batch of headlines.

    `polarity` is bounded to [-1, +1]: -1 is unrelieved bad news, 0 is neutral
    or toneless, +1 is unrelieved good news. It is a *proportion of tone words*,
    not of all words, so a long neutral article and a short neutral one both
    read 0 rather than the long one being pulled toward it.
    """

    polarity: float
    #: Share of all words that express uncertainty ("may", "approximate",
    #: "risk"). Reported alongside polarity rather than folded into it: a
    #: hedged positive statement is a different thing from a negative one, and
    #: collapsing them would lose the distinction.
    uncertainty: float
    positive_words: int
    negative_words: int
    uncertain_words: int
    total_words: int
    #: How many separate texts fed this reading. 1 for `score_text`.
    documents: int


def tokenise(text: str) -> list[str]:
    """Upper-case alphabetic tokens, matching the lexicon's own form."""
    return [match.group(0).upper() for match in _WORD.finditer(text)]


def _negated(tokens: list[str], index: int) -> bool:
    """Is the token at `index` inside a negator's shadow?"""
    start = max(0, index - NEGATION_WINDOW)
    return any(tokens[j] in NEGATORS for j in range(start, index))


def score_text(text: str) -> SentimentReading | None:
    """Tone of one piece of text, or None when there is nothing to score.

    None rather than a neutral zero when no tone words appear at all. A headline
    with no tone is a non-observation, and averaging it in as 0.0 would drag a
    batch toward neutral in proportion to how much toneless text it contained —
    which would make sentiment a measure of article length.
    """
    tokens = tokenise(text)
    if not tokens:
        return None

    positive = negative = uncertain = 0
    for index, token in enumerate(tokens):
        if token in UNCERTAINTY:
            uncertain += 1
        is_positive = token in POSITIVE
        is_negative = token in NEGATIVE
        if not (is_positive or is_negative):
            continue
        if _negated(tokens, index):
            # "not profitable" counts against, "no losses" counts for.
            is_positive, is_negative = is_negative, is_positive
        positive += is_positive
        negative += is_negative

    tone_words = positive + negative
    if tone_words == 0 and uncertain == 0:
        return None
    polarity = (positive - negative) / tone_words if tone_words else 0.0
    return SentimentReading(
        polarity=polarity,
        uncertainty=uncertain / len(tokens),
        positive_words=positive,
        negative_words=negative,
        uncertain_words=uncertain,
        total_words=len(tokens),
        documents=1,
    )


def score_headlines(headlines: list[str]) -> SentimentReading | None:
    """Aggregate tone across headlines, or None when none of them scored.

    Pooled rather than averaged: the counts from every headline are summed and
    the polarity computed once over the total. Averaging per-headline polarities
    would give a three-word headline with one tone word the same weight as a
    paragraph with twenty, which is backwards — the longer text is the better
    evidence, not the worse.

    Headlines with no tone words contribute their length to `total_words` (so
    uncertainty stays a true fraction of everything read) but nothing to
    polarity, which is exactly the intended treatment of a neutral report.
    """
    positive = negative = uncertain = total = scored = 0
    for headline in headlines:
        tokens = tokenise(headline)
        if not tokens:
            continue
        total += len(tokens)
        reading = score_text(headline)
        if reading is None:
            continue
        scored += 1
        positive += reading.positive_words
        negative += reading.negative_words
        uncertain += reading.uncertain_words

    if scored == 0:
        return None
    tone_words = positive + negative
    return SentimentReading(
        polarity=(positive - negative) / tone_words if tone_words else 0.0,
        uncertainty=uncertain / total if total else 0.0,
        positive_words=positive,
        negative_words=negative,
        uncertain_words=uncertain,
        total_words=total,
        documents=scored,
    )
