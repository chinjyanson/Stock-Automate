"""Ingest SEC Form 4 filings and score them (§4, §6).

Two halves:

  * **Ingest** — poll EDGAR, resolve each issuer ticker to a local instrument,
    and upsert. Idempotent on (accession, line): the feed is re-read constantly
    and the same filing is reachable by several paths.

  * **Score** — turn stored filings into a single 0-100 factor the scanner can
    blend, where 50 is neutral, buying pushes up and selling pushes down.

The scoring rules encode what actually carries information, rather than treating
every Form 4 as equal:

  * **Buying counts for more than selling.** An executive buys for one reason
    and sells for many — diversification, a tax bill, a house. The asymmetry is
    deliberate and is the single most important thing here.
  * **Mechanical rows are ignored entirely.** Only codes P and S survive ingest
    as signalling; withholding, grants, exercises and gifts express no view.
  * **10b5-1 sales are ignored.** A sale scheduled months ahead is not an opinion
    formed today, and measurably does not behave like one.
  * **Only chief officers can trigger the sell penalty.** Selling by officers and
    directors showed no measurable effect in either direction; buying keeps the
    full seniority ladder, where it does.
  * **The signal decays**, and dies at `LOOKBACK_DAYS`.
  * **Already-moved signals are stood down** — see `_priced_in_damping`.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.edgar import EDGARClient, InsiderFiling
from app.models.enums import Interval
from app.models.insider import SIGNALLING_CODES, InsiderTransaction
from app.models.instrument import Exchange, Instrument
from app.models.market_data import Candle

log = structlog.get_logger(__name__)

#: How far back a filing stays relevant. Shortened from 90 days on evidence: a
#: backtest over 686 Form 4 events found discretionary C-suite selling preceded
#: -6.28% excess return at one month (t = -2.43) and nothing measurable at three
#: (-1.72%, t = -0.38). The reason is mechanical rather than mysterious — the
#: measured move happens in the weeks *after* the filing, so once a filing is
#: itself a month old that move has already occurred and there is nothing left
#: to trade. Complements `_priced_in_damping`, which catches the same staleness
#: from the price side rather than the calendar side.
LOOKBACK_DAYS = 45

#: Neutral score. The factor is a deviation from "no information", not a
#: quality measure — an instrument with no filings must land exactly here so it
#: neither helps nor hurts.
NEUTRAL = 50.0

#: Buying and selling are expressed through *different mechanisms*, not as two
#: directions of one number:
#:
#:   * Buying raises a weighted factor. It can only ever help, so it cannot
#:     disadvantage the ~60% of a UK-tradable catalogue that Form 4 never covers.
#:   * Selling applies a multiplicative penalty to the final score. A penalty can
#:     be aggressive without breaking comparability, because a stock with no
#:     filings is simply untouched rather than ranked below one that has them.
#:
#: Folding both into a single weighted factor was the earlier design and it had a
#: bad property: raising the weight enough for selling to bite also inflated
#: every US stock with a buy, so the ranking drifted toward "do we have filings
#: for this?" rather than "is this a good company?".
MAX_BUY_POINTS = 50.0
#: Ceiling on the sell penalty: at full strength a stock loses 40% of its score.
#: Aggressive, but it now fires on a far narrower set than before — only
#: discretionary C-suite sales, about 17% of signalling filings — and that is
#: the one cohort the backtest found a real effect for.
MAX_SELL_PENALTY = 0.40

#: Multipliers on a filing's weight, by how much the filer plausibly knows.
#: Anyone below director is excluded outright rather than given a small weight —
#: a 10% holder or a family trust rebalancing is not an insider forming a view,
#: and letting it contribute at all was letting fund mechanics masquerade as
#: signal (a $48m "insider sale" in the live data was filed by a holding company
#: with no officer title).
_ROLE_WEIGHT = {"c_suite": 1.0, "officer": 0.7, "director": 0.5}
#: A pre-scheduled (Rule 10b5-1) sale is ignored outright, not merely discounted.
#: Nobody decided anything on the day it executed — the schedule was set months
#: earlier — and the backtest found these sales preceded *out*performance
#: (+5.56% at one month, t = +3.39; +9.26% at three, t = +2.56, over 198
#: events). Penalising them at the previous 0.2 was therefore not cautious but
#: backwards. Set to zero rather than negative: "ignore" is what the evidence
#: supports, whereas rewarding scheduled selling would be fitting a rule to one
#: nine-month window in a rising market.
_PLAN_DAMPING = 0.0
#: Value at which a filing counts for a full unit of conviction. Chosen so an
#: ordinary six-figure trade registers and a seven-figure one saturates.
_SATURATION_USD = Decimal("1000000")


@dataclass(slots=True)
class InsiderScore:
    """The scanner-facing result for one instrument."""

    #: Buy-side factor, 50..100, or None when nobody senior has been buying.
    #: None rather than 50 so a non-observation does not dilute other factors.
    score: float | None
    #: Fraction of the final score to remove for insider selling, 0..MAX_SELL_PENALTY.
    sell_penalty: float
    buy_value: Decimal
    sell_value: Decimal
    filings: int
    #: How much the raw signal was cut because price already moved (0..1, where
    #: 1.0 means untouched).
    priced_in_factor: float
    #: Signed move since the earliest relevant filing, in ATR units. Negative
    #: means the stock has already fallen. Kept separate from
    #: `priced_in_factor`, which is deliberately direction-blind for ranking:
    #: an *exit* decision needs the sign, because a stock that has already
    #: dropped is priced in (selling now realises the loss at the bottom) while
    #: one that has risen is the best moment to leave.
    move_since_filing_atr: float | None
    detail: list[str]


class InsiderIngestionService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- Ingest -------------------------------------------------------------

    async def ingest_recent(self, client: EDGARClient, *, limit: int = 100) -> dict[str, int]:
        """Poll the newest Form 4 filings and store them."""
        urls = await client.recent_form4_index_urls(limit=limit)
        stored = signalling = resolved = 0
        for url in urls:
            try:
                filings = await client.fetch_filing(url)
            except Exception as exc:  # one bad filing must not stop the sweep
                log.warning("insider.filing_failed", url=url, error=str(exc))
                continue
            for filing in filings:
                instrument_id = await self._resolve(filing.issuer_ticker)
                await self._upsert(filing, instrument_id)
                stored += 1
                signalling += filing.transaction_code in SIGNALLING_CODES
                resolved += instrument_id is not None
        return {
            "filings_seen": len(urls),
            "rows_stored": stored,
            "signalling": signalling,
            "resolved_to_instrument": resolved,
        }

    async def _resolve(self, ticker: str | None) -> uuid.UUID | None:
        """Map a US ticker to a local instrument, or None.

        Deliberately conservative. A bare ticker is ambiguous across venues —
        the same letters can name different companies in New York and London —
        so only US-listed matches are accepted. A wrong mapping would attribute
        one company's insider activity to another, which is worse than no
        signal at all.
        """
        if not ticker:
            return None
        # Filter in SQL rather than loading instruments and inspecting
        # `.exchange` in Python: that attribute is a lazy relationship, and
        # touching one on an asyncio session raises MissingGreenlet instead of
        # emitting a query.
        rows = (
            (
                await self._session.execute(
                    select(Instrument.id)
                    .join(Exchange, Exchange.id == Instrument.exchange_id)
                    .where(
                        Instrument.exchange_ticker == ticker.upper(),
                        Exchange.country == "US",
                    )
                )
            )
            .scalars()
            .all()
        )
        # Exactly one match, or nothing. Two US venues listing the same ticker is
        # ambiguous, and guessing would attribute one company's insider activity
        # to another.
        return rows[0] if len(rows) == 1 else None

    async def _upsert(self, filing: InsiderFiling, instrument_id: uuid.UUID | None) -> None:
        values = {
            "accession_number": filing.accession_number,
            "line_index": filing.line_index,
            "instrument_id": instrument_id,
            "issuer_ticker": filing.issuer_ticker,
            "issuer_name": filing.issuer_name,
            "insider_name": filing.insider_name,
            "officer_title": filing.officer_title,
            "is_officer": filing.is_officer,
            "is_director": filing.is_director,
            "is_ten_percent_owner": filing.is_ten_percent_owner,
            "is_c_suite": filing.is_c_suite,
            "transaction_code": filing.transaction_code,
            "acquired_disposed": filing.acquired_disposed,
            "transaction_date": filing.transaction_date,
            "filed_at": filing.filed_at,
            "shares": filing.shares,
            "price_per_share": filing.price_per_share,
            "value_usd": filing.value_usd,
            "shares_owned_after": filing.shares_owned_after,
            "is_10b5_1_plan": filing.is_10b5_1_plan,
            "is_signalling": filing.transaction_code in SIGNALLING_CODES,
            "filing_url": filing.filing_url,
        }
        statement = pg_insert(InsiderTransaction).values(**values)
        # Re-polling must converge, not duplicate. Later reads can also correct
        # an earlier one (a ticker that now resolves), so update rather than
        # ignore on conflict.
        await self._session.execute(
            statement.on_conflict_do_update(
                index_elements=["accession_number", "line_index"],
                set_={
                    k: v for k, v in values.items() if k not in ("accession_number", "line_index")
                },
            )
        )

    # -- Score --------------------------------------------------------------

    async def score_instrument(
        self, instrument_id: uuid.UUID, *, as_of: date | None = None
    ) -> InsiderScore | None:
        """A 0-100 insider factor, or None when there is nothing to say.

        None rather than 50 when there are no filings: the scanner drops missing
        factors and renormalises, so returning neutral would dilute every other
        factor with a non-observation.
        """
        as_of = as_of or datetime.now(UTC).date()
        cutoff = as_of - timedelta(days=LOOKBACK_DAYS)
        rows = (
            (
                await self._session.execute(
                    select(InsiderTransaction).where(
                        InsiderTransaction.instrument_id == instrument_id,
                        InsiderTransaction.is_signalling.is_(True),
                        InsiderTransaction.transaction_date >= cutoff,
                    )
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            return None

        buy_points = sell_points = 0.0
        buy_value = sell_value = Decimal(0)
        detail: list[str] = []
        for row in rows:
            # Selling is only read as a signal from a chief officer. Officer and
            # director sales showed no measurable effect either way in the
            # backtest (+3.86%, t=1.05 and -0.67%, t=-0.51), so penalising them
            # would be marking stocks down on noise.
            if row.transaction_code == "S" and not row.is_c_suite:
                continue
            weight = self._filing_weight(row, as_of)
            if weight <= 0:
                continue  # too junior, scheduled, or decayed past the horizon
            value = row.value_usd or Decimal(0)
            if row.transaction_code == "P":
                buy_points += weight
                buy_value += value
                detail.append(f"buy ${value:,.0f} by {row.officer_title or 'insider'}")
            else:
                sell_points += weight
                sell_value += value
                detail.append(f"sell ${value:,.0f} by {row.officer_title or 'insider'}")

        damping, move_atr = await self._price_move_since(instrument_id, rows)

        # Saturating, so one enormous filing cannot dominate and ten small ones
        # are not mistaken for conviction.
        score = (
            round(NEUTRAL + MAX_BUY_POINTS * _saturate(buy_points) * damping, 2)
            if buy_points > 0
            else None
        )
        penalty = round(MAX_SELL_PENALTY * _saturate(sell_points) * damping, 4)
        if score is None and penalty <= 0:
            return None  # filings exist but none of them said anything
        return InsiderScore(
            score=score,
            sell_penalty=penalty,
            buy_value=buy_value,
            sell_value=sell_value,
            filings=len(rows),
            priced_in_factor=round(damping, 3),
            move_since_filing_atr=(None if move_atr is None else round(move_atr, 3)),
            detail=detail[:6],
        )

    def _filing_weight(self, row: InsiderTransaction, as_of: date) -> float:
        """Conviction contributed by one filing, before saturation."""
        if row.is_c_suite:
            weight = _ROLE_WEIGHT["c_suite"]
        elif row.is_officer:
            weight = _ROLE_WEIGHT["officer"]
        elif row.is_director:
            weight = _ROLE_WEIGHT["director"]
        else:
            return 0.0  # not senior enough to be read as a view

        value = float(row.value_usd or 0)
        weight *= min(1.0, value / float(_SATURATION_USD))

        # Linear decay to zero at the lookback horizon.
        if row.transaction_date is not None:
            age = (as_of - row.transaction_date).days
            weight *= max(0.0, 1.0 - age / LOOKBACK_DAYS)

        # A sale scheduled months in advance is not an opinion formed today, and
        # the data says it is not even weakly bearish. With _PLAN_DAMPING at 0
        # this zeroes the filing outright.
        if row.is_10b5_1_plan and row.transaction_code == "S":
            weight *= _PLAN_DAMPING
        return weight

    async def _price_move_since(
        self, instrument_id: uuid.UUID, rows: Sequence[InsiderTransaction]
    ) -> tuple[float, float | None]:
        """How much of the signal survives, given the price already moved.

        An insider filing is only an edge until the market has acted on it. This
        measures the move from the earliest relevant filing date to the latest
        close, in units of ATR so it means the same thing on a quiet stock and a
        wild one, and cuts the signal as that move grows:

            <= 1 ATR   full signal      (the market has not reacted)
            >= 3 ATR   nothing left     (whatever the edge was, it is gone)

        Returns `(damping, signed_move_in_atr)`.

        The damping is deliberately direction-agnostic and the signed move is
        returned alongside it, because the two consumers ask different
        questions and are *meant* to disagree:

          * **Ranking** (the scanner) asks "is this worth considering?". A stock
            that rose after an insider sale is less concerning — the market
            shrugged — so the penalty damps on magnitude alone.
          * **Exiting** (the strategy) asks "should I sell what I hold?". A rise
            after an insider sale is the *best* moment to leave, not a reason to
            stand down, so it reads the sign.

        The strategy governs sells; the scanner never sells anything. Collapsing
        these into one rule would make one of the two answers wrong.
        """
        dates = [r.transaction_date for r in rows if r.transaction_date is not None]
        if not dates:
            return 1.0, None
        since = min(dates)

        candles = (
            (
                await self._session.execute(
                    select(Candle)
                    .where(
                        Candle.instrument_id == instrument_id,
                        Candle.interval == Interval.D1,
                        Candle.is_closed.is_(True),
                    )
                    .order_by(Candle.timestamp.desc())
                    .limit(120)
                )
            )
            .scalars()
            .all()
        )
        if len(candles) < 20:
            return 1.0, None  # cannot tell; do not invent a reason to discount

        ordered = list(reversed(candles))
        latest = float(ordered[-1].close)
        at_filing = next((float(c.close) for c in ordered if c.timestamp.date() >= since), None)
        if at_filing is None or at_filing <= 0:
            return 1.0, None

        highs = [float(c.high) for c in ordered]
        lows = [float(c.low) for c in ordered]
        closes = [float(c.close) for c in ordered]
        atr = _average_true_range(highs, lows, closes)
        if atr is None or atr <= 0:
            return 1.0, None

        signed = (latest - at_filing) / atr
        magnitude = abs(signed)
        if magnitude <= 1.0:
            return 1.0, signed
        if magnitude >= 3.0:
            return 0.0, signed
        return 1.0 - (magnitude - 1.0) / 2.0, signed


def _saturate(points: float) -> float:
    """Diminishing returns: 1 unit -> 0.5, 3 -> 0.75, unbounded input -> 1."""
    return points / (points + 1.0) if points > 0 else 0.0


def _average_true_range(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> float | None:
    if min(len(highs), len(lows), len(closes)) < period + 1:
        return None
    trs = [
        max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        for i in range(1, len(closes))
    ]
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period
