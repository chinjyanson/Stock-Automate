"""Replay the scanner over history: rank, hold the best, compare to owning it all.

Pure functions over a `universe.Panel`. No I/O, no database, no network — the
script that calls this owns all three.

## The one thing that keeps it honest

Every score comes from `scoring.score_series`, the same call the nightly scan
makes, evaluated on a slice that ends at the decision bar. Nothing here
re-implements a signal, so the backtest cannot drift from production the way a
parallel implementation always eventually does.

The slice is `WINDOW` bars rather than the whole history, and that is not an
approximation: every lookback in `indicators.functions` is bounded at 252 bars
or fewer, so 500 bars produces bit-identical scores to passing 5,000 and costs a
fraction of the time. `TestWindowIsEnough` pins that claim.

## Decide on one bar, trade on the next

A decision taken from the close of day *i* is executed at the close of day
*i + 1*. The alternative — deciding and trading on the same close — is the
commonest silent look-ahead in a monthly backtest, and it flatters exactly the
signals this scanner is built from, since a name that closed at a 52-week low is
disproportionately likely to bounce the following morning.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from app.backtest.universe import BENCHMARK, EQUAL_WEIGHT, RATES, SECTOR_ETFS, Panel
from app.indicators.functions import TRADING_DAYS_PER_YEAR
from app.indicators.series import PriceSeries
from app.scanner import scoring

#: Bars handed to `score_series`. Comfortably above the longest lookback (252),
#: so scores match a full-history call exactly.
WINDOW = 500

#: A name is not ranked below this many bars. The scanner scores thinner series
#: at reduced confidence, but a 52-week statistic computed from 60 bars is a
#: different measurement wearing the same name, and in a ranking it competes
#: directly against names where it means what it says.
MIN_BARS = 300

TRADING_DAYS_PER_MONTH = 21


@dataclass(frozen=True, slots=True)
class Ranked:
    """One name's score at one decision date.

    `groups` carries each group's own 0-100 reading alongside the blend, because
    the blend can only ever answer "does the scanner work". The interesting
    question once the answer is no is *which part* failed — and a group that
    ranks backwards is invisible in a combined score that averages it against
    four others.
    """

    symbol: str
    score: float
    column: int
    groups: dict[str, float | None] = field(default_factory=dict)


@dataclass(slots=True)
class Result:
    """One simulated book: its equity curve and what it did to get there."""

    name: str
    dates: np.ndarray
    equity: np.ndarray
    turnover: list[float] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        return float(self.equity[-1] / self.equity[0] - 1.0)

    @property
    def years(self) -> float:
        span = (self.dates[-1] - self.dates[0]).astype("timedelta64[D]").astype(float)
        return float(span) / 365.25

    @property
    def cagr(self) -> float:
        if self.years <= 0:
            return 0.0
        return float((self.equity[-1] / self.equity[0]) ** (1.0 / self.years) - 1.0)

    @property
    def max_drawdown(self) -> float:
        peak = np.maximum.accumulate(self.equity)
        return float(np.max((peak - self.equity) / peak))

    @property
    def volatility(self) -> float:
        rets = self.equity[1:] / self.equity[:-1] - 1.0
        return float(np.std(rets, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))

    @property
    def sharpe(self) -> float:
        """Excess-of-nothing Sharpe. Cash pays something, so this flatters every
        row equally; it is a comparison between the rows, not a claim about any
        one of them."""
        vol = self.volatility
        return self.cagr / vol if vol > 0 else 0.0

    @property
    def average_turnover(self) -> float:
        return float(np.mean(self.turnover)) if self.turnover else 0.0


def rebalance_days(dates: np.ndarray, *, every: int = TRADING_DAYS_PER_MONTH) -> list[int]:
    """Indices of the decision bars — the last trading day of each month.

    Month ends rather than every-*n*-days because a calendar month is what the
    scanner's own rotation budget is expressed in, and because it keeps the
    result comparable to how anyone would actually run this.
    """
    months = dates.astype("datetime64[M]")
    ends = np.flatnonzero(months[1:] != months[:-1])
    return [int(i) for i in ends if i > 0]


def _slice(panel: Panel, column: int, cut: int, *, window: int = WINDOW) -> PriceSeries | None:
    """Bars for one symbol ending at `cut`, or None if it is not scoreable.

    Leading NaNs (the name had not listed) are excluded by construction: the
    window is taken backwards from `cut` and rejected if it is not wholly
    finite, so a series that begins mid-window is simply not ranked that month
    rather than being scored against NaN-poisoned indicators.
    """
    start = max(0, cut + 1 - window)
    close = panel.fields["adjusted_close"][start : cut + 1, column]
    if close.size < MIN_BARS:
        return None
    finite = np.isfinite(close)
    if not finite[-MIN_BARS:].all():
        return None
    first = int(np.argmax(finite)) if not finite.all() else 0
    lo = start + first
    if cut + 1 - lo < MIN_BARS:
        return None

    def take(name: str) -> np.ndarray:
        return panel.fields[name][lo : cut + 1, column]

    return PriceSeries(
        open=take("open"),
        high=take("high"),
        low=take("low"),
        close=take("close"),
        adjusted_close=take("adjusted_close"),
        volume=take("volume"),
    )


def rank_at(
    panel: Panel,
    cut: int,
    *,
    tradable: tuple[str, ...],
    weights: dict[str, float] | None = None,
    fundamentals: Callable[[str], dict[str, Decimal | None] | None] | None = None,
) -> list[Ranked]:
    """Score every tradable name from data ending at `cut`, best first.

    Sector and rates proxies are cut at the same bar as the name itself. Passing
    an uncut proxy is the subtle look-ahead that a per-name backtest invites:
    the stock would be scored on Monday's information against a sector that
    already knew Friday's.

    `fundamentals` supplies the other half of the score — `value` and the
    business third of `quality`, 45 of the 100 points. It is a callable rather
    than a mapping because the figures depend on the decision date as well as
    the name, and the caller binds the date once per rebalance. Left out, the
    scanner scores on price alone and `combine_score` renormalises, which is a
    real shipping configuration but a minority of the model.
    """
    sector_cache: dict[str, PriceSeries | None] = {}
    rates = _slice(panel, panel.column(RATES), cut)

    ranked: list[Ranked] = []
    for symbol in tradable:
        column = panel.column(symbol)
        series = _slice(panel, column, cut)
        if series is None:
            continue

        etf = SECTOR_ETFS.get(panel.sectors.get(symbol, ""), "")
        if etf and etf not in sector_cache:
            sector_cache[etf] = _slice(panel, panel.column(etf), cut)
        sector = sector_cache.get(etf)

        result = scoring.score_series(
            series,
            weights=weights,
            sector=sector,
            rates=rates,
            fundamentals=None if fundamentals is None else fundamentals(symbol),
        )
        ranked.append(
            Ranked(
                symbol=symbol,
                score=result.score,
                column=column,
                groups={name: group.score for name, group in result.groups.items()},
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def _weights_for(members: list[int], size: int) -> np.ndarray:
    holding = np.zeros(size)
    if members:
        holding[members] = 1.0 / len(members)
    return holding


def simulate(
    panel: Panel,
    schedule: list[tuple[int, list[int]]],
    *,
    name: str,
    cost: float,
) -> Result:
    """Walk a book of equal-weighted holdings day by day.

    `schedule` pairs a decision bar with the columns to hold from the *next*
    bar's close until the next decision is executed. Costs are charged on the
    traded fraction, one side each way.
    """
    adjusted = panel.fields["adjusted_close"]
    daily = np.full(adjusted.shape, np.nan)
    daily[1:] = adjusted[1:] / adjusted[:-1] - 1.0

    start = schedule[0][0] + 1
    stop = adjusted.shape[0]
    equity = 1.0
    held = np.zeros(adjusted.shape[1])
    curve = np.empty(stop - start)
    dates = panel.dates[start:stop]
    turnover: list[float] = []

    changes = {bar + 1: members for bar, members in schedule}

    for offset, i in enumerate(range(start, stop)):
        # Earn first, then trade. `daily[i]` is the move from *i - 1* to *i*, so
        # it belongs to whatever was held at the close of *i - 1* — crediting it
        # to a book that is only bought at the close of *i* hands the strategy a
        # day of hindsight at every rebalance. That is a one-bar error worth a
        # percentage point a year here, and it flatters exactly the signals this
        # scanner is made of, since names bought at a 52-week low are the ones
        # most likely to have bounced overnight. `TestNoLookAhead` pins it.
        step = daily[i]
        active = held > 0
        if active.any():
            moves = np.where(np.isfinite(step[active]), step[active], 0.0)
            equity *= 1.0 + float(np.dot(held[active], moves))

        if i in changes:
            wanted = _weights_for(changes[i], adjusted.shape[1])
            traded = float(np.abs(wanted - held).sum())
            equity -= equity * traded * cost / 2.0
            turnover.append(traded / 2.0)
            held = wanted

        curve[offset] = equity

    return Result(name=name, dates=dates, equity=curve, turnover=turnover)


def buy_and_hold(panel: Panel, symbol: str, start: int) -> Result:
    """Own one thing from `start` to the end and never touch it again."""
    column = panel.column(symbol)
    prices = panel.fields["adjusted_close"][start:, column]
    finite = np.isfinite(prices)
    equity = np.where(finite, prices, np.nan)
    equity = _forward_fill(equity) / prices[np.argmax(finite)]
    return Result(name=f"buy and hold {symbol}", dates=panel.dates[start:], equity=equity)


def _forward_fill(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    bad = ~np.isfinite(out)
    if bad.any():
        idx = np.where(~bad, np.arange(out.size), 0)
        np.maximum.accumulate(idx, out=idx)
        out = out[idx]
    return out


def reading(row: Ranked, key: str) -> float | None:
    """The number to rank by: the blended score, or one group's own reading."""
    return row.score if key == "score" else row.groups.get(key)


def ordered(ranked: list[Ranked], key: str) -> list[Ranked]:
    """`ranked` re-sorted best-first on `key`, dropping names the key is None for.

    A group that could not be measured for a name is *excluded* from that
    group's ranking rather than sorted to the bottom — the same reasoning that
    makes `combine_score` renormalise instead of scoring absence as zero.
    """
    scored = [r for r in ranked if reading(r, key) is not None]
    scored.sort(key=lambda r: reading(r, key), reverse=True)  # type: ignore[arg-type,return-value]
    return scored


def forward_returns(
    panel: Panel,
    rankings: list[tuple[int, list[Ranked]]],
    *,
    horizon: int,
    key: str = "score",
) -> dict[str, np.ndarray]:
    """Every ranked name's score against what it did next.

    The portfolio result answers "would this have made money"; this answers the
    prior question — "does the score carry any information at all" — without a
    holding period, a cost or a weighting scheme in the way. If the scores and
    the forward returns are unrelated here, no amount of portfolio construction
    downstream can rescue them.
    """
    adjusted = panel.fields["adjusted_close"]
    scores: list[float] = []
    rets: list[float] = []
    for cut, ranked in rankings:
        entry = cut + 1
        exit_bar = min(entry + horizon, adjusted.shape[0] - 1)
        if entry >= adjusted.shape[0]:
            continue
        for row in ranked:
            value = reading(row, key)
            if value is None:
                continue
            a = adjusted[entry, row.column]
            b = adjusted[exit_bar, row.column]
            if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
                continue
            scores.append(value)
            rets.append(b / a - 1.0)
    return {"score": np.asarray(scores), "forward": np.asarray(rets)}


def by_decile(
    scores: np.ndarray, forward: np.ndarray, *, buckets: int = 10
) -> list[dict[str, float]]:
    """Forward return per score decile, reported three ways on purpose.

    **`mean` is the number that misleads here, and it took a wrong conclusion to
    notice.** The arithmetic mean of a volatile name's returns exceeds what
    holding it actually compounds to — alternate +50% and -33% and the mean is
    +8.5% a period while the investor ends where they started. So any signal
    correlated with volatility ranks on the mean for reasons that have nothing
    to do with being right, and this scanner has a group (`quality`) built
    largely out of *low* volatility. Ranked on means it appeared to predict
    backwards, strongly and significantly.

    `growth` is the mean **log** return, which is exactly what compounds, and
    `vol` is the dispersion inside the bucket that explains the difference
    between the two. Read `growth`; use `mean` only to see the confound.
    """
    if scores.size == 0:
        return []
    edges = np.quantile(scores, np.linspace(0.0, 1.0, buckets + 1))
    edges[-1] = np.inf
    rows: list[dict[str, float]] = []
    for b in range(buckets):
        mask = (scores >= edges[b]) & (scores < edges[b + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        values = forward[mask]
        # -100% is a real outcome and log(0) is not; clip just above it so a
        # total loss contributes a large finite penalty rather than a NaN that
        # would quietly drop the worst results out of the average.
        logs = np.log1p(np.clip(values, -0.99, None))
        rows.append(
            {
                "decile": b + 1,
                "score_from": float(edges[b]),
                "score_to": float(edges[b + 1])
                if np.isfinite(edges[b + 1])
                else float(scores.max()),
                "n": n,
                "mean": float(values.mean()),
                "stderr": float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
                "median": float(np.median(values)),
                "growth": float(logs.mean()),
                "growth_stderr": float(logs.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
                "vol": float(values.std(ddof=1)) if n > 1 else float("nan"),
            }
        )
    return rows


def yearly(result: Result) -> dict[int, float]:
    """Calendar-year returns, for the per-year win counts that decide whether a
    headline is one event or a repeatable effect."""
    years = result.dates.astype("datetime64[Y]").astype(int) + 1970
    out: dict[int, float] = {}
    for year in np.unique(years):
        mask = years == year
        segment = result.equity[mask]
        if segment.size > 1:
            out[int(year)] = float(segment[-1] / segment[0] - 1.0)
    return out


def market_symbols() -> tuple[str, ...]:
    """Everything in the panel that is a proxy rather than a rankable holding."""
    return (*sorted(set(SECTOR_ETFS.values())), BENCHMARK, RATES, EQUAL_WEIGHT)
