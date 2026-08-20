"""What would the scanner have picked on a past date, and what happened next? (§6)

    python -m app.scripts.scanner_picks_forward --as-of 2021-01-04 --top 20

`backtest_scanner.py` asks whether the ranking beats owning the S&P 500 over
many rebalances. This asks the blunter question: score **the whole broker
catalogue** on one day years ago, take the best twenty, and see how they did.

The answer needs three numbers, not one. "Sixteen of twenty went up" sounds like
a result and is meaningless alone: if four-fifths of every stock in the universe
rose over the same years, sixteen of twenty is exactly average. So the top slice
is reported beside **every ranked name** (the base rate) and beside the
**bottom twenty** by the same score (the falsification). A score that knows
something puts its best names above the base rate and its worst below it.

## Three things this cannot do, stated before the numbers

**No fundamentals.** `value` (30 points) and the business third of `quality`
come from a yfinance snapshot that exists only as of today. Scoring a 2021
decision with 2026 earnings is look-ahead of the worst kind, and a flattering
one: a company whose numbers look good now is disproportionately one that did
well since. They are therefore absent, and `combine_score` renormalises over the
rest exactly as production does for the ~45% of the live catalogue that has no
fundamentals either. **This measures the price-derived scanner, not the whole
model** — cheapness, sector and the risk/liquidity part of quality.

**Survivorship, and it is severe here.** The universe is the broker's *current*
tradeable list. Every company that delisted, was taken over or went to zero
between the as-of date and today is simply absent — not mismarked, absent. So
the absolute hit rate is flattered, badly. The defence is the same as
everywhere else in this project: the base rate carries the identical bias, so
the *gap* between the top slice and the base rate is close to unbiased even
though both levels are too kind.

**One date.** A single as-of is one draw. January 2021 was a particular moment —
near the top of a speculative run — and a score that looks good or bad from it
may say more about that month than about the score. `--as-of` takes any date;
one is an anecdote.

## Decide on one bar, trade on the next

The rank is computed from data ending at the as-of close, and the return is
measured from the **following** close. Deciding and buying on the same close is
the standard silent look-ahead, and it flatters exactly the mean-reversion
signals this scanner is built from.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.backtest import scanner_replay as replay
from app.backtest import universe

#: yfinance labels sectors differently from GICS, and `SECTOR_ETFS` is keyed by
#: GICS. Five of the eleven happen to match; without the rest, most of the
#: catalogue would silently lose its sector group and be scored on a
#: renormalised remainder — a quiet change in what is being measured.
GICS_FROM_YFINANCE: dict[str, str] = {
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Communication Services": "Communication Services",
}

CATALOGUE_SQL = """
    SELECT m.provider_symbol, i.sector
    FROM instruments i
    JOIN market_data_mappings m
      ON m.instrument_id = i.id AND m.is_active AND m.provider = 'yfinance'
    WHERE i.suspended_at IS NULL
"""


@dataclass(frozen=True, slots=True)
class Outcome:
    name: str
    count: int
    rose: int
    median: float
    mean: float

    @property
    def hit_rate(self) -> float:
        return self.rose / self.count if self.count else float("nan")


def catalogue() -> tuple[tuple[str, ...], dict[str, str]]:
    """Every tradeable symbol the broker lists, with its sector in GICS terms."""
    from sqlalchemy import create_engine, text

    from app.config import get_settings

    engine = create_engine(get_settings().sync_database_url)
    with engine.connect() as connection:
        rows = list(connection.execute(text(CATALOGUE_SQL)))
    engine.dispose()

    sectors: dict[str, str] = {}
    symbols: list[str] = []
    for symbol, sector in rows:
        symbols.append(symbol)
        mapped = GICS_FROM_YFINANCE.get(sector or "")
        if mapped:
            sectors[symbol] = mapped
    return tuple(sorted(set(symbols))), sectors


def summarise(name: str, returns: np.ndarray) -> Outcome:
    usable = returns[np.isfinite(returns)]
    return Outcome(
        name=name,
        count=int(usable.size),
        rose=int((usable > 0.0).sum()),
        median=float(np.median(usable)) if usable.size else float("nan"),
        mean=float(np.mean(usable)) if usable.size else float("nan"),
    )


def forward(panel: universe.Panel, columns: np.ndarray, entry: int, exit_bar: int) -> np.ndarray:
    prices = panel.fields["adjusted_close"]
    start = prices[entry, columns]
    end = prices[exit_bar, columns]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(start) & (start > 0), end / start - 1.0, np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the catalogue on a past day.")
    parser.add_argument("--as-of", default="2021-01-04")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--cache", type=Path, default=Path("data/catalogue_panel.npz"))
    parser.add_argument("--since", default="2016-01-01")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0 = the whole catalogue")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    symbols, sectors = catalogue()
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"\n  {len(symbols):,} tradeable symbols, {len(sectors):,} with a known sector")
    print(f"  downloading daily bars since {args.since} (cached at {args.cache})", flush=True)

    panel = universe.build(symbols, sectors, args.cache, args.since, refresh=args.refresh)
    dates = panel.dates
    cut = int(np.searchsorted(dates, np.datetime64(args.as_of, "D"), side="right")) - 1
    if cut < 0 or cut + 1 >= dates.size:
        raise SystemExit(f"--as-of {args.as_of} is outside the downloaded range")
    entry, exit_bar = cut + 1, dates.size - 1

    print(f"  ranking on {dates[cut]}, buying at the close of {dates[entry]},")
    print(f"  measured to {dates[exit_bar]}\n", flush=True)

    tradable = tuple(s for s in symbols if s in panel.symbols)
    ranked = replay.rank_at(panel, cut, tradable=tradable)
    if not ranked:
        raise SystemExit("nothing was scoreable on that date")

    columns = np.array([r.column for r in ranked])
    returns = forward(panel, columns, entry, exit_bar)

    top = summarise(f"top {args.top}", returns[: args.top])
    bottom = summarise(f"bottom {args.top}", returns[-args.top :])
    everything = summarise("every ranked name", returns)

    print(
        f"  {len(ranked):,} names had enough history to be scored "
        f"({replay.MIN_BARS} bars minimum)\n"
    )
    print(f"  {'slice':<20} {'names':>6} {'went up':>10} {'median':>10} {'mean':>10}")
    print("  " + "-" * 60)
    for outcome in (top, everything, bottom):
        print(
            f"  {outcome.name:<20} {outcome.count:>6} "
            f"{outcome.rose:>4}/{outcome.count:<5} {outcome.median:>+9.1%} "
            f"{outcome.mean:>+9.1%}"
        )

    gap = top.hit_rate - everything.hit_rate
    print(
        f"\n  The top {args.top} beat the base rate by {gap * 100:+.1f} percentage points"
        f" on hit rate,"
    )
    median_gap = (top.median - everything.median) * 100
    print(f"  and by {median_gap:+.1f} points on the median return.")
    print("  The base rate is the comparison that matters: it carries the same")
    print("  survivorship bias, so the gap survives what the levels do not.\n")

    print(f"  The {args.top} it picked, best first:\n")
    print(f"  {'#':>3} {'symbol':<12} {'score':>7} {'return since':>14}")
    print("  " + "-" * 42)
    for i, (row, change) in enumerate(
        zip(ranked[: args.top], returns[: args.top], strict=False), 1
    ):
        shown = "no price" if not np.isfinite(change) else f"{change:+.1%}"
        print(f"  {i:>3} {row.symbol:<12} {row.score:>7.1f} {shown:>14}")
    print()


if __name__ == "__main__":
    main()
