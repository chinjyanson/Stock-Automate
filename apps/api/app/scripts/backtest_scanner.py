"""Does the scanner's ranking beat owning the whole universe? (§6)

    python -m app.scripts.backtest_scanner --top 20

Every backtest in this project so far has measured a *strategy* — when to enter,
where to stop, when to step aside. The layer that chooses **which companies to
own at all** has never been measured, and it is the layer the whole product is
built around. This script measures it.

## The three benchmarks, and why the middle one is the real test

  * **Buy and hold SPY** — the question everyone asks, and the least
    informative, because the ranked book is equal-weighted and the index is not.
    Half of any difference is the weighting scheme.
  * **Equal-weight the same 503 names** — the control. Same universe, same
    survivorship bias, same rebalance cadence, same costs, no ranking. What is
    left when this is subtracted is what the *score* contributed, and nothing
    else. This is the row that decides the verdict.
  * **The bottom N by the same score** — the falsification. A score that carries
    information should have its worst names underperform its best ones. If top
    and bottom are indistinguishable, the ranking is noise however the top-N
    book happens to have done.

## Two configurations, and which one you are reading

By default this measures the **price-derived scanner**: cheapness 24, sector 10,
and the risk/liquidity two-thirds of quality 21 — 55 of the 100 points,
renormalised. `value` and the business third of `quality` are absent, because
the only fundamentals the system holds are a yfinance snapshot of *today*, and
feeding today's P/E into a 2009 decision would be look-ahead of the worst kind.
That is a real shipping configuration — it is what production does for the ~45%
of the live catalogue with no fundamentals — but it is a minority of the model.

`--fundamentals` measures the **whole 100 points**, using
`app.backtest.fundamentals`: figures read from SEC filings and filtered by the
date each filing became public, so a decision only ever sees what had been
published. That is the configuration to quote when asking whether the scanner
works, because the fundamentals are the half the scanner is built around. It
covers US filers only, so the handful of names EDGAR does not carry fall back to
the price-derived score, exactly as production would.

## What is not modelled, stated rather than discovered later

No dividend timing beyond adjusted closes, no slippage beyond the flat `--cost`,
no position limits, no risk engine, no market impact. Fundamentals are annual
rather than trailing-twelve-month, so they lag by up to a fiscal year — a
handicap, not an advantage; the score is behind the market, never ahead of it.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import numpy as np

from app.backtest import fundamentals, universe
from app.backtest import scanner_replay as replay
from app.scanner import scoring


def _table(results: list[replay.Result]) -> None:
    print(
        f"\n  {'book':<28} {'return':>10} {'CAGR':>8} {'vol':>7} "
        f"{'maxDD':>8} {'ret/vol':>8} {'turnover':>9}"
    )
    print("  " + "-" * 82)
    for r in results:
        turnover = f"{r.average_turnover:>8.1%}" if r.turnover else "       --"
        print(
            f"  {r.name:<28} {r.total_return:>9.0%} {r.cagr:>7.1%} {r.volatility:>6.1%} "
            f"{r.max_drawdown:>7.1%} {r.sharpe:>7.2f} {turnover:>9}"
        )


def _per_year(results: list[replay.Result], reference: replay.Result) -> None:
    print(f"\n  Calendar years, each book against {reference.name}:\n")
    years = replay.yearly(reference)
    header = "  year  " + "".join(f"{r.name[:14]:>16}" for r in results) + f"{'control':>16}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    wins = {r.name: 0 for r in results}
    for year in sorted(years):
        row = f"  {year}  "
        for r in results:
            value = replay.yearly(r).get(year)
            if value is None:
                row += f"{'--':>16}"
                continue
            if value > years[year]:
                wins[r.name] += 1
            row += f"{value:>15.1%} "
        row += f"{years[year]:>15.1%} "
        print(row)
    print("  " + "-" * (len(header) - 2))
    total = len(years)
    for r in results:
        print(f"  {r.name:<28} beat the control in {wins[r.name]}/{total} years")


def _survivorship(control: replay.Result, real: replay.Result) -> None:
    """Price the harness's own bias, rather than caveating it in prose.

    The control equal-weights *today's* index members over the whole period; RSP
    equal-weights whoever was actually in it at the time. The two differ only by
    survivorship, so the gap between them is this backtest's bias measured in
    percent per year — and it belongs next to the results, because it is
    routinely larger than the effect anyone is trying to detect.
    """
    gap = control.cagr - real.cagr
    print(
        f"\n  Survivorship: the control returns {control.cagr:.1%} a year holding today's members;"
        f"\n  {universe.EQUAL_WEIGHT} held whoever was a member at the time and"
        f" returned {real.cagr:.1%}."
        f"\n  This harness is therefore worth about {gap * 100:+.1f}pp a year to *every* book"
        "\n  table above, including buy and hold. Only differences between books survive it."
    )


def _spread(rows: list[dict[str, float]], *, on: str = "growth") -> tuple[float, float]:
    """Top decile minus bottom, and the 2-sigma band on that difference.

    Defaults to `growth` (mean log return) rather than `mean`: see `by_decile`
    for why the arithmetic mean ranks volatility instead of skill.
    """
    if len(rows) < 2:
        return float("nan"), float("nan")
    error = "stderr" if on == "mean" else "growth_stderr"
    return (
        rows[-1][on] - rows[0][on],
        2.0 * float(np.hypot(rows[-1][error], rows[0][error])),
    )


def _groups(
    panel: universe.Panel,
    rankings: list[tuple[int, list[replay.Ranked]]],
    *,
    horizon: int,
    top: int,
    cost: float,
) -> None:
    """Each group's own ranking power, measured the same way as the blend.

    A combined score that does not rank has two possible causes with opposite
    remedies: every group is noise, or some groups rank and are being cancelled
    by others that rank backwards. Only a per-group measurement separates them,
    and the second case is the one worth acting on.
    """
    print(f"\n  Ranking power of each group on its own ({horizon}-day forward return):\n")
    print(
        f"  {'ranked by':<12} {'n':>7} {'growth spread':>14} {'+/- 2se':>9}  "
        f"{'verdict':<16} {'(mean spread)':>14}"
    )
    print("  " + "-" * 78)

    books: list[replay.Result] = []
    for key in ("score", *scoring.GROUP_NAMES):
        pooled = replay.forward_returns(panel, rankings, horizon=horizon, key=key)
        if pooled["score"].size == 0:
            print(f"  {key:<12} {'--':>7}  never measurable in this configuration")
            continue
        rows = replay.by_decile(pooled["score"], pooled["forward"])
        spread, band = _spread(rows)
        naive, _ = _spread(rows, on="mean")
        verdict = "no" if abs(spread) <= band else ("ranks" if spread > 0 else "RANKS BACKWARDS")
        print(
            f"  {key:<12} {pooled['score'].size:>7} {spread:>+13.2%} {band:>9.2%}  "
            f"{verdict:<16} {naive:>+13.2%}"
        )

        schedule = [
            (cut, [r.column for r in replay.ordered(ranked, key)[:top]]) for cut, ranked in rankings
        ]
        if all(members for _, members in schedule):
            books.append(replay.simulate(panel, schedule, name=f"top {top} by {key}", cost=cost))

    if books:
        _table(books)


def _eras(
    panel: universe.Panel,
    rankings: list[tuple[int, list[replay.Ranked]]],
    *,
    horizon: int,
    parts: int,
) -> None:
    """The same per-group spread, measured separately in each era.

    A signal measured once over 22 years has been measured once. Splitting the
    period is the cheapest available check on whether a spread is a property of
    the signal or of a decade — and this project has already had one headline
    (the crash overlay's drawdown win) turn out to be a single event repeated
    across settings. A group that ranks in one half and not the other has not
    been shown to rank.
    """
    edges = np.array_split(np.arange(len(rankings)), parts)
    labels: list[str] = []
    for chunk in edges:
        first = panel.dates[rankings[int(chunk[0])][0]].astype(str)[:4]
        last = panel.dates[rankings[int(chunk[-1])][0]].astype(str)[:4]
        labels.append(f"{first}-{last}")

    print("\n  Growth spread by era — does any group rank in every one of them?\n")
    print("  " + f"{'ranked by':<12}" + "".join(f"{label:>18}" for label in labels))
    print("  " + "-" * (12 + 18 * len(labels)))
    for key in ("score", *scoring.GROUP_NAMES):
        cells: list[str] = []
        measurable = False
        for chunk in edges:
            slice_ = [rankings[int(i)] for i in chunk]
            pooled = replay.forward_returns(panel, slice_, horizon=horizon, key=key)
            if pooled["score"].size == 0:
                cells.append(f"{'--':>18}")
                continue
            measurable = True
            spread, band = _spread(replay.by_decile(pooled["score"], pooled["forward"]))
            mark = " " if abs(spread) > band else "?"
            cells.append(f"{spread:>+16.2%}{mark} ")
        if measurable:
            print(f"  {key:<12}" + "".join(cells))
    print("\n  ? marks a spread whose 2-sigma band includes zero — i.e. not measured, not zero.")


def _deciles(rows: list[dict[str, float]], horizon: int) -> None:
    print(f"\n  Forward {horizon}-day return by score decile (all ranked names, pooled):\n")
    print(
        f"  {'decile':>7} {'score':>12} {'n':>7} {'growth':>9} {'+/- 2se':>9} "
        f"{'median':>9} {'mean':>9} {'sd':>8}"
    )
    print("  " + "-" * 76)
    for row in rows:
        bounds = f"{row['score_from']:.0f}-{row['score_to']:.0f}"
        print(
            f"  {row['decile']:>7.0f} {bounds:>12} {row['n']:>7.0f} {row['growth']:>8.2%} "
            f"{2 * row['growth_stderr']:>8.2%} {row['median']:>8.2%} "
            f"{row['mean']:>8.2%} {row['vol']:>7.1%}"
        )
    spread, band = _spread(rows)
    if np.isfinite(spread):
        verdict = "excludes zero" if abs(spread) > band else "INCLUDES ZERO"
        print(
            f"\n  Top decile minus bottom: {spread:+.2%} +/- {band:.2%} ({verdict})."
            "\n  A spread whose band includes zero is a ranking that did not rank."
        )


def _fundamentals_at(
    panel: universe.Panel,
    books: dict[str, fundamentals.Company],
    splits: dict[str, list[tuple[str, float]]],
    cut: int,
) -> Callable[[str], dict[str, Decimal | None] | None]:
    """Bind a decision bar to a lookup the ranker can call per name.

    The price handed to `reading` is the **adjusted** close at the decision
    bar, which is why the split factor has to come with it: EDGAR reports per
    share figures as filed, and the two are otherwise on different bases by
    whatever the company has split since.
    """
    day = str(panel.dates[cut])
    prices = panel.fields["adjusted_close"]

    def at(symbol: str) -> dict[str, Decimal | None] | None:
        company = books.get(symbol)
        if company is None:
            return None
        price = float(prices[cut, panel.column(symbol)])
        if not np.isfinite(price) or price <= 0:
            return None
        factor = fundamentals.split_factor(splits.get(symbol, []), day)
        return fundamentals.reading(company, day, price, split_factor=factor)

    return at


def run(
    *,
    cache: Path,
    since: str,
    top: int,
    cost: float,
    horizon: int,
    eras: int,
    refresh: bool,
    with_fundamentals: bool = False,
) -> None:
    warnings.filterwarnings("ignore")

    panel = universe.load(cache, since=since, refresh=refresh)
    proxies = set(replay.market_symbols())
    tradable = tuple(s for s in panel.symbols if s not in proxies)
    days = replay.rebalance_days(panel.dates)

    print(
        f"\n  Universe   {len(tradable)} names (S&P 500 as of today — survivorship, see module doc)"
    )
    print(f"  Period     {panel.dates[0]} -> {panel.dates[-1]}")
    print(f"  Decisions  {len(days)} month ends, traded at the next close, {cost:.2%} round trip")

    books_of_record: dict[str, fundamentals.Company] = {}
    splits: dict[str, list[tuple[str, float]]] = {}
    if with_fundamentals:
        print("\n  Fetching point-in-time fundamentals from SEC filings...", flush=True)
        books_of_record = fundamentals.fetch(tradable, cache.parent / "sec_facts.json")
        splits = fundamentals.splits(tradable, cache.parent / "sec_splits.json")
        print(f"  {len(books_of_record)} of {len(tradable)} names have SEC filings")

    rankings: list[tuple[int, list[replay.Ranked]]] = []
    for cut in days:
        provider = None
        if with_fundamentals:
            provider = _fundamentals_at(panel, books_of_record, splits, cut)
        ranked = replay.rank_at(panel, cut, tradable=tradable, fundamentals=provider)
        if ranked:
            rankings.append((cut, ranked))

    if not rankings:
        raise SystemExit("nothing was rankable; check the panel")

    sizes = [len(r) for _, r in rankings]
    median = int(np.median(sizes))
    print(f"  Ranked     {min(sizes)}-{max(sizes)} names per decision (median {median})")

    best = [(cut, [r.column for r in ranked[:top]]) for cut, ranked in rankings]
    worst = [(cut, [r.column for r in ranked[-top:]]) for cut, ranked in rankings]
    everything = [(cut, [r.column for r in ranked]) for cut, ranked in rankings]

    control = replay.simulate(panel, everything, name="equal weight, all names", cost=cost)
    books = [
        replay.simulate(panel, best, name=f"scanner top {top}", cost=cost),
        replay.simulate(panel, worst, name=f"scanner bottom {top}", cost=cost),
        control,
        replay.buy_and_hold(panel, universe.BENCHMARK, rankings[0][0] + 1),
    ]
    real = replay.buy_and_hold(panel, universe.EQUAL_WEIGHT, rankings[0][0] + 1)
    books.append(real)
    _table(books)
    _survivorship(control, real)
    _per_year(books[:2], control)

    pooled = replay.forward_returns(panel, rankings, horizon=horizon)
    _deciles(replay.by_decile(pooled["score"], pooled["forward"]), horizon)
    _groups(panel, rankings, horizon=horizon, top=top, cost=cost)
    _eras(panel, rankings, horizon=horizon, parts=eras)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/sp500_panel.npz"))
    parser.add_argument("--since", default="2004-01-01")
    parser.add_argument("--top", type=int, default=20, help="names held in the ranked book")
    parser.add_argument("--cost", type=float, default=0.001, help="round-trip cost fraction")
    parser.add_argument("--horizon", type=int, default=21, help="bars for the decile test")
    parser.add_argument("--eras", type=int, default=3, help="equal periods to re-measure in")
    parser.add_argument("--refresh", action="store_true", help="re-download the panel")
    parser.add_argument(
        "--fundamentals",
        action="store_true",
        help="score the full 100 points using point-in-time SEC filings",
    )
    args = parser.parse_args()

    run(
        cache=args.cache,
        since=args.since,
        top=args.top,
        cost=args.cost,
        horizon=args.horizon,
        eras=args.eras,
        refresh=args.refresh,
        with_fundamentals=args.fundamentals,
    )


if __name__ == "__main__":
    main()
