"""Download the index once and freeze it.

    python fetch.py

Everything downstream reads the cache this writes. That is deliberate: yfinance
restates history quietly, and a comparison whose inputs move between runs cannot
be checked by anyone, including us tomorrow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import data


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache S&P 500 daily bars and labels.")
    parser.add_argument("--since", default="1990-01-01")
    parser.add_argument("--until", default=None)
    parser.add_argument("--out", type=Path, default=Path("cache/sp500.npz"))
    args = parser.parse_args()

    market = data.fetch(since=args.since, until=args.until)
    data.save(market, args.out)

    known = np.isfinite(market.label)
    usable = market.usable()
    print(
        f"\n  {market.index.size:,} daily bars, {market.index[0].date()} to {market.index[-1].date()}"
    )
    print(
        f"  {int(market.label[known].sum()):,} of them are followed by a 2% fall "
        f"({market.label[known].mean():.2%} of days)"
    )
    print(f"  {usable.size:,} usable decision bars (a full {data.LOOKBACK}-day window behind,")
    print(f"  a known answer ahead), from {market.index[usable[0]].date()}\n")
    print(f"  written to {args.out}\n")


if __name__ == "__main__":
    main()
