"""Run Kronos over every day once and keep what it was thinking.

    python features.py --size small

The model's final hidden state is its summary of the last 512 trading days,
formed just before it would commit to a guess about tomorrow. Extracting it once
and caching it turns the expensive part of every later experiment into a matrix
that fits in memory, which is what makes it practical to try twenty variants of
a classifier on a laptop instead of one.

Nothing here is fitted, so this cache is not contaminated by any train/test
split: the same array is used for both sides, and the split happens downstream.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import data
import engine


def extract(
    eng: engine.Engine,
    market: data.Market,
    at: np.ndarray,
    *,
    batch: int = 64,
    lookback: int = data.LOOKBACK,
) -> np.ndarray:
    out = np.empty((at.size, eng.width), dtype=np.float32)
    started = time.time()
    for start in range(0, at.size, batch):
        chunk = at[start : start + batch]
        out[start : start + chunk.size] = engine.hidden_states(
            eng, market.windows(chunk, lookback), market.index, chunk
        )
        done = start + chunk.size
        if done % (batch * 20) == 0 or done == at.size:
            elapsed = time.time() - started
            rate = done / elapsed
            print(
                f"    {done:>6,}/{at.size:,}  {elapsed / 60:5.1f} min elapsed, "
                f"{(at.size - done) / rate / 60:5.1f} to go",
                flush=True,
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache Kronos hidden states per day.")
    parser.add_argument("--cache", type=Path, default=Path("cache/sp500.npz"))
    parser.add_argument("--size", default="small", choices=("mini", "small", "base"))
    parser.add_argument("--since", default=None)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lookback", type=int, default=data.LOOKBACK)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    market = data.load(args.cache)
    at = market.usable(since=args.since, lookback=args.lookback)
    eng = engine.load(args.size)
    suffix = "" if args.lookback == data.LOOKBACK else f"_{args.lookback}"
    out = args.out or Path(f"cache/features_{args.size}{suffix}.npz")

    print(f"\n  Kronos-{args.size} on {eng.device}, width {eng.width}, lookback {args.lookback}")
    print(f"  {at.size:,} days, {market.index[at[0]].date()} to {market.index[at[-1]].date()}\n")

    states = extract(eng, market, at, batch=args.batch, lookback=args.lookback)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        bar=at,
        date=market.index[at].values.astype("datetime64[D]"),
        state=states,
        label=market.label[at],
        forward=market.forward[at],
    )
    print(f"\n  {states.shape} written to {out}\n")


if __name__ == "__main__":
    main()
