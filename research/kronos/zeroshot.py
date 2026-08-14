"""Ask Kronos for tomorrow, many times, and count the crashes. (No training.)

    python zeroshot.py --size small --count 256

The probe fits a classifier on the model's hidden state, which can only ever
express what a straight line through that state can express. This asks the model
a different way: let it *generate* tomorrow, hundreds of times, and take the
share of those futures that fall 2% or more. Nothing is fitted, so there is no
train/test split to argue about and no way to overfit — whatever this scores, it
scored without seeing a single label.

## Why the nucleus is switched off

Upstream defaults to `top_p=0.9`, which before sampling discards the least
likely tenth of the probability mass. That is a sensible default for generating
a plausible-looking chart and a fatal one here, because the tenth being
discarded is the tail, and the tail is the entire question. We sample from the
full distribution and say so; `--top-p` remains available for comparison.

## What gets saved

Every sampled return, not just the summary. The samples are the expensive part,
and keeping them means later questions — a different threshold, a quantile
instead of a count, how the spread itself ranks — cost nothing to answer and
cannot be accused of having re-run anything to get a nicer number.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import data
import engine


def run(
    eng: engine.Engine,
    market: data.Market,
    at: np.ndarray,
    *,
    count: int,
    temperature: float,
    top_p: float,
    batch: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    out = np.empty((at.size, count), dtype=np.float32)
    started = time.time()
    for start in range(0, at.size, batch):
        chunk = at[start : start + batch]
        out[start : start + chunk.size] = engine.sample_next_bar(
            eng,
            market.windows(chunk),
            market.index,
            chunk,
            count=count,
            temperature=temperature,
            top_p=top_p,
        )
        done = start + chunk.size
        if done % (batch * 10) == 0 or done == at.size:
            elapsed = time.time() - started
            print(
                f"    {done:>6,}/{at.size:,}  {elapsed / 60:5.1f} min elapsed, "
                f"{(at.size - done) / (done / elapsed) / 60:5.1f} to go",
                flush=True,
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample futures and count the 2% falls.")
    parser.add_argument("--cache", type=Path, default=Path("cache/sp500.npz"))
    parser.add_argument("--size", default="small", choices=("mini", "small", "base"))
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--since", default="2014-01-01")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    market = data.load(args.cache)
    at = market.usable(since=args.since)
    eng = engine.load(args.size)
    out = args.out or Path(f"cache/zeroshot_{args.size}.npz")

    print(f"\n  Kronos-{args.size} on {eng.device}, {args.count} futures per day")
    print(
        f"  {at.size:,} days from {market.index[at[0]].date()}, "
        f"temperature {args.temperature}, top_p {args.top_p}\n"
    )

    samples = run(
        eng,
        market,
        at,
        count=args.count,
        temperature=args.temperature,
        top_p=args.top_p,
        batch=args.batch,
        seed=args.seed,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        bar=at,
        date=market.index[at].values.astype("datetime64[D]"),
        samples=samples,
        label=market.label[at],
        forward=market.forward[at],
    )

    chance = (samples <= -0.02).mean(axis=1)
    fell = market.label[at] > 0.5
    print(f"\n  it thought a 2% fall was {chance.mean():.2%} likely on an average day")
    print(f"  actual rate over the same days: {fell.mean():.2%}")
    print(f"  on the days that did fall, it had said {chance[fell].mean():.2%};")
    print(f"  on the days that did not, {chance[~fell].mean():.2%}")
    print(f"\n  written to {out}\n")


if __name__ == "__main__":
    main()
