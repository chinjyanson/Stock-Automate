"""Prove the fast path is the same path.

    python verify.py

`engine.py` earns its speed by computing only the final position of the second
stage, and by optionally shortening the window used to decode a generated token
back into a price. Both are claims about equivalence, and claims about
equivalence are worth exactly what their tests are worth. This script checks
three things against upstream's own code:

  1. **Second stage.** Our hand-written final position against
     `model.decode_s2(...)[:, -1, :]` on identical inputs. This must agree to
     floating-point noise; anything larger means the rotary positions are wrong.
  2. **End to end, greedy.** With a nucleus narrow enough that both paths must
     take the most likely token, our sampled bar against
     `KronosPredictor.predict`. Removes the randomness so the comparison is an
     equality rather than a distribution test.
  3. **Decode window.** How far the price moves when the decoder is shown a
     shorter trailing window. This one is not expected to be zero — it is
     reported so the shortcut can be accepted or refused on evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data
import engine


def check_second_stage(eng: engine.Engine, market: data.Market, at: np.ndarray) -> float:
    """Ours against theirs, on the same context and the same candidate tokens."""
    windows = market.windows(at)
    scaled, _, _ = engine.normalise(windows)
    x = torch.from_numpy(scaled.astype(np.float32)).to(eng.device)
    stamp = torch.from_numpy(engine._stamps_for(market.index, at, windows.shape[1]))

    with torch.no_grad():
        s1, s2 = eng.tokenizer.encode(x, half=True)
        _, context = eng.model.decode_s1(s1, s2, stamp.to(eng.device))

        batch, length = s1.shape
        count = 5
        rng = torch.Generator(device="cpu").manual_seed(7)
        drawn = torch.randint(0, 2**eng.model.s1_bits, (batch, count), generator=rng)
        drawn = drawn.to(eng.device)

        ours = engine.second_stage_logits(eng.model, context, drawn)

        # Theirs, one candidate at a time, through the unmodified method.
        gaps = []
        for k in range(count):
            full_s1 = s1.clone()
            full_s1[:, -1] = drawn[:, k]
            theirs = eng.model.decode_s2(context, full_s1)[:, -1, :]
            gaps.append((ours[:, k, :] - theirs).abs().max().item())
    return max(gaps)


def check_greedy(eng: engine.Engine, market: data.Market, bar: int) -> tuple[float, float]:
    """The whole pipeline against `KronosPredictor.predict`, randomness removed."""
    from model import KronosPredictor

    predictor = KronosPredictor(eng.model, eng.tokenizer, device=eng.device, max_context=512)
    frame = pd.DataFrame(market.bars[bar - 511 : bar + 1], columns=list(data.COLUMNS))
    x_stamp = pd.Series(market.index[bar - 511 : bar + 1])
    y_stamp = pd.Series(market.index[bar + 1 : bar + 2])

    theirs = predictor.predict(
        frame, x_stamp, y_stamp, pred_len=1, T=1.0, top_p=1e-6, sample_count=1, verbose=False
    )
    their_return = float(theirs["close"].iloc[0]) / market.close[bar] - 1.0

    at = np.array([bar])
    ours = engine.sample_next_bar(
        eng, market.windows(at), market.index, at, count=1, temperature=1.0, top_p=1e-6
    )
    return their_return, float(ours[0, 0])


def check_cached_decode(eng: engine.Engine, market: data.Market, at: np.ndarray) -> float:
    """The cached decoder against the plain one, same draws, same everything."""
    windows = market.windows(at)
    torch.manual_seed(5)
    fast = engine.sample_next_bar(eng, windows, market.index, at, count=24, cached=True)
    torch.manual_seed(5)
    slow = engine.sample_next_bar(eng, windows, market.index, at, count=24, cached=False)
    return float(np.abs(fast - slow).max())


def check_decode_tail(
    eng: engine.Engine, market: data.Market, at: np.ndarray, tails: tuple[int, ...]
) -> dict[int, tuple[float, float]]:
    """Shorten the decoder's window and see what it costs, in return terms."""
    windows = market.windows(at)
    reference = None
    out: dict[int, tuple[float, float]] = {}
    for tail in tails:
        torch.manual_seed(3)
        sampled = engine.sample_next_bar(eng, windows, market.index, at, count=32, decode_tail=tail)
        if reference is None:
            reference = sampled
        gap = np.abs(sampled - reference)
        out[tail] = (float(gap.mean()), float(gap.max()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the fast path against upstream.")
    parser.add_argument("--cache", type=Path, default=Path("cache/sp500.npz"))
    parser.add_argument("--size", default="small")
    args = parser.parse_args()

    market = data.load(args.cache)
    at = market.usable(since="2019-01-01", until="2019-02-01")[:6]
    eng = engine.load(args.size)

    print(
        f"\n  Kronos-{args.size} on {eng.device}, {at.size} sample days from "
        f"{market.index[at[0]].date()}\n"
    )

    gap = check_second_stage(eng, market, at)
    verdict = "same" if gap < 1e-3 else "DIFFERENT"
    print(f"  second stage, ours vs model.decode_s2   max logit gap {gap:.2e}   {verdict}")

    theirs, ours = check_greedy(eng, market, int(at[3]))
    verdict = "same" if abs(theirs - ours) < 1e-6 else "DIFFERENT"
    print(
        f"  greedy bar, ours vs KronosPredictor     {theirs:+.6%} vs {ours:+.6%}   "
        f"gap {abs(theirs - ours):.2e}   {verdict}"
    )

    gap = check_cached_decode(eng, market, at)
    verdict = "same" if gap < 1e-5 else "DIFFERENT"
    print(f"  cached decoder, ours vs tokenizer.decode  max return gap {gap:.2e}   {verdict}")

    print("\n  decoder window, against the full 512:")
    for tail, (mean_gap, worst) in check_decode_tail(
        eng, market, at, (512, 128, 64, 32, 16)
    ).items():
        note = "  <- exact by construction" if tail == 512 else ""
        print(f"    tail {tail:>4}   mean move {mean_gap:.4%}   worst {worst:.4%}{note}")
    print("\n  (a move here is a change in the predicted next-day return, so compare")
    print("   it against the ~1% daily spread the model actually forecasts)\n")


if __name__ == "__main__":
    main()
