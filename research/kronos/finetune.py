"""Actually fine-tune Kronos on the question, rather than reading it off.

    python finetune.py --size small --unfreeze 2

`probe.py` freezes the backbone and fits a straight line on top, which answers
"is the answer already in there, in a form a line can reach?". The answer came
back no. This asks the harder version: given the chance to *change* its weights,
can Kronos learn to spot a 2% fall?

It is worth doing even though the odds are poor, because the odds being poor is
an argument and not a measurement, and because "we only tried the cheap version"
is a fair objection to the probe's result.

## What the odds actually are

The training years contain about 224 days followed by a 2% fall. Kronos-small
has 24.7 million parameters. There is no version of this where the whole model
is fitted to that; it would memorise the 224 days exactly and generalise not at
all. So the last few transformer blocks are unfrozen and the rest is held still,
which is the standard way of adapting a pre-trained model on a small dataset,
and even that is generous at this ratio.

Two guards, both fixed in advance:

  * **Early stopping on a validation window the fit never sees**, checked every
    epoch, keeping the weights from the best epoch rather than the last.
  * **The floor is in the table.** Recent choppiness is one number and no
    training at all. If the fine-tune lands below it, the fine-tune has lost, no
    matter how respectable its loss curve looked.

## Why the tokens are cached

The tokenizer is frozen throughout, so it maps each window to the same tokens on
every epoch. Running it once and keeping the result turns a 2-minute-per-epoch
cost into a one-off, which is most of what makes this practical on a laptop.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import data
import engine
import metrics


class Detector(nn.Module):
    """Kronos, plus one linear layer that turns its state into a probability."""

    def __init__(self, model: engine.Kronos, width: int) -> None:
        super().__init__()
        self.model = model
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, stamp: torch.Tensor) -> torch.Tensor:
        _, context = self.model.decode_s1(s1, s2, stamp)
        return self.head(context[:, -1, :]).squeeze(-1)


def cache_tokens(
    eng: engine.Engine,
    market: data.Market,
    at: np.ndarray,
    *,
    batch: int = 64,
    lookback: int = data.LOOKBACK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenise every window once. The tokenizer never trains, so this is safe."""
    s1_all, s2_all, stamp_all = [], [], []
    started = time.time()
    for start in range(0, at.size, batch):
        chunk = at[start : start + batch]
        s1, s2, stamp = engine.tokenize(eng, market.windows(chunk, lookback), market.index, chunk)
        s1_all.append(s1.to(torch.int16).cpu())
        s2_all.append(s2.to(torch.int16).cpu())
        stamp_all.append(stamp.cpu())
    print(f"    tokenised {at.size:,} windows in {time.time() - started:.0f}s", flush=True)
    return torch.cat(s1_all), torch.cat(s2_all), torch.cat(stamp_all)


def unfreeze(detector: Detector, blocks: int) -> list[torch.nn.Parameter]:
    """Freeze everything, then release the head and the last `blocks` layers."""
    for parameter in detector.model.parameters():
        parameter.requires_grad_(False)
    released = list(detector.head.parameters())
    if blocks > 0:
        for layer in detector.model.transformer[-blocks:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
            released += list(layer.parameters())
        for parameter in detector.model.norm.parameters():
            parameter.requires_grad_(True)
        released += list(detector.model.norm.parameters())
    return released


@torch.no_grad()
def score(
    detector: Detector,
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rows: np.ndarray,
    device: str,
    batch: int = 64,
) -> np.ndarray:
    detector.eval()
    s1, s2, stamp = tokens
    out = np.empty(rows.size, dtype=np.float64)
    for start in range(0, rows.size, batch):
        pick = rows[start : start + batch]
        logits = detector(
            s1[pick].long().to(device), s2[pick].long().to(device), stamp[pick].to(device)
        )
        out[start : start + pick.size] = torch.sigmoid(logits).float().cpu().numpy()
    return out


def train(
    detector: Detector,
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    label: np.ndarray,
    fit_rows: np.ndarray,
    check_rows: np.ndarray,
    *,
    device: str,
    epochs: int,
    batch: int,
    head_rate: float,
    body_rate: float,
    blocks: int,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    unfreeze(detector, blocks)
    s1, s2, stamp = tokens
    target = torch.from_numpy(label).float()

    body = [p for p in detector.model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(
        [
            {"params": detector.head.parameters(), "lr": head_rate},
            {"params": body, "lr": body_rate},
        ],
        weight_decay=0.01,
    )

    # One day in thirty is a fall, so an unweighted loss is minimised by always
    # answering "no". This is the same correction `class_weight="balanced"` makes
    # in the probe, written out.
    positives = float(label[fit_rows].sum())
    pos_weight = torch.tensor((fit_rows.size - positives) / max(positives, 1.0), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best, best_state, best_epoch = -np.inf, None, -1
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        # Deliberately *not* `detector.train()`. Two reasons, one forced and one
        # chosen. Forced: Apple's fused attention kernel raises on dropout, so
        # training mode does not run on this machine at all. Chosen: with the
        # backbone almost entirely frozen and early stopping on a held-out
        # window, dropout is not what is holding overfitting back here, and
        # turning it off makes each epoch's validation number comparable to the
        # last rather than noisy in its own right. Gradients flow either way.
        detector.eval()
        order = rng.permutation(fit_rows)
        total = 0.0
        for start in range(0, order.size, batch):
            pick = order[start : start + batch]
            logits = detector(
                s1[pick].long().to(device), s2[pick].long().to(device), stamp[pick].to(device)
            )
            loss = criterion(logits, target[pick].to(device))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimiser.param_groups for p in g["params"]], 1.0
            )
            optimiser.step()
            total += float(loss.detach()) * pick.size

        value = metrics.auc(score(detector, tokens, check_rows, device), label[check_rows])
        flag = ""
        if value > best:
            best, best_epoch = value, epoch
            best_state = {k: v.detach().clone() for k, v in detector.state_dict().items()}
            flag = "  <- best so far"
        print(
            f"    epoch {epoch + 1:>2}  loss {total / order.size:.4f}  "
            f"validation AUC {value:.3f}{flag}",
            flush=True,
        )

    if best_state is not None:
        detector.load_state_dict(best_state)
    print(f"    kept epoch {best_epoch + 1}", flush=True)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Kronos to spot 2% falls.")
    parser.add_argument("--cache", type=Path, default=Path("cache/sp500.npz"))
    parser.add_argument("--size", default="small", choices=("mini", "small", "base"))
    parser.add_argument("--test-from", default="2016-01-01")
    parser.add_argument("--check-from", default="2012-01-01")
    parser.add_argument("--unfreeze", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--head-rate", type=float, default=1e-3)
    parser.add_argument("--body-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookback", type=int, default=data.LOOKBACK)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    market = data.load(args.cache)
    at = market.usable(lookback=args.lookback)
    date = market.index[at]
    label = market.label[at]

    fit_rows = np.flatnonzero(date < pd.Timestamp(args.check_from))
    check_rows = np.flatnonzero(
        (date >= pd.Timestamp(args.check_from)) & (date < pd.Timestamp(args.test_from))
    )
    test_rows = np.flatnonzero(date >= pd.Timestamp(args.test_from))

    eng = engine.load(args.size)
    print(
        f"\n  Kronos-{args.size} on {eng.device}, last {args.unfreeze} blocks unfrozen, "
        f"lookback {args.lookback}"
    )
    print(
        f"  fit {fit_rows.size:,} days ({int(label[fit_rows].sum())} falls), "
        f"early-stop on {check_rows.size:,} ({int(label[check_rows].sum())} falls), "
        f"test {test_rows.size:,} ({int(label[test_rows].sum())} falls)\n"
    )

    tokens = cache_tokens(eng, market, at, lookback=args.lookback)
    detector = Detector(eng.model, eng.width).to(eng.device)

    # Counted after the freeze, not before it. `train` calls `unfreeze`, so
    # taking this here would print the whole model and quietly overstate what is
    # being fitted — which is the one number this script exists to put next to
    # the number of training falls.
    unfreeze(detector, args.unfreeze)
    trainable = sum(p.numel() for p in detector.parameters() if p.requires_grad)

    best = train(
        detector,
        tokens,
        label,
        fit_rows,
        check_rows,
        device=eng.device,
        epochs=args.epochs,
        batch=args.batch,
        head_rate=args.head_rate,
        body_rate=args.body_rate,
        blocks=args.unfreeze,
        seed=args.seed,
    )

    scores = score(detector, tokens, test_rows, eng.device)
    tag = args.size if args.lookback == data.LOOKBACK else f"{args.size}_{args.lookback}"
    out = args.out or Path(f"cache/pred_finetune_{tag}.csv")
    pd.DataFrame(
        {"date": date[test_rows], "label": label[test_rows], f"finetune_{tag}": scores}
    ).to_csv(out, index=False)

    print(
        f"\n  {trainable:,} trainable parameters against "
        f"{int(label[fit_rows].sum())} training falls"
    )
    print(f"  best validation AUC {best:.3f}\n")
    print(metrics.table([metrics.evaluate(f"finetune_{args.size}", scores, label[test_rows])]))
    print(f"\n  written to {out}\n")


if __name__ == "__main__":
    main()
