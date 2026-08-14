# Does Kronos predict a 2% fall better than what we already have?

**No.** Tested four ways — read off frozen, fine-tuned, sampled from, and
blended in — it lands below the detector we ship, and the gap is real rather
than a matter of overlapping error bars. This directory is the evidence, kept so
the question does not have to be re-opened from scratch.

[Kronos](https://github.com/shiyu-coder/Kronos) (MIT, AAAI 2026) is a
foundation model for candlesticks: a tokenizer that turns OHLCV bars into
discrete tokens, and a decoder-only transformer pre-trained on 12 billion bars
from 45 exchanges. It is the obvious thing to try, which is why it was tried.

## The result

Held out from 2016-01-01 to 2026-08-12 — 2,667 days, 92 of them followed by a
close 2% or more below the day before. Every detector scored on identical days.
AUC is "pick a day that fell and a day that did not; how often is the faller
ranked higher" — 0.5 is a coin toss.

| detector | AUC | 90% range | top 10% hit rate |
|---|---|---|---|
| **shipped detector** (logistic, 10 market features) | **0.798** | 0.757-0.836 | 13.5% |
| shipped + best Kronos, ranks averaged | 0.793 | 0.752-0.831 | 14.2% |
| Kronos-small state + choppiness, 256-day window | 0.767 | 0.721-0.808 | 14.2% |
| **Kronos-small state, 256-day window** | **0.766** | 0.720-0.807 | **14.6%** |
| **recent choppiness alone** (20-day spread, no model) | **0.749** | 0.700-0.794 | 12.7% |
| Kronos-small, **fine-tuned**, 256-day window | 0.749 | 0.702-0.793 | 13.1% |
| Kronos-small state, 128-day window | 0.745 | 0.701-0.785 | 12.4% |
| shipped detector, with insider data | 0.724 | 0.675-0.768 | 11.2% |
| Kronos-small state, 512-day window | 0.704 | 0.658-0.747 | 10.1% |
| Kronos-base state, 512-day window | 0.690 | 0.643-0.740 | 10.9% |
| Kronos-small, **fine-tuned** (last 2 blocks unfrozen) | 0.686 | 0.635-0.734 | 10.9% |
| Kronos zero-shot: spread of sampled futures | 0.614 | 0.565-0.660 | 5.6% |
| Kronos zero-shot: **its own chance of a 2% fall** | 0.499 | 0.448-0.547 | 3.4% |

Base rate 3.45%. Guessing scores 0.500.

### Read the paired test, not the overlapping ranges

Those 90% ranges are wide and they overlap, which tempts the conclusion that
nothing is distinguishable. That conclusion would be wrong, and for an
interesting reason: both intervals are wide mostly because the test window is
short, and that noise is *shared* — the same 92 events drive every row. Resample
the days and take the **difference** and the shared noise cancels:

| against the shipped detector | AUC gap | 90% range | better in |
|---|---|---|---|
| shipped + best Kronos, averaged | -0.005 | -0.020 to +0.009 | 29% |
| Kronos-small, 256-day window | -0.032 | -0.063 to -0.003 | 3% |
| recent choppiness | -0.049 | -0.074 to -0.023 | 0% |
| Kronos-small, 512-day window | -0.094 | -0.126 to -0.064 | 0% |
| Kronos-base | -0.107 | -0.152 to -0.065 | 0% |
| zero-shot chance of a 2% fall | -0.299 | -0.369 to -0.229 | 0% |

Every Kronos variant is genuinely behind, including the best one. And the
blend — shipped plus Kronos — sits at -0.005 with a range straddling zero, which
is the sharpest statement in this directory: **adding Kronos to what we have
changes nothing measurable.** If it were right about a different set of days, the
blend would beat both halves. It does not.

## What did happen, that is worth keeping

**Its own crash probability is worthless, and badly calibrated with it.**
Sampling 512 futures per day from the pre-trained model and counting how many
fall 2% scores 0.499 over 8,709 days — a coin toss. It also puts an average
11.5% chance on a 2% fall against an actual rate of 3.6%, and says 13.0% on days
that fell versus 11.5% on days that did not. Kronos has a broad sense of how
volatile markets are and no sense of which particular day is dangerous.

**How much history you show it matters more than which model you use.** The same
probe scores 0.704 on a 512-day window and 0.766 on a 256-day one — a bigger
swing than between the 24.7M and 102M models. Two years of daily bars is more
context than it can use. `sweep.py` picks the window on cross-validated training
score; on this data that column and the held-out column agree on 256, so nothing
here is hindsight.

**Fine-tuning did not help, which is the expected shape.** Unfreezing the last
two transformer blocks puts 5.25 million weights against 180 training falls;
validation peaks at epoch 1 and declines from there in every run. On the
512-day window it scored 0.686, *below* the frozen probe's 0.704; on the 256-day
window 0.749, again below the frozen probe's 0.766. Training the model on our
question makes it worse than reading it off — there is not enough of the rare
event to move that many weights toward it.

**Bigger did not help.** Kronos-base (102M) scored below Kronos-small (24.7M) on
the same window. That is a hint that the limit is what price history contains,
not model capacity.

## The finding that matters more than Kronos

**Most of what our shipped detector knows is "the market has been jumpy
lately."** Twenty-day return spread, on its own, scores 0.749 against the
detector's 0.798 with ten features and a fitted model. It is genuinely behind
(-0.049, entirely below zero on the paired test) — but not by much, for one line
of code against the whole apparatus.

Every Kronos variant here also converges toward that same number from below,
which is the tell: what a price-only model can extract from an index chart is
essentially the volatility signal. The productive direction is a signal that is
**not** a volatility proxy — the insider and credit-spread features are attempts
at that — rather than a larger model of the same information.

## Caveats, stated rather than buried

**Pre-training contamination cannot be ruled out.** Neither the paper's abstract
nor the model card gives a cut-off date for the corpus, and it spans 45
exchanges. If US index history through 2024 is in there, most of our test window
sits inside the model's training data — which would make these results
*flattering* to Kronos, not harsh. `compare.py` reports the 2025-onward slice
separately as the only clean part; it holds 9 falls, every interval spans
roughly 0.3 to 0.9, and it settles nothing either way. It is printed so nobody
has to take that on trust, not because it is evidence.

**One market, one horizon.** Everything here is `^GSPC`, daily, next-day, 2%.
Kronos may well be good at things that are not this. In particular it was built
for multi-step forecasting of the whole bar, and we ask it for a one-day tail
probability — the narrowest question it can be asked.

**A protocol mistake, kept in the record.** The first version chose settings on a
single 2012-2015 validation window holding fifteen falls. One variant scored 0.70
there and 0.54 on the test set, and the first Kronos numbers were flattered by
the same noise — an early draft of this file reported 0.749 where the honest
figure was 0.704. Switching to expanding-window cross-validation over all the
training years fixed it. If these scripts are extended, do not reintroduce a
single small validation window.

## Layout

| file | what it does |
|---|---|
| `data.py` | `^GSPC` daily bars, and the 2%-fall label as production defines it |
| `engine.py` | running Kronos: hidden states out, futures sampled from it |
| `metrics.py` | AUC, lift, bootstrap intervals, and the paired duel |
| `fetch.py` | download and freeze the index (run this first) |
| `verify.py` | proves the fast paths in `engine.py` equal upstream's slow ones |
| `features.py` | cache the hidden state for every day |
| `probe.py` | fit a classifier on the frozen state |
| `sweep.py` | choose the lookback on training data, then show what it cost |
| `finetune.py` | unfreeze the last blocks and actually train it |
| `zeroshot.py` | sample futures and count the falls — no training at all |
| `compare.py` | every detector, identical days, one table, plus the paired test |
| `tests/` | the label definition and the scoring, which everything else rests on |

## Running it

Kronos pins `pandas==2.2.2` and needs torch; the API runs pandas 3 and must not
ship torch. So this lives in its own virtualenv and talks to the rest of the
system through CSV files rather than imports.

```bash
git clone https://github.com/shiyu-coder/Kronos.git ~/.cache/kronos-work/Kronos
uv venv ~/.cache/kronos-work/venv --python 3.12
VIRTUAL_ENV=~/.cache/kronos-work/venv uv pip install \
    torch numpy "pandas>=2.2,<3" einops huggingface_hub safetensors \
    tqdm yfinance scikit-learn pytest

export KRONOS_HOME=~/.cache/kronos-work/Kronos      # defaults to this anyway
export PYTHONPATH=$PWD
P=~/.cache/kronos-work/venv/bin/python

$P fetch.py                                  # cache the index and labels
$P verify.py                                 # check our fast paths against upstream
$P -m pytest tests -q

$P features.py --size small --lookback 256   # ~2 min on an M3 Pro
$P probe.py    --size small --features cache/features_small_256.npz
$P sweep.py    --size small                  # after extracting 128/256/512
$P finetune.py --size small --unfreeze 2 --check-from 2010-01-01 --lookback 256
$P zeroshot.py --size small --count 512 --since 1990-01-01   # ~15 min
```

The baseline half runs in the API's virtualenv and writes the CSVs this side
reads:

```bash
cd ../../apps/api
.venv/bin/python -m app.scripts.dump_crash_probabilities \
    --since 1990-01-01 --split-date 2016-01-01 \
    --out ../../research/kronos/cache/logistic_1990.csv
```

Then `compare.py` joins everything on date and prints both tables above.

## A note on `engine.py`

Two of its functions reimplement upstream logic, both because the upstream
version could not answer the question:

* `KronosPredictor.predict` **averages** the sampled futures before returning
  them. The average is a forecast of the middle; we need the tail. So the
  sampling loop is written out here, and `verify.py` pins it against theirs with
  the randomness removed (they agree to 3e-8).
* The naive one-sample-at-a-time path was heading for about three hours per run.
  Two exact rearrangements — computing only the final position of the second
  stage, and caching the shared prefix in the token decoder — bring it to six
  minutes. Both are checked in `verify.py` and agree to floating-point noise.
  Shortening the decoder's window would *also* have been fast and is **not**
  exact: `verify.py` measures it moving predictions by as much as the daily
  spread, which is why it is not used.

Two upstream quirks worth knowing if you extend this:

* `top_k_top_p_filtering` guards both of its bodies and has no final `return`,
  so asking it not to filter hands back `None`. Sampling the untruncated
  distribution is the whole point of the zero-shot test — nucleus sampling
  discards the tail, which is the event being counted — so `engine._draw` skips
  the call instead.
* Apple's fused attention kernel raises on dropout, so `model.train()` cannot run
  on an M-series Mac at all. `finetune.py` trains in eval mode; gradients are
  unaffected.
