"""Local Kronos inference, isolated behind a lazy import (§8).

Kronos (AAAI 2026, MIT licensed) is a decoder-only foundation model over K-line
sequences. A tokenizer quantises OHLCV bars into discrete tokens and a
transformer generates the continuation, so a "forecast" is a *sample* of future
candles rather than a number.

**Torch is imported inside the constructor, never at module scope.** That is the
whole design. Importing torch costs ~250-300MB resident before any weights load,
against a 448MB worker on the free-tier box — so a top-level import would make
the entire API and worker unbootable there whether or not anything used Kronos.
This module can be imported anywhere; only *constructing* the client pays.

Consequently the dependency is optional. `is_available()` answers whether the
extra is installed, and everything downstream treats an absent Kronos as "no
prediction today" rather than an error, exactly as the sentiment sweep treats a
missing API key.

Install with:  uv sync --extra kronos     (torch, huggingface-hub, pandas)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import structlog

from app.indicators.functions import FloatArray

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    pass

log = structlog.get_logger(__name__)

#: Hugging Face repositories. mini is 4.1M parameters with a 2048-bar context;
#: small is 24.7M and base 102.3M, both with 512. The weights are small — the
#: memory cost is torch itself, which is why the variant barely matters for
#: fitting on a box and matters a lot for throughput.
MODELS: dict[str, tuple[str, str, int]] = {
    # name: (model repo, tokenizer repo, context bars)
    "kronos-mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "kronos-small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "kronos-base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}

DEFAULT_MODEL = "kronos-small"

#: Sampled paths per instrument. The model is stochastic, so one path is a draw
#: rather than a forecast; the spread across paths is the useful part. More
#: paths cost linear time, and time is the binding constraint on universe size.
DEFAULT_SAMPLES = 8

#: Sampling temperature and nucleus cutoff. The defaults from the model card —
#: not tuned here, because tuning a generative sampler against the same data the
#: strategy is measured on is how a backtest is talked into agreeing with itself.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9


@dataclass(frozen=True, slots=True)
class KronosForecast:
    """What one instrument's sampled paths reduce to."""

    predicted_return: float
    #: Spread of terminal returns across paths — the model's own uncertainty.
    path_dispersion: float
    #: Share of paths ending above the last observed close.
    prob_up: float
    #: Deepest drawdown along the mean path, positive.
    predicted_drawdown: float
    horizon_days: int
    sample_count: int
    context_bars: int
    model_name: str
    generation_ms: int


def is_available() -> bool:
    """Whether the optional Kronos extra is installed.

    Checked by importlib rather than a try/import so that asking the question
    does not itself load torch.
    """
    from importlib.util import find_spec

    return find_spec("torch") is not None


class KronosUnavailableError(RuntimeError):
    """Raised when inference is attempted without the extra installed."""


class KronosClient:
    """Wraps one loaded model. Construct once per job, never per instrument.

    Loading weights takes seconds; constructing this inside a loop over a
    universe would dominate the runtime.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if not is_available():
            raise KronosUnavailableError("Kronos needs the optional extra: uv sync --extra kronos")
        if model_name not in MODELS:
            raise ValueError(f"unknown model {model_name!r}; choose from {sorted(MODELS)}")

        # Deliberately inside __init__ — see the module docstring.
        import torch  # type: ignore[import-not-found]

        self._torch = torch
        self.model_name = model_name
        repo, tokenizer_repo, self.context_bars = MODELS[model_name]

        if device is None:
            # MPS is Apple Silicon's GPU. Worth preferring: these models are
            # small enough that the transfer overhead does not eat the gain, and
            # generation is autoregressive so it is latency-bound.
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore[import-not-found]

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_repo, cache_dir=cache_dir)
        model = Kronos.from_pretrained(repo, cache_dir=cache_dir)
        self._predictor = KronosPredictor(
            model, tokenizer, device=device, max_context=self.context_bars
        )
        log.info("kronos.loaded", model=model_name, device=device, context=self.context_bars)

    def forecast(
        self,
        open_: FloatArray,
        high: FloatArray,
        low: FloatArray,
        close: FloatArray,
        volume: FloatArray,
        timestamps: list[datetime],
        *,
        horizon_days: int = 20,
        sample_count: int = DEFAULT_SAMPLES,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> KronosForecast | None:
        """Sample `sample_count` continuations and reduce them to one reading.

        Returns None when the series is too short to fill any useful context —
        an unanswerable question, distinct from a forecast of no move.

        The caller is responsible for passing only bars that had already closed
        at the time being simulated. This method cannot check that, and it is
        the one way a Kronos feature could silently leak the future.
        """
        import pandas as pd

        n = close.size
        if n < 64 or len(timestamps) != n:
            return None

        context = min(self.context_bars, n)
        frame = pd.DataFrame(
            {
                "open": open_[-context:],
                "high": high[-context:],
                "low": low[-context:],
                "close": close[-context:],
                "volume": volume[-context:],
            }
        )
        x_timestamp = pd.Series(pd.to_datetime(timestamps[-context:]))
        # Business days ahead. The exact calendar matters less than the count:
        # the model conditions on the sequence, and these are only labels for
        # the positions it is being asked to fill.
        y_timestamp = pd.Series(
            pd.bdate_range(start=x_timestamp.iloc[-1], periods=horizon_days + 1)[1:]
        )

        # One call per path, rather than one call for all of them.
        #
        # `predict(sample_count=n)` averages internally and returns a single
        # path, which throws away the very thing worth having: how much the
        # samples disagreed. Eight sketches that all say "+3%" and eight that
        # average to +3% from +20/-15/+8/-12 are the same number and completely
        # different information — the first is a forecast, the second is noise
        # wearing a suit.
        #
        # The cost is a factor of `sample_count` in runtime. That is affordable
        # precisely because the strategy universe is ~20 names; at catalogue
        # scale this would have to go back to the averaged call, and the
        # dispersion feature would have to be dropped rather than faked.
        started = time.perf_counter()
        paths: list[FloatArray] = []
        for _ in range(max(sample_count, 1)):
            try:
                drawn = self._predictor.predict(
                    df=frame,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=horizon_days,
                    T=temperature,
                    top_p=top_p,
                    sample_count=1,
                )
            except Exception as exc:  # one bad series must not end a sweep
                log.warning("kronos.forecast_failed", error=str(exc))
                return None
            if drawn is None or len(drawn) == 0:
                continue
            closes_out = np.asarray(drawn["close"], dtype=np.float64)
            if closes_out.size and np.isfinite(closes_out).all():
                paths.append(closes_out)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if not paths:
            return None
        return self._reduce(paths, float(close[-1]), horizon_days, elapsed_ms)

    def _reduce(
        self,
        paths: list[FloatArray],
        last_close: float,
        horizon_days: int,
        elapsed_ms: int,
    ) -> KronosForecast | None:
        """Reduce the sampled paths to the four numbers that get stored.

        Every statistic here is taken *across paths*, which is what makes them
        meaningful: the mean is the forecast, and the spread around it is the
        model's own confidence in that forecast.
        """
        if not paths or last_close <= 0:
            return None

        # Ragged paths would mean the model returned different lengths; align to
        # the shortest rather than padding, since a padded value is invented.
        length = min(p.size for p in paths)
        if length == 0:
            return None
        stacked = np.vstack([p[:length] for p in paths])

        terminals = stacked[:, -1] / last_close - 1.0
        mean_path = stacked.mean(axis=0)

        # Deepest drawdown along the *mean* path. Measured on the average rather
        # than averaged across paths, because a typical journey is what a holder
        # would experience; averaging each path's worst point would report a
        # drawdown no single path actually took.
        with_entry = np.concatenate(([last_close], mean_path))
        peak = np.maximum.accumulate(with_entry)
        drawdown = float(np.max((peak - with_entry) / peak))

        return KronosForecast(
            predicted_return=float(terminals.mean()),
            # Standard deviation of where the paths *ended*. High means the model
            # is guessing, and that is usable in its own right.
            path_dispersion=float(terminals.std(ddof=1)) if terminals.size > 1 else 0.0,
            # Share of paths ending above the entry — a genuine probability now,
            # rather than the share of one averaged path spent above it.
            prob_up=float((terminals > 0).mean()),
            predicted_drawdown=max(drawdown, 0.0),
            horizon_days=horizon_days,
            sample_count=len(paths),
            context_bars=self.context_bars,
            model_name=self.model_name,
            generation_ms=elapsed_ms,
        )
