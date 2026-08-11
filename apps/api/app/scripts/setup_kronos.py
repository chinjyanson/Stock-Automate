"""Clone Kronos and check it can actually run (§8).

    uv sync --extra kronos
    python -m app.scripts.setup_kronos

Kronos publishes no PyPI package and carries no setup.py, so "installing" it
means cloning the repository and importing its `model` package from the
checkout. This does the clone, records where it went, and — the part that
matters — **runs one real forecast** so a broken setup fails here rather than
silently at 21:00 in a job nobody is watching.

Deliberately not part of `uv sync`. Torch is ~250-300MB resident on import and
does not fit the 448MB worker on the deployment box; this is a local-machine
step, and the box reads predictions from a table instead.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

REPO_URL = "https://github.com/shiyu-coder/Kronos.git"
DEFAULT_PATH = Path.home() / ".cache" / "kronos-src"


def _clone(destination: Path) -> bool:
    if (destination / "model").is_dir():
        print(f"Already present: {destination}")
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {REPO_URL} -> {destination}")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(destination)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  clone failed: {result.stderr.strip()}")
        return False
    return (destination / "model").is_dir()


def _smoke_test(destination: Path, model_name: str) -> bool:
    """Load the model and forecast a synthetic series.

    Synthetic on purpose: this is checking that weights download, the tokenizer
    loads and generation runs end to end. Whether the forecast is any *good* is
    a separate question, and one the feature ranking answers with real data.
    """
    from app.signals.kronos_client import KronosClient

    print(f"\nLoading {model_name} (first run downloads weights)...")
    try:
        client = KronosClient(model_name)
    except Exception as exc:
        print(f"  failed to load: {exc}")
        return False
    print(f"  loaded on {client.device}, context {client.context_bars} bars")

    n = 256
    rng = np.random.default_rng(0)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.02, n))
    timestamps = [datetime.now(UTC) - timedelta(days=n - i) for i in range(n)]

    print("Forecasting a synthetic 256-bar series...")
    started = datetime.now(UTC)
    forecast = client.forecast(
        open_=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=np.full(n, 1_000_000.0),
        timestamps=timestamps,
        horizon_days=20,
        sample_count=4,
    )
    elapsed = (datetime.now(UTC) - started).total_seconds()

    if forecast is None:
        print("  returned nothing — generation failed")
        return False

    print(
        f"  return {forecast.predicted_return:+.2%}  "
        f"dispersion {forecast.path_dispersion:.2%}  "
        f"prob_up {forecast.prob_up:.0%}  "
        f"drawdown {forecast.predicted_drawdown:.2%}"
    )
    print(f"  {elapsed:.1f}s for 4 paths — about {elapsed / 4:.1f}s per path")
    print(
        f"\n  At that rate, 20 instruments x 8 paths is roughly "
        f"{20 * 8 * elapsed / 4 / 60:.0f} minutes."
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone and verify Kronos locally.")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH, help="Where to clone.")
    parser.add_argument("--model", default="kronos-small", help="Variant to smoke-test.")
    parser.add_argument("--skip-test", action="store_true", help="Clone only.")
    args = parser.parse_args()

    from app.signals.kronos_client import is_available

    if not is_available():
        print("torch is not installed. Run:  uv sync --extra kronos")
        sys.exit(1)

    destination = args.path.expanduser().resolve()
    if not _clone(destination):
        sys.exit(1)

    print(f"\nAdd this to your .env:\n  KRONOS_REPO_PATH={destination}")

    if args.skip_test:
        return

    # The smoke test resolves the path through settings, so it has to be visible
    # to this process before the client is constructed.
    import os

    os.environ["KRONOS_REPO_PATH"] = str(destination)
    from app.config import get_settings

    get_settings.cache_clear()

    if not _smoke_test(destination, args.model):
        sys.exit(1)
    print("\nKronos is working.")


if __name__ == "__main__":
    main()
