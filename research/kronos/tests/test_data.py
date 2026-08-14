"""The label, and the boundary between what the model sees and what it is asked.

Everything in this directory is a claim about accuracy, and every such claim
rests on two things being right: that we are predicting the same event
production predicts, and that no window contains its own answer. Those are the
two things tested here. A subtle error in either does not crash anything — it
produces a slightly better number, which is the worst possible failure mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import data


def production_label(close: np.ndarray, i: int, *, fall: float, horizon: int) -> float:
    """A transcription of `app.signals.crash_features.label_fall`.

    Copied rather than imported: the API package lives in a different virtualenv
    and pulls in a database layer. Copying is a risk, so the copy is kept
    literal — if the original changes, this should be updated to match and the
    test below will keep the vectorised version honest in the meantime.
    """
    ahead = close[i : i + 1 + horizon]
    return 1.0 if float(np.min(ahead) / close[i] - 1.0) <= -fall else 0.0


class TestTheLabelIsProductionsLabel:
    def test_matches_bar_by_bar_on_noise(self) -> None:
        rng = np.random.default_rng(4)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 600)))
        ours = data.label_fall(close, fall=0.02, horizon=1)
        for i in range(close.size - 1):
            assert ours[i] == production_label(close, i, fall=0.02, horizon=1)

    def test_matches_at_longer_horizons(self) -> None:
        rng = np.random.default_rng(5)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.015, 300)))
        ours = data.label_fall(close, fall=0.03, horizon=5)
        for i in range(close.size - 5):
            assert ours[i] == production_label(close, i, fall=0.03, horizon=5)

    def test_the_boundary_is_at_or_below(self) -> None:
        exactly_two = np.array([100.0, 98.0, 100.0])
        assert data.label_fall(exactly_two, fall=0.02, horizon=1)[0] == 1.0
        just_under = np.array([100.0, 98.01, 100.0])
        assert data.label_fall(just_under, fall=0.02, horizon=1)[0] == 0.0

    def test_unknown_rather_than_no(self) -> None:
        """The last bar has no answer yet, and must not be recorded as a zero.

        Production's scalar version reads a one-element slice there and returns
        0.0, which would silently add a non-event to every test set.
        """
        close = np.array([100.0, 99.0, 98.0])
        out = data.label_fall(close, fall=0.02, horizon=1)
        assert np.isnan(out[-1])
        assert not np.isnan(out[:-1]).any()


class TestNoWindowContainsItsOwnAnswer:
    @pytest.fixture
    def market(self) -> data.Market:
        n = data.LOOKBACK + 60
        rng = np.random.default_rng(9)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
        frame = pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.004,
                "Low": close * 0.996,
                "Close": close,
                "Volume": np.full(n, 1e9),
            },
            index=pd.bdate_range("2010-01-04", periods=n),
        )
        return data._assemble(frame)

    def test_a_window_ends_on_its_own_decision_bar(self, market: data.Market) -> None:
        at = market.usable()
        windows = market.windows(at)
        for k in (0, 5, at.size // 2, at.size - 1):
            assert windows[k, -1, :] == pytest.approx(market.bars[at[k], :])

    def test_a_window_never_reaches_the_bar_being_predicted(self, market: data.Market) -> None:
        at = market.usable()
        windows = market.windows(at)
        for k in (0, at.size // 3, at.size - 1):
            future = market.bars[at[k] + 1, 3]
            # The predicted close may coincide with an earlier one by chance on
            # real data, so this checks position rather than value: the window
            # spans exactly the bars up to the decision, and no further.
            assert windows[k].shape[0] == data.LOOKBACK
            assert windows[k, -1, 3] == pytest.approx(market.close[at[k]])
            assert market.close[at[k] + 1] == pytest.approx(future)

    def test_every_usable_bar_has_a_known_answer(self, market: data.Market) -> None:
        at = market.usable()
        assert at.size > 0
        assert np.isfinite(market.label[at]).all()
        assert at.max() < market.index.size - 1

    def test_dates_filter_without_shifting_alignment(self, market: data.Market) -> None:
        everything = market.usable()
        later = market.usable(since=str(market.index[everything[10]].date()))
        assert later[0] == everything[10]
        assert set(later).issubset(set(everything))


class TestStampsLineUpWithWindows:
    def test_one_timestamp_per_bar_in_window_order(self) -> None:
        n = data.LOOKBACK + 20
        rng = np.random.default_rng(2)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        frame = pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": np.ones(n)},
            index=pd.bdate_range("2015-01-05", periods=n),
        )
        market = data._assemble(frame)
        at = market.usable()[:3]
        stamps = market.stamps(at)
        assert stamps.size == at.size * data.LOOKBACK
        assert stamps.iloc[data.LOOKBACK - 1] == market.index[at[0]]
        assert stamps.iloc[2 * data.LOOKBACK - 1] == market.index[at[1]]
