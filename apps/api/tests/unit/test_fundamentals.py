"""Point-in-time fundamentals, and the three ways they quietly go wrong.

Every test here exists because the first version of the module failed it, and
each failure produced a *plausible* number rather than an error. That is the
whole hazard of this file's subject: a look-ahead leak, a split, or a dead XBRL
tag do not raise, they hand back a ratio that a reader would accept.
"""

from __future__ import annotations

from app.backtest import fundamentals as fu


def _facts(*rows: tuple[str, str, float]) -> tuple[fu.Fact, ...]:
    return tuple(fu.Fact(end=end, filed=filed, value=value) for end, filed, value in rows)


class TestNothingIsUsedBeforeItWasFiled:
    """The rule the whole module exists to enforce."""

    def test_a_figure_filed_tomorrow_is_invisible_today(self) -> None:
        facts = _facts(("2020-12-31", "2021-02-15", 10.0))
        assert fu._latest(facts, "2021-01-04") is None
        assert fu._latest(facts, "2021-02-15") is not None

    def test_the_most_recent_visible_period_wins(self) -> None:
        facts = _facts(
            ("2018-12-31", "2019-02-01", 1.0),
            ("2019-12-31", "2020-02-01", 2.0),
            ("2020-12-31", "2021-02-01", 3.0),
        )
        assert fu._latest(facts, "2020-06-30").value == 2.0
        assert fu._latest(facts, "2021-06-30").value == 3.0

    def test_skip_reaches_the_prior_year_not_the_prior_filing(self) -> None:
        """`skip` must step back a *period*, and restatements add filings.

        A year restated twice would otherwise make "last year" mean "the same
        year, an earlier draft", and every growth figure would read as zero.
        """
        facts = _facts(
            ("2019-12-31", "2020-02-01", 2.0),
            ("2020-12-31", "2021-02-01", 3.0),
            ("2020-12-31", "2021-08-01", 3.5),
        )
        assert fu._latest(facts, "2021-09-01").value == 3.5
        assert fu._latest(facts, "2021-09-01", skip=1).value == 2.0

    def test_a_restatement_is_seen_only_after_it_lands(self) -> None:
        facts = _facts(
            ("2020-12-31", "2021-02-01", 3.0),
            ("2020-12-31", "2021-08-01", 9.9),
        )
        assert fu._latest(facts, "2021-05-01").value == 3.0
        assert fu._latest(facts, "2021-09-01").value == 9.9


class TestSplitsAreReconciled:
    """EDGAR reports per share as filed; a price series is split-adjusted.

    Left alone this made Apple's January 2016 price/earnings ratio read 2.6
    against a true figure near 11 — cheap by the split factor, in the group
    carrying the most weight in the score, for every company in the years before
    a split.
    """

    def test_factor_counts_only_splits_after_the_date(self) -> None:
        events = [("2014-06-09", 7.0), ("2020-08-31", 4.0)]
        assert fu.split_factor(events, "2013-01-01") == 28.0
        assert fu.split_factor(events, "2016-01-04") == 4.0
        assert fu.split_factor(events, "2021-01-04") == 1.0

    def test_the_ratio_moves_by_exactly_the_factor(self) -> None:
        company = fu.Company(
            ticker="X",
            cik=1,
            series={"eps": _facts(("2015-09-30", "2015-10-30", 9.22))},
        )
        naive = fu.reading(company, "2016-01-04", 24.33, split_factor=1.0)
        fixed = fu.reading(company, "2016-01-04", 24.33, split_factor=4.0)
        assert float(naive["trailing_pe"]) * 4.0 == float(fixed["trailing_pe"])
        # The corrected figure is the plausible one for a large-cap in 2016.
        assert 9.0 < float(fixed["trailing_pe"]) < 12.0

    def test_growth_is_immune_because_it_uses_totals(self) -> None:
        """Growth must not move when a split happens mid-comparison."""
        company = fu.Company(
            ticker="X",
            cik=1,
            series={
                "net_income": _facts(
                    ("2019-12-31", "2020-02-01", 100.0),
                    ("2020-12-31", "2021-02-01", 120.0),
                )
            },
        )
        plain = fu.reading(company, "2021-06-01", 50.0, split_factor=1.0)
        split = fu.reading(company, "2021-06-01", 50.0, split_factor=10.0)
        assert plain["earnings_growth"] == split["earnings_growth"]
        assert abs(float(plain["earnings_growth"]) - 0.2) < 1e-9


class TestConceptsAreMerged:
    """A filer may report one metric under several XBRL tags, or switch.

    Taking the first tag that exists picked Apple's abandoned `Revenues`
    (eleven facts) over its current concept (a hundred and seventeen), and every
    revenue-growth reading after 2018 compared the same two ancient years.
    """

    def _gaap(self) -> dict:
        return {
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "start": "2015-01-01",
                            "end": "2015-12-31",
                            "filed": "2016-02-01",
                            "val": 100.0,
                        }
                    ]
                }
            },
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {
                    "USD": [
                        {
                            "start": "2019-01-01",
                            "end": "2019-12-31",
                            "filed": "2020-02-01",
                            "val": 200.0,
                        },
                        {
                            "start": "2020-01-01",
                            "end": "2020-12-31",
                            "filed": "2021-02-01",
                            "val": 260.0,
                        },
                    ]
                }
            },
        }

    def test_both_eras_survive(self) -> None:
        facts = fu._extract(self._gaap(), fu.TAGS["revenue"])
        assert [f.end for f in facts] == ["2015-12-31", "2019-12-31", "2020-12-31"]

    def test_growth_uses_adjacent_years_not_the_dead_tag(self) -> None:
        facts = fu._extract(self._gaap(), fu.TAGS["revenue"])
        assert fu._latest(facts, "2021-06-01").value == 260.0
        assert fu._latest(facts, "2021-06-01", skip=1).value == 200.0

    def test_the_preferred_concept_wins_a_tie(self) -> None:
        gaap = {
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "start": "2020-01-01",
                            "end": "2020-12-31",
                            "filed": "2021-02-01",
                            "val": 1.0,
                        }
                    ]
                }
            },
            "SalesRevenueNet": {
                "units": {
                    "USD": [
                        {
                            "start": "2020-01-01",
                            "end": "2020-12-31",
                            "filed": "2021-02-01",
                            "val": 2.0,
                        }
                    ]
                }
            },
        }
        facts = fu._extract(gaap, fu.TAGS["revenue"])
        assert len(facts) == 1
        assert facts[0].value == 1.0


class TestOnlyAnnualPeriodsAreKept:
    def test_a_quarter_is_rejected(self) -> None:
        gaap = {
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {
                            "start": "2020-01-01",
                            "end": "2020-03-31",
                            "filed": "2020-05-01",
                            "val": 5.0,
                        },
                        {
                            "start": "2020-01-01",
                            "end": "2020-12-31",
                            "filed": "2021-02-01",
                            "val": 20.0,
                        },
                    ]
                }
            }
        }
        facts = fu._extract(gaap, fu.TAGS["net_income"])
        assert [f.value for f in facts] == [20.0]

    def test_an_instant_fact_is_kept(self) -> None:
        """Equity has no duration; rejecting undated spans would drop it."""
        gaap = {
            "StockholdersEquity": {
                "units": {"USD": [{"end": "2020-12-31", "filed": "2021-02-01", "val": 500.0}]}
            }
        }
        assert len(fu._extract(gaap, fu.TAGS["equity"])) == 1


class TestUnusableInputsBecomeAbsentRatherThanWrong:
    def test_a_loss_maker_has_no_price_to_earnings(self) -> None:
        """A negative ratio would rank as the cheapest name in the universe."""
        company = fu.Company(
            ticker="X", cik=1, series={"eps": _facts(("2020-12-31", "2021-02-01", -2.0))}
        )
        assert fu.reading(company, "2021-06-01", 30.0)["trailing_pe"] is None

    def test_missing_metrics_are_none_not_zero(self) -> None:
        company = fu.Company(ticker="X", cik=1, series={})
        reading = fu.reading(company, "2021-06-01", 30.0)
        assert set(reading) == {
            "trailing_pe",
            "price_to_book",
            "earnings_growth",
            "revenue_growth",
            "profit_margin",
            "debt_to_equity",
            "dividend_yield",
        }
        assert all(value is None for value in reading.values())
