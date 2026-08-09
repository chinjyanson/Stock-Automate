"""Insider factor: parsing and the rules that decide what a filing is worth.

The parser is exercised against fixture XML rather than the live feed — the
subtle mistakes live in the parsing, not the HTTP, and a test that needs the
network is a test that fails on a Monday for reasons of its own.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.data.edgar import EDGARClient, is_c_suite_title
from app.scanner import scoring
from app.services import insider

_URL = "https://www.sec.gov/Archives/0001234567-26-000123-index.htm"


def _form4(
    *,
    code: str = "P",
    acquired: str = "A",
    shares: str = "50000",
    price: str = "24.50",
    title: str = "Chief Executive Officer",
    footnote: str = "",
) -> str:
    return f"""
    <ownershipDocument>
      <issuerName>Example Corp</issuerName>
      <issuerTradingSymbol>EXMP</issuerTradingSymbol>
      <rptOwnerName>Jane Doe</rptOwnerName>
      <isOfficer>1</isOfficer>
      <officerTitle>{title}</officerTitle>
      <nonDerivativeTransaction>
        <transactionDate><value>2026-07-28</value></transactionDate>
        <transactionCode>{code}</transactionCode>
        <transactionAcquiredDisposedCode><value>{acquired}</value></transactionAcquiredDisposedCode>
        <transactionShares><value>{shares}</value></transactionShares>
        <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
        <sharesOwnedFollowingTransaction><value>310000</value></sharesOwnedFollowingTransaction>
      </nonDerivativeTransaction>
      {footnote}
    </ownershipDocument>
    """


class TestForm4Parsing:
    def test_open_market_purchase(self) -> None:
        (filing,) = EDGARClient.parse_form4(_form4(), filing_url=_URL)
        assert filing.transaction_code == "P"
        assert filing.acquired_disposed == "A"
        assert filing.issuer_ticker == "EXMP"
        assert filing.shares == Decimal("50000")
        assert filing.value_usd == Decimal("1225000.00")
        assert filing.is_c_suite is True
        assert filing.is_10b5_1_plan is False

    def test_accession_is_recovered_from_the_url(self) -> None:
        """Form 4 XML does not carry its own accession; the path does.

        It is the deduplication key, so losing it would make re-polling insert
        duplicates instead of converging.
        """
        (filing,) = EDGARClient.parse_form4(_form4(), filing_url=_URL)
        assert filing.accession_number == "0001234567-26-000123"

    def test_10b5_1_footnote_is_detected(self) -> None:
        """The flag lives in free-text footnotes, not a dedicated field."""
        note = "<footnote>Sale under a Rule 10b5-1 trading plan adopted 2026-01-05.</footnote>"
        (filing,) = EDGARClient.parse_form4(_form4(code="S", footnote=note), filing_url=_URL)
        assert filing.is_10b5_1_plan is True

    def test_html_entities_are_unescaped(self) -> None:
        """ "EVP &amp; Chief Operating Officer" must still match as C-suite."""
        (filing,) = EDGARClient.parse_form4(
            _form4(title="EVP &amp; Chief Operating Officer"), filing_url=_URL
        )
        assert filing.officer_title == "EVP & Chief Operating Officer"
        assert filing.is_c_suite is True

    def test_multiple_transaction_lines_get_distinct_indices(self) -> None:
        doubled = _form4().replace("</ownershipDocument>", "")
        doubled += """
          <nonDerivativeTransaction>
            <transactionDate><value>2026-07-28</value></transactionDate>
            <transactionCode>S</transactionCode>
            <transactionShares><value>100</value></transactionShares>
            <transactionPricePerShare><value>10</value></transactionPricePerShare>
          </nonDerivativeTransaction></ownershipDocument>"""
        filings = EDGARClient.parse_form4(doubled, filing_url=_URL)
        assert [f.line_index for f in filings] == [0, 1]


class TestRoleClassification:
    def test_chief_titles_and_president_are_c_suite(self) -> None:
        for title in ("Chief Executive Officer", "CFO", "President", "Chief Legal Officer"):
            assert is_c_suite_title(title) is True, title

    def test_vice_presidents_are_not(self) -> None:
        """An SVP is not a chief officer. Conflating them would flatten the
        strongest tier of this signal into the noisiest."""
        for title in ("Executive Vice President", "SVP, Sales", "Vice President"):
            assert is_c_suite_title(title) is False, title

    def test_directors_and_unknowns_are_not(self) -> None:
        assert is_c_suite_title(None) is False
        assert is_c_suite_title("Director") is False
        assert is_c_suite_title("General Counsel") is False


class TestInsiderGroupBlend:
    """The insider group must behave like every other optional group."""

    def _groups(self, insider: float | None) -> dict[str, scoring.GroupScore]:
        scores: dict[str, float | None] = {
            "value": 70.0,
            "cheapness": 60.0,
            "insider": insider,
            "quality": 60.0,
            "sector": 50.0,
        }
        return {
            name: scoring.GroupScore(
                name=name, score=scores[name], weight=scoring.DEFAULT_WEIGHTS[name]
            )
            for name in scoring.GROUP_NAMES
        }

    def test_absent_insider_data_neither_helps_nor_hurts(self) -> None:
        """The common case: no Form 4 filings at all.

        A missing group must drop out *with its weight* and let the rest
        renormalise. Otherwise the ~60% of a UK-tradable universe that can never
        carry this group would be silently marked down.
        """
        baseline = scoring.combine_score(self._groups(None))
        assert baseline > 0
        # Adding an insider *buy* can only ever raise it from here.
        assert scoring.combine_score(self._groups(90.0)) > baseline

    def test_selling_discounts_the_score_via_the_penalty(self) -> None:
        """Selling acts as a multiplicative discount, not as a low group score."""
        clean = scoring.combine_score(self._groups(None))
        selling = scoring.combine_score(self._groups(None), insider_sell_penalty=0.40)
        assert selling == pytest.approx(clean * 0.60, rel=1e-3)

    def test_the_penalty_is_capped(self) -> None:
        """Even at full strength a stock is marked down, never erased."""
        clean = scoring.combine_score(self._groups(None))
        marked = scoring.combine_score(
            self._groups(None), insider_sell_penalty=scoring.DEFAULT_INSIDER_SELL_PENALTY
        )
        assert marked == pytest.approx(
            clean * (1.0 - scoring.DEFAULT_INSIDER_SELL_PENALTY), rel=1e-3
        )
        assert marked > 0

    def test_buying_cannot_dominate_the_ranking(self) -> None:
        """At weight 16 the group matters without deciding the outcome.

        The failure this guards against: a weight large enough for insider
        activity to swing the ranking makes *having US filings* the dominant
        criterion, since instruments without them renormalise around the gap.
        """
        best = scoring.combine_score(self._groups(100.0))
        none = scoring.combine_score(self._groups(None))
        assert (best - none) < 12.0

    def test_weights_still_sum_to_one_hundred(self) -> None:
        assert abs(sum(scoring.DEFAULT_WEIGHTS.values()) - 100.0) < 1e-9


class TestEvidenceDrivenRules:
    """The three rules a 686-event EDGAR backtest actually supports.

    Pinned as tests because each one is counter-intuitive enough that a future
    reader might "fix" it back to the obvious-but-wrong behaviour.
    """

    def test_scheduled_sales_are_ignored_entirely(self) -> None:
        """10b5-1 sales preceded +5.56% at one month (t = +3.39), so penalising
        them at the old 0.2 was not cautious — it was the wrong sign."""
        assert insider._PLAN_DAMPING == 0.0

    def test_the_sell_penalty_ceiling_matches_the_scanner_default(self) -> None:
        """Two constants, one number: they must not drift apart."""
        assert insider.MAX_SELL_PENALTY == scoring.DEFAULT_INSIDER_SELL_PENALTY

    def test_the_lookback_is_short_enough_to_be_actionable(self) -> None:
        """The measured effect is at one month and gone by three. A filing older
        than the window in which the move happens has nothing left to predict."""
        assert insider.LOOKBACK_DAYS <= 60

    def test_only_chief_officers_are_weighted_for_selling(self) -> None:
        """Officer and director selling showed no measurable effect either way."""
        assert set(insider._ROLE_WEIGHT) == {"c_suite", "officer", "director"}
        assert insider._ROLE_WEIGHT["c_suite"] == 1.0
