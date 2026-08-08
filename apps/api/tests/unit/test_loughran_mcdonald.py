"""The Loughran-McDonald scorer. Pure, offline, no fixtures.

The tests worth having here are the ones that pin *why this lexicon* rather than
any other. A general-purpose sentiment lexicon reads "liability", "cost", "tax"
and "depreciation" as negative, and a bag-of-words counter reads "not profitable"
as positive. Both failures would put a confident, wrong number into the risk
category — so both are pinned from the direction that would catch a regression
back to a naive implementation.
"""

from __future__ import annotations

import pytest

from app.sentiment import lexicon, score_headlines, score_text, tokenise


class TestTokenising:
    def test_it_upper_cases_and_drops_punctuation(self) -> None:
        assert tokenise("Profits rose, sharply!") == ["PROFITS", "ROSE", "SHARPLY"]

    def test_numbers_are_not_words(self) -> None:
        """Digits carry no tone and would otherwise dilute the denominator."""
        assert tokenise("Revenue up 12.5%") == ["REVENUE", "UP"]

    def test_empty_text_yields_nothing(self) -> None:
        assert tokenise("   ") == []


class TestPolarity:
    def test_negative_words_push_polarity_down(self) -> None:
        reading = score_text("The company reported a severe loss and a fraud investigation.")
        assert reading is not None
        assert reading.polarity < 0
        assert reading.negative_words > 0

    def test_positive_words_push_polarity_up(self) -> None:
        reading = score_text("Strong gains and an excellent recovery in profitability.")
        assert reading is not None
        assert reading.polarity > 0
        assert reading.positive_words > 0

    def test_polarity_is_bounded(self) -> None:
        worst = score_text("loss losses fraud litigation bankruptcy adverse")
        best = score_text("gain gains profitable excellent strong opportunities")
        assert worst is not None and best is not None
        assert worst.polarity == pytest.approx(-1.0)
        assert best.polarity == pytest.approx(1.0)

    def test_toneless_text_reports_nothing(self) -> None:
        """None, not zero.

        A headline with no tone is a non-observation. Scoring it as neutral
        would let a batch's reading be driven by how much toneless text it
        happened to contain, i.e. by article length.
        """
        assert score_text("The board will meet on Tuesday in London.") is None


class TestFinancialVocabulary:
    """The whole reason for this lexicon rather than a general-purpose one."""

    @pytest.mark.parametrize(
        "word", ["LIABILITY", "LIABILITIES", "COST", "COSTS", "TAX", "CAPITAL", "DEPRECIATION"]
    )
    def test_ordinary_accounting_terms_are_not_negative(self, word: str) -> None:
        """Harvard-IV marks these negative; in financial text they are furniture.

        Loughran and McDonald's own finding was that roughly three-quarters of
        Harvard-IV's negative hits in 10-Ks are words like these. A lexicon that
        flagged them would score every balance-sheet story as bad news.
        """
        assert word not in lexicon.NEGATIVE

    def test_a_routine_accounting_headline_is_not_bearish(self) -> None:
        reading = score_text("Total liabilities and deferred tax costs rose with depreciation.")
        assert reading is None or reading.polarity >= 0


class TestNegation:
    def test_a_negated_positive_reads_negative(self) -> None:
        """'Not profitable' is the case a bag-of-words counter gets backwards."""
        plain = score_text("The division was profitable.")
        negated = score_text("The division was not profitable.")
        assert plain is not None and negated is not None
        assert plain.polarity > 0
        assert negated.polarity < 0

    def test_a_negated_negative_reads_positive(self) -> None:
        reading = score_text("There were no losses this quarter.")
        assert reading is not None
        assert reading.polarity > 0

    def test_negation_does_not_reach_past_its_window(self) -> None:
        """A negator early in a sentence must not poison the whole line.

        With a three-token window, the tone word here sits outside it and keeps
        its own sign — otherwise one 'not' would invert an entire headline.
        """
        reading = score_text("No decision was taken although the quarter was profitable")
        assert reading is not None
        assert reading.positive_words >= 1


class TestUncertainty:
    def test_hedged_language_registers_as_uncertainty(self) -> None:
        reading = score_text("Results may possibly approximate the earlier estimate.")
        assert reading is not None
        assert reading.uncertain_words > 0
        assert reading.uncertainty > 0

    def test_uncertainty_is_not_folded_into_polarity(self) -> None:
        """A hedged statement is a different thing from a negative one.

        Collapsing the two would make "we may do well" indistinguishable from
        "we did badly", which are not the same claim about a company.

        (Note "growth" is deliberately absent from LM's positive list — it is
        a neutral business term, not a tone word. Hence "profitable" here.)
        """
        reading = score_text("The division may possibly remain profitable.")
        assert reading is not None
        assert reading.uncertainty > 0
        assert reading.polarity > 0


class TestBatches:
    def test_headlines_are_pooled_not_averaged(self) -> None:
        """Counts are summed and the polarity computed once over the total.

        Averaging per-headline polarities would give a three-word headline with
        one tone word the same weight as a paragraph with twenty — which is
        backwards, since the longer text is the better evidence.
        """
        long_bad = "loss losses fraud litigation adverse bankruptcy declines"
        short_good = "gains"
        pooled = score_headlines([long_bad, short_good])
        assert pooled is not None
        # 1 positive against 7 negative: pooling lands near -1, an average of the
        # two headlines' polarities (-1 and +1) would land at 0.
        assert pooled.polarity < -0.5

    def test_toneless_headlines_do_not_drag_the_reading_toward_neutral(self) -> None:
        strong = ["Severe losses and a fraud investigation"]
        padded = [*strong, "The board will meet on Tuesday", "A new office opened in Leeds"]
        assert score_headlines(strong) is not None
        assert score_headlines(padded) is not None
        assert score_headlines(padded).polarity == pytest.approx(  # type: ignore[union-attr]
            score_headlines(strong).polarity  # type: ignore[union-attr]
        )

    def test_no_headlines_reports_nothing(self) -> None:
        assert score_headlines([]) is None

    def test_only_toneless_headlines_reports_nothing(self) -> None:
        assert score_headlines(["The board will meet on Tuesday in London."]) is None

    def test_document_count_reflects_what_actually_scored(self) -> None:
        reading = score_headlines(["Record profits and strong gains", "Meeting on Tuesday"])
        assert reading is not None
        assert reading.documents == 1


class TestLexiconIntegrity:
    def test_the_word_lists_are_the_published_ones(self) -> None:
        """Guards the generated module against a truncated regeneration.

        These are the counts Loughran and McDonald publish for the master
        dictionary. A silent partial extraction would leave the scorer working
        but blind to most of its vocabulary.
        """
        assert len(lexicon.NEGATIVE) == 2355
        assert len(lexicon.POSITIVE) == 354
        assert len(lexicon.UNCERTAINTY) == 297

    def test_no_word_is_both_positive_and_negative(self) -> None:
        assert not (lexicon.POSITIVE & lexicon.NEGATIVE)

    def test_words_are_upper_case_matching_the_tokeniser(self) -> None:
        for words in (lexicon.NEGATIVE, lexicon.POSITIVE, lexicon.UNCERTAINTY):
            sample = next(iter(words))
            assert sample.isupper() and sample.isalpha()
