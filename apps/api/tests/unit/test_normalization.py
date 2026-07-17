"""Price-unit normalisation (§20: currency normalisation, GBX-to-GBP).

These tests encode a fact confirmed against the live yfinance API: the London
Stock Exchange hosts both pence-quoted and pound-quoted instruments, so the
`.L` suffix does not determine the unit and the reported currency must win.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.data.normalization import (
    denominated_currency,
    infer_price_unit,
    is_minor_unit,
    major_unit_for,
    normalise_optional,
    normalise_price,
)
from app.models.enums import PriceUnit


class TestNormalisePrice:
    def test_gbx_divides_by_one_hundred(self) -> None:
        assert normalise_price(Decimal("5758.84"), PriceUnit.GBX) == Decimal("57.5884")

    def test_gbx_conversion_is_exact_not_floating(self) -> None:
        # The whole reason for Decimal: 0.1 + 0.2 != 0.3 in binary float, and a
        # price that drifts changes position sizing.
        assert normalise_price(Decimal("110.755"), PriceUnit.GBX) == Decimal("1.10755")

    def test_gbp_is_untouched(self) -> None:
        # VUAG.L genuinely quotes in GBP; dividing it would be the bug.
        assert normalise_price(Decimal("108.36"), PriceUnit.GBP) == Decimal("108.36")

    @pytest.mark.parametrize("unit", [PriceUnit.USD, PriceUnit.EUR, PriceUnit.CHF])
    def test_major_units_are_untouched(self, unit: PriceUnit) -> None:
        assert normalise_price(Decimal("750.72"), unit) == Decimal("750.72")

    def test_normalising_twice_is_not_idempotent(self) -> None:
        """Guards against a second normalisation being applied downstream.

        Encoded deliberately: if someone later makes this idempotent to "be
        safe", they have hidden a double-conversion bug rather than fixed it.
        Normalisation happens exactly once, at the adapter boundary.
        """
        once = normalise_price(Decimal("5758.84"), PriceUnit.GBX)
        twice = normalise_price(once, PriceUnit.GBX)
        assert twice != once
        assert twice == Decimal("0.575884")

    def test_optional_passes_none_through(self) -> None:
        assert normalise_optional(None, PriceUnit.GBX) is None
        assert normalise_optional(Decimal("100"), PriceUnit.GBX) == Decimal("1")


class TestUnitMetadata:
    def test_gbx_is_minor_gbp_is_not(self) -> None:
        assert is_minor_unit(PriceUnit.GBX)
        assert not is_minor_unit(PriceUnit.GBP)

    def test_gbx_normalises_to_gbp(self) -> None:
        assert major_unit_for(PriceUnit.GBX) is PriceUnit.GBP

    def test_major_unit_is_its_own_major(self) -> None:
        assert major_unit_for(PriceUnit.USD) is PriceUnit.USD

    def test_gbx_settles_in_gbp(self) -> None:
        assert denominated_currency(PriceUnit.GBX) == "GBP"


class TestInferPriceUnit:
    def test_yfinance_lowercase_gbp_means_pence(self) -> None:
        # yfinance signals pence as "GBp" — the only reliable marker it gives.
        assert infer_price_unit("LLOY.L", "GBp") is PriceUnit.GBX

    def test_gbx_spelling_means_pence(self) -> None:
        assert infer_price_unit("SGLN.L", "GBX") is PriceUnit.GBX

    def test_lse_instrument_quoted_in_pounds_is_not_pence(self) -> None:
        """The case the `.L` suffix gets wrong.

        Confirmed live: VUAG.L reports currency=GBP at ~108, while SGLN.L on the
        same exchange reports GBp at ~5758. Currency must beat suffix.
        """
        assert infer_price_unit("VUAG.L", "GBP") is PriceUnit.GBP

    def test_suffix_is_only_a_fallback_when_currency_is_absent(self) -> None:
        assert infer_price_unit("LLOY.L", None) is PriceUnit.GBX

    def test_us_symbol_resolves_to_usd(self) -> None:
        assert infer_price_unit("SPY", "USD") is PriceUnit.USD

    def test_unknown_currency_falls_back_rather_than_raising(self) -> None:
        # An unrecognised currency must not take down a scan mid-rotation.
        assert infer_price_unit("FOO", "ZZZ") is PriceUnit.USD
