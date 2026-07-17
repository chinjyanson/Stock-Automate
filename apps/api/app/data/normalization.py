"""Price-unit normalisation (§4).

LSE instruments are quoted in pence (GBX) but denominated in pounds (GBP).
yfinance returns GBX for `.L` symbols without saying so. Treating that 8750 as
GBP overstates the price by 100x, which does not fail loudly — it produces an
order for 1/100th of the intended size, or a stop distance that never triggers.

Normalisation happens exactly once, at the adapter boundary, and everything
downstream is in the instrument's denominated currency. Doing it later, or
twice, is the failure mode this module exists to prevent.
"""

from __future__ import annotations

from decimal import Decimal

from app.models.enums import PriceUnit

#: Minor-unit quotes and the major unit they divide into.
_MINOR_UNITS: dict[PriceUnit, tuple[PriceUnit, Decimal]] = {
    PriceUnit.GBX: (PriceUnit.GBP, Decimal(100)),
}


def is_minor_unit(unit: PriceUnit) -> bool:
    return unit in _MINOR_UNITS


def major_unit_for(unit: PriceUnit) -> PriceUnit:
    """The unit prices become after normalisation. Identity for major units."""
    mapping = _MINOR_UNITS.get(unit)
    return mapping[0] if mapping else unit


def normalise_price(price: Decimal, from_unit: PriceUnit) -> Decimal:
    """Convert a quoted price into its major unit.

    Exact: Decimal division by 100 is exact, so no rounding is applied here.
    Rounding is a presentation and order-submission concern, and doing it this
    early would compound across indicator maths.
    """
    mapping = _MINOR_UNITS.get(from_unit)
    if mapping is None:
        return price
    _, divisor = mapping
    return price / divisor


def normalise_optional(price: Decimal | None, from_unit: PriceUnit) -> Decimal | None:
    return None if price is None else normalise_price(price, from_unit)


def infer_price_unit(symbol: str, currency: str | None) -> PriceUnit:
    """Best-effort unit inference for a provider that will not tell us.

    yfinance reports `currency == "GBp"` (lowercase p) for pence-quoted LSE
    listings, which is the one reliable signal it gives. The `.L` suffix alone
    is not sufficient — some LSE lines genuinely quote in GBP or USD — so
    currency is trusted first and the suffix is only a fallback.

    Inference is a last resort. A confirmed `MarketDataMapping.price_unit`
    always wins over this.
    """
    if currency:
        # "GBp" is yfinance's pence marker; "GBX" is the ISO-ish spelling.
        if currency in {"GBp", "GBX", "GBp "}:
            return PriceUnit.GBX
        try:
            return PriceUnit(currency.upper())
        except ValueError:
            pass

    if symbol.upper().endswith(".L"):
        return PriceUnit.GBX

    return PriceUnit.USD


def denominated_currency(unit: PriceUnit) -> str:
    """The ISO 4217 code an instrument quoted in `unit` settles in."""
    return str(major_unit_for(unit))
