"""Declarative base and shared column conventions.

Money and quantity columns use `Numeric`, never `Float`. Binary floating point
cannot represent decimal cash amounts exactly, and rounding drift in a system
that sizes positions and computes stop distances is a correctness bug, not a
cosmetic one. Values surface in Python as `decimal.Decimal`.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON, Numeric, String, TypeDecorator

# Explicit naming so Alembic autogenerate produces stable, reversible names
# instead of backend-chosen ones.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class StrEnumType(TypeDecorator[Any]):
    """Stores a StrEnum as VARCHAR, and returns the *member* on load.

    Without this, a `Mapped[PriceUnit]` column backed by `String(3)` round-trips
    to a plain `str`. Because these are StrEnums, `value == PriceUnit.GBX` still
    passes, so the defect hides — until code uses `is`, which silently never
    matches. That is a genuinely dangerous failure here: a lifecycle check that
    never fires would let an unvalidated instrument slip forward.

    A TypeDecorator over String is chosen over `sa.Enum(..., native_enum=False)`
    because it emits identical DDL (plain VARCHAR, no CHECK constraint, no
    PostgreSQL ENUM type), so adding an enum member stays a code-only change
    (see `app.models.enums`).

    Binding also *validates*: an unknown value raises here rather than being
    written and discovered later on read.
    """

    impl = String
    cache_ok = True

    def __init__(self, enum_class: type[Any], length: int = 32, **kwargs: Any) -> None:
        self._enum_class = enum_class
        # Kept explicitly: `self.impl` is the String *class*, whose `length` is
        # instance-only, so copy() cannot read it back off the impl.
        self._length = length
        super().__init__(length=length, **kwargs)

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        # Accept either a member or its value; normalise to the canonical value.
        return str(self._enum_class(value).value)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return self._enum_class(value)

    def copy(self, **kwargs: Any) -> StrEnumType:
        return StrEnumType(self._enum_class, self._length)


class JSONBOrJSON(TypeDecorator[Any]):
    """JSONB on PostgreSQL, generic JSON elsewhere.

    Lets the model layer stay portable so unit tests can run against SQLite
    without a container, while production still gets JSONB indexing.
    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


#: Prices: 18 significant digits, 8 dp. Accommodates GBX pence quotes and
#: fractional-share unit prices without truncation.
Price = Numeric(18, 8)

#: Quantities: Trading 212 supports fractional shares.
Quantity = Numeric(18, 8)

#: Money: settled cash amounts.
Money = Numeric(18, 4)

#: Ratios, percentages, scores, z-scores.
Ratio = Numeric(12, 6)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    # ClassVar so ruff does not read this as a mutable default; SQLAlchemy reads
    # it off the class, and it is never mutated at runtime.
    type_annotation_map: ClassVar[dict[Any, Any]] = {
        dict[str, Any]: JSONBOrJSON,
    }


class UUIDPrimaryKeyMixin:
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
