"""Alembic environment.

Runs migrations against a *synchronous* driver even though the application is
async. Alembic's migration context is synchronous, and driving it through an
async engine adds a greenlet layer for no benefit — migrations are a one-shot
administrative operation, not a hot path.
"""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from app.config import get_settings

# Importing the package attaches every table to Base.metadata. Without this,
# autogenerate produces an empty migration and silently drops the schema.
from app.models import Base
from app.models.base import JSONBOrJSON, StrEnumType

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
# psycopg2 rather than asyncpg; see the module docstring.
config.set_main_option("sqlalchemy.url", settings.sync_database_url)

target_metadata = Base.metadata


def include_object(
    obj: object, name: str | None, type_: str, reflected: bool, compare_to: object
) -> bool:
    """Filter objects out of autogenerate.

    Alembic cannot see the audit immutability trigger (it is raw DDL), and
    would otherwise propose nothing about it — which is correct. This hook
    exists for future cases where a database object is managed outside the ORM.
    """
    return True


def render_item(type_: str, obj: object, autogen_context: object) -> str | bool:
    """Render custom types with their import.

    Autogenerate emits `app.models.base.JSONBOrJSON()` for our TypeDecorator but
    does not know it must import the module, producing a migration that fails at
    runtime with `NameError: name 'app' is not defined`. Registering the import
    here fixes every future migration rather than requiring a manual edit each
    time.
    """
    if type_ == "type" and isinstance(obj, JSONBOrJSON):
        autogen_context.imports.add("import app.models.base")  # type: ignore[attr-defined]
        return "app.models.base.JSONBOrJSON()"
    if type_ == "type" and isinstance(obj, StrEnumType):
        # StrEnumType is a validating wrapper over VARCHAR; its DDL is exactly a
        # String of the same length. Render it as such so migrations carry no
        # dependency on the enum class and read as the plain columns they are.
        return f"sa.String(length={obj.impl.length})"
    # False means "fall back to alembic's default rendering".
    return False


def run_migrations_offline() -> None:
    """Emit SQL without a connection, for review or manual application."""
    context.configure(
        url=settings.sync_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Detect column type and server-default drift, not just added and
            # dropped tables. A Numeric silently becoming a Float is exactly the
            # change that must never pass unnoticed here.
            compare_type=True,
            compare_server_default=True,
            include_object=include_object,
            render_item=render_item,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
