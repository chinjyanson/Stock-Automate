# syntax=docker/dockerfile:1

# uv's own image supplies the binary; we copy it into a slim Python base rather
# than building on it, to keep the runtime layer small.
FROM ghcr.io/astral-sh/uv:0.11-python3.13-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app/apps/api

# Dependency layer first, from the lockfile only. Application source changes far
# more often than dependencies do, so this layer stays cached across most builds.
COPY apps/api/pyproject.toml apps/api/uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev 2>/dev/null \
    || uv sync --no-install-project --no-dev

COPY apps/api/ ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

ENV PATH="/opt/venv/bin:$PATH"

# Non-root. A process that can place trades should not also be able to rewrite
# its own image.
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser appuser \
    && chown -R appuser:appuser /app /opt/venv
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Migrations run on start so a fresh `docker compose up` yields a working
# database with no manual step (acceptance criterion 21). Safe to repeat:
# `alembic upgrade head` is a no-op when already current.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
