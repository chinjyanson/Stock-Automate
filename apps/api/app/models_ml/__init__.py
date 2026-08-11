"""Fitted models — the arithmetic, not the SQLAlchemy tables.

Named `models_ml` rather than `models` because `app.models` is already the
persistence layer. What lives here is the fitting and evaluation of statistical
models; where a fitted model is *stored* is `app.models.strategy_model`.
"""
