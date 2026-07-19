"""The risk engine (§9, §10).

A strategy proposes; the risk engine disposes. Nothing in the system may submit
an order without passing through `RiskEngine`, which can reduce it, reject it, or
find trading halted entirely. See `docs/risk-model.md` for the model this
implements.
"""
