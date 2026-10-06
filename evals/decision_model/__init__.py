"""System-1 decision-model advice for the adaptive helper policy.

Modules:

- ``config``: settings and CLI flags.
- ``schema``: versioned question profiles and answer parsing (SystemOne API).
- ``state``: bounded, redacted decision state rendered from the attempt context.
- ``policy``: pure role logic (shadow, policy, features, prior, atomic mapping).
- ``client``: HTTP client for ``POST /v1/systemone``.
- ``runtime``: glue between the attempt flow, the client and the policy controller.
- ``probe``: offline probe (``python -m evals.decision_model.probe``) scoring answers against
  case labels without repair budget.

This package intentionally imports nothing at package level so the policy controller can import
``policy`` without pulling in the attempt-flow modules.
"""
