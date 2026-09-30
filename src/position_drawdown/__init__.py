"""Position-level drawdown alert engine.

For each user-tracked position, compares today's close against the
historical peak (computed from the position's open date), and triggers
a Feishu alert when the drawdown from peak crosses configured thresholds.

Design constraints:
  - peak is recomputed every run (no persistent cache); GitHub Actions
    runs at 18:00 Beijing, well after market close, so today's close
    is stable.
  - Source: portfolio.json (committed to a private GitHub repo). Local
    Web UI (DSA Portfolio module) is the future editor.
  - Trigger gates: peak > cost (the position must have been profitable
    at some point) AND drawdown from peak crosses a tiered threshold.
  - Delivery: reuses DSA's NotificationService so every channel the
    user has already configured (Feishu / WeCom / Telegram / Discord /
    Slack / Email / DingTalk / PushPlus / ...) just works.
"""
from .engine import (
    Alert,
    AlertSeverity,
    PositionDrawdown,
    evaluate_portfolio,
)
from .notifier import build_markdown, send

__all__ = [
    "Alert",
    "AlertSeverity",
    "PositionDrawdown",
    "evaluate_portfolio",
    "build_markdown",
    "send",
]
