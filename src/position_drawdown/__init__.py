"""Position-level alert engine (drawdown + stop-loss).

For each user-tracked position, compares today's close against the
historical peak computed from daily closes since the position's open date.

Two independent alert kinds:
  - DRAWDOWN  — the position ran up at least ``min_gain`` (5%) and has since
                given back a large share of that peak (20/50/80% tiers).
  - STOP_LOSS — the position fell straight through the cost line (-5%),
                regardless of any run-up (5/10/20% loss tiers).

Design constraints:
  - peak is recomputed every run (no persistent cache); GitHub Actions
    runs after market close, so today's close is stable.
  - peak and current both use the daily CLOSE, so the drawdown never mixes
    intraday highs with closing prices.
  - Source: portfolio.json (private Gist or repo file).
  - Delivery: reuses DSA's NotificationService so every channel the
    user has already configured (Feishu / WeCom / Telegram / Discord /
    Slack / Email / DingTalk / PushPlus / Server酱 ...) just works.
"""
from .engine import (
    Alert,
    AlertKind,
    AlertSeverity,
    PositionDrawdown,
    evaluate_portfolio,
)
from .notifier import build_markdown, send

__all__ = [
    "Alert",
    "AlertKind",
    "AlertSeverity",
    "PositionDrawdown",
    "evaluate_portfolio",
    "build_markdown",
    "send",
]
