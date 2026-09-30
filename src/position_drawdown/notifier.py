"""Notifier for the position-drawdown alert.

Thin wrapper over DSA's existing :class:`NotificationService` so the
position-drawdown check can reuse every notification channel
(Feishu / WeChat / Telegram / Discord / Slack / Email / DingTalk /
PushPlus / ...) and the user's existing .env configuration.

Build a markdown summary, hand it to ``NotificationService.send`` with
``route_type='alert'`` and a severity tag, and let the existing
pipeline route it to every channel the user has enabled.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .engine import Alert, AlertSeverity, SEVERITY_GLYPH, SEVERITY_RANK, PositionDrawdown

logger = logging.getLogger(__name__)


def build_markdown(result: Dict[str, Any], *, title: str = "持仓回撤预警") -> str:
    """Build a compact markdown report. Returns "" when nothing to report and
    the user explicitly opts into silence (see :func:`should_send`)."""
    alerts: list[Alert] = result.get("alerts") or []
    positions: list[PositionDrawdown] = result.get("positions") or []
    as_of: str = result.get("as_of") or ""

    lines: list[str] = [f"**{title}** · {as_of}", ""]
    if not alerts:
        lines.append(f"扫描 {len(positions)} 只持仓，无回撤预警。")
        return "\n".join(lines)

    # Sort alerts: most severe first, then by code
    ordered = sorted(
        alerts,
        key=lambda a: (
            -SEVERITY_RANK[a.severity],
            a.position.code,
        ),
    )
    lines.append(f"扫描 **{len(positions)}** 只持仓，触发 **{len(alerts)}** 只：")
    lines.append("")
    for a in ordered:
        glyph = SEVERITY_GLYPH[a.severity]
        p = a.position
        lines.append(
            f"{glyph} **{p.name}** ({p.code})  "
            f"峰值 {p.peak_price:.2f} → 收盘 {p.today_close:.2f}  "
            f"回撤 **{p.drawdown_pct:.1f}%**"
        )
    lines.append("")
    lines.append("---")
    for a in ordered:
        p = a.position
        block = [
            f"**{p.name} ({p.code})**",
            f"- 建仓：{p.position_date}　成本：{p.cost_price:.2f}",
            f"- 历史峰值：{p.peak_price:.2f}（{p.peak_date}）",
            f"- 今日收盘：{p.today_close:.2f}（{p.today_date}）",
            f"- 峰值回撤：**{p.drawdown_pct:.2f}%**",
        ]
        if p.remaining_upside_pct is not None:
            if p.remaining_upside_pct >= 0:
                block.append(f"- 剩余空间：**+{p.remaining_upside_pct:.2f}%**")
            else:
                block.append(f"- ⚠️ **已跌破成本 {-p.remaining_upside_pct:.2f}%**")
        if p.cost_price and p.peak_price > p.cost_price:
            peak_gain = (p.peak_price - p.cost_price) / p.cost_price * 100
            consumed = (p.peak_price - p.today_close) / (p.peak_price - p.cost_price) * 100
            block.append(f"- 曾经涨过 {peak_gain:.1f}%，现已回吐 {consumed:.1f}%")
        lines.append("")
        lines.extend(block)
    return "\n".join(lines)


def pick_severity(result: Dict[str, Any]) -> Optional[str]:
    """Pick a NotificationService severity tag based on the worst trigger."""
    alerts = result.get("alerts") or []
    if not alerts:
        return None
    severities = [a.severity for a in alerts]
    if AlertSeverity.CRITICAL in severities:
        return "critical"
    if AlertSeverity.SEVERE in severities:
        return "high"
    return "warning"


def send(result: Dict[str, Any], *, dry_run: bool = False) -> bool:
    """Send the alert via DSA's NotificationService. Returns True on success.

    Falls back to a console preview when:
      - ``dry_run`` is True, or
      - no notification channel is configured, or
      - any send-time error is raised.
    """
    if dry_run:
        logger.info("[dry-run] markdown payload:\n%s", build_markdown(result))
        return True

    try:
        from src.notification import NotificationService
    except Exception as exc:  # noqa: BLE001
        logger.error("NotificationService unavailable: %s", exc)
        print(build_markdown(result))
        return False

    try:
        service = NotificationService()
        severity = pick_severity(result) or "info"
        return bool(
            service.send(
                build_markdown(result),
                route_type="alert",
                severity=severity,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("NotificationService.send failed: %s", exc)
        print(build_markdown(result))
        return False
