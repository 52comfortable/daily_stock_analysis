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

from .engine import (
    Alert,
    AlertKind,
    AlertSeverity,
    SEVERITY_RANK,
    PositionDrawdown,
    alert_glyph,
)

logger = logging.getLogger(__name__)


def build_markdown(result: Dict[str, Any], *, title: str = "持仓预警") -> str:
    """Build a compact markdown report grouped by the two alert kinds."""
    alerts: list[Alert] = result.get("alerts") or []
    positions: list[PositionDrawdown] = result.get("positions") or []
    thresholds: Dict[str, Any] = result.get("thresholds") or {}
    as_of: str = result.get("as_of") or ""
    min_gain = float(thresholds.get("min_gain", 5.0))
    stop_loss = float(thresholds.get("stop_loss", 5.0))

    lines: list[str] = [f"**{title}** · {as_of}", ""]
    if not alerts:
        lines.append(
            f"扫描 {len(positions)} 只持仓，均未触发"
            f"（浮盈回撤需涨超 {min_gain:g}% 后大幅回落；"
            f"止损需跌破成本 {stop_loss:g}%）"
        )
        return "\n".join(lines)

    # Two independent buckets. A position can appear in both.
    stop_alerts = [a for a in alerts if a.kind is AlertKind.STOP_LOSS]
    draw_alerts = [a for a in alerts if a.kind is AlertKind.DRAWDOWN]

    def _rank(a: Alert) -> tuple:
        metric = (
            -a.position.remaining_upside_pct
            if a.kind is AlertKind.STOP_LOSS
            else -(a.position.drawdown_pct or 0.0)
        )
        return (-SEVERITY_RANK[a.severity], metric, a.position.code)

    draw_alerts.sort(key=_rank)
    stop_alerts.sort(key=_rank)

    n_pos = len({a.position.code for a in alerts})
    lines.append(
        f"扫描 {len(positions)} 只持仓，**{len(alerts)}** 条预警"
        f"（涉及 {n_pos} 只）"
    )

    if draw_alerts:
        lines.append("")
        lines.append(
            f"**① 浮盈回撤**（曾涨超 {min_gain:g}%，现大幅回落）　{len(draw_alerts)} 条"
        )
        for a in draw_alerts:
            g = alert_glyph(a.severity, a.kind)
            p = a.position
            lines.append(
                f"- {g} **{p.name}** ({p.code})　"
                f"浮盈 +{p.peak_gain_pct:.1f}% → 回撤 **{p.drawdown_pct:.1f}%**　"
                f"{p.peak_price:.2f} → {p.today_close:.2f}"
            )

    if stop_alerts:
        lines.append("")
        lines.append(
            f"**② 跌破止损线**（成本 -{stop_loss:g}%）　{len(stop_alerts)} 条"
        )
        for a in stop_alerts:
            g = alert_glyph(a.severity, a.kind)
            p = a.position
            lines.append(
                f"- {g} **{p.name}** ({p.code})　"
                f"已亏 **{-p.remaining_upside_pct:.1f}%**　"
                f"成本 {p.cost_price:.2f} → 现价 {p.today_close:.2f}"
            )

    lines.append("")
    lines.append("---")

    def _detail(a: Alert) -> list[str]:
        p = a.position
        rows = [
            f"- 建仓 {p.position_date}　成本 {p.cost_price:.2f}",
            f"- 历史最高收盘 {p.peak_price:.2f}（{p.peak_date}）"
            f"　最新收盘 {p.today_close:.2f}（{p.today_date}）",
        ]
        if a.kind is AlertKind.DRAWDOWN:
            rows.append(
                f"- 峰值时浮盈 **+{p.peak_gain_pct:.2f}%**"
                f"　现价距峰值 **{p.drawdown_pct:.2f}%**"
            )
        else:
            rows.append(
                f"- 已跌破成本 **{-p.remaining_upside_pct:.2f}%**"
                f"（止损线 -{stop_loss:g}%）"
            )
        if p.remaining_upside_pct is not None and p.remaining_upside_pct >= 0:
            rows.append(f"- 剩余空间 **+{p.remaining_upside_pct:.2f}%**")
        return rows

    for group, label in (
        (draw_alerts, "浮盈回撤"),
        (stop_alerts, "跌破止损"),
    ):
        if not group:
            continue
        lines.append("")
        lines.append(f"**{label}明细**")
        for a in group:
            p = a.position
            g = alert_glyph(a.severity, a.kind)
            lines.append("")
            lines.append(f"{g} **{p.name} ({p.code})**")
            lines.extend(_detail(a))

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
