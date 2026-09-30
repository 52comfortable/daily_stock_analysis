"""CLI entry point for the position-drawdown alert.

Used by:
  - ``python main.py --position-drawdown-check`` (production)
  - Local debugging: ``python -m src.position_drawdown.cli --help``

Behavior
--------
1. Load ``portfolio.json`` from the path given by env
   ``POSITION_DRAWDOWN_JSON_PATH`` (default: ``data/portfolio.json``).
2. Run ``evaluate_portfolio`` to compute drawdowns.
3. Print a plain-text summary to stdout.
4. If ``FEISHU_WEBHOOK_URL`` is set, push a Feishu card. Otherwise
   print a JSON preview (dry-run mode).
5. Always write a markdown report to
   ``reports/position-drawdown/YYYY-MM-DD.md`` for archiving.

Exit code
---------
  0  -  ran successfully (alerts or not)
  2  -  portfolio.json missing or invalid
  3  -  Feishu push failed (after evaluation succeeded)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .engine import (
    AlertSeverity,
    SEVERITY_GLYPH,
    evaluate_portfolio,
)
from .notifier import build_markdown, send as send_notification

logger = logging.getLogger(__name__)


DEFAULT_PORTFOLIO_PATH = "data/portfolio.json"


def _load_portfolio(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"portfolio file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict) or "positions" not in data:
        raise ValueError(
            f"portfolio file {path} must be a JSON object with a 'positions' array"
        )
    return data


def _print_text_summary(result: Dict[str, Any]) -> None:
    """Print a plain-text summary to stdout (for log readability).

    Emojis (🟡🟠🔴) are used as severity glyphs. When the terminal
    encoding can't render them (e.g. Windows GBK console), we fall back
    to ASCII markers so the run still produces useful output instead of
    crashing with a UnicodeEncodeError.
    """
    alerts = result.get("alerts") or []
    positions = result.get("positions") or []
    as_of = result.get("as_of") or ""
    lines = [f"\n=== 持仓回撤预警 · {as_of} ==="]
    lines.append(f"扫描 {len(positions)} 只持仓，触发 {len(alerts)} 只\n")
    for p in positions:
        if p.error:
            lines.append(f"  [{p.code}] error: {p.error}")
            continue
        peak_str = (
            f"peak {p.peak_price:.2f}({p.peak_date})"
            if p.peak_price is not None
            else "peak -"
        )
        close_str = (
            f"close {p.today_close:.2f}"
            if p.today_close is not None
            else "close -"
        )
        dd = f"{p.drawdown_pct:.2f}%" if p.drawdown_pct is not None else "-"
        upside = (
            f"{p.remaining_upside_pct:+.2f}%"
            if p.remaining_upside_pct is not None
            else "-"
        )
        sev = next(
            (a.severity for a in alerts if a.position.code == p.code), None
        )
        marker = (SEVERITY_GLYPH[sev] + " " + sev.value) if sev else "ok"
        lines.append(
            f"  [{marker:>10}] {p.name}({p.code}) "
            f"cost={p.cost_price:.2f} {peak_str} {close_str} "
            f"dd={dd} upside={upside}"
        )
    if not alerts:
        lines.append("\n(无触发)")
    lines.append("")
    payload = "\n".join(lines)
    # Write to stdout with errors="replace" so GBK consoles don't crash.
    try:
        print(payload)
    except UnicodeEncodeError:
        sys.stdout.reconfigure(errors="replace")
        print(payload)
        sys.stdout.reconfigure(errors="strict")


def run(
    *,
    portfolio_path: Optional[Path] = None,
    webhook_url: Optional[str] = None,
    dry_run: Optional[bool] = None,
    thresholds: Optional[Dict[str, float]] = None,
    report_file: Optional[Path] = None,
) -> int:
    """Run the alert pipeline and return an exit code."""
    portfolio_path = portfolio_path or Path(
        os.environ.get("POSITION_DRAWDOWN_JSON_PATH", DEFAULT_PORTFOLIO_PATH)
    )
    webhook_url = (
        webhook_url
        if webhook_url is not None
        else os.environ.get("FEISHU_WEBHOOK_URL", "")
    )
    if dry_run is None:
        # Default: real push. NotificationService 自动发现所有已配渠道
        # (Feishu/Server酱³/PushPlus/Email/Telegram/...); 若一个都没配，
        # 它会在日志里报错，notifier.send() 也会落到 dry-run 预览兜底。
        dry_run = os.environ.get("POSITION_DRAWDOWN_DRY_RUN", "").lower() in (
            "1", "true", "yes",
        )

    try:
        portfolio = _load_portfolio(Path(portfolio_path))
    except (FileNotFoundError, ValueError) as exc:
        logger.error("加载持仓文件失败: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # env 变量是默认，thresholds 参数用来覆盖。
    # 这样 workflow 不用传 CLI 参数，部署时只改 .env / GitHub vars 即可。
    env_thresholds: Dict[str, float] = {}
    for key, env_name in (
        ("warning", "POSITION_DRAWDOWN_WARNING_PCT"),
        ("severe", "POSITION_DRAWDOWN_SEVERE_PCT"),
        ("critical", "POSITION_DRAWDOWN_CRITICAL_PCT"),
        ("severe_upside", "POSITION_DRAWDOWN_SEVERE_UPSIDE_PCT"),
    ):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            try:
                env_thresholds[key] = float(raw)
            except ValueError:
                logger.warning("Invalid %s=%r; ignoring", env_name, raw)
    # Caller-provided thresholds take precedence over env.
    merged: Dict[str, float] = {**env_thresholds, **(thresholds or {})}
    logger.info(
        "thresholds: warning=%s severe=%s critical=%s severe_upside=%s (source: %s)",
        merged.get("warning", 10.0),
        merged.get("severe", 20.0),
        merged.get("critical", 30.0),
        merged.get("severe_upside", 5.0),
        "cli+env" if (thresholds and env_thresholds) else ("cli" if thresholds else "env"),
    )

    result = evaluate_portfolio(portfolio, thresholds=merged)
    _print_text_summary(result)

    # Markdown 报告：默认只 print 到 stdout（避免本地文件累积）。
    # CI 场景下，workflow 启动时把 stdout 重定向到 --report-file 指定路径，
    # 给 actions/upload-artifact 抓。这样本地跑永远不落盘，CI 跑落盘一次。
    markdown = build_markdown(result)
    if report_file is not None:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(markdown, encoding="utf-8")
        logger.info("markdown report written to %s", report_file)
    else:
        # 写到 stdout。logger 走 stderr，print 走 stdout，两者不冲突，
        # CI 重定向 >file 时只抓走 markdown 本体。
        sys.stdout.write(markdown + "\n")
        sys.stdout.flush()

    pushed = send_notification(result, dry_run=dry_run)
    if not pushed and not dry_run:
        # Real push attempted but no channel accepted it (already logged).
        return 3

    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="持仓回撤预警：拉每日 K 线算回撤，触发时推飞书"
    )
    parser.add_argument(
        "--portfolio",
        default=None,
        help="持仓 JSON 路径（默认读 POSITION_DRAWDOWN_JSON_PATH 或 data/portfolio.json）",
    )
    parser.add_argument(
        "--webhook-url",
        default=None,
        help="飞书 webhook URL（默认读 FEISHU_WEBHOOK_URL）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不真正推送飞书，只打 payload 预览",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="把 markdown 报告写到指定文件（默认只 print 到 stdout；CI 场景下用这个写到 reports/position-drawdown/YYYY-MM-DD.md）",
    )
    parser.add_argument(
        "--warning",
        type=float,
        default=None,
        help="警告档回撤阈值（默认 10）",
    )
    parser.add_argument(
        "--severe",
        type=float,
        default=None,
        help="严重档回撤阈值（默认 20）",
    )
    parser.add_argument(
        "--critical",
        type=float,
        default=None,
        help="危急档回撤阈值（默认 30）",
    )
    parser.add_argument(
        "--severe-upside",
        type=float,
        default=None,
        help="剩余空间 < 该值时升级为严重档（默认 5）",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="输出 debug 日志"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    thresholds: Dict[str, float] = {}
    for key in ("warning", "severe", "critical", "severe_upside"):
        v = getattr(args, key)
        if v is not None:
            thresholds[key] = v

    return run(
        portfolio_path=Path(args.portfolio) if args.portfolio else None,
        webhook_url=args.webhook_url,
        dry_run=args.dry_run or None,
        thresholds=thresholds or None,
        report_file=Path(args.report_file) if args.report_file else None,
    )


if __name__ == "__main__":
    sys.exit(main())
