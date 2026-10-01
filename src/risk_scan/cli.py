"""排雷扫描 CLI 入口。

用法
----
生产（GitHub Actions）::

    python -m src.risk_scan.cli --portfolio data/portfolio.json \\
        --report-file reports/risk-scan/YYYY-MM-DD.md

本地调试::

    python -m src.risk_scan.cli --help
    python -m src.risk_scan.cli --dry-run --window-days 30

行为
----
1. 从 ``RISK_SCAN_JSON_PATH``（默认 ``data/portfolio.json``）读持仓。
2. 逐只抓取巨潮公告（窗口默认 30 天，见 ``--window-days``）。
3. 按风险阶梯 + 状态机判定，输出纯文本摘要到 stdout。
4. markdown 报告写到 ``--report-file``（未指定则只写 stdout）。
5. 达到 ``RISK_SCAN_PUSH_MIN_LEVEL``（默认 ``warning``）时推送通知。

退出码
------
0  执行成功（无论是否命中）
2  持仓文件缺失或格式错误
3  推送失败（判定本身已成功）
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from .engine import evaluate_holdings, split_coverage
from .fetcher import default_window, fetch_announcements
from .notifier import (
    build_markdown,
    load_state,
    save_state,
    select_pushable,
    send as send_notification,
)
from .rules import RiskLevel

logger = logging.getLogger(__name__)

DEFAULT_PORTFOLIO_PATH = "data/portfolio.json"

#: 公告扫描窗口默认值。
#:
#: 取 180（半年）而不是对齐立案 TTL 的 540，依据是「重大利空的价格冲击前置」：
#: 立案、处罚、风险提示这类公告出来当天就跌停，反应在几分钟内完成，股价在
#: 事件日就反映完了。一年前的未结案立案不再是「该不该现在卖」的信息。
#:
#: 实测（9 只真实持仓）：180 / 360 / 540 / 730 四个窗口判定结果完全一致，
#: 180 天最快且公告量最少（529 vs 1561 条）。
#:
#: 注意：重复推送**不是**由窗口控制的，而是由事件指纹去重控制 —— 沃格光电
#: 那条 139 天前的立案，无论窗口多长都只会推一次。
#:
#: 副作用：180 天窗口内，只有 TTL=90 的规则（异常波动）还能触发时效衰减，
#: 其余规则的 TTL 都大于窗口，表现为"今年发生过就仍然有效"。
DEFAULT_WINDOW_DAYS = 180
_LEVEL_BY_NAME = {level.label: level for level in RiskLevel}
_LEVEL_BY_NAME.update({level.name.lower(): level for level in RiskLevel})
# 允许用整数直接指定
_LEVEL_BY_NAME.update({str(int(level)): level for level in RiskLevel})


def _load_portfolio(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"持仓文件不存在: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict) or not isinstance(data.get("positions"), list):
        raise ValueError(f"{path} 需为含顶层 positions 数组的 JSON 对象")
    return data


def _parse_level(raw: str, *, default: RiskLevel) -> RiskLevel:
    text = (raw or "").strip()
    if not text:
        return default
    level = _LEVEL_BY_NAME.get(text.lower()) or _LEVEL_BY_NAME.get(text)
    if level is None:
        logger.warning("无法解析等级 %r，回退到 %s", raw, default.label)
        return default
    return level


def _print_text_summary(report: Any) -> None:
    """stdout 纯文本摘要。GBK 控制台下 emoji 会崩，这里统一 errors='replace'。"""
    lines: List[str] = [
        f"\n=== 持仓排雷 · {report.as_of} ===",
        f"窗口 {report.window_start} ~ {report.window_end}　"
        f"持仓 {report.total_positions} 只（覆盖 {report.covered_count}，"
        f"未覆盖 {len(report.uncovered)}）",
        "",
    ]
    for verdict in report.sorted_verdicts():
        marker = f"{verdict.level.glyph} {verdict.level.label}"
        head = verdict.top_event.title if verdict.top_event else "无风险事件"
        extra = " [ST]" if verdict.is_st else ""
        lines.append(f"  [{marker:>6}]{extra} {verdict.name}({verdict.code})  {head[:44]}")
        if verdict.downgrade_reason:
            lines.append(f"           ⤷ {verdict.downgrade_reason}")
        if verdict.error:
            lines.append(f"           ✖ {verdict.error}")
    if report.uncovered:
        lines.append("")
        lines.append(f"  未纳入排雷 {len(report.uncovered)} 只：")
        for item in report.uncovered:
            lines.append(f"    - {item['name']}({item['code']})  {item['reason']}")
    if report.errors:
        lines.append("")
        lines.append(f"  数据异常 {len(report.errors)} 项：")
        for err in report.errors[:10]:
            lines.append(f"    - {err}")
    lines.append("")

    payload = "\n".join(lines)
    try:
        print(payload)
    except UnicodeEncodeError:
        sys.stdout.reconfigure(errors="replace")
        print(payload)
        sys.stdout.reconfigure(errors="strict")


def run(
    *,
    portfolio_path: Optional[Path] = None,
    window_days: Optional[int] = None,
    push_min_level: Optional[RiskLevel] = None,
    mitigation_tiers: Optional[int] = None,
    dry_run: Optional[bool] = None,
    report_file: Optional[Path] = None,
    state_path: Optional[str] = None,
    with_news: Optional[bool] = None,
    with_triage: Optional[bool] = None,
    with_earnings: Optional[bool] = None,
    crosscheck: bool = True,
) -> int:
    """执行一次排雷扫描，返回退出码。"""
    portfolio_path = portfolio_path or Path(
        os.environ.get("RISK_SCAN_JSON_PATH", DEFAULT_PORTFOLIO_PATH)
    )
    window = int(
        window_days
        if window_days is not None
        else os.environ.get("RISK_SCAN_WINDOW_DAYS", str(DEFAULT_WINDOW_DAYS))
    )
    level = _parse_level(
        "" if push_min_level is None else push_min_level.label,
        default=_parse_level(
            os.environ.get("RISK_SCAN_PUSH_MIN_LEVEL", "warning"),
            default=RiskLevel.WARNING,
        ),
    )
    tiers = int(
        mitigation_tiers
        if mitigation_tiers is not None
        else os.environ.get("RISK_SCAN_MITIGATION_TIERS", "2")
    )
    if dry_run is None:
        dry_run = os.environ.get("RISK_SCAN_DRY_RUN", "").lower() in ("1", "true", "yes")
    if with_news is None:
        with_news = os.environ.get("RISK_SCAN_WITH_NEWS", "1").lower() not in (
            "0", "false", "no",
        )
    if with_earnings is None:
        with_earnings = os.environ.get("RISK_SCAN_WITH_EARNINGS", "1").lower() not in (
            "0", "false", "no",
        )
    if with_triage is None:
        with_triage = os.environ.get("RISK_SCAN_LLM_TRIAGE", "1").lower() not in (
            "0", "false", "no",
        )

    try:
        portfolio = _load_portfolio(Path(portfolio_path))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        logger.error("加载持仓文件失败: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    start, end = default_window(window, end=date.today())
    logger.info("扫描窗口 %s ~ %s", start, end)

    covered, uncovered = split_coverage(portfolio)
    if uncovered:
        logger.info(
            "%d 只持仓不适用公告排雷（基金/ETF/LOF）: %s",
            len(uncovered),
            "、".join(f"{i['name']}({i['code']})" for i in uncovered),
        )
    if not covered:
        logger.warning("没有可排雷的持仓")
        print("WARNING: 没有可排雷的持仓（全部为基金/ETF/LOF）", file=sys.stderr)
        return 0

    codes = [code for code, _ in covered]
    names = {code: name for code, name in covered}

    # ── 通道一：交易所公告（仅 A 股）──
    try:
        announcements = fetch_announcements(
            codes, names=names, start=start, end=end, crosscheck=crosscheck
        )
    except Exception as exc:  # noqa: BLE001 - 抓取层异常不应让 CLI 崩掉
        logger.exception("公告抓取失败: %s", exc)
        print(f"ERROR: 公告抓取失败: {exc}", file=sys.stderr)
        return 4

    # 抓取阶段记录的整条通道故障。必须一路带到报告里 —— 通道整条挂掉时
    # 受影响的标的只会显示「无风险事件」，光看逐只结果发现不了。
    channel_errors: List[str] = []

    # ── 通道二：长桥（资讯/社区情绪/财报，全市场）──
    lb_rows: Dict[str, Any] = {}
    if with_news or with_earnings:
        try:
            from .fetcher import fetch_longbridge

            lb_rows = fetch_longbridge(
                codes, names=names,
                include_news=with_news, include_topics=with_news,
                include_earnings=with_earnings,
            )
            # 覆盖清单要打出来：只报"覆盖 N 只"时无法判断长桥到底支持哪些
            # 市场，是 A 股资讯真为空、还是调用失败，光看计数分不出来。
            covered = sorted(lb_rows)
            logger.info(
                "长桥通道取回 %d 条（覆盖 %d 只：%s）",
                sum(len(v) for v in lb_rows.values()),
                len(covered),
                "、".join(
                    f"{names.get(c, '')}({c})" if names.get(c) else c for c in covered
                ) or "无",
            )
            for code, rows in lb_rows.items():
                announcements.setdefault(code, []).extend(rows)
        except Exception as exc:  # noqa: BLE001 - 补充通道，失败不影响主判定
            logger.warning("长桥通道失败（不影响公告判定）: %s", exc)
            channel_errors.append(f"长桥通道不可用，港股无外部信号：{exc}")

    # ── LLM 兜底判定（可选，失败不影响主流程）──
    triage: Optional[Dict[Any, Any]] = None
    if with_triage:
        try:
            from .engine import triage_announcements

            triage = triage_announcements(announcements)
            hits = sum(len(v) for v in triage.values())
            logger.info("LLM 兜底：判定 %d 只标的共 %d 条", len(triage), hits)
        except Exception as exc:  # noqa: BLE001 - 兜底失败不得影响主判定
            logger.warning("LLM 兜底判定跳过: %s", exc)

    report = evaluate_holdings(
        portfolio,
        announcements,
        as_of=end,
        window_start=start,
        window_end=end,
        mitigation_tiers=tiers,
        triage=triage,
        channel_errors=channel_errors,
    )
    _print_text_summary(report)

    # ── 跨运行去重 ──
    state = load_state(state_path)
    pushable, new_map = select_pushable(report, push_min_level=level, state=state)
    if report.verdicts:
        logger.info(
            "达到门槛 %d 只，其中本次有新增事件 %d 只",
            len(report.actionable(level)),
            len(pushable),
        )

    markdown = build_markdown(
        report, push_min_level=level, state=state, new_map=new_map
    )
    if report_file is not None:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(markdown, encoding="utf-8")
        logger.info("markdown 报告已写入 %s", report_file)
    else:
        sys.stdout.write(markdown + "\n")
        sys.stdout.flush()

    pushed = send_notification(
        report,
        dry_run=dry_run,
        push_min_level=level,
        state=state,
        new_map=new_map,
    )
    if not pushed and not dry_run:
        # 推送失败时**不写状态** —— 否则这次的事件会被当成"已告知"，
        # 下次就不会再推，等于静默吞掉一条告警。
        logger.error("推送失败，保留旧状态以便下次重推")
        return 3

    save_state(report, state_path)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="持仓排雷：逐只拉取巨潮公告，按风险阶梯判定监管风险等级"
    )
    parser.add_argument("--portfolio", default=None, help="持仓 JSON 路径")
    parser.add_argument(
        "--window-days",
        type=int,
        default=None,
        help=f"公告扫描窗口天数（默认 {DEFAULT_WINDOW_DAYS}；不建议调小，会漏掉长期未结案的立案）",
    )
    parser.add_argument(
        "--push-min-level",
        type=str,
        default=None,
        help="达到该等级才推送：clear/watch/warning/high/critical（默认 warning）",
    )
    parser.add_argument(
        "--mitigation-tiers",
        type=int,
        default=None,
        help="出现结案/摘帽等缓解事件时下调几级（默认 2，下限 watch）",
    )
    parser.add_argument("--dry-run", action="store_true", help="不真正推送，只打预览")
    parser.add_argument(
        "--no-crosscheck", action="store_true", help="跳过零结果的巨潮 raw 反查"
    )
    parser.add_argument(
        "--no-news", action="store_true", help="跳过长桥资讯/社区情绪（只用交易所公告 + 财报）"
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="跨运行状态文件路径（默认 data/risk_scan_state.json，用于事件指纹去重）",
    )
    parser.add_argument(
        "--no-triage", action="store_true", help="跳过 LLM 兜底判定（只用确定性规则）"
    )
    parser.add_argument(
        "--no-earnings",
        action="store_true",
        help="跳过长桥财报通道（连续亏损检测）",
    )
    parser.add_argument("--report-file", type=str, default=None, help="markdown 报告输出路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出 debug 日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    push_level = (
        _parse_level(args.push_min_level, default=RiskLevel.WARNING)
        if args.push_min_level
        else None
    )
    return run(
        portfolio_path=Path(args.portfolio) if args.portfolio else None,
        window_days=args.window_days,
        push_min_level=push_level,
        mitigation_tiers=args.mitigation_tiers,
        dry_run=args.dry_run or None,
        report_file=Path(args.report_file) if args.report_file else None,
        state_path=args.state,
        with_news=False if args.no_news else None,
        with_triage=False if args.no_triage else None,
        with_earnings=False if args.no_earnings else None,

        crosscheck=not args.no_crosscheck,
    )


if __name__ == "__main__":
    sys.exit(main())
