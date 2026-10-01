"""排雷通知层。

薄封装 DSA 的 :class:`NotificationService`，复用已配置的全部渠道
（企业微信 / 飞书 / 钉钉 / Telegram / Discord / Slack / Email / Server酱³ /
PushPlus / ...），不新增任何推送实现。

跨运行去重
----------
GitHub Actions runner 是一次性的，两次运行之间没有磁盘。要回答
"我有没有告诉过你了"这个问题，必须记住一点东西 —— 公告本身不记录这件事。

因此这里维护一份 :func:`load_state` / :func:`save_state` 读写的小快照
（每个标的当前的风险事件指纹集合，约几 KB），由 workflow 通过
``actions/cache`` 在运行之间传递。

**去重只影响"推不推"，完全不影响"判得准不准"** —— 判定永远从当次抓回的
完整公告时间线重新推导，状态文件不参与任何等级计算。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .engine import RiskReport, RiskVerdict
from .rules import EventClass, RiskLevel, SourceType

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = "data/risk_scan_state.json"


# RiskLevel -> NotificationService severity
_SEVERITY_MAP = {
    RiskLevel.CLEAR: "info",
    RiskLevel.WATCH: "info",
    RiskLevel.WARNING: "warning",
    RiskLevel.HIGH: "error",
    RiskLevel.CRITICAL: "critical",
}


# ---------------------------------------------------------------------------
# 跨运行状态：事件指纹去重
# ---------------------------------------------------------------------------


def load_state(path: Optional[str] = None) -> Dict[str, Any]:
    """读取上次的推送快照。文件不存在或损坏时返回空状态。

    空状态意味着"全部当新事件"，也就是首次运行会推一遍 —— 这是期望行为，
    不是降级。
    """
    target = Path(path or os.environ.get("RISK_SCAN_STATE_PATH", DEFAULT_STATE_PATH))
    if not target.exists():
        logger.info("未找到历史状态 %s，按首次运行处理", target)
        return {"as_of": "", "seen": {}}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("状态文件读取失败 %s: %s；按首次运行处理", target, exc)
        return {"as_of": "", "seen": {}}
    if not isinstance(data, dict) or not isinstance(data.get("seen"), dict):
        return {"as_of": "", "seen": {}}
    return data


def save_state(report: RiskReport, path: Optional[str] = None) -> Path:
    """把本次的风险事件指纹覆盖写入状态文件。

    用**整体覆盖**而非追加：上一次有、这次没有的指纹会自动消失，
    文件不会无限增长，也不存在清理逻辑。
    """
    target = Path(path or os.environ.get("RISK_SCAN_STATE_PATH", DEFAULT_STATE_PATH))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of": report.as_of,
        "seen": {v.code: v.fingerprints for v in report.verdicts if v.fingerprints},
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "状态已写入 %s（%d 个标的，%d 字节）",
        target,
        len(payload["seen"]),
        target.stat().st_size,
    )
    return target


def _new_fingerprints(
    verdict: RiskVerdict, seen: Dict[str, Any]
) -> List[str]:
    """该标的本次新增的、用户还没被告知过的事件指纹。"""
    known = set(seen.get(verdict.code) or [])
    return [fp for fp in verdict.fingerprints if fp not in known]


def select_pushable(
    report: RiskReport,
    *,
    push_min_level: RiskLevel = RiskLevel.WARNING,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[RiskVerdict], Dict[str, List[str]]]:
    """挑出本次真正需要推送的持仓。

    返回 ``(verdicts, new_map)``，其中 ``new_map[code]`` 是该标的的新增事件指纹。

    判定条件（任一满足即推）：

    * 上次没有这个标的 → 新进入警戒
    * 有新增事件指纹   → 出了新情况（哪怕等级没变，比如"处罚落地"挂在
      未结案的立案下面，等级仍是严重但值得单独告知）
    * 等级较上次上升   → 风险升级

    等级未变且无新事件 → 静默。这就是"同一件事不重复推"的全部逻辑。
    """
    prior = state or {}
    seen = prior.get("seen") or {}

    pushable: List[RiskVerdict] = []
    new_map: Dict[str, List[str]] = {}
    for verdict in report.sorted_verdicts():
        if verdict.level < push_min_level:
            continue
        new = _new_fingerprints(verdict, seen)
        if not new and verdict.code in seen:
            continue  # 同一件事，不重推
        pushable.append(verdict)
        new_map[verdict.code] = new
    return pushable, new_map


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------


def _fmt_date(event) -> str:
    """展示事件日期。哨兵日期（URL 里抽不到）显示"日期未知"而不是 1970-01-01。"""
    from .fetcher import _UNKNOWN_DATE

    return "日期未知" if event.publish_date == _UNKNOWN_DATE else event.publish_date


def _fmt_event(verdict: RiskVerdict, limit: int = 4) -> List[str]:
    lines: List[str] = []
    events = verdict.events[:limit]
    for event in events:
        sign = "↑" if event.is_aggravating else "↓"
        link = f" [原文]({event.url})" if event.url else ""
        lines.append(
            f"    {sign} `{event.rule.keyword}` {event.level.label}"
            f"（{event.rule.category}） {event.publish_date}"
            f" · {event.title[:40]}{link}"
        )
    if len(verdict.events) > limit:
        lines.append(f"    … 另有 {len(verdict.events) - limit} 条相关公告")
    return lines


def build_markdown(
    report: RiskReport,
    *,
    push_min_level: RiskLevel = RiskLevel.WARNING,
    state: Optional[Dict[str, Any]] = None,
    new_map: Optional[Dict[str, List[str]]] = None,
    title: str = "持仓排雷",
) -> str:
    """生成排雷 markdown 报告（三段式）。

    **报告永远展示全貌**（所有达到门槛的持仓），不因为"这次没有新增"就
    清空内容 —— 报告是给人看现状的，推送与否由 :func:`send` 单独决定。
    ``new_map`` 只用于给本次新增的标的打 🆕 标记。
    """
    actionable = report.actionable(push_min_level)
    new_codes = set(new_map or {})
    lines: List[str] = [
        f"**{title}** · {report.as_of}　"
        f"窗口 {report.window_start}~{report.window_end}　"
        f"持仓 {report.total_positions} 只（覆盖 {report.covered_count}，"
        f"未覆盖 {len(report.uncovered)}）",
        "",
    ]

    # ── 第一段：确认风险（交易所 / 证监会公告）──
    filing_hits = [
        v
        for v in actionable
        if v.top_event is None or v.top_event.source_type is SourceType.EXCHANGE_FILING
    ]
    news_hits = [
        v for v in actionable if v.top_event is not None
        and v.top_event.source_type is SourceType.NEWS
    ]

    if filing_hits:
        lines.append(f"🚨 **确认风险**（交易所/证监会公告）　{len(filing_hits)} 只")
        for verdict in filing_hits:
            flag = "🆕 " if verdict.code in new_codes else ""
            lines.append("")
            lines.append(
                f"{flag}{verdict.level.glyph} **{verdict.name}** ({verdict.code})　"
                f"**{verdict.level.label}**" + ("　[ST]" if verdict.is_st else "")
            )
            if verdict.categories:
                for cat in verdict.categories:
                    is_primary = (
                        verdict.top_event is not None
                        and cat.top_event is verdict.top_event
                    )
                    date = cat.top_event.publish_date if cat.top_event else "—"
                    kw = cat.top_event.rule.keyword if cat.top_event else "—"
                    link = (
                        f" [原文]({cat.top_event.url})"
                        if cat.top_event and cat.top_event.url
                        else ""
                    )
                    extra = f"　（{'；'.join(cat.reasons)}）" if cat.reasons else ""
                    tail = "　**← 主因**" if is_primary else ""
                    lines.append(
                        f"- {'◀ ' if is_primary else ''}**{cat.category}**"
                        f"（{cat.level.label}）　`{kw}` {date}{link}{extra}{tail}"
                    )
            if verdict.downgrade_reason:
                lines.append(f"- ⤷ {verdict.downgrade_reason}")
    else:
        clean = [v for v in report.sorted_verdicts() if v.level == RiskLevel.CLEAR]
        lines.append(
            f"✅ **确认风险**：无持仓达到「{push_min_level.label}」档"
            f"（{report.covered_count} 只已覆盖持仓中，{len(clean)} 只无风险事件）"
        )

    # ── 第二段：舆情信号（新闻，未经官方确认）──
    if news_hits:
        lines.append("")
        lines.append(f"🔎 **舆情信号**（新闻，未经官方确认）　{len(news_hits)} 只")
        for verdict in news_hits:
            flag = "🆕 " if verdict.code in new_codes else ""
            head = verdict.top_event.title if verdict.top_event else "—"
            link = (
                f" [链接]({verdict.top_event.url})"
                if verdict.top_event and verdict.top_event.url
                else ""
            )
            lines.append("")
            lines.append(
                f"{flag}{verdict.level.glyph} **{verdict.name}** ({verdict.code})　"
                f"**{verdict.level.label}**（舆情档上限）"
            )
            lines.append(f"    {head[:70]}{link}")
            if verdict.downgrade_reason:
                lines.append(f"    ⤷ {verdict.downgrade_reason}")

    # ── 财务异常（交易所口径的确定性数据）──
    fin_rows = [
        (v, e) for v in report.sorted_verdicts()
        for e in v.events if e.event_class.value == "fundamental"
    ]
    if fin_rows:
        lines.append("")
        lines.append("📉 **财务异常**（财报口径，不受舆情封顶限制）")
        seen_fin: set = set()
        for verdict, event in fin_rows:
            key = (verdict.code, event.publish_date)
            if key in seen_fin:
                continue
            seen_fin.add(key)
            lines.append(f"- {event.title}")

    others = list(
        (v, e) for v in report.sorted_verdicts() for e in v.other_events
    )
    if others:
        lines.append("")
        lines.append("📋 **其它动向**（股东行为 / 公司运作）")
        seen_pairs: set = set()
        for verdict, event in others:
            key = (verdict.code, event.fingerprint)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            lines.append(
                f"- {verdict.name} ({verdict.code})　`{event.rule.keyword}`"
                f" {event.publish_date}　[{event.event_class.value}]"
            )

    # 热度：单条公告里看不到的信号。密集出现说明关注度在抬升。
    hot = [v for v in report.sorted_verdicts() if v.is_hot]
    if hot:
        lines.append("")
        lines.append("🔥 **关注度异常**（窗口内风险类公告密集）")
        for verdict in hot:
            lines.append(
                f"- {verdict.name} ({verdict.code})　窗口内 {verdict.heat} 条风险类公告"
            )

    if report.uncovered:
        lines.append("")
        lines.append(f"**⚠️ 未纳入本次排雷**　{len(report.uncovered)} 只")
        lines.append("")
        for item in report.uncovered:
            lines.append(f"- {item['name']} ({item['code']})　— {item['reason']}")

    if report.errors:
        lines.append("")
        lines.append(f"**数据异常**　{len(report.errors)} 项")
        for err in report.errors[:10]:
            lines.append(f"- {err}")

    return "\n".join(lines)

    # 关注档（即使已有 actionable 也补一段，方便观察变化）
    watching = [
        v
        for v in report.sorted_verdicts()
        if RiskLevel.WATCH <= v.level < push_min_level
    ]
    if actionable and watching:
        lines.append("")
        lines.append(f"**关注档**　{len(watching)} 只")
        for verdict in watching:
            head = verdict.top_event.title if verdict.top_event else "无公告事件"
            lines.append(
                f"- {verdict.level.glyph} **{verdict.name}** ({verdict.code})　"
                f"{head[:36]}"
            )

    if report.uncovered:
        lines.append("")
        lines.append(f"**⚠️ 未纳入本次排雷**　{len(report.uncovered)} 只")
        lines.append("")
        for item in report.uncovered:
            lines.append(f"- {item['name']} ({item['code']})　— {item['reason']}")

    if report.errors:
        lines.append("")
        lines.append(f"**数据异常**　{len(report.errors)} 项")
        for err in report.errors[:10]:
            lines.append(f"- {err}")

    return "\n".join(lines)


def pick_severity(
    report: RiskReport,
    *,
    push_min_level: RiskLevel = RiskLevel.WARNING,
    new_map: Optional[Dict[str, List[str]]] = None,
) -> str:
    """按最高命中的持仓选择通知 severity 标签。"""
    if new_map is not None:
        actionable = [v for v in report.sorted_verdicts() if v.code in new_map]
    else:
        actionable = report.actionable(push_min_level)
    if not actionable:
        return "info"
    worst = max((v.level for v in actionable), default=RiskLevel.CLEAR)
    return _SEVERITY_MAP.get(worst, "warning")


def send(
    report: RiskReport,
    *,
    dry_run: bool = False,
    push_min_level: RiskLevel = RiskLevel.WARNING,
    state: Optional[Dict[str, Any]] = None,
    new_map: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """推送排雷结果。返回 True 表示"已推送或无需推送"。

    跳过推送的情况（均视为成功，避免 workflow 翻红）：

    * ``dry_run=True``
    * 无持仓达到 ``push_min_level``（全绿日不打扰）
    * 所有命中的标的都没有新事件（同一件事不重复推）
    * 未配置任何通知渠道（NotificationService 内部会记录错误）
    """
    actionable = (
        report.actionable(push_min_level)
        if new_map is None
        else [v for v in report.sorted_verdicts() if v.code in new_map]
    )
    if not actionable:
        logger.info(
            "本次无需推送：无达到 %s 档的新增事件（%s）",
            push_min_level.label,
            "全绿" if new_map is None else "与上次相同",
        )
        return True

    markdown = build_markdown(
        report, push_min_level=push_min_level, state=state, new_map=new_map
    )
    if dry_run:
        logger.info("[dry-run] markdown payload:\n%s", markdown)
        return True

    try:
        from src.notification import NotificationService
    except Exception as exc:  # noqa: BLE001
        logger.error("NotificationService 不可用: %s", exc)
        print(markdown)
        return False

    try:
        service = NotificationService()
        severity = pick_severity(
            report, push_min_level=push_min_level, new_map=new_map
        )
        return bool(
            service.send(
                markdown,
                route_type="alert",
                severity=severity,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("NotificationService.send 失败: %s", exc)
        print(markdown)
        return False
