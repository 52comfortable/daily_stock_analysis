"""排雷判定引擎：公告 / 舆情 → 持仓风险等级。

判定流程（每只持仓独立计算）
---------------------------
1. 各数据通道产出 :class:`RiskEvent`，每条带 ``source_type``（信息源性质）
   与 ``event_class``（事件归类）。
2. 只把 ``event_class == REGULATORY`` 的事件送进风险阶梯；
   股东行为 / 公司运作类事件单独成段，不参与定级。
3. 拆成"恶化事件"和"缓解事件"两组，按类别（立案 / 处罚 / 退市 / 交易 / 诉讼）
   **各自结算**。
4. **时效衰减**：每条恶化规则带 TTL。年龄 ≤ TTL 保持原级；≤ 2×TTL 降一级；
   > 2×TTL 清零。
5. **取最高级**作为基础等级 —— 注意是"取最大"而非旧版的互斥 mask 分类，
   因此一只票可以同时是"被立案 + 已受处罚"，而不是被其中一个桶吃掉。
6. **缓解结算**：晚于最高级恶化事件的缓解事件（结案 / 摘帽）只下调**本类别**
   的等级，且不低于 WATCH —— 结案不等于清白，仍需留意后续处罚。
7. **确定性封顶**：按 :data:`CEILING_BY_SOURCE` 截断。舆情（NEWS）无论
   结算出多高，硬性压到「警告」。这一步写在代码里而不是文档约定，
   避免有人把舆情规则误标成 CRITICAL 后外溢到「严重/致命」。
8. **ST 兜底**：简称带 ST/*ST 说明风险警示仍在生效，给 WARNING 地板。
   （旧版把 ST 票整行隐藏，恰好把最该看的标的删掉了。）
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .fetcher import Announcement
from .rules import (
    AGGRAVATING,
    CEILING_BY_SOURCE,
    HEAT_ALERT_THRESHOLD,
    MITIGATING,
    EventClass,
    RiskLevel,
    RiskRule,
    SourceType,
    is_st_name,
    match_rules,
    needs_broad_scan,
)

logger = logging.getLogger(__name__)

# 简称里带这些词的一律不在公告排雷覆盖范围（靠名称而非代码判定，
# 因为基金/ETF/LOF 与个股的代码前缀高度重叠，无法从代码本身可靠区分）
_UNCOVERED_NAME_HINTS = ("LOF", "ETF", "指数", "基金", "债", "货币")
# 明确按代码前缀排除的可转债/基金类
_UNCOVERED_CODE_PREFIXES = ("16", "50", "51", "52", "56", "58")


@dataclass(frozen=True)
class RiskEvent:
    """一次命中的规则及其证据公告。"""

    rule: RiskRule
    title: str
    publish_date: str
    url: str = ""
    age_days: int = 0

    @property
    def level(self) -> RiskLevel:
        return self.rule.level

    @property
    def is_aggravating(self) -> bool:
        return self.rule.polarity == AGGRAVATING

    @property
    def source_type(self) -> SourceType:
        return self.rule.source_type

    @property
    def event_class(self) -> EventClass:
        return self.rule.event_class

    @property
    def fingerprint(self) -> str:
        """事件指纹：跨运行识别"是不是同一条"的稳定标识。

        刻意**不包含等级**：等级会随缓解事件和 TTL 衰减变化，若把它放进
        指纹，同一件事每次运行都会算出不同指纹，去重就失效了。
        """
        return f"{self.rule.category}|{self.rule.keyword}|{self.publish_date}"


@dataclass
class RiskVerdict:
    """单只持仓的判定结果。"""

    code: str
    name: str
    level: RiskLevel
    events: List[RiskEvent] = field(default_factory=list)
    categories: List["CategoryVerdict"] = field(default_factory=list)
    top_event: Optional[RiskEvent] = None
    downgrade_reason: str = ""
    is_st: bool = False
    announcement_count: int = 0
    error: str = ""
    #: 不参与定级的其它动向（股东行为 / 公司运作 / 利好），报告里单独成段
    other_events: List[RiskEvent] = field(default_factory=list)
    #: 窗口内风险类公告条数；超过阈值视为关注度异常
    heat: int = 0
    is_hot: bool = False

    @property
    def positive_events(self) -> List[RiskEvent]:
        return [e for e in self.other_events if e.event_class is EventClass.POSITIVE]

    @property
    def mitigation_events(self) -> List[RiskEvent]:
        return [e for e in self.events if e.rule.polarity == MITIGATING]

    @property
    def fingerprints(self) -> List[str]:
        """进入风险阶梯的所有事件指纹，用于跨运行去重。"""
        return sorted({e.fingerprint for e in self.events})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "level": int(self.level),
            "level_label": self.level.label,
            "is_st": self.is_st,
            "announcement_count": self.announcement_count,
            "downgrade_reason": self.downgrade_reason,
            "top_event": self.top_event.title if self.top_event else "",
            "top_event_url": self.top_event.url if self.top_event else "",
            "categories": [
                {
                    "category": c.category,
                    "level": int(c.level),
                    "level_label": c.level.label,
                    "top_keyword": c.top_event.rule.keyword if c.top_event else "",
                    "top_date": c.top_event.publish_date if c.top_event else "",
                    "reasons": list(c.reasons),
                }
                for c in self.categories
            ],
            "events": [
                {
                    "keyword": e.rule.keyword,
                    "category": e.rule.category,
                    "level": int(e.level),
                    "polarity": e.rule.polarity,
                    "title": e.title,
                    "publish_date": e.publish_date,
                    "age_days": e.age_days,
                    "url": e.url,
                }
                for e in self.events
            ],
            "error": self.error,
        }


@dataclass
class RiskReport:
    """一次扫描的完整结果。"""

    as_of: str
    window_start: str
    window_end: str
    verdicts: List[RiskVerdict] = field(default_factory=list)
    uncovered: List[Dict[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def covered_count(self) -> int:
        return len(self.verdicts)

    @property
    def total_positions(self) -> int:
        return len(self.verdicts) + len(self.uncovered)

    def sorted_verdicts(self) -> List[RiskVerdict]:
        """按等级降序、同级按最新事件日期降序。"""
        def _sort_key(v: RiskVerdict) -> tuple[int, int, str]:
            # 日期转成 YYYYMMDD 整数再取负，保证数值排序
            stamp = 0
            if v.top_event is not None:
                try:
                    stamp = int(v.top_event.publish_date.replace("-", ""))
                except ValueError:
                    stamp = 0
            return (-int(v.level), -stamp, v.code)

        return sorted(self.verdicts, key=_sort_key)

    def actionable(self, min_level: RiskLevel) -> List[RiskVerdict]:
        """达到或超过 ``min_level`` 的持仓。"""
        return [v for v in self.sorted_verdicts() if v.level >= min_level]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "as_of": self.as_of,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "covered_count": self.covered_count,
            "total_positions": self.total_positions,
            "verdicts": [v.as_dict() for v in self.sorted_verdicts()],
            "uncovered": list(self.uncovered),
            "errors": list(self.errors),
            "config": dict(self.config),
        }


# ---------------------------------------------------------------------------
# LLM 兜底判定
#
# 定位：**规则表的探针，不是决策者**。
#
# 规则表会永远滞后于文件类型的多样性 —— 实测 876 条公告里 12 词表只认出
# 3 条，补了词之后舆情通道 72 条新闻仍然一条都匹配不上。与其无限扩词
# （扩到"监管""处罚"这种程度就会误伤"最近五年未被处罚"这类正面公告），
# 不如让模型只回答一个封闭问题：这条到底属于哪一类。
#
# 三条硬约束：
#   1. 只做**分类**，不产等级。等级由确定性映射从 class/severity 推出。
#   2. 批量调用（一次 20 条），不是逐条 —— 152 条逐条要 152 次往返。
#   3. 任何失败（无 key / 超时 / 解析失败）都**静默降级为"不判"**，
#      绝不阻塞主判定流程。
# ---------------------------------------------------------------------------

_LLM_CLASSES = (
    "regulatory",   # 监管风险：立案、处罚、退市、调查、诉讼、监管措施
    "shareholder",  # 股东行为：减持、质押、解禁
    "corporate",    # 公司运作：担保、增发、收购、重组
    "positive",     # 利好：回购、增持、中标、业绩预增
    "neutral",      # 其它：例行公告、行业新闻、价格波动描述
)

_TRIAGE_PROMPT = """你是 A 股 / 港股的风险信息分类器。下面每条是一个上市公司公告或新闻标题。

请对每条判断它属于哪一类：
- regulatory：监管风险。立案调查、行政处罚、问询函、诉讼、退市风险警示、
  核数师保留意见、停牌、盈警、被调查、被做空等
- shareholder：股东行为。减持、增持、质押、解押、股份被冻结
- corporate：公司常规运作。担保、授信、增发、收购、重组、理财、会议决议
- positive：实质利好。回购、中标、业绩预增、获批、股权激励
- neutral：其它。例行公告、行业新闻、股价涨跌描述、年报季报

每条开头的「公司名(代码)」是这条信息**所属的标的**。如果标题讲的是别家公司
（例如标的是华特气体，标题却是「江淮汽车排放造假」），一律判 neutral。

severity 只在 class 为 regulatory 时有意义：high=监管定性或退市临界，
mid=立案/处罚/诉讼，low=问询函/一般关注。

严格输出 JSON，不要任何解释文字：
{"items":[{"i":0,"class":"neutral","severity":"low","summary":"10字内中文概括"}]}

待判定条目：
{items}"""


@dataclass(frozen=True)
class TriageResult:
    """LLM 对单条标题的分类结论。"""

    index: int
    event_class: str
    severity: str = "low"
    summary: str = ""

    @property
    def is_regulatory(self) -> bool:
        return self.event_class == "regulatory"


def _severity_to_level(severity: str) -> RiskLevel:
    """LLM 的 severity → 确定性的等级。映射写死在代码里，不由模型定级。"""
    return {
        "high": RiskLevel.CRITICAL,
        "mid": RiskLevel.HIGH,
        "low": RiskLevel.WARNING,
    }.get((severity or "").strip().lower(), RiskLevel.WARNING)


def _resolve_llm_endpoint(config: Any) -> Optional[Tuple[str, str, str]]:
    """解析 LLM 端点，返回 ``(model, api_key, api_base)``。

    必须走 ``config.llm_channels`` 而不是 ``config.litellm_model``：后者是
    legacy env 兜底，模型名可能形如 ``openai/MiniMax-M3`` 却**不带** base_url 和
    api_key，直接用会打到 OpenAI 官方端点并报 "Missing credentials"。
    真实凭证在通道配置里。
    """
    for channel in list(getattr(config, "llm_channels", None) or []):
        if not isinstance(channel, dict) or not channel.get("enabled", True):
            continue
        models = [m for m in (channel.get("models") or []) if str(m).strip()]
        if not models:
            continue
        # api_keys 既可能是 list 也可能是逗号分隔字符串，两种都要认
        raw_keys = channel.get("api_keys")
        if isinstance(raw_keys, str):
            keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        elif isinstance(raw_keys, (list, tuple)):
            keys = [str(k).strip() for k in raw_keys if str(k).strip()]
        else:
            keys = []
        api_key = keys[0] if keys else ""
        base = str(channel.get("base_url") or "").strip()
        if not api_key and not base:
            # 既无 key 也无 base：多半是本地 ollama 之类，但它仍需要 base
            continue
        return str(models[0]).strip(), api_key, base
    return None


def _llm_json(messages: List[Dict[str, str]], *, timeout: int = 90,
              max_tokens: int = 4000) -> Optional[Dict[str, Any]]:
    """调用 DSA 现有的 litellm 链路拿 JSON。失败返回 None，绝不抛异常。"""
    try:
        import litellm
        from src.config import get_config
        from src.llm.errors import call_litellm_with_param_recovery
        from src.llm.generation_params import apply_litellm_generation_params
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM 依赖不可用，跳过兜底判定: %s", exc)
        return None

    try:
        endpoint = _resolve_llm_endpoint(get_config())
        if endpoint is None:
            logger.info("未配置可用的 LLM 通道，跳过兜底判定")
            return None
        model, api_key, base = endpoint
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "timeout": float(timeout),
            "num_retries": 0,
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
        }
        if api_key:
            kwargs["api_key"] = api_key
        if base:
            kwargs["api_base"] = base
        kwargs = apply_litellm_generation_params(
            kwargs, model=model, temperature=0.0
        )
        response = call_litellm_with_param_recovery(
            lambda request_kwargs: litellm.completion(**request_kwargs),
            model=model,
            call_kwargs=kwargs,
        )
        content = response["choices"][0]["message"]["content"] or ""
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            return None
        return json.loads(content[start:end + 1])
    except Exception as exc:  # noqa: BLE001 - 兜底判定失败不得影响主流程
        logger.warning("LLM 兜底判定失败，跳过: %s", exc)
        return None


def llm_triage(
    items: Sequence[Tuple[int, str]], *, batch_size: int = 15
) -> Dict[int, TriageResult]:
    """批量判定一批 ``(index, title)``，返回 ``{index: TriageResult}``。

    ``items`` 里的 index 由调用方给出，贯穿整个判定过程；返回值也用它做键，
    这样即使某一批解析失败，其它批的结果依然可用。
    """
    if not items:
        return {}
    results: Dict[int, TriageResult] = {}
    for offset in range(0, len(items), batch_size):
        chunk = items[offset:offset + batch_size]
        payload = "\n".join(f"{i}. {t[:160]}" for i, t in chunk)
        data = _llm_json(
            [{"role": "user", "content": _TRIAGE_PROMPT.replace("{items}", payload)}]
        )
        if not isinstance(data, dict):
            continue
        for row in data.get("items") or []:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("i"))
            except (TypeError, ValueError):
                continue
            klass = str(row.get("class") or "").strip().lower()
            if klass not in _LLM_CLASSES:
                continue
            results[idx] = TriageResult(
                index=idx,
                event_class=klass,
                severity=str(row.get("severity") or "low").strip().lower(),
                summary=str(row.get("summary") or "").strip()[:60],
            )
    logger.info("LLM 兜底判定：送 %d 条，判定 %d 条", len(items), len(results))
    return results


def _llm_rule(result: TriageResult, source_type: SourceType) -> RiskRule:
    """把 LLM 结论转成一条确定性规则，参与常规结算。

    ``source_type`` **必须**从原公告继承，不能用 dataclass 默认值。默认值是
    EXCHANGE_FILING，若沿用默认，舆情来源的结论会绕过
    :data:`CEILING_BY_SOURCE` 直接冲到「致命」—— 这正是封顶机制要防的漏洞，
    已有回归测试锁定。
    """
    return RiskRule(
        keyword=f"llm:{result.event_class}",
        level=_severity_to_level(result.severity),
        category=f"llm-{result.event_class}",
        polarity=AGGRAVATING,
        ttl_days=180,
        note=result.summary or "LLM 兜底判定",
        source_type=source_type,
    )


# ---------------------------------------------------------------------------
# 覆盖判定
# ---------------------------------------------------------------------------

def _is_uncovered(code: str, name: str) -> str:
    """返回未覆盖原因；空字符串表示已覆盖。

    覆盖判定按"有没有可用数据源"，不按市场。港股没有交易所公告接口，
    但有舆情通道，因此**不算未覆盖**——它会以 NEWS 身份进入报告的第二段，
    只是拿不到交易所公告级别的确定性结论。
    """
    text = str(code or "").strip()
    bare = text.split(".")[0] if "." in text else text
    # 纯 6 位数字 → A 股；带 .HK / .US 后缀 → 港股 / 美股
    if len(bare) == 6 and bare.isdigit():
        if bare.startswith(_UNCOVERED_CODE_PREFIXES):
            return "基金 / ETF / LOF / 可转债，不适用公告排雷"
        upper = (name or "").upper()
        for hint in _UNCOVERED_NAME_HINTS:
            if hint in upper:
                return f"疑似基金 / ETF / LOF（名称含「{hint}」），不适用公告排雷"
        return ""
    if text.upper().endswith((".HK", ".US")):
        return ""  # 走舆情通道
    return "无法识别的代码格式，且无舆情通道可用"


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------

def _decayed_level(event: RiskEvent, rule_ttl: int) -> RiskLevel:
    """按事件年龄对等级做时效衰减。"""
    if event.age_days <= rule_ttl:
        return event.level
    if event.age_days <= rule_ttl * 2:
        # 降一级地板，不会跌破 CLEAR
        return max(RiskLevel(int(event.level) - 1), RiskLevel.CLEAR)
    return RiskLevel.CLEAR


def _build_events(
    announcements: Sequence[Announcement],
    as_of: date,
    triage: Optional[Dict[Tuple[str, str], TriageResult]] = None,
) -> List[RiskEvent]:
    """把公告转成命中规则的事件。

    ``triage`` 是该股单只的 LLM 兜底判定结果，键为 ``(标题, 公告日期)``。
    只有"规则没命中、但广撒网预筛认为值得问模型"的公告才会去查这张表。
    """
    triage = triage or {}
    events: List[RiskEvent] = []
    for ann in announcements:
        try:
            publish = date.fromisoformat(ann.publish_date)
            age = (as_of - publish).days
        except ValueError:
            age = 0
        rules = match_rules(ann.title, ann.source_type)
        if not rules:
            verdict = triage.get((ann.title, ann.publish_date))
            if verdict is not None and verdict.is_regulatory:
                rules = [_llm_rule(verdict, ann.source_type)]
        for rule in rules:
            events.append(
                RiskEvent(
                    rule=rule,
                    title=ann.title,
                    publish_date=ann.publish_date,
                    url=ann.url,
                    age_days=max(age, 0),
                )
            )
    # 严重级优先，同级按新优先
    events.sort(key=lambda e: (int(e.level), e.publish_date), reverse=True)
    return events


def collect_broad_scan_candidates(
    announcements: Dict[str, List[Announcement]],
) -> List[Announcement]:
    """挑出需要 LLM 兜底判定的公告：规则未命中，但广撒网预筛认为值得问模型。"""
    candidates: List[Announcement] = []
    for code in sorted(announcements):
        for ann in announcements[code]:
            if match_rules(ann.title, ann.source_type):
                continue
            if needs_broad_scan(ann.title):
                candidates.append(ann)
    return candidates


def triage_announcements(
    announcements: Dict[str, List[Announcement]],
    *,
    batch_size: int = 15,
) -> Dict[str, Dict[Tuple[str, str], TriageResult]]:
    """对广撒网命中的公告跑一次 LLM 兜底判定。

    返回 ``{代码: {(标题, 日期): TriageResult}}``，供
    :func:`evaluate_holdings` 逐股传入。**任何失败都返回空表** ——
    主判定流程照常走，只是少一层兜底。
    """
    candidates = collect_broad_scan_candidates(announcements)
    if not candidates:
        return {}
    # **必须带上公司标识**。定向 query 塞了 20 个负面词，捞回来的是"含这些词
    # 的文章"，里面混着别家公司的新闻（实测华特气体那条命中的是
    # 「江淮汽车排放造假被罚1.7亿元」）。只给模型看标题，它无从判断这条
    # 讲的是不是这只票 —— 给了名字和代码，模型可以直接判相关性。
    payload = [
        (index, f"{ann.name}({ann.code}) {ann.title}")
        for index, ann in enumerate(candidates)
    ]
    results = llm_triage(payload, batch_size=batch_size)
    table: Dict[str, Dict[Tuple[str, str], TriageResult]] = {}
    for index, ann in enumerate(candidates):
        verdict = results.get(index)
        if verdict is None:
            continue
        table.setdefault(ann.code, {})[(ann.title, ann.publish_date)] = verdict
    return table


@dataclass(frozen=True)
class CategoryVerdict:
    """单个风险类别（立案 / 处罚 / 退市）的净效果。"""

    category: str
    level: RiskLevel
    top_event: Optional[RiskEvent]
    reasons: List[str] = field(default_factory=list)


def _settle_categories(
    events: Sequence[RiskEvent], *, mitigation_tiers: int
) -> List[CategoryVerdict]:
    """按类别分别结算风险，再由调用方取最高者。

    为什么要分类结算
    ----------------
    沃格光电(603773) 的真实生命周期是：立案(5/15) → 行政处罚事先告知书(8/8)
    → 行政处罚决定书(8/22)。若做全局取最大值 + 全局降级，要么被"拟处罚"这条
    已经过期的公告继续挂在头上（市场早已消化），要么一降级把仍未结案的
    立案也一起降掉 —— 后者会漏掉真正的持续风险。

    因此按类别独立结算：
      * 立案类别：立案告知书 HIGH，无结案 → 保持 HIGH
      * 处罚类别：事先告知书 HIGH 被 8/22 的决定书结算 → WARNING
    股票最终等级 = max(各类别)，主因 = 等级最高类别的 top_event，
    于是报告指向"立案未结案"，这才是交易上真正要盯的那条线。
    """
    aggravating = [e for e in events if e.is_aggravating]
    mitigations = [e for e in events if e.rule.polarity == MITIGATING]

    # 没有恶化事件时，仍要用"只出现缓解事件"的类别来记录既成事实
    # （例如仅有行政处罚决定书 = 已被罚，应为 WARNING 而非 CLEAR）
    touched = {e.rule.category for e in aggravating}
    for event in mitigations:
        touched.update(event.rule.mitigates_list or [event.rule.category])

    results: List[CategoryVerdict] = []
    for category in sorted(touched):
        own = [e for e in aggravating if e.rule.category == category]
        if not own:
            # 只剩缓解事件：以其自身等级作为既成事实
            settled = [
                e
                for e in mitigations
                if category in (e.rule.mitigates_list or [e.rule.category])
            ]
            if not settled:
                continue
            newest = max(settled, key=lambda e: e.publish_date)
            results.append(
                CategoryVerdict(
                    category=category,
                    level=newest.level,
                    top_event=newest,
                    reasons=[f"仅有「{newest.rule.keyword}」，属既成事实"],
                )
            )
            continue

        best: Optional[RiskEvent] = None
        best_level = RiskLevel.CLEAR
        for event in own:
            level = _decayed_level(event, event.rule.ttl_days)
            if level > best_level or (
                level == best_level
                and best is not None
                and event.publish_date > best.publish_date
            ):
                best, best_level = event, level

        if best is None:
            continue

        level = best_level
        reasons: List[str] = []
        cutoff = best.publish_date
        newer = [
            e
            for e in mitigations
            if category in (e.rule.mitigates_list or [e.rule.category])
            and e.publish_date >= cutoff
        ]
        if newer:
            keywords = "、".join(sorted({e.rule.keyword for e in newer}))
            floor = max(
                RiskLevel(int(level) - max(mitigation_tiers, 0)),
                RiskLevel.WATCH,
            )
            # 缓解事件自身等级作为下限：只有决定书而无告知书时仍记 WARNING
            level = max(floor, max(e.level for e in newer), RiskLevel.WATCH)
            reasons.append(f"「{keywords}」晚于最高级事件，{category}类别下调 {mitigation_tiers} 级")

        if best.age_days > best.rule.ttl_days:
            reasons.append(
                f"事件已发生 {best.age_days} 天（TTL {best.rule.ttl_days} 天），等级已衰减"
            )

        results.append(
            CategoryVerdict(
                category=category, level=level, top_event=best, reasons=reasons
            )
        )
    # 严重类别排前面，报告里第一行就是主因

    def _cat_sort_key(c: CategoryVerdict) -> tuple[int, int, str]:
        stamp = 0
        if c.top_event is not None:
            try:
                stamp = int(c.top_event.publish_date.replace("-", ""))
            except ValueError:
                stamp = 0
        return (-int(c.level), -stamp, c.category)

    results.sort(key=_cat_sort_key)
    return results


def evaluate_stock(
    code: str,
    name: str,
    announcements: Sequence[Announcement],
    *,
    as_of: date,
    mitigation_tiers: int = 2,
    st_floor: RiskLevel = RiskLevel.WARNING,
    triage: Optional[Dict[Any, TriageResult]] = None,
) -> RiskVerdict:
    """对单只持仓做完整判定。"""
    all_events = _build_events(announcements, as_of, triage)

    # 进风险阶梯的是「监管」与「财务异常」两类：
    #   监管  —— 来自公告或舆情
    #   财务  —— 来自长桥 corp_action 的财报口径（确定性数据，不受舆情封顶）
    # 其余（股东行为 / 公司运作 / 利好）单独收集，报告里另起段落。
    # 财务事件**不能**进 other_events —— 它已经进了阶梯，两边都放会重复。
    in_ladder = (EventClass.REGULATORY, EventClass.FUNDAMENTAL)
    events = [e for e in all_events if e.event_class in in_ladder]
    other_events = [e for e in all_events if e.event_class not in in_ladder]

    verdict = RiskVerdict(
        code=code,
        name=name,
        level=RiskLevel.CLEAR,
        events=events,
        other_events=other_events,
        announcement_count=len(announcements),
    )
    # 热度：窗口内风险类公告的条数。这条信息任何单条公告里都看不到 ——
    # 单条"风险提示公告"只说明有事，20 条密集出现说明市场关注度在飙升。
    verdict.heat = sum(1 for e in events if e.is_aggravating)
    verdict.is_hot = verdict.heat >= HEAT_ALERT_THRESHOLD

    categories = _settle_categories(events, mitigation_tiers=mitigation_tiers)
    verdict.categories = categories

    if not categories:
        verdict.is_st = is_st_name(name)
        if verdict.is_st:
            verdict.level = st_floor
            verdict.downgrade_reason = (
                f"窗口内无风险公告事件，但简称带风险警示标识，按 {st_floor.label} 档处理"
            )
        return verdict

    # 主因选取：等级相同时，**交易所公告优先于舆情**。
    # 舆情条目常常缺日期（被按"今天"处理），若只按日期排序，一条昨天的
    # 档案页新闻会顶掉今天刚发布的真实风险提示公告，让报告指向最不可靠的
    # 那条证据。
    def _primary_key(c: CategoryVerdict) -> tuple[int, int, str]:
        event = c.top_event
        is_filing = 1 if (event and event.source_type is SourceType.EXCHANGE_FILING) else 0
        return (int(c.level), is_filing, event.publish_date if event else "")

    worst = max(categories, key=_primary_key)
    verdict.level = worst.level
    verdict.top_event = worst.top_event

    reasons: List[str] = []
    if len(categories) > 1:
        others = "、".join(
            f"{c.category}{c.level.label}" for c in categories if c is not worst
        )
        reasons.append(f"其他类别：{others}")
    reasons.extend(worst.reasons)

    # 确定性封顶：按主因事件的信息源性质硬性截断。
    # 放在最后一步执行，因此无论前面结算出多高都拦得住。
    if worst.top_event is not None:
        source = worst.top_event.source_type
        ceiling = CEILING_BY_SOURCE[source]
        if verdict.level > ceiling:
            reasons.append(
                f"主因来自{'舆情检索' if source is SourceType.NEWS else '公告'}，"
                f"按信息源性质封顶在「{ceiling.label}」档"
            )
            verdict.level = ceiling

    verdict.is_st = is_st_name(name)
    if verdict.is_st and verdict.level < st_floor:
        verdict.level = st_floor
        reasons.append(f"简称带风险警示标识，抬升至 {st_floor.label} 档")
    verdict.downgrade_reason = "；".join(r for r in reasons if r)
    return verdict


def split_coverage(
    portfolio: Dict[str, Any],
) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    """把持仓拆成"已覆盖"和"未覆盖"两组。

    抓取阶段就应调用它，避免为港股/基金白跑一次巨潮请求。
    返回 ``([(code, name), ...], [{code, name, reason}, ...])``。
    """
    positions = portfolio.get("positions")
    if not isinstance(positions, list):
        # 不能用 `or []`：空 dict / 空字符串是 falsy，会被悄悄当成空持仓通过
        raise ValueError("portfolio.positions must be a list")

    covered: List[tuple[str, str]] = []
    uncovered: List[Dict[str, str]] = []
    seen: set[str] = set()
    for raw in positions:
        if not isinstance(raw, dict):
            logger.warning("跳过非 dict 持仓条目: %r", raw)
            continue
        code = str(raw.get("code") or "").strip()
        name = str(raw.get("name") or code).strip()
        if not code or code in seen:
            continue
        seen.add(code)
        reason = _is_uncovered(code, name)
        if reason:
            uncovered.append({"code": code, "name": name, "reason": reason})
        else:
            covered.append((code, name))
    return covered, uncovered


def evaluate_holdings(
    portfolio: Dict[str, Any],
    announcements: Dict[str, List[Announcement]],
    *,
    as_of: Optional[date] = None,
    window_start: Optional[date] = None,
    window_end: Optional[date] = None,
    mitigation_tiers: int = 2,
    st_floor: RiskLevel = RiskLevel.WARNING,
    triage: Optional[Dict[Any, TriageResult]] = None,
    channel_errors: Optional[List[str]] = None,
) -> RiskReport:
    """对整个持仓清单做排雷判定。

    ``portfolio`` 需含顶层 ``positions`` 数组（与 ``src.position_drawdown``
    使用的是同一个 ``data/portfolio.json`` 契约）。

    ``channel_errors`` 是抓取阶段记录的**整条通道故障**（如长桥凭证缺失）。
    必须显式传进来落到报告里：一条通道整条挂掉时，受影响的标的会显示
    「无风险事件」—— 这正是"把没查到说成没事"这个最危险的失败模式，
    光看逐只结果发现不了。
    """
    as_of_date = as_of or date.today()
    covered, report_uncovered = split_coverage(portfolio)

    report = RiskReport(
        as_of=as_of_date.isoformat(),
        window_start=(window_start or as_of_date).isoformat(),
        window_end=(window_end or as_of_date).isoformat(),
        config={
            "mitigation_tiers": mitigation_tiers,
            "st_floor": st_floor.label,
        },
        uncovered=report_uncovered,
        errors=list(channel_errors or []),
    )

    for code, name in covered:
        rows = announcements.get(code)
        if rows is None:
            # 抓取阶段就没拿到这只票（抓取失败）
            verdict = RiskVerdict(
                code=code,
                name=name,
                level=RiskLevel.CLEAR,
                error="未获取到公告数据（抓取失败）",
            )
            report.verdicts.append(verdict)
            report.errors.append(f"{code} {name}: 未获取到公告数据")
            continue
        # 零公告是正常情况：窗口内公司没发公告
        display_name = name or (rows[0].name if rows else code)
        report.verdicts.append(
            evaluate_stock(
                code, display_name, rows, as_of=as_of_date,
                mitigation_tiers=mitigation_tiers, st_floor=st_floor,
                triage=(triage or {}).get(code),
            )
        )

    return report
