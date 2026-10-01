"""风险阶梯与关键词规则。

设计要点（相对旧版排雷脚本的四点修正）
----------------------------------------
1. **分级替代分组**：旧版用互斥 mask 做"立案/结案/处罚/退市/摘帽"六个桶，
   只看"有没有"不看"什么时候"和"先后顺序"，导致结案/摘帽被立案/退市组吃掉。
   这里改为数值风险阶梯 + 状态机：取所有事件的最高等级，缓解事件（结案/摘帽）
   只做降级，不做清空。
2. **事先告知书 ≠ 已处罚**：``行政处罚事先告知书``（拟处罚，在途）判为 HIGH，
   ``行政处罚决定书``（已落地，靴子掉地）判为 WARNING。旧版两者同组，
   统一输出"已收到行政处罚"。
3. **时效衰减**：每条规则带 ``ttl_days``，超过 TTL 后等级降到地板值；
   超过 2 倍 TTL 直接清零。旧版 ``groupby.max()`` 完全不看事件年龄。
4. **ST 不隐藏**：ST 是风险信号本身，由 :func:`is_st_name` 单独识别并提级，
   绝不像旧版那样从所有展示里 ``df[~is_st]`` 掉。

关键词冲突处理
--------------
"申请撤销退市风险警示" 里同时包含缓解词"申请撤销退市风险警示"和恶化词
"退市风险警示"。直接子串匹配会让缓解词反而触发恶化规则，因此匹配后统一做
一次**抑制**：若某条恶化规则的关键词是某条已命中缓解关键词的子串，则丢弃
该恶化规则。取最长匹配，语义上符合"更具体的表述优先"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import List

# 全文检索接口会在命中词外裹 <em> 标签，且标题里常混入全角空格/换行
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[\s　]+")
# 交易所风险警示前缀，出现在简称开头：ST / *ST / S*ST / SST
# 末尾的 (?![A-Za-z]) 用于排除 "STAR科技" 这类以 ST 开头的正常简称。
_ST_RE = re.compile(r"^(?:S\*?|\*)?ST(?![A-Za-z])", re.IGNORECASE)


class RiskLevel(IntEnum):
    """监管风险等级。数值越大越严重，可直接比较大小做升降级。"""

    CLEAR = 0
    WATCH = 1
    WARNING = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return _LEVEL_LABEL[self]

    @property
    def glyph(self) -> str:
        return _LEVEL_GLYPH[self]


_LEVEL_LABEL = {
    RiskLevel.CLEAR: "无",
    RiskLevel.WATCH: "关注",
    RiskLevel.WARNING: "警告",
    RiskLevel.HIGH: "严重",
    RiskLevel.CRITICAL: "致命",
}

_LEVEL_GLYPH = {
    RiskLevel.CLEAR: "⚪",
    RiskLevel.WATCH: "🔵",
    RiskLevel.WARNING: "🟡",
    RiskLevel.HIGH: "🟠",
    RiskLevel.CRITICAL: "🔴",
}


class SourceType(IntEnum):
    """信息源性质 —— 决定定级上限。

    分界不是市场（A 股 / 港股），而是信息本身的可信度：

    * ``EXCHANGE_FILING`` 交易所/证监会正式文件。词汇封闭、日期精确、可枚举，
      允许定到致命档。
    * ``NEWS`` 新闻报道与舆情。词汇开放、日期常缺、可能失实，
      由 :data:`CEILING_BY_SOURCE` 硬性封顶在「警告」。

    刻意不按市场划分：舆情检索对 A 股同样可用，港股公告（若接口打通）
    同样属于 EXCHANGE_FILING。
    """

    EXCHANGE_FILING = 2
    NEWS = 1


class EventClass(str, Enum):
    """事件归类 —— 决定进报告哪一段。"""

    REGULATORY = "regulatory"    # 监管风险，进风险阶梯
    SHAREHOLDER = "shareholder"  # 减持/质押/解禁，不进阶梯
    CORPORATE = "corporate"      # 担保/增发/收购，不进阶梯
    POSITIVE = "positive"        # 利好，不进阶梯，单列一段


#: 广撒网预筛词 —— 只决定"这条要不要送 LLM 判"，本身不产生任何等级。
#:
#: 目的不是精确分类，而是**兜住规则表的盲区**。规则表会永远滞后于
#: 文件类型的多样性（实测就漏过"风险提示公告"），而 LLM 调用要花钱，
#: 不能把 876 条公告全丢进去。先用这张宽口径的网筛一遍，只把可能是风险
#: 事件的送进 LLM。
#:
#: 词表刻意**不含**裸的「监管」「处罚」「风险」——那会误伤
#: "关于最近五年未被证券监管部门和证券交易所采取监管措施或处罚情况的公告"
#: 这类正面公告。
BROAD_SCAN_KEYWORDS: tuple[str, ...] = (
    "立案", "处罚", "警示", "退市", "调查", "诉讼", "问询", "关注函",
    "违规", "异常波动", "风险提示", "冻结", "质押", "担保", "预亏",
    "亏损", "暴雷", "停产", "违约", "召回", "破产", "重整", "接管",
    "减持", "解禁", "做空", "造假", "贪腐", "被查", "失联", "留置",
)

#: 利好规则表 —— 独立于风险阶梯，命中后进报告的「利好」段。
#: 定义在 :data:`FILING_RULES` 之后（需要 :class:`RiskRule` 已就绪）。
#:
#: 为什么要有它：一个只报风险的报告会训练你忽略它。知道持仓里哪只
#: 刚拿到大单、哪只完成回购，本身是有用的信息，也能帮你判断"公司主动
#: 披露"与"被动出事"的对比。

#: 风险信号的热度阈值。同一只票在窗口内命中风险词的公告条数超过它，
#: 说明市场关注度异常抬升 —— 这是一条**任何单条公告里都看不到**的信息。
HEAT_ALERT_THRESHOLD = 8


#: 每种信息源允许达到的最高等级。引擎在结算时强制执行，不依赖调用方自觉。
CEILING_BY_SOURCE: dict[SourceType, RiskLevel] = {
    SourceType.EXCHANGE_FILING: RiskLevel.CRITICAL,
    SourceType.NEWS: RiskLevel.WARNING,
}


class Polarity(str):
    """规则极性标记。"""


# 恶化事件：出现即抬升风险
AGGRAVATING = "aggravating"
# 缓解事件：只做降级，永不清空
MITIGATING = "mitigating"


@dataclass(frozen=True)
class RiskRule:
    """一条关键词规则。

    ``mitigates`` 只对 ``MITIGATING`` 规则有意义：声明它能"结算"哪个类别的
    恶化事件。结算而非全局清零，才能让一只票同时呈现
    "立案未结案（真实持续风险）+ 处罚已落地（靴子掉地）"两个独立事实。
    """

    keyword: str
    level: RiskLevel
    category: str
    polarity: str
    ttl_days: int
    note: str = ""
    mitigates: str = ""
    source_type: SourceType = SourceType.EXCHANGE_FILING
    event_class: EventClass = EventClass.REGULATORY

    @property
    def is_aggravating(self) -> bool:
        return self.polarity == AGGRAVATING

    @property
    def mitigates_list(self) -> List[str]:
        """本规则能结算哪些类别（逗号分隔）。"""
        if not self.mitigates:
            return []
        return [item.strip() for item in self.mitigates.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# 关键词表
#
# ttl_days 含义：该等级在多长时间内"不打折"。超过 1×TTL 降一级地板，
# 超过 2×TTL 清零。退市类给了极长 TTL（10 年），因为这类状态一旦触发
# 在公司退市或摘帽前不会自行消失。
# ---------------------------------------------------------------------------
KEYWORD_RULES: tuple[RiskRule, ...] = (
    # --- 致命：终止上市 / 重大违法退市 ---
    RiskRule(
        "终止上市事先告知书", RiskLevel.CRITICAL, "退市", AGGRAVATING, 3650,
        "已进入终止上市决定前的最后告知阶段",
    ),
    RiskRule(
        "终止上市相关事项监管工作函", RiskLevel.CRITICAL, "退市", AGGRAVATING, 3650,
        "监管已就终止上市事项下发工作函",
    ),
    RiskRule(
        "重大违法强制退市", RiskLevel.CRITICAL, "退市", AGGRAVATING, 3650,
        "触及重大违法强制退市标准",
    ),
    # --- 严重：立案调查 / 拟处罚 ---
    RiskRule(
        "立案告知书", RiskLevel.HIGH, "立案", AGGRAVATING, 540,
        "证监会已立案调查，结论未定",
    ),
    RiskRule(
        "立案调查", RiskLevel.HIGH, "立案", AGGRAVATING, 540,
        "证监会已立案调查，结论未定",
    ),
    RiskRule(
        "行政处罚事先告知书", RiskLevel.HIGH, "处罚", AGGRAVATING, 180,
        "拟处罚，处罚在途（与已作出的处罚决定书性质不同）",
    ),
    # --- 警告：已落地处罚 / 退市风险警示 ---
    # 处罚决定书既是"已受处罚"这一既成事实（WARNING），也把同类别里在途的
    # "事先告知书"（HIGH）结算掉。结算在 evaluate_stock 里按 mitigates 生效，
    # 这里保持它为 MITIGATING 以便同时承担"事实记录"和"结算"两个角色。
    RiskRule(
        "行政处罚决定书", RiskLevel.WARNING, "处罚", MITIGATING, 365,
        "处罚已落地，靴子掉地；在途的拟处罚同时结算", mitigates="处罚",
    ),
    RiskRule(
        "行政处罚", RiskLevel.WARNING, "处罚", AGGRAVATING, 365,
        "处罚类公告兜底",
    ),
    RiskRule(
        "退市风险警示", RiskLevel.WARNING, "退市", AGGRAVATING, 365,
        "已被实施退市风险警示",
    ),
    RiskRule(
        "其他风险警示", RiskLevel.WARNING, "退市", AGGRAVATING, 365,
        "被实施其他风险警示（通常带 ST 标识）",
    ),
    RiskRule(
        "立案", RiskLevel.WARNING, "立案", AGGRAVATING, 540,
        "立案类公告兜底",
    ),
    # --- 交易风险提示（2026-10-01 补）---
    # 这几条来自对 9 只真实持仓 12 个月、876 条公告的实测缺口：
    # 巨潮全部抓到了，但原 12 词表一条都没命中。沃格光电 2026-05-12 就在发
    # 《风险提示公告》，比规则表能识别出的立案(5-15)还早一个月。
    # 它们是公司主动披露的确定性文件，日期精确、噪音低，不该漏。
    RiskRule(
        "风险提示", RiskLevel.WARNING, "交易", AGGRAVATING, 365,
        "公司主动发布交易风险提示（风险提示公告 / 风险提示性公告）",
    ),
    RiskRule(
        "严重异常波动", RiskLevel.WARNING, "交易", AGGRAVATING, 90,
        "股票交易严重异常波动，公司需说明原因",
    ),
    RiskRule(
        "异常波动", RiskLevel.WATCH, "交易", AGGRAVATING, 90,
        "股票交易异常波动（严重异常波动的兜底）",
    ),
    RiskRule(
        "重大诉讼", RiskLevel.WARNING, "诉讼", AGGRAVATING, 365,
        "公司提起或被诉重大诉讼",
    ),
    RiskRule(
        "提起诉讼", RiskLevel.WARNING, "诉讼", AGGRAVATING, 365,
        "提起重大诉讼",
    ),
    RiskRule(
        "诉讼", RiskLevel.WATCH, "诉讼", AGGRAVATING, 365,
        "诉讼事项公告兜底（含子公司诉讼进展）",
    ),
    # --- 缓解：只结算对应类别，不清空其他类别 ---
    # 结案同时结算"立案"和"退市"：终止上市流程走完并结案，等同该风险线了结。
    RiskRule(
        "结案告知书", RiskLevel.WATCH, "结案", MITIGATING, 180,
        "调查已结案，立案/退市类别降级", mitigates="立案,退市",
    ),
    RiskRule(
        "结案", RiskLevel.WATCH, "结案", MITIGATING, 180,
        "结案类公告兜底", mitigates="立案,退市",
    ),
    RiskRule(
        "申请撤销退市风险警示", RiskLevel.WATCH, "摘帽", MITIGATING, 180,
        "已申请摘帽，等待交易所审核", mitigates="退市",
    ),
    RiskRule(
        "撤销退市风险警示", RiskLevel.WATCH, "摘帽", MITIGATING, 365,
        "风险警示已撤销", mitigates="退市",
    ),
    RiskRule(
        "撤销其他风险警示", RiskLevel.WATCH, "摘帽", MITIGATING, 365,
        "其他风险警示已撤销", mitigates="退市",
    ),
)

# 交易所公告规则表：仅 EXCHANGE_FILING 通道使用，可定级到致命档。
FILING_RULES: tuple[RiskRule, ...] = KEYWORD_RULES

# 利好规则表：确定性命中即定级，不走 LLM。
POSITIVE_RULES: tuple[RiskRule, ...] = (
    RiskRule(
        "回购", RiskLevel.WARNING, "利好", AGGRAVATING, 90,
        "股份回购", event_class=EventClass.POSITIVE,
    ),
    RiskRule(
        "增持", RiskLevel.WATCH, "利好", AGGRAVATING, 90,
        "股东增持", event_class=EventClass.POSITIVE,
    ),
    RiskRule(
        "股权激励", RiskLevel.WATCH, "利好", AGGRAVATING, 365,
        "股权激励计划", event_class=EventClass.POSITIVE,
    ),
    RiskRule(
        "中标", RiskLevel.WATCH, "利好", AGGRAVATING, 90,
        "中标/签署重大合同", event_class=EventClass.POSITIVE,
    ),
    RiskRule(
        "业绩预增", RiskLevel.WARNING, "利好", AGGRAVATING, 180,
        "业绩预增（明确向好）", event_class=EventClass.POSITIVE,
    ),
    # 刻意**不收**「业绩预告」：它方向中性，预告内容可能是预亏。
    # 判断方向需要读正文，属于 LLM 的活，不该由关键词表武断下结论。
    RiskRule(
        "重组", RiskLevel.WATCH, "利好", AGGRAVATING, 365,
        "重大资产重组", event_class=EventClass.POSITIVE,
    ),
    RiskRule(
        "获批", RiskLevel.WATCH, "利好", AGGRAVATING, 180,
        "取得监管批准", event_class=EventClass.POSITIVE,
    ),
)

# 舆情新闻规则表：仅 NEWS 通道使用。
#
# 只收录"硬监管事件"这类即使一句话也能确定性质的词。理由有两条：
#   1. 引擎会把 NEWS 通道的结算结果硬封顶在「警告」，所以这里就算误标成
#      CRITICAL 也不会外溢到「严重/致命」两档。
#   2. 保留这份表作为 LLM 不可用时的兜底 —— 即使 LLM 整个挂掉，
#      盈警/停牌/清盘这类词仍然能靠子串匹配报出来。
#
# 港股标题为繁体（劍橋科技 / 速騰聚創），关键词本身在简体与繁体中字形相同，
# 不需要维护两套。
NEWS_RULES: tuple[RiskRule, ...] = (
    RiskRule(
        "盈警", RiskLevel.WARNING, "港股业绩", AGGRAVATING, 180,
        "公司发布盈利警告", source_type=SourceType.NEWS,
    ),
    RiskRule(
        "盈利警告", RiskLevel.WARNING, "港股业绩", AGGRAVATING, 180,
        "盈利警告（盈警全称）", source_type=SourceType.NEWS,
    ),
    RiskRule(
        "核数师", RiskLevel.WARNING, "港股财报", AGGRAVATING, 365,
        "核数师（审计师）出具保留意见 / 无法表示意见",
        source_type=SourceType.NEWS,
    ),
    RiskRule(
        "保留意见", RiskLevel.WARNING, "港股财报", AGGRAVATING, 365,
        "审计报告保留意见", source_type=SourceType.NEWS,
    ),
    RiskRule(
        "无法表示意见", RiskLevel.WARNING, "港股财报", AGGRAVATING, 365,
        "审计报告无法表示意见", source_type=SourceType.NEWS,
    ),
    RiskRule(
        "停牌", RiskLevel.WARNING, "港股交易", AGGRAVATING, 90,
        "港股停牌", source_type=SourceType.NEWS,
    ),
    RiskRule(
        "清盘", RiskLevel.WARNING, "港股交易", AGGRAVATING, 365,
        "面临清盘风险", source_type=SourceType.NEWS,
    ),
    RiskRule(
        "除牌", RiskLevel.WARNING, "港股交易", AGGRAVATING, 365,
        "面临除牌风险", source_type=SourceType.NEWS,
    ),
    RiskRule(
        "证监会调查", RiskLevel.WARNING, "港股监管", AGGRAVATING, 365,
        "香港证监会介入调查", source_type=SourceType.NEWS,
    ),
    RiskRule(
        "证监会", RiskLevel.WATCH, "港股监管", AGGRAVATING, 180,
        "出现「证监会」字样（兜底，可能指 SFC 或内地证监会）",
        source_type=SourceType.NEWS,
    ),
    # --- 繁体变体 ---
    # 港股新闻标题基本全是繁体（劍橋科技、證監會、調查）。字形不同的词
    # 必须各写一条；字形相同的（盈警 / 停牌 / 核数师）上面已覆盖。
    RiskRule(
        "證監會", RiskLevel.WARNING, "港股监管", AGGRAVATING, 365,
        "香港证监会介入（證監會 繁体）", source_type=SourceType.NEWS,
    ),
    RiskRule(
        "证监会调查", RiskLevel.WARNING, "港股监管", AGGRAVATING, 365,
        "内地证监会调查（简体）", source_type=SourceType.NEWS,
    ),
    RiskRule(
        "調查", RiskLevel.WATCH, "港股监管", AGGRAVATING, 365,
        "出现「調查」字样（繁体兜底）", source_type=SourceType.NEWS,
    ),
    RiskRule(
        "清盤", RiskLevel.WARNING, "港股交易", AGGRAVATING, 365,
        "清盤（繁体）", source_type=SourceType.NEWS,
    ),
    RiskRule(
        "核數師", RiskLevel.WARNING, "港股财报", AGGRAVATING, 365,
        "核數師（繁体）", source_type=SourceType.NEWS,
    ),
    RiskRule(
        "停牌", RiskLevel.WARNING, "港股交易", AGGRAVATING, 90,
        "港股停牌（港台用词，两体同形）", source_type=SourceType.NEWS,
    ),
)

# 两张表的合并视图，供 :func:`match_rules` 按 source_type 过滤使用。
ALL_RULES: tuple[RiskRule, ...] = FILING_RULES + NEWS_RULES


def rules_for(source_type: SourceType) -> tuple[RiskRule, ...]:
    """取某个信息源对应的规则表。

    利好规则对两种来源都生效：公告里的回购公告要认，舆情里捞到的回购消息
    同样要认 —— 否则港股（没有公告通道）会完全看不到利好。
    """
    if source_type is SourceType.NEWS:
        return NEWS_RULES + POSITIVE_RULES
    return FILING_RULES + POSITIVE_RULES


def needs_broad_scan(title: str) -> bool:
    """这条标题是否值得送 LLM 兜底判定。

    命中广撒网预筛词即返回 True。它**不判断**这条是不是风险，只回答
    "值不值得花一次调用问模型"。
    """
    text = normalize_title(title)
    if not text:
        return False
    return any(word in text for word in BROAD_SCAN_KEYWORDS)


# 兜底正则：关键词表覆盖不到的高危表述
_EXTRA_PATTERNS: tuple[tuple[re.Pattern[str], RiskRule], ...] = (
    (
        re.compile(r"被(中国证监会|证监会).{0,6}立案"),
        RiskRule("证监会立案", RiskLevel.HIGH, "立案", AGGRAVATING, 540, "正则兜底"),
    ),
    (
        re.compile(r"终止上市"),
        RiskRule("终止上市", RiskLevel.CRITICAL, "退市", AGGRAVATING, 3650, "正则兜底"),
    ),
)


def normalize_title(title: str) -> str:
    """清洗公告标题：去 HTML 标签、压掉空白，便于子串匹配。"""
    text = _TAG_RE.sub("", str(title or ""))
    return _WS_RE.sub("", text)


def is_st_name(name: str) -> bool:
    """判断是否为风险警示股票（*ST / ST / S*ST / SST）。

    ST 是监管风险的**结果**而非原因，因此在 :mod:`src.risk_scan.engine` 里
    作为独立信号提级，而不是像旧版那样把这类票从报告里整行删掉。
    """
    return bool(_ST_RE.search(str(name or "").strip()))


def match_rules(
    title: str, source_type: SourceType = SourceType.EXCHANGE_FILING
) -> List[RiskRule]:
    """匹配标题命中的规则（按信息源过滤 + 冲突抑制）。

    ``source_type`` 决定查哪张表：公告通道查 ``FILING_RULES``，
    舆情通道查 ``NEWS_RULES``。两张表词条互不重叠，因此一条新闻不会被
    A 股公告规则误命中。

    抑制规则：若某条恶化规则的关键词是某条已命中缓解关键词的子串，则丢弃。
    这样"申请撤销退市风险警示"只会命中缓解词，不会同时触发"退市风险警示"。
    """
    text = normalize_title(title)
    if not text:
        return []

    table = rules_for(source_type)

    hit: List[RiskRule] = []
    for rule in table:
        if rule.keyword in text:
            hit.append(rule)

    if not any(r.polarity == MITIGATING for r in hit):
        # 没有缓解词时才允许正则兜底，避免"申请撤销"这类表述被兜底规则打回恶化级
        for pattern, rule in _EXTRA_PATTERNS:
            if rule.source_type is source_type and pattern.search(text):
                hit.append(rule)

    mitigations = [r for r in hit if r.polarity == MITIGATING]
    if not mitigations:
        return hit

    mitigation_keywords = [r.keyword for r in mitigations]
    suppressed = {
        r.keyword
        for r in hit
        if r.is_aggravating
        and any(r.keyword in mk and r.keyword != mk for mk in mitigation_keywords)
    }
    return [r for r in hit if r.keyword not in suppressed]
