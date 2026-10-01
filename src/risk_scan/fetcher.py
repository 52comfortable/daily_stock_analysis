"""公告抓取层（巨潮资讯）。

按持仓逐只拉取全量公告，而不是旧版脚本那样按关键词全市场扫。
持仓只有十几只时，逐只查询的优势很实在：

* **一次查询拿全部公告**，关键词在本地匹配。旧版 12 个关键词各扫一次全市场，
  既慢又会漏掉关键词表没覆盖的表述。
* 巨潮返回的 ``公告链接`` 字段自带原文 URL，预警可一键验证。
* 只有"确实一条公告都没有"时才回落到巨潮 raw 接口反查，避免 akshare 静默
  失败被误判成"这只票没有雷"。

覆盖范围：仅 A 股。港股在巨潮无数据，基金/ETF/LOF 不适用本判定。
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .rules import SourceType

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[\s　]+")

CNINFO_SEARCH_URL = "https://www.cninfo.com.cn/new/fulltextSearch/full"


@dataclass(frozen=True)
class Announcement:
    """一条公告或一条舆情新闻。

    ``source_type`` 决定它走哪张规则表、以及结算后能封顶到哪一档，
    由 :func:`src.risk_scan.engine.evaluate_stock` 强制执行。
    """

    code: str
    name: str
    title: str
    publish_date: str  # YYYY-MM-DD
    url: str = ""
    source: str = "cninfo_akshare"
    source_type: SourceType = SourceType.EXCHANGE_FILING

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "title": self.title,
            "publish_date": self.publish_date,
            "url": self.url,
            "source": self.source,
            "source_type": self.source.name,
        }


class CninfoSession:
    """巨潮 raw 检索会话（仅在 akshare 零结果时做反查）。"""

    def __init__(self) -> None:
        import requests

        self._requests = requests
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                ),
                "Accept": "application/json",
                "Referer": "https://www.cninfo.com.cn/",
            }
        )

    def search(self, keyword: str, start: str, end: str, page_size: int = 50) -> List[Dict[str, Any]]:
        """按关键词全市场检索，返回原始 announcement dict 列表。"""
        try:
            resp = self._session.get(
                CNINFO_SEARCH_URL,
                params={
                    "searchkey": keyword,
                    "sdate": start,
                    "edate": end,
                    "isfulltext": "false",
                    "sortName": "pubdate",
                    "sortType": "desc",
                    "pageNum": 1,
                    "pageSize": page_size,
                },
                timeout=15,
            )
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - 反查失败不应影响主流程
            logger.warning("巨潮 raw 反查失败 keyword=%r: %s", keyword, exc)
            return []
        return list(data.get("announcements") or [])


def _clean(text: Any) -> str:
    return _WS_RE.sub("", _TAG_RE.sub("", str(text or "")))


def _to_iso(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    if " " in text:
        text = text.split(" ", 1)[0]
    return text


def _fetch_one_via_akshare(
    code: str, start_compact: str, end_compact: str
) -> Optional[List[Announcement]]:
    """用 akshare 拉单只股票的全部公告。

    **失败返回 None，成功但无数据返回 []**。两者必须区分：

    * ``[]``  → 这家公司窗口内确实没发公告
    * ``None`` → 抓取失败（网络/限流/代码错误）

    混为一谈会让一次网络抖动被报成「⚪ 无风险」，这是排雷最危险的
    失败模式 —— 用户看到的是"没事"，实际是"没查到"。
    """
    try:
        import akshare as ak
    except Exception as exc:  # noqa: BLE001
        logger.error("akshare 不可用: %s", exc)
        return None

    try:
        df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=code,
            market="沪深京",
            keyword="",
            start_date=start_compact,
            end_date=end_compact,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("巨潮抓取失败 %s: %s", code, exc)
        return None

    if df is None or getattr(df, "empty", True):
        return []

    # akshare 返回列：代码 / 简称 / 公告标题 / 公告时间 / 公告链接
    results: List[Announcement] = []
    for row in df.itertuples(index=False):
        publish_date = _to_iso(getattr(row, "公告时间", ""))
        if not publish_date:
            continue
        results.append(
            Announcement(
                code=code,
                name=_clean(getattr(row, "简称", "")),
                title=_clean(getattr(row, "公告标题", "")),
                publish_date=publish_date,
                url=str(getattr(row, "公告链接", "") or "").strip(),
            )
        )
    return results


def _crosscheck_via_raw(
    session: CninfoSession, code: str, name: str, start: str, end: str
) -> int:
    """零结果反查：用股票名做巨潮全文检索，确认不是抓取失败。

    返回命中的公告条数，仅用于日志与告警，不直接写入结果。
    """
    probe = (name or code).strip()
    if not probe:
        return 0
    try:
        rows = session.search(probe, start, end, page_size=50)
    except Exception as exc:  # noqa: BLE001
        logger.warning("巨潮 raw 反查异常 %s/%s: %s", code, probe, exc)
        return 0
    hits = [
        r
        for r in rows
        if str(r.get("secCode", "")).strip() == code
    ]
    if hits:
        logger.warning(
            "⚠ %s(%s) akshare 返回 0 条，但巨潮 raw 检索到 %d 条 —— 疑似数据源不一致",
            name or code,
            code,
            len(hits),
        )
    return len(hits)


def fetch_announcements(
    codes: Iterable[str],
    *,
    names: Optional[Dict[str, str]] = None,
    start: date,
    end: date,
    delay: float = 0.35,
    crosscheck: bool = True,
    ah_map: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Announcement]]:
    """逐只抓取公告，返回 ``{code: [Announcement, ...]}``。

    参数
    ----
    codes:
        6 位 A 股代码列表。
    start / end:
        闭区间日期。巨潮 akshare 接口要求 ``YYYYMMDD`` 紧凑格式。
    delay:
        每只之间的间隔秒数，避免连续请求触发风控。
    crosscheck:
        是否对"零结果"的股票做一次巨潮 raw 反查。
    """
    start_compact = start.strftime("%Y%m%d")
    end_compact = end.strftime("%Y%m%d")
    start_dash = start.isoformat()
    end_dash = end.isoformat()

    if not any(str(c or "").strip() for c in codes):
        return {}

    session: Optional[CninfoSession] = None
    results: Dict[str, Optional[List[Announcement]]] = {}
    empty_codes: List[str] = []
    found_names: Dict[str, str] = {}
    skipped: List[str] = []
    failed: List[str] = []
    # 优先用调用方给的（来自缓存的）映射；没给才现算。
    if ah_map is None:
        ah_map = build_ah_map(names)

    # 展开成"实际去查巨潮的代码"，并记住它属于哪只持仓
    target: List[str] = []
    owner: Dict[str, str] = {}      # 巨潮代码 -> 持仓原始代码
    for raw in codes:
        text = str(raw or "").strip()
        if not text:
            continue
        bare = text.split(".")[0] if "." in text else text
        if _is_cn_filing_code(text):
            target.append(text)
            owner[text] = text
            continue
        a_share = ah_map.get(bare.zfill(5))
        if a_share:
            # A+H：用 A 线代码去查巨潮，公告归回原持仓
            target.append(a_share)
            owner[a_share] = text
            logger.info(
                "A+H 映射：%s 改查 A 线 %s 取公告", text, a_share
            )
        else:
            # 港股/美股且非 A+H。巨潮只收 A 股，强行传进去必然 KeyError，
            # 白等一轮还在日志里刷 WARNING。直接跳过，由舆情通道覆盖。
            skipped.append(text)

    for code in target:
        rows = _fetch_one_via_akshare(code, start_compact, end_compact)
        held = owner.get(code, code)
        if rows is None:
            # 抓取失败：置 None 而不是空列表，让 evaluate_holdings 标成
            # 「未获取到公告数据」而不是「无风险」
            results[held] = None
            failed.append(held)
            if delay:
                time.sleep(delay)
            continue
        # 去重：同一只票同一天可能有多条公告，主键用 (日期, 标题)
        seen = set()
        unique: List[Announcement] = []
        for item in rows:
            key = (item.publish_date, item.title)
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        unique.sort(key=lambda a: a.publish_date, reverse=True)
        # 归回持仓原始代码：A+H 的 A 线公告要挂回港股代码上
        results[held] = [Announcement(
            code=held, name=a.name, title=a.title, publish_date=a.publish_date,
            url=a.url, source=a.source, source_type=a.source_type,
        ) for a in unique]
        if unique:
            found_names[held] = unique[0].name
        else:
            empty_codes.append(held)
        if delay:
            time.sleep(delay)

    if skipped:
        # 关键：必须回填**空列表**而不是留空。
        # 留空的话下游 `announcements.get(code)` 拿到 None，无法区分
        # 「查了但失败」和「这本就不该查（纯港股）」，会把它误报成
        # 「未获取到公告数据（抓取失败）」—— 明明是设计如此，却报成故障。
        # 两者在字典里的区分：None = 查了失败；[] = 查了/不适用，都不是故障。
        for code in skipped:
            results[code] = []
        logger.debug(
            "巨潮通道跳过 %d 个非 A 股代码（由长桥通道覆盖）: %s",
            len(skipped), "、".join(skipped),
        )

    if crosscheck and empty_codes:
        try:
            session = CninfoSession()
        except Exception as exc:  # noqa: BLE001
            logger.warning("巨潮 raw 会话初始化失败，跳过反查: %s", exc)
            session = None
        if session is not None:
            for code in empty_codes:
                _crosscheck_via_raw(
                    session, code, found_names.get(code, code), start_dash, end_dash
                )
                time.sleep(delay)

    return results


def _is_cn_filing_code(code: str) -> bool:
    """是否支持巨潮公告通道。巨潮只收录沪深京 A 股。"""
    text = str(code or "").strip()
    return len(text) == 6 and text.isdigit()


#: A+H 双重上市：港股代码(5 位) -> A 股代码(6 位)。
#:
#: A+H 公司在港股线也受同一套证监会/交易所监管，公告会同步披露；巨潮只收录
#: A 股线，所以查港股时改用它的 A 股代码去取，公告再归回原持仓代码。
#:
#: 映射不靠手工维护：仓库自带的 ``stocks.index.json``（中文名 → 规范代码
#: + 市场）就是全集，用 ``portfolio.json`` 里本来就有的中文名去查即可。
#: 零网络调用、零限流、无需人工维护，也无需缓存（实测全量解析只要 7 ms）。

#: 股票索引的候选路径。按"是否被 git 跟踪"排序 —— CI 只能读到入库的。
#: ``static/`` 被 .gitignore 排除，``apps/dsa-web/public/`` 才是入库的那份。
_STOCK_INDEX_PATHS = (
    "apps/dsa-web/public/stocks.index.json",
    "static/stocks.index.json",
    "data/cache/stocks.index.json",
)
#: 索引记录里「市场」字段表示 A 股的取值
_CN_MARKET = "CN"
#: 记录里规范代码的字段序号 / 中文名字段序号
_IDX_SYMBOL, _IDX_NAME, _IDX_MARKET = 0, 2, 6

_NAME_INDEX: Optional[Dict[str, List[str]]] = None


def _load_cn_name_index() -> Dict[str, List[str]]:
    """构建「A 股中文名 → 规范代码列表」索引。文件缺失返回空表。"""
    global _NAME_INDEX
    if _NAME_INDEX is not None:
        return _NAME_INDEX

    import json

    index: Dict[str, List[str]] = {}
    for rel in _STOCK_INDEX_PATHS:
        path = Path(__file__).resolve().parents[2] / rel
        try:
            if not path.exists():
                continue
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.debug("股票索引不可用 %s: %s", path, exc)
            continue
        for row in rows:
            try:
                if row[_IDX_MARKET] != _CN_MARKET:
                    continue
                name = str(row[_IDX_NAME]).strip()
                symbol = str(row[_IDX_SYMBOL]).strip()
            except (IndexError, TypeError):
                continue
            if name and symbol:
                index.setdefault(name, []).append(symbol)
        if index:
            logger.info("A/H 映射索引已加载：%s（%d 个中文名）", rel, len(index))
            break

    _NAME_INDEX = index
    return index


def build_ah_map(names: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """现算 A+H 映射：``{港股5位码: A股6位裸码}``。两级来源，东财优先。

    **第一级 —— 东财 A/H 官方对照表**：代码对代码的官方配对，最权威，优先用。

    **第二级 —— 本地股票索引**：东财没解出来的港股，按持仓中文名查同名的
    A 股（同一公司在两地通常同名）。零网络调用，实测 7 ms。

    为什么东财在前而不是本地在前：这两级**不是二选一，东财每轮基本都会被
    调用**（持仓里的纯港股永远本地解析不出，必然触发二级），所以换顺序不多
    花任何一次网络请求。同样的代价下，当然用更权威的源打底。本地降级成兜底
    后，恰好补上东财的两个盲区：连不上（代理/限流/改列名）、或表里没这只票。

    两级都拿不到就**不映射**，该港股走长桥通道，不假装已覆盖。任何一级失败
    都静默降级（缺 akshare / 代理拦截 / 限流 / 列名变更 / 空表 / 索引缺失），
    不中断整轮扫描。

    注意映射值必须是**裸 6 位**：akshare 的 ``symbol`` 不接受 ``603083.SH``
    这种带后缀的形式，会 KeyError。
    """
    mapping: Dict[str, str] = {}
    wanted = _hk_codes(names)

    # ── 一级：东财官方对照表 ──
    remote = _fetch_ah_map_eastmoney()
    if remote:
        hits = {code: remote[code] for code in wanted if code in remote}
        mapping.update(hits)
        logger.info("A/H 东财官方对照表解析出 %d 组：%s", len(hits), hits)
    else:
        logger.info("A/H 东财对照表不可用，改用本地索引")

    # ── 二级：本地中文名索引，只补东财没解出来的 ──
    index = _load_cn_name_index()
    if index:
        for code, name in (names or {}).items():
            text = str(code or "").strip()
            if "." not in text or text.endswith(".SH"):
                continue
            bare = text.split(".")[0].zfill(5)
            if bare in mapping:
                continue  # 东财已解出，本地不覆盖权威源
            candidates = index.get(str(name or "").strip(), [])
            if len(candidates) == 1:
                mapping[bare] = candidates[0].split(".")[0]
                logger.info("A/H 本地索引补上 %s(%s) → %s", text, name, mapping[bare])
            elif len(candidates) > 1:
                # 同名多只 A 股：无法确定，不猜
                logger.warning(
                    "A/H 映射跳过 %s(%s)：中文名「%s」匹配到多只 A 股 %s",
                    text, name, name, candidates,
                )
    else:
        logger.info("A/H 本地索引不可用")

    if mapping:
        logger.info("A/H 映射合计 %d 组：%s", len(mapping), mapping)
    else:
        logger.info("A/H 映射为空，%d 只港股全部走长桥通道", len(wanted))
    return mapping


def _hk_codes(names: Optional[Dict[str, str]]) -> List[str]:
    """持仓里的港股 5 位码（去重、保序）。A 股码与 .SH 后缀不算。"""
    out: List[str] = []
    for code in (names or {}):
        text = str(code or "").strip()
        if "." not in text or text.endswith(".SH"):
            continue
        bare = text.split(".")[0].zfill(5)
        if bare not in out:
            out.append(bare)
    return out


#: 东财 A/H 对照表的候选列名。**不硬编单列** —— akshare 改列名就会静默失配，
#: 多试几个别名，命中哪个用哪个。
_EM_HK_COLS = ("港股代码", "港股代码 ", "hk_code", "H股代码")
_EM_A_COLS = ("代码", "A股代码", "a_code")

#: 进程内缓存。同一轮扫描只请求一次，不是跨运行缓存。
_EM_AH_MAP: Optional[Dict[str, str]] = None


def _normalize_hk_code(value: Any) -> str:
    """东财港股代码归一成 5 位裸码。容忍 ``HK06166`` / ``06166.HK`` / ``06166``。"""
    text = str(value or "").strip().upper()
    if "." in text:
        text = text.split(".")[0]
    if text.startswith("HK"):
        text = text[2:]
    if text.isdigit() and 1 <= len(text) <= 5:
        return text.zfill(5)
    return ""


def _normalize_a_code(value: Any) -> str:
    """东财 A 股代码归一成 6 位裸码（akshare 只认裸码，带后缀会 KeyError）。"""
    text = str(value or "").strip().upper()
    if "." in text:
        text = text.split(".")[0]
    if text.startswith("SH") or text.startswith("SZ"):
        text = text[2:]
    if text.isdigit() and len(text) == 6:
        return text
    return ""


def _fetch_ah_map_eastmoney() -> Dict[str, str]:
    """东财 A/H 官方对照表 → ``{港股5位码: A股6位裸码}``。

    只作**兜底**：本地中文名索引解析不出的港股才来这里补。理由是本地索引
    零网络、6ms，而这里是网络调用、有被墙/限流/改列名的风险，所以放在后面。

    任何失败（akshare 缺失、代理拦截、限流、列名变了、返回空表）都返回空
    字典并记 INFO —— 拿不到就退回"不映射"，该港股走长桥通道，绝不因此
    中断整轮扫描。
    """
    global _EM_AH_MAP
    if _EM_AH_MAP is not None:
        return _EM_AH_MAP

    try:
        import akshare as ak
    except Exception as exc:  # noqa: BLE001
        logger.info("akshare 不可用，跳过 A/H 东财兜底: %s", exc)
        _EM_AH_MAP = {}
        return _EM_AH_MAP

    try:
        df = ak.stock_zh_ah_spot_em()
    except Exception as exc:  # noqa: BLE001 - 网络/限流/接口变更一律降级
        logger.info("东财 A/H 对照表取不到，跳过兜底: %s", exc)
        _EM_AH_MAP = {}
        return _EM_AH_MAP

    try:
        columns = list(df.columns)
    except Exception as exc:  # noqa: BLE001
        logger.info("东财 A/H 对照表结构异常，跳过兜底: %s", exc)
        _EM_AH_MAP = {}
        return _EM_AH_MAP

    hk_col = next((c for c in _EM_HK_COLS if c in columns), None)
    a_col = next((c for c in _EM_A_COLS if c in columns), None)
    if not hk_col or not a_col:
        logger.info(
            "东财 A/H 对照表列名不匹配（现有列：%s），跳过兜底", columns[:12]
        )
        _EM_AH_MAP = {}
        return _EM_AH_MAP

    mapping: Dict[str, str] = {}
    for hk_raw, a_raw in zip(df[hk_col], df[a_col]):
        hk = _normalize_hk_code(hk_raw)
        a = _normalize_a_code(a_raw)
        # 同一港股码理论上只对应一只 A 股；撞车时不覆盖，保留先到的。
        if hk and a:
            mapping.setdefault(hk, a)

    _EM_AH_MAP = mapping
    if mapping:
        logger.info("东财 A/H 对照表解析出 %d 组", len(mapping))
    else:
        logger.info("东财 A/H 对照表为空，跳过兜底")
    return _EM_AH_MAP


def default_window(days: int, *, end: date | None = None) -> tuple[date, date]:
    """返回排雷扫描窗口 ``(start, end)``。"""
    end_date = end or date.today()
    return end_date - timedelta(days=max(int(days), 1)), end_date


#: 认不出来日期时的哨兵值。**绝不能用"今天"** —— 事件指纹里含 publish_date，
#: 每天都填今天的话指纹天天变，同一条新闻会被判定成“新事件”反复推送。
_UNKNOWN_DATE = "1970-01-01"


# ---------------------------------------------------------------------------
# 长桥通道
#
# 港股唯一的外部信息源。巨潮只收 A 股，而港交所公告接口
# （HKEXnews titleSearchServlet）对 2024 年后新格式 stockId 一律返回
# recordCnt=0，目前无可用入口 —— 港股风险全靠这里。
#
# 代价：ContentContext / FundamentalContext 不像 QuoteContext 那样由 SDK
# 内部限流，服务端限制 1 秒 1 次，超出报 429002。每日批量任务总调用量不过
# 50 次，1.2 秒间隔即可 —— 与运行频率无关，不是吞吐问题。
# ---------------------------------------------------------------------------

#: 长桥内容类接口的最小调用间隔（秒）
_LB_MIN_INTERVAL = 1.2

#: 社区话题的负面情绪预筛词。topics 每天 50 条，**不能全送 LLM**，
#: 只把命中这些词的帖子留下。
_NEGATIVE_SENTIMENT_TERMS: tuple[str, ...] = (
    "trapped", "bagholder", "cut loss", "stop loss", "panic", "liquidat",
    "margin call", "forced", "bearish", "crash", "plunge", "scam",
    "fraud", "probe", "investigation", "lawsuit", "delist", "suspend",
    "halt", "winding", "bankrupt", "default", "miss payment", "blow up",
    "套牢", "割肉", "止损", "爆仓", "平仓", "被套", "阴跌", "暴跌", "崩",
    "维权", "做空", "造假", "调查", "起诉", "停牌", "退市", "破产", "违约",
)

#: 财报净利为负。corp_action 的 act_desc 形如
#: ``FY2026 Q2 Earning Release (CNY) Revenue 633.93 M, Net Income -72.39 M``
_LOSS_RE = re.compile(r"Net\s+Income\s+(-?[\d,]+\.?\d*)\s*([MKB]?)", re.IGNORECASE)
_LOSS_SCALE = {"M": 1e6, "B": 1e9, "K": 1e3, "": 1.0}


def _lb_contexts():
    """构建长桥 Content / Fundamental 上下文。任一失败返回 None。

    必须先 ``import src.config``：它负责把 ``.env`` 载入 ``os.environ``。
    长桥 SDK 的 ``Config.from_apikey_env()`` 只读环境变量，绕开 config 就会
    报 ``missing environment variable: LONGBRIDGE_APP_KEY``（本地必现）。
    CI 里环境变量由 workflow env 注入，但统一走 config 更稳。
    """
    try:
        import longbridge.openapi as lbo

        from src.config import get_config  # noqa: F401 - 副作用：加载 .env

        get_config()
        cfg = lbo.Config.from_apikey_env()
        return lbo.ContentContext(cfg), lbo.FundamentalContext(cfg)
    except Exception as exc:  # noqa: BLE001 - 缺凭证/SDK 未装都走这里
        # 用 warning 而非 info：这是**故障**，不是正常降级。CI 里最常见的
        # 成因是 workflow 漏传 LONGBRIDGE_APP_KEY，表现为长桥整条 0 条而
        # 港股照常显示「无风险事件」—— 光看逐只结果发现不了。
        logger.warning("长桥通道不可用: %s", exc)
        return None, None


def _is_negative_sentiment(text: str) -> bool:
    low = str(text or "").lower()
    return any(term in low for term in _NEGATIVE_SENTIMENT_TERMS)


def _lb_date(value: Any) -> str:
    """长桥时间戳 → YYYY-MM-DD；解析不出用哨兵值。"""
    text = str(value or "").strip()
    if not text:
        return _UNKNOWN_DATE
    head = text.replace("/", "-").split(" ")[0].split("T")[0]
    if len(head) == 8 and head.isdigit():
        return f"{head[:4]}-{head[4:6]}-{head[6:8]}"
    try:
        return date.fromisoformat(head).isoformat()
    except ValueError:
        return _UNKNOWN_DATE


def _earnings_loss_events(fc, code: str, name: str) -> List[Announcement]:
    """从 corp_action 提取「财报净利为负」，并统计**真正的**连续亏损季数。

    两个关键点：

    1. ``it.date`` 是 8 位带年份（``20260923``），``it.date_str`` 只有月.日
       （``09.23``）。必须用前者，否则年份丢失会落进哨兵值。
    2. corp_action 的列表**不是按时间连续排列的**（同一财年的 Q2/Q4 混排），
       所以必须先按日期**正序**再从最新往回数；遇到第一个非负就**停止**，
       而不是清零后继续数 —— 后者会把"FY2024Q2 亏、FY2023Q4 亏"误算成连续。
    3. 只输出一条汇总，不逐季刷屏。
    """
    try:
        items = list(fc.corp_action(code).items)
    except Exception as exc:  # noqa: BLE001 - 非 A+H 标的会报网络错，属正常
        logger.debug("corp_action 不可用 %s: %s", code, exc)
        return []

    # 正序：最老 → 最新
    ordered = sorted(items, key=lambda x: str(x.date or ""))
    losses: List[Tuple[str, float]] = []
    for it in ordered:
        if "Earning" not in str(it.act_type or ""):
            continue
        match = _LOSS_RE.search(str(it.act_desc or ""))
        if not match:
            continue
        try:
            value = float(match.group(1).replace(",", "")) * _LOSS_SCALE.get(
                match.group(2).upper(), 1.0
            )
        except ValueError:
            continue
        losses.append((_lb_date(it.date or it.date_str), value))

    if not losses:
        return []

    # 从最新往回数，遇到非负立即停止
    consecutive = 0
    for _, value in reversed(losses):
        if value >= 0:
            break
        consecutive += 1

    if consecutive == 0:
        return []

    latest_date, latest_value = losses[-1]
    desc = next(
        (str(i.act_desc) for i in reversed(ordered)
         if "Earning" in str(i.act_type or "") and _LOSS_RE.search(str(i.act_desc or ""))),
        "财报",
    )
    label = (
        f"{name} {desc[:56]}（净利 {_fmt_amount(latest_value)}，"
        f"已连续 {consecutive} 个季度为负）"
    )
    return [Announcement(
        code=code, name=name, title=label, publish_date=latest_date,
        url="", source="longbridge:corp_action",
        source_type=SourceType.EXCHANGE_FILING,
    )]


def _fmt_amount(value: float) -> str:
    """把净利金额格式化成可读形式。``value`` 已是负数，不要再拼负号。"""
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"{value / 1e9:.2f}B"
    if magnitude >= 1e6:
        return f"{value / 1e6:.1f}M"
    if magnitude >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{value:.0f}"


def fetch_longbridge(
    codes: Iterable[str],
    *,
    names: Optional[Dict[str, str]] = None,
    news_markets: str = "ALL",
    include_news: bool = True,
    include_topics: bool = True,
    include_earnings: bool = True,
    max_news: int = 10,
    max_topics: int = 10,
    interval: float = _LB_MIN_INTERVAL,
) -> Dict[str, List[Announcement]]:
    """经长桥取资讯、社区负面情绪与财报异常。

    ``news_markets`` 控制哪些市场跑 news/topics，默认 **ALL（全市场）**：

    * 传 ``"HK"`` 可退回只跑港股。A 股的长桥 news 实测基本是「三大指数低开」
      「某概念板块普跌」这类大盘综述，不是该股的事；且 A 股本来就有巨潮这个
      高质量结构化源，长桥的增量价值有限。删掉 Web 舆情通道后，A股若还想有
      任何外部新闻信号，就只能靠这里 —— 故默认全市场跑。
    * 港股无论如何必跑：没有公告通道，长桥的资讯/社区是唯一外部信号。

    ``corp_action``（财报）不受此限制，永远全市场跑 —— 它是确定性数据，
    沃格光电那种「连续四个季度净利为负」只能从这里拿到。

    注意：全市场跑 news/topics 会把每只票的调用次数从 1 次拉到 3 次
    （news + topics + corp_action），16 只持仓约 48 次，按 1.2 秒节流
    约 58 秒。相比只跑港股多花约 40 秒。
    """
    cc, fc = _lb_contexts()
    if cc is None and fc is None:
        # 凭证缺失/SDK 未装 → **显式抛错**，不要静默返回 {}。
        # 静默返回会让港股照常显示「无风险事件」，报告里看不出任何异常，
        # 这正是"把没查到说成没事"这个最危险的失败模式。抛出去由调用方
        # 记进报告的「数据异常」段。
        raise RuntimeError(
            "长桥通道不可用（凭证缺失或 longbridge 未安装）。"
            "检查 workflow 是否传入 LONGBRIDGE_APP_KEY / APP_SECRET / ACCESS_TOKEN。"
        )

    name_map = names or {}
    hk_news_only = str(news_markets or "").strip().upper() in ("HK", "港股")
    out: Dict[str, List[Announcement]] = {}
    last = [0.0]

    def _throttle() -> None:
        gap = time.monotonic() - last[0]
        if gap < interval:
            time.sleep(interval - gap)
        last[0] = time.monotonic()

    for code in codes:
        key = str(code or "").strip()
        if not key:
            continue
        display = name_map.get(key, key)
        rows: List[Announcement] = []
        want_news = include_news and (
            not hk_news_only or key.upper().endswith(".HK")
        )

        if want_news and cc is not None:
            _throttle()
            try:
                for item in (cc.news(key) or [])[:max_news]:
                    rows.append(Announcement(
                        code=key, name=display, title=item.title or "",
                        publish_date=_lb_date(item.published_at),
                        url=item.url or "", source="longbridge:news",
                        source_type=SourceType.NEWS,
                    ))
            except Exception as exc:  # noqa: BLE001
                logger.debug("长桥资讯失败 %s: %s", key, exc)

        if want_news and include_topics and cc is not None:
            _throttle()
            try:
                for item in (cc.topics(key) or []):
                    text = item.description or ""
                    if not _is_negative_sentiment(text):
                        continue
                    if len(rows) >= max_news + max_topics:
                        break
                    rows.append(Announcement(
                        code=key, name=display,
                        title=f"[社区情绪] {text[:100]}",
                        publish_date=_lb_date(item.published_at),
                        url=item.url or "", source="longbridge:topics",
                        source_type=SourceType.NEWS,
                    ))
            except Exception as exc:  # noqa: BLE001
                logger.debug("长桥社区话题失败 %s: %s", key, exc)

        if include_earnings and fc is not None:
            _throttle()
            rows.extend(_earnings_loss_events(fc, key, display))

        if rows:
            rows.sort(key=lambda a: a.publish_date, reverse=True)
            out[key] = rows
    return out


# ---------------------------------------------------------------------------
# 通道发现
# ---------------------------------------------------------------------------
