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
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional

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
) -> List[Announcement]:
    """用 akshare 拉单只股票的全部公告。"""
    try:
        import akshare as ak
    except Exception as exc:  # noqa: BLE001
        logger.error("akshare 不可用: %s", exc)
        return []

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
        return []

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
    start: date,
    end: date,
    delay: float = 0.35,
    crosscheck: bool = True,
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

    target = [str(c).strip() for c in codes if str(c or "").strip()]
    if not target:
        return {}

    session: Optional[CninfoSession] = None
    results: Dict[str, List[Announcement]] = {}
    empty_codes: List[str] = []
    names: Dict[str, str] = {}

    for code in target:
        rows = _fetch_one_via_akshare(code, start_compact, end_compact)
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
        results[code] = unique
        if unique:
            names[code] = unique[0].name
        else:
            empty_codes.append(code)
        if delay:
            time.sleep(delay)

    if crosscheck and empty_codes:
        try:
            session = CninfoSession()
        except Exception as exc:  # noqa: BLE001
            logger.warning("巨潮 raw 会话初始化失败，跳过反查: %s", exc)
            session = None
        if session is not None:
            for code in empty_codes:
                _crosscheck_via_raw(
                    session, code, names.get(code, code), start_dash, end_dash
                )
                time.sleep(delay)

    return results


def default_window(days: int, *, end: date | None = None) -> tuple[date, date]:
    """返回排雷扫描窗口 ``(start, end)``。"""
    end_date = end or date.today()
    return end_date - timedelta(days=max(int(days), 1)), end_date


# ---------------------------------------------------------------------------
# 舆情通道
#
# 巨潮只收录 A 股公告，港股公告接口（HKEXnews titleSearchServlet）对 2024 年
# 后新格式 stockId 一律返回 recordCnt=0，目前无可用入口。因此港股改走
# DSA 既有的 SearchService 做新闻检索。
#
# 两条通道的差别不在市场，而在信息源性质：
#   cn_filing → EXCHANGE_FILING，词汇封闭、日期精确，可定级到致命
#   any_news  → NEWS，词汇开放、日期常缺，由 CEILING_BY_SOURCE 硬顶在警告
# A 股同时挂两条；港股只挂 any_news。哪天港交所接口打通，加一条 hk_filing
# 进来港股自动变成双通道，不需要改动判定逻辑。
# ---------------------------------------------------------------------------


#: 定向负面 query 的检索词。
#:
#: 为什么不用泛化 query：实测同一批持仓，"公司名 + 代码 + 股票最新消息" 捞回来
#: 的 18 条里负面信号 0 条，全是「盘中涨超5%」「港股通占比异动」「雪球股价页」
#: 这类价格与资金流噪音；而定向 query 同样 18 条里捞出约 8 条真负面
#: （被罚没千万 / 再遭监管降级 / 控股股东减持 / 盘中大跌近70%）。
#:
#: 刻意不写成 `| OR` 之类的布尔语法：Tavily 等 provider 对长 query 的处理
#: 差异大，堆词比列词更稳。
NEGATIVE_QUERY_TERMS: tuple[str, ...] = (
    "亏损", "调查", "处罚", "违规", "停牌", "被查", "造假", "减持",
    "诉讼", "退市", "问询", "爆雷", "暴跌", "重挫", "清盘", "核数师",
    "盈利警告", "股东质押", "债务违约",
)


def _build_news_query(code: str, name: str) -> str:
    """构造定向负面 query。

    **必须同时带公司名和裸代码**：只带名字会串到 A 股同名公司
    （实测「剑桥科技」会捞到 603083 的违规记录，那是一只完全不同的股票）。
    """
    bare = str(code or "").split(".")[0].strip()
    parts = [name, bare] if bare and bare != name else [name]
    return " ".join(parts) + " " + " ".join(NEGATIVE_QUERY_TERMS)


#: 静态档案页 / 行情页特征。这类 URL 是券商和门户的**常驻个股页面**，
#: 标题里带负面词但没有事件时间（实测新浪 `违规记录` 页、���球 `股价行情` 页
#: 会被定向 query 稳定捞到）。它们不是新闻，必须丢掉。
_ARCHIVE_PAGE_RE = re.compile(
    r"(违规记录|股价_股价行情|历史行情|股票股价_|公司高管_|资料档案"
    r"|_行情中心|个股主页|F10|十大股东|公司简介)"
)

#: 大盘级汇总清单。这些文章列举了 N 家公司，某只票恰好在名单里就被检索命中，
#: 但它并没有针对这只票发生任何事。中文财经媒体的"周末利空盘点"极其泛滥，
#: 模式必须写得很宽 —— 实测「雷来了！周末31股利空」「5家被罚，4个立案，
#: 38家诉讼」这类标题用最初那版规则完全挡不住。
_ROUNDUP_RE = re.compile(
    r"(\d+\s*家公司|\d+\s*家(被罚|立案|退市|诉讼|问询)"
    r"|\d+\s*股(利空|涨停|跌停|爆雷|异动|集体)"
    r"|周末\d*|隔夜|一觉醒来|雷来了|集体爆雷|集体重挫|集体涨停|批量爆雷|批量退市"
    r"|避雷清单|避雷|盘点|名单|汇总|排行"
    r"|涨停板复盘|跌停板复盘|龙虎榜|今日涨停|今日跌停"
    r"|盘前必读|早间速览|市场参考|财经早餐|每日复盘|收评|午评|复盘)"
)


def is_usable_news(title: str, url: str = "") -> bool:
    """判断一条检索结果是不是**针对该标的**的真实负面舆情。

    挡掉两类噪音：静态档案页、大盘汇总清单。
    """
    text = f"{title} {url}"
    if _ARCHIVE_PAGE_RE.search(text):
        return False
    if _ROUNDUP_RE.search(title or ""):
        return False
    return True


#: URL 里的日期。中文财经媒体的 URL 常见四种形态：
#:   /article/20260716/herald/   连续 8 位
#:   /a/202603263686001829.html  14 位毫秒时间戳，8 位日期**嵌在更长数字串中间**
#:   /2019-07-05/101436127.html  带横线
#:   /detail/2461205             无日期
#: 所以要同时匹配"8 位连续"和"带横线"两种，且**不能**给连续位加 ``(?!\d)``
#: 边界（会漏掉时间戳形态）；误配靠"日期是否合法 + 是否是未来"来挡。
_URL_DATE_RES = (
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})"),
    re.compile(r"(?<!\d)(20\d{2})[-/](\d{2})[-/](\d{2})(?!\d)"),
)
#: 认不出来时的哨兵日期。**绝不能用"今天"** —— 事件指纹里含 publish_date，
#: 每天都填今天的话指纹天天变，同一条新闻会被判定成"新事件"反复推送。
_UNKNOWN_DATE = "1970-01-01"


def extract_date(url: str, fallback: str = _UNKNOWN_DATE) -> str:
    """从 URL 抽发布日期，抽不到返回 ``fallback``。

    通用网页搜索不返回 ``published_date``，但中文财经媒体的 URL 普遍带日期。
    校验不通过就返回哨兵日期而不是"今天"：指纹必须稳定，否则同一条新闻
    每天都被当成新事件重推。
    """
    text = str(url or "")
    today = date.today()
    for pattern in _URL_DATE_RES:
        for match in pattern.finditer(text):
            year, month, day = match.groups()
            try:
                parsed = date(int(year), int(month), int(day))
            except ValueError:
                continue
            if parsed > today + timedelta(days=2):
                continue  # 未来日期，多半是编号而非日期
            return parsed.isoformat()
    return fallback


def fetch_news(
    codes: Iterable[str],
    *,
    names: Optional[Dict[str, str]] = None,
    days: int = 30,
    max_results: int = 8,
    delay: float = 0.4,
) -> Dict[str, List[Announcement]]:
    """按只检索**负面舆情**，返回 ``{code: [Announcement(source_type=NEWS), ...]}``。

    与 ``SearchService.search_stock_news`` 的区别有三点，都是必须的：

    1. **定向 query**：自带负面检索词，而不是"股票最新消息"这种泛化查询。
    2. **不走 ``focus_keywords``**：该参数在 ``search_service.py:4050`` 会把
       query 整个替换成关键词拼接，公司名和代码被丢弃，实测三家不同公司会
       返回完全相同的一批结果。这里自己拼 query，直接调 provider 层。
    3. **丢掉相关性准入**：``search_stock_news`` 的相关性打分是为泛化 query
       设计的；定向 query 下不再需要，兜底交给 LLM 分类。

    ``days`` 是**舆情专用窗口**，与公告窗口（``RISK_SCAN_WINDOW_DAYS``）
    无关：舆情讲"最近出了什么事"，公告要还原"这件事走到哪一步了"。

    窗口由**本模块自己**执行过滤，不依赖 provider：通用网页搜索的
    ``days`` 参数不被强制执行（Tavily 只在 ``topic="news"`` 下认它，
    而 ``topic="news"`` 会把中文小盘港股查询路由到国际新闻索引、零覆盖）。
    所以策略是「provider 照常返回，本地按 URL 抽出的日期筛」。
    抽不出日期的条目一律保留 —— 无法证明它旧，就不该丢。

    ``published_date`` 通常为空；万一有值优先用它。
    """
    try:
        from src.search_service import get_search_service
    except Exception as exc:  # noqa: BLE001 - 缺依赖时舆情通道整体降级
        logger.warning("SearchService 不可用，舆情通道跳过: %s", exc)
        return {}

    name_map = names or {}
    try:
        service = get_search_service()
    except Exception as exc:  # noqa: BLE001
        logger.warning("SearchService 初始化失败，舆情通道跳过: %s", exc)
        return {}

    if not service._providers:
        logger.warning("未配置任何搜索 provider，舆情通道静默跳过")
        return {}

    cutoff = date.today() - timedelta(days=max(int(days), 1))
    results: Dict[str, List[Announcement]] = {}
    for code in codes:
        key = str(code or "").strip()
        if not key:
            continue
        display = name_map.get(key, key)
        query = _build_news_query(key, display)

        response = None
        for provider in service._providers:
            # 刻意**不传 topic="news"**。实测带 topic 会把查询路由到 Tavily 的
            # 国际新闻索引，中文小盘港股零覆盖：速腾聚创返回
            # 「UWM Holdings Sued for Securities Fraud」、剑桥科技返回
            # 「Super Micro Computer Investigation」，完全无关。
            # 通用网页搜索反而能捞到「美尚生态财务造假，五家券商被连带起诉」
            # 这类真信号（广发就在其中）。
            try:
                response = provider.search(query, max_results=max_results, days=days)
            except Exception as exc:  # noqa: BLE001 - 单个 provider 失败换下一个
                logger.debug("provider %s 查询失败 %s: %s", provider.name, key, exc)
                continue
            if response is not None and getattr(response, "results", None):
                break
            response = None

        if response is None:
            logger.debug("舆情检索无结果 %s(%s)", display, key)
            results[key] = []
            continue

        rows: List[Announcement] = []
        dropped = 0
        stale = 0
        for item in response.results or []:
            if not is_usable_news(item.title or "", item.url or ""):
                dropped += 1
                continue
            publish = item.published_date or extract_date(item.url or "")
            # 时间窗口由本模块执行。哨兵日期（URL 里抽不到）**保留** ——
            # 无法证明它旧，就不该因为猜了个日期把它丢掉。
            if publish != _UNKNOWN_DATE:
                try:
                    if date.fromisoformat(publish) < cutoff:
                        stale += 1
                        continue
                except ValueError:
                    pass
            rows.append(
                Announcement(
                    code=key,
                    name=display,
                    title=item.title,
                    # 绝不填"今天"：事件指纹含 publish_date，天天变会导致
                    # 同一条新闻被当成新事件反复推送。
                    publish_date=publish,
                    url=item.url,
                    source=f"news:{response.provider or 'unknown'}",
                    source_type=SourceType.NEWS,
                )
            )
        if dropped or stale:
            logger.debug(
                "%s(%s) 滤掉 %d 条档案页/汇总、%d 条超窗(>%d天)",
                display, key, dropped, stale, days,
            )
        results[key] = rows
        if delay:
            time.sleep(delay)
    return results


# ---------------------------------------------------------------------------
# 通道发现
# ---------------------------------------------------------------------------
