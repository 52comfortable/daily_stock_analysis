"""Core drawdown engine for the position-drawdown alert.

Algorithm
---------
For each position in ``portfolio.json``:

1. Read the position's ``position_date`` (open date) and ``cost_price``.
2. Fetch daily K-line from ``position_date`` to today (inclusive). Use
   efinance; fall back to akshare on failure.
3. Compute the historical peak from K-line highs:
       peak_price = max(high) on (position_date, today]
       peak_date  = date on which that high was reached
4. Read today's close from the K-line.
5. Compute two metrics:
       drawdown_pct        = (peak - close) / peak * 100
       remaining_upside_pct = (close - cost) / cost * 100
6. Pick a severity tier (only when peak > cost):
       critical: drawdown >= 30%  OR  remaining_upside <= 0
       severe:   drawdown >= 20%  OR  remaining_upside <  5
       warning:  drawdown >= 10%
       (none if no tier matches OR peak <= cost)
7. Emit an ``Alert`` for any non-None severity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    WARNING = "warning"
    SEVERE = "severe"
    CRITICAL = "critical"


SEVERITY_GLYPH = {
    AlertSeverity.WARNING: "🟡",
    AlertSeverity.SEVERE: "🟠",
    AlertSeverity.CRITICAL: "🔴",
}

SEVERITY_RANK = {
    AlertSeverity.WARNING: 1,
    AlertSeverity.SEVERE: 2,
    AlertSeverity.CRITICAL: 3,
}


@dataclass
class PositionDrawdown:
    """Computed metrics for one position."""
    code: str
    name: str
    cost_price: float
    position_date: str
    peak_price: Optional[float] = None
    peak_date: Optional[str] = None
    today_close: Optional[float] = None
    today_date: Optional[str] = None
    drawdown_pct: Optional[float] = None
    remaining_upside_pct: Optional[float] = None
    ever_risen: bool = False  # peak > cost
    error: Optional[str] = None


@dataclass
class Alert:
    """Triggered alert to be sent to Feishu."""
    severity: AlertSeverity
    position: PositionDrawdown
    summary: str
    detail_lines: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# K-line fetcher (delegates to DSA's DataFetcherManager)
# ---------------------------------------------------------------------------

_DATA_MANAGER: Optional[Any] = None


def _get_data_manager() -> Optional[Any]:
    """Lazily build a DataFetcherManager so the import doesn't fail when
    data_provider is unavailable (e.g., minimal CI test envs)."""
    global _DATA_MANAGER
    if _DATA_MANAGER is not None:
        return _DATA_MANAGER
    try:
        from data_provider.base import DataFetcherManager
        _DATA_MANAGER = DataFetcherManager()
    except Exception as exc:  # noqa: BLE001
        logger.warning("DataFetcherManager unavailable: %s", exc)
        _DATA_MANAGER = False  # cache the failure
    return _DATA_MANAGER if _DATA_MANAGER else None


def fetch_kline(code: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch daily K-line via DSA's existing DataFetcherManager.

    The manager chains: Tushare → Efinance → Akshare → Pytdx → Baostock
    → Yfinance → Tencent → Longbridge, with built-in rate-limiting,
    retries, and anti-ban heuristics. ``get_daily_data`` returns a
    ``(DataFrame, status_string)`` tuple; we keep just the frame.
    """
    manager = _get_data_manager()
    if manager is None:
        return None
    try:
        result = manager.get_daily_data(
            stock_code=code,
            start_date=f"{start[:4]}-{start[4:6]}-{start[6:8]}",
            end_date=f"{end[:4]}-{end[4:6]}-{end[6:8]}",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("DataFetcherManager.get_daily_data failed for %s: %s", code, exc)
        return None
    # get_daily_data returns (DataFrame, status_str) — unwrap.
    if isinstance(result, tuple):
        df = result[0]
    else:
        df = result
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    return df


# ---------------------------------------------------------------------------
# Per-position evaluation
# ---------------------------------------------------------------------------

def _to_yyyymmdd(d: str) -> str:
    """Normalize YYYY-MM-DD or already-compact YYYYMMDD to YYYYMMDD."""
    s = (d or "").strip()
    if not s:
        return s
    if "-" in s:
        return s.replace("-", "")
    return s


def _to_iso(d: Any) -> str:
    """Best-effort conversion of a date-like value to YYYY-MM-DD string."""
    if isinstance(d, (date, datetime)):
        return d.strftime("%Y-%m-%d")
    s = str(d or "").strip()
    if not s:
        return s
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def _compute_position(
    pos: Dict[str, Any], today: date
) -> PositionDrawdown:
    """Compute drawdown metrics for one position dict."""
    code = str(pos.get("code") or "").strip()
    name = str(pos.get("name") or code).strip()
    cost = float(pos.get("cost_price") or 0.0)
    position_date_raw = str(pos.get("position_date") or "").strip()
    result = PositionDrawdown(
        code=code, name=name, cost_price=cost, position_date=position_date_raw
    )
    if not code or cost <= 0 or not position_date_raw:
        result.error = f"missing required field (code={code!r} cost={cost} position_date={position_date_raw!r})"
        return result

    start = _to_yyyymmdd(position_date_raw)
    end = today.strftime("%Y%m%d")
    df = fetch_kline(code, start, end)
    if df is None or df.empty:
        result.error = f"no K-line data for {code} [{start}..{end}]"
        return result

    # DataFetcherManager returns STANDARD_COLUMNS: date/open/high/low/close/volume/amount/pct_chg
    if "high" not in df.columns or "close" not in df.columns or "date" not in df.columns:
        result.error = f"unexpected K-line columns: {list(df.columns)}"
        return result

    # Some fetchers return float columns already, some as Decimal/string.
    high_series = pd.to_numeric(df["high"], errors="coerce")
    close_series = pd.to_numeric(df["close"], errors="coerce")
    if high_series.isna().all() or close_series.isna().all():
        result.error = "all numeric values are NaN"
        return result

    peak_value = float(high_series.max())
    peak_idx = high_series.idxmax()
    peak_date_val = df.loc[peak_idx, "date"]
    last_idx = close_series.last_valid_index()
    today_close_val = float(close_series.loc[last_idx])
    today_date_val = df.loc[last_idx, "date"]

    result.peak_price = peak_value
    result.peak_date = _to_iso(peak_date_val)
    result.today_close = today_close_val
    result.today_date = _to_iso(today_date_val)
    result.ever_risen = peak_value > cost
    if peak_value > 0:
        result.drawdown_pct = round((peak_value - today_close_val) / peak_value * 100, 4)
    if cost > 0:
        result.remaining_upside_pct = round((today_close_val - cost) / cost * 100, 4)
    return result


def _pick_severity(p: PositionDrawdown, thresholds: Dict[str, float]) -> Optional[AlertSeverity]:
    """Pick the highest matching tier. Returns None if peak <= cost or no tier matches."""
    if not p.ever_risen:
        return None
    if p.drawdown_pct is None:
        return None

    candidates: List[AlertSeverity] = []
    if p.drawdown_pct >= thresholds.get("critical", 30):
        candidates.append(AlertSeverity.CRITICAL)
    if p.drawdown_pct >= thresholds.get("severe", 20):
        candidates.append(AlertSeverity.SEVERE)
    if p.drawdown_pct >= thresholds.get("warning", 10):
        candidates.append(AlertSeverity.WARNING)
    # Status upgrades: close approaching or crossing cost
    if p.remaining_upside_pct is not None:
        if p.remaining_upside_pct <= 0:
            candidates.append(AlertSeverity.CRITICAL)
        elif p.remaining_upside_pct < thresholds.get("severe_upside", 5):
            candidates.append(AlertSeverity.SEVERE)
    if not candidates:
        return None
    return max(candidates, key=lambda s: SEVERITY_RANK[s])


def _build_alert(p: PositionDrawdown, severity: AlertSeverity) -> Alert:
    glyph = SEVERITY_GLYPH[severity]
    line1 = (
        f"{glyph} {severity.value.title()}：{p.name}({p.code}) "
        f"从历史峰值 {p.peak_price:.2f} 回撤 {p.drawdown_pct:.2f}%"
    )
    details: List[str] = [
        f"成本 {p.cost_price:.2f}（建仓 {p.position_date}，曾涨至 {p.peak_price:.2f}）"
        f" / 今日收盘 {p.today_close:.2f}",
    ]
    if p.remaining_upside_pct is not None:
        if p.remaining_upside_pct >= 0:
            details.append(f"剩余空间 +{p.remaining_upside_pct:.2f}%")
        else:
            details.append(
                f"⚠️ 已跌破成本 {-p.remaining_upside_pct:.2f}%"
            )
    if p.peak_price and p.cost_price and p.peak_price > p.cost_price:
        gain_pct = (p.peak_price - p.cost_price) / p.cost_price * 100
        consumed_pct = (p.peak_price - p.today_close) / (p.peak_price - p.cost_price) * 100 if p.peak_price > p.cost_price else 0
        details.append(
            f"曾经涨过 {gain_pct:.1f}%，现已回吐 {consumed_pct:.1f}%"
        )
    summary = line1
    return Alert(severity=severity, position=p, summary=summary, detail_lines=details)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_portfolio(
    portfolio: Dict[str, Any],
    *,
    thresholds: Optional[Dict[str, float]] = None,
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """Evaluate every position in ``portfolio`` and return alerts + per-position metrics.

    Returns a dict with:
        alerts: List[Alert] (empty if no triggers)
        positions: List[PositionDrawdown] (one per input position)
        as_of: ISO date string of the evaluation
    """
    cfg = {
        "warning": 10.0,
        "severe": 20.0,
        "critical": 30.0,
        "severe_upside": 5.0,
    }
    if thresholds:
        cfg.update({k: float(v) for k, v in thresholds.items() if v is not None})

    as_of = today or date.today()
    positions_raw = portfolio.get("positions") or []
    if not isinstance(positions_raw, list):
        raise ValueError("portfolio.positions must be a list")

    results: List[PositionDrawdown] = []
    alerts: List[Alert] = []
    for raw in positions_raw:
        if not isinstance(raw, dict):
            logger.warning("skip non-dict position entry: %r", raw)
            continue
        p = _compute_position(raw, as_of)
        results.append(p)
        if p.error:
            logger.warning("[%s] %s", p.code, p.error)
            continue
        sev = _pick_severity(p, cfg)
        if sev is not None:
            alerts.append(_build_alert(p, sev))

    return {
        "alerts": alerts,
        "positions": results,
        "as_of": as_of.isoformat(),
        "thresholds": cfg,
    }
