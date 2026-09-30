"""Core drawdown engine for the position-drawdown alert.

Algorithm
---------
For each position in ``portfolio.json``:

1. Read the position's ``position_date`` (open date) and ``cost_price``.
2. Fetch daily K-line from ``position_date`` to today (inclusive). Use
   efinance; fall back to akshare on failure.
3. Compute the historical peak from K-line **closes**:
       peak_price = max(close) on (position_date, today]
       peak_date  = date on which that close was reached
   The intraday ``high`` is deliberately ignored: peak and current must share
   one price basis, otherwise the drawdown compares 盘中最高价 against 收盘价.
4. Read today's close from the last K-line row.
5. Compute three metrics:
       drawdown_pct         = (peak - close) / peak * 100
       remaining_upside_pct = (close - cost) / cost * 100
       peak_gain_pct        = (peak - cost) / cost * 100
6. Emit an ``Alert`` for each rule that fires, independently:
   a. **浮盈回撤 (drawdown)** — requires ``peak_gain_pct >= min_gain`` (5%),
      then grades on the drawdown from that peak:
          warning  >= 20%   severe >= 50%   critical >= 80%
   b. **跌破止损 (stop_loss)** — no run-up required, single tier: fires when
      the close is at or past the stop line versus cost
      (``remaining_upside <= -stop_loss``, default 5%). Always CRITICAL and
      always rendered with ⛔, because it is an action signal rather than a
      "how bad" scale.
   A position can fire (a), (b), both, or neither.
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


class AlertKind(str, Enum):
    """The two independent alert types.

    DRAWDOWN — the position had a meaningful run-up (peak gain >= min_gain)
               and has since given back a large share of that peak.
    STOP_LOSS — the position fell straight through the cost line, no run-up
               required. Triggers on remaining_upside <= -stop_loss.
    """

    DRAWDOWN = "drawdown"
    STOP_LOSS = "stop_loss"


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

# Stop-loss always renders as ⛔ regardless of severity, so a breached stop line
# is never mistaken for a merely-deep drawdown.
def alert_glyph(severity: AlertSeverity, kind: AlertKind) -> str:
    """Glyph for an alert: ⛔ for stop-loss, severity colour for drawdown."""
    if kind is AlertKind.STOP_LOSS:
        return "⛔"
    return SEVERITY_GLYPH[severity]


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
    peak_gain_pct: Optional[float] = None  # (peak - cost) / cost * 100
    ever_risen: bool = False  # peak > cost
    error: Optional[str] = None


@dataclass
class Alert:
    """Triggered alert to be sent out."""
    severity: AlertSeverity
    kind: AlertKind
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
    if "close" not in df.columns or "date" not in df.columns:
        result.error = f"unexpected K-line columns: {list(df.columns)}"
        return result

    # Some fetchers return float columns already, some as Decimal/string.
    close_series = pd.to_numeric(df["close"], errors="coerce")
    close_series = close_series.dropna()
    if close_series.empty:
        result.error = "all numeric values are NaN"
        return result

    # peak uses the daily CLOSE, not the intraday high: both the peak and the
    # current value must be the same price basis, otherwise the drawdown mixes
    # 盘中最高价 with 收盘价 and no longer means "从最高收盘跌到今天收盘".
    peak_value = float(close_series.max())
    peak_idx = close_series.idxmax()
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
        result.peak_gain_pct = round((peak_value - cost) / cost * 100, 4)
    return result


def _pick_severity(p: PositionDrawdown, thresholds: Dict[str, float]) -> Optional[AlertSeverity]:
    """Pick the drawdown tier. Returns None if the position never ran up.

    Gate: the position must have floated at least ``min_gain`` percent above
    cost at its peak. A "drawdown" from a peak that never rose meaningfully is
    noise (e.g. a money-market fund pinned at its cost).
    """
    if p.peak_gain_pct is None or p.drawdown_pct is None:
        return None
    if p.peak_gain_pct < thresholds.get("min_gain", 5.0):
        return None

    candidates: List[AlertSeverity] = []
    if p.drawdown_pct >= thresholds.get("critical", 80):
        candidates.append(AlertSeverity.CRITICAL)
    if p.drawdown_pct >= thresholds.get("severe", 50):
        candidates.append(AlertSeverity.SEVERE)
    if p.drawdown_pct >= thresholds.get("warning", 20):
        candidates.append(AlertSeverity.WARNING)
    if not candidates:
        return None
    return max(candidates, key=lambda s: SEVERITY_RANK[s])


def _pick_stop_loss(p: PositionDrawdown, thresholds: Dict[str, float]) -> Optional[AlertSeverity]:
    """Stop-loss is a single tier on purpose: one line, one meaning.

    Fires when the close sits at or below the stop line:
        remaining_upside_pct <= -stop_loss
    Breaching the stop is an action signal, not a "how bad" scale, so it always
    lands on CRITICAL — depth is already visible in the reported loss %.
    """
    if p.remaining_upside_pct is None:
        return None
    if p.remaining_upside_pct > -thresholds.get("stop_loss", 5.0):
        return None
    return AlertSeverity.CRITICAL


def _build_alert(
    p: PositionDrawdown, severity: AlertSeverity, kind: AlertKind
) -> Alert:
    glyph = alert_glyph(severity, kind)
    details: List[str] = [
        f"建仓 {p.position_date}　成本 {p.cost_price:.2f}",
        f"历史最高收盘 {p.peak_price:.2f}（{p.peak_date}）"
        f"　最新收盘 {p.today_close:.2f}（{p.today_date}）",
    ]

    if kind is AlertKind.DRAWDOWN:
        line1 = (
            f"{glyph} 浮盈回撤：{p.name}({p.code}) "
            f"曾浮盈 +{p.peak_gain_pct:.1f}%，现价距最高收盘 -{p.drawdown_pct:.1f}%"
        )
        details.append(
            f"峰值回撤：**{p.drawdown_pct:.2f}%**"
            f"（{p.peak_price:.2f} → {p.today_close:.2f}）"
        )
        details.append(f"峰值时浮盈 **+{p.peak_gain_pct:.1f}%**")
    else:
        line1 = (
            f"{glyph} 跌破止损线：{p.name}({p.code}) "
            f"成本 {p.cost_price:.2f}，现价 {p.today_close:.2f}，"
            f"已亏 **{-p.remaining_upside_pct:.1f}%**"
        )
        details.append(f"⚠️ 已跌破成本 **{-p.remaining_upside_pct:.2f}%**")

    if p.remaining_upside_pct is not None and p.remaining_upside_pct >= 0:
        details.append(f"剩余空间 +{p.remaining_upside_pct:.2f}%")

    return Alert(
        severity=severity,
        kind=kind,
        position=p,
        summary=line1,
        detail_lines=details,
    )


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

    Each position is checked against two independent rules and may therefore
    produce zero, one, or two alerts:

    * ``drawdown``  — ran up >= ``min_gain`` then gave back >= warning/severe/critical
    * ``stop_loss`` — fell straight through ``-stop_loss`` versus cost, no run-up needed

    Returns a dict with:
        alerts: List[Alert] (empty if no triggers)
        positions: List[PositionDrawdown] (one per input position)
        skipped: List[PositionDrawdown] (evaluated fine but triggered nothing)
        as_of: ISO date string of the evaluation
    """
    cfg = {
        # drawdown-from-peak tiers
        "warning": 20.0,
        "severe": 50.0,
        "critical": 80.0,
        "min_gain": 5.0,
        # stop-loss is a single threshold, measured as loss versus cost
        "stop_loss": 5.0,
    }
    if thresholds:
        cfg.update({k: float(v) for k, v in thresholds.items() if v is not None})

    as_of = today or date.today()
    positions_raw = portfolio.get("positions") or []
    if not isinstance(positions_raw, list):
        raise ValueError("portfolio.positions must be a list")

    results: List[PositionDrawdown] = []
    alerts: List[Alert] = []
    skipped: List[PositionDrawdown] = []
    for raw in positions_raw:
        if not isinstance(raw, dict):
            logger.warning("skip non-dict position entry: %r", raw)
            continue
        p = _compute_position(raw, as_of)
        results.append(p)
        if p.error:
            logger.warning("[%s] %s", p.code, p.error)
            continue

        triggered = False

        drawdown_sev = _pick_severity(p, cfg)
        if drawdown_sev is not None:
            alerts.append(_build_alert(p, drawdown_sev, AlertKind.DRAWDOWN))
            triggered = True

        stop_sev = _pick_stop_loss(p, cfg)
        if stop_sev is not None:
            alerts.append(_build_alert(p, stop_sev, AlertKind.STOP_LOSS))
            triggered = True

        if not triggered:
            skipped.append(p)

    return {
        "alerts": alerts,
        "positions": results,
        "skipped": skipped,
        "as_of": as_of.isoformat(),
        "thresholds": cfg,
    }
