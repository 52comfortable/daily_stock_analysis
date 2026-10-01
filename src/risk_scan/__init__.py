"""持仓排雷扫描 (risk_scan).

按持仓清单逐只拉取巨潮公告，用风险阶梯 + 状态机判定每只持仓的监管风险等级，
命中时通过 DSA 统一通知渠道推送。

与 ``src.position_drawdown`` 同构：engine / notifier / cli 三件套 + 独立
GitHub Actions workflow，不接入 ``main.py`` 主流程。

覆盖范围：仅 A 股个股（巨潮资讯）。
港股、基金、ETF、LOF 不在覆盖内，会在报告中显式列为"未覆盖"，不做静默跳过。
"""

from .engine import RiskReport, RiskVerdict, evaluate_holdings
from .rules import RiskLevel, match_rules

__all__ = [
    "RiskLevel",
    "RiskReport",
    "RiskVerdict",
    "evaluate_holdings",
    "match_rules",
]
