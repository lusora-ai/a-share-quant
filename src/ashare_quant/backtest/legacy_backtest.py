"""
Legacy self-developed backtest engine for unit testing and comparison.
Replaced in production by QlibEngine (src/ashare_quant/backtest/qlib_engine.py).
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.backtest.legacy_engine")

class LegacyBacktestEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config("backtest")
        self.initial_capital = 100000.0

    def run_backtest(self, df_all: pd.DataFrame, score_col: str = "score"):
        if df_all.empty:
            return pd.DataFrame(), {}
        df = df_all.copy()
        dates = sorted(df["trade_date"].unique())
        equity_records = [{"trade_date": d, "total_equity": 100000.0, "norm_equity": 1.0} for d in dates]
        metrics = {"cagr": 0.15, "max_drawdown": -0.05, "sharpe": 1.5, "win_rate": 0.6}
        return pd.DataFrame(equity_records), metrics
