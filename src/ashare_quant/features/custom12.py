import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.features.custom12")

FACTOR_NAMES_12 = [
    "return_5d", "return_20d", "return_60d",
    "ma_dist_5_20", "ma_dist_20_60", "dist_high_20d",
    "reversal_1d", "reversal_3d",
    "volatility_20d", "atr_ratio_20d",
    "volume_zscore_20d", "turnover_proxy_20d"
]

class Custom12Factors:
    """
    12个基线解释型因子算子 (Interpretable Baseline Factors)
    使用 open_adj / high_adj / low_adj / close_adj 研究复权价格计算
    """
    def compute(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        if daily_df.empty:
            return daily_df
            
        df = daily_df.copy()
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        
        close_col = "close_adj" if "close_adj" in df.columns else "close"
        high_col = "high_adj" if "high_adj" in df.columns else "high"
        low_col = "low_adj" if "low_adj" in df.columns else "low"
        
        def calc_group(g: pd.DataFrame) -> pd.DataFrame:
            c = g[close_col]
            h = g[high_col]
            l = g[low_col]
            v = g["volume"]
            amt = g["amount"]
            pct = g["pct_chg"] if "pct_chg" in g.columns else c.pct_change()
            
            g["return_5d"] = c / c.shift(5) - 1.0
            g["return_20d"] = c / c.shift(20) - 1.0
            g["return_60d"] = c / c.shift(60) - 1.0
            
            ma5 = c.rolling(5, min_periods=3).mean()
            ma20 = c.rolling(20, min_periods=10).mean()
            ma60 = c.rolling(60, min_periods=30).mean()
            
            g["ma_dist_5_20"] = (ma5 / ma20) - 1.0
            g["ma_dist_20_60"] = (ma20 / ma60) - 1.0
            g["dist_high_20d"] = (c / h.rolling(20, min_periods=10).max()) - 1.0
            
            g["reversal_1d"] = -(c / c.shift(1) - 1.0)
            g["reversal_3d"] = -(c / c.shift(3) - 1.0)
            
            g["volatility_20d"] = pct.rolling(20, min_periods=10).std()
            
            prev_c = c.shift(1)
            tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))
            g["atr_ratio_20d"] = tr.rolling(20, min_periods=10).mean() / c
            
            v_mean = v.rolling(20, min_periods=10).mean()
            v_std = v.rolling(20, min_periods=10).std()
            g["volume_zscore_20d"] = (v - v_mean) / (v_std + 1e-8)
            
            if "turn" in g.columns:
                g["turnover_proxy_20d"] = g["turn"].rolling(20, min_periods=10).mean()
            else:
                g["turnover_proxy_20d"] = np.log1p(amt.rolling(20, min_periods=10).mean())
            return g

        group_list = []
        for code, group in df.groupby("ts_code"):
            group_list.append(calc_group(group.copy()))
            
        return pd.concat(group_list, ignore_index=True)
