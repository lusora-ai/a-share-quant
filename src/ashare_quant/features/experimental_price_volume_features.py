import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.features.experimental_price_volume")

class ExperimentalPriceVolumeFeatures:
    """
    实验性量价多周期衍生特征库 (前身：简化量价特征)
    包含基础 K 线形态、多周期 Rolling 均值、动量、波动率与换手率算子
    """
    def __init__(self):
        self.windows = [5, 10, 20, 30, 60]

    def compute(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        if daily_df.empty:
            return daily_df
            
        df = daily_df.copy()
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        
        close_col = "close_adj" if "close_adj" in df.columns else "close"
        open_col = "open_adj" if "open_adj" in df.columns else "open"
        high_col = "high_adj" if "high_adj" in df.columns else "high"
        low_col = "low_adj" if "low_adj" in df.columns else "low"
        
        logger.info(f"Computing experimental price volume features for {len(df['ts_code'].unique())} stocks...")
        
        def calc_group_features(g: pd.DataFrame) -> pd.DataFrame:
            c = g[close_col]
            o = g[open_col]
            h = g[high_col]
            l = g[low_col]
            v = g["volume"]
            
            # K线形态
            g["KMID"] = (c - o) / (o + 1e-8)
            g["KLEN"] = (h - l) / (o + 1e-8)
            g["KMID2"] = (c - o) / (h - l + 1e-8)
            g["KUP"] = (h - np.maximum(c, o)) / (o + 1e-8)
            g["KLOW"] = (np.minimum(c, o) - l) / (o + 1e-8)
            g["KSFT"] = (2 * c - h - l) / (h - l + 1e-8)
            
            # 多周期统计量
            for w in self.windows:
                g[f"ROC_{w}"] = c / (c.shift(w) + 1e-8) - 1.0
                g[f"MA_{w}"] = c.rolling(w, min_periods=max(2, w//2)).mean() / (c + 1e-8) - 1.0
                g[f"STD_{w}"] = c.pct_change().rolling(w, min_periods=max(2, w//2)).std()
                g[f"VMA_{w}"] = v.rolling(w, min_periods=max(2, w//2)).mean() / (v + 1e-8) - 1.0
                g[f"VSTD_{w}"] = v.rolling(w, min_periods=max(2, w//2)).std() / (v.mean() + 1e-8)
                
            return g

        group_list = []
        for code, group in df.groupby("ts_code"):
            group_list.append(calc_group_features(group.copy()))
            
        res_df = pd.concat(group_list, ignore_index=True)
        return res_df
