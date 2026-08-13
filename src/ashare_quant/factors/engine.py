import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.factors.engine")

FACTOR_NAMES = [
    "return_5d",
    "return_20d",
    "return_60d",
    "ma_dist_5_20",
    "ma_dist_20_60",
    "dist_high_20d",
    "reversal_1d",
    "reversal_3d",
    "volatility_20d",
    "atr_ratio_20d",
    "volume_zscore_20d",
    "turnover_proxy_20d",
]

class FactorEngine:
    """
    12个基线因子计算引擎
    严格遵守无未来函数约束，所有 Rolling 计算仅作用于当前及历史日期
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config("factors")

    def compute_factors(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 12 个基线因子
        输入 DataFrame 包含按 [ts_code, trade_date] 排序的行情
        """
        if daily_df.empty:
            logger.warning("Empty daily DataFrame passed to FactorEngine.")
            return daily_df
            
        df = daily_df.copy()
        
        # 确保数据按 ts_code 和 trade_date 正向排序
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        
        logger.info(f"Computing 12 baseline factors for {len(df['ts_code'].unique())} stocks...")
        
        # 按照 ts_code 分组计算
        def compute_group_factors(g: pd.DataFrame) -> pd.DataFrame:
            close = g["close"]
            high = g["high"]
            low = g["low"]
            volume = g["volume"]
            amount = g["amount"]
            pct_chg = g["pct_chg"] if "pct_chg" in g.columns else close.pct_change()
            
            # 1. 动量因子
            g["return_5d"] = close / close.shift(5) - 1.0
            g["return_20d"] = close / close.shift(20) - 1.0
            g["return_60d"] = close / close.shift(60) - 1.0
            
            # 2. 趋势因子
            ma5 = close.rolling(5, min_periods=3).mean()
            ma20 = close.rolling(20, min_periods=10).mean()
            ma60 = close.rolling(60, min_periods=30).mean()
            
            g["ma_dist_5_20"] = (ma5 / ma20) - 1.0
            g["ma_dist_20_60"] = (ma20 / ma60) - 1.0
            
            high_20 = high.rolling(20, min_periods=10).max()
            g["dist_high_20d"] = (close / high_20) - 1.0
            
            # 3. 反转因子
            g["reversal_1d"] = -(close / close.shift(1) - 1.0)
            g["reversal_3d"] = -(close / close.shift(3) - 1.0)
            
            # 4. 波动率因子
            g["volatility_20d"] = pct_chg.rolling(20, min_periods=10).std()
            
            prev_close = close.shift(1)
            tr = np.maximum(
                high - low,
                np.maximum(
                    (high - prev_close).abs(),
                    (low - prev_close).abs()
                )
            )
            atr_20 = tr.rolling(20, min_periods=10).mean()
            g["atr_ratio_20d"] = atr_20 / close
            
            # 5. 量价/流动性因子
            vol_mean_20 = volume.rolling(20, min_periods=10).mean()
            vol_std_20 = volume.rolling(20, min_periods=10).std()
            g["volume_zscore_20d"] = (volume - vol_mean_20) / (vol_std_20 + 1e-8)
            
            if "turn" in g.columns:
                g["turnover_proxy_20d"] = g["turn"].rolling(20, min_periods=10).mean()
            else:
                g["turnover_proxy_20d"] = np.log1p(amount.rolling(20, min_periods=10).mean())
                
            return g
            
        group_list = []
        for code, group in df.groupby("ts_code"):
            g_factors = compute_group_factors(group.copy())
            group_list.append(g_factors)
            
        result_df = pd.concat(group_list, ignore_index=True)
        logger.info("Successfully computed all 12 baseline factors.")
        return result_df
