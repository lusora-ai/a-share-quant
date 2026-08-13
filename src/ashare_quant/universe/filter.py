import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.universe.filter")

class UniverseFilter:
    """
    可交易股票池（Universe）过滤器
    依据当时可获取的信息构建截面股票池，规避未来函数与幸存者偏差
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config("data").get("universe", {})
        self.min_list_days = self.config.get("min_list_days", 120)
        self.exclude_st = self.config.get("exclude_st", True)
        self.exclude_suspended = self.config.get("exclude_suspended", True)
        self.min_avg_amount_20d = self.config.get("min_avg_amount_20d", 10000000.0)

    def filter_universe(self, daily_df: pd.DataFrame, master_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        根据配置过滤日线股票池
        输入 DataFrame 必须包含: [ts_code, trade_date, open, high, low, close, volume, amount, is_suspended]
        输出: 过滤后的 DataFrame (附带 is_in_universe 标识)
        """
        if daily_df.empty:
            logger.warning("Empty daily DataFrame passed to UniverseFilter.")
            return daily_df
            
        df = daily_df.copy()
        
        # 1. 股票名称/代码 ST 判断
        if "name" in df.columns:
            st_mask = df["name"].astype(str).str.contains(r"ST|\*ST|退", regex=True)
        else:
            st_mask = pd.Series(False, index=df.index)
            
        # 2. 20日平均成交额计算
        df["amount_20d_avg"] = df.groupby("ts_code")["amount"].transform(lambda x: x.rolling(20, min_periods=5).mean())
        
        # 3. 上市天数过滤 (基于历史记录出现的总交易日)
        df["list_days_count"] = df.groupby("ts_code").cumcount() + 1
        
        # 综合过滤条件
        valid_mask = pd.Series(True, index=df.index)
        
        if self.exclude_st:
            valid_mask &= (~st_mask)
            
        if self.exclude_suspended and "is_suspended" in df.columns:
            valid_mask &= (~df["is_suspended"].fillna(False))
            
        if self.min_list_days > 0:
            valid_mask &= (df["list_days_count"] >= self.min_list_days)
            
        if self.min_avg_amount_20d > 0:
            valid_mask &= (df["amount_20d_avg"] >= self.min_avg_amount_20d)
            
        df["is_in_universe"] = valid_mask
        
        filtered_count = valid_mask.sum()
        total_count = len(df)
        logger.info(f"Universe filter applied: {filtered_count}/{total_count} records ({filtered_count/total_count:.1%}) retained in universe.")
        
        return df
