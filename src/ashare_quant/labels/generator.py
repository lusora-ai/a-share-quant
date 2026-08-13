import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.labels.generator")

class LabelGenerator:
    """
    横截面 5 日相对收益 Rank 标签生成器
    - 预测未来 5 个交易日个股收益率减去同期基准收益率
    - 转换成每日横截面 Percentile Rank (0.0 ~ 1.0)
    - 严禁进入特征矩阵
    """
    def __init__(self, horizon: int = 5, benchmark_code: str = "000300.SH"):
        self.horizon = horizon
        self.benchmark_code = benchmark_code

    def generate_labels(
        self,
        daily_df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        生成未来 N 日相对收益 percentile rank 标签: rank_label_5d
        """
        if daily_df.empty:
            logger.warning("Empty DataFrame passed to LabelGenerator.")
            return daily_df
            
        df = daily_df.copy()
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        
        logger.info(f"Generating {self.horizon}-day forward relative excess return rank labels...")
        
        # 1. 计算未来 N 日个股收益率: Return(i, t+1 -> t+N) = Close_{t+N} / Close_{t} - 1.0
        def compute_forward_return(g: pd.DataFrame) -> pd.DataFrame:
            close = g["close"]
            # shift(-N) 表示未来第 N 天的收盘价
            future_close = close.shift(-self.horizon)
            g[f"forward_return_{self.horizon}d"] = (future_close / close) - 1.0
            return g
            
        group_list = []
        for code, group in df.groupby("ts_code"):
            g_lbl = compute_forward_return(group.copy())
            group_list.append(g_lbl)
            
        df = pd.concat(group_list, ignore_index=True)
        
        # 2. 计算或扣除基准收益 (若未提供 benchmark_df，则取当日截面平均收益作为代理)
        if benchmark_df is not None and not benchmark_df.empty:
            bm = benchmark_df.copy()
            bm["bm_forward_return"] = (bm["close"].shift(-self.horizon) / bm["close"]) - 1.0
            bm_map = bm.set_index("trade_date")["bm_forward_return"].to_dict()
            df["bm_forward_return"] = df["trade_date"].map(bm_map)
        else:
            # 取截面均值作为代理基准
            df["bm_forward_return"] = df.groupby("trade_date")[f"forward_return_{self.horizon}d"].transform("mean")
            
        # raw_label = forward_return - bm_forward_return
        raw_label_col = f"raw_label_{self.horizon}d"
        df[raw_label_col] = df[f"forward_return_{self.horizon}d"] - df["bm_forward_return"]
        
        # 3. 计算每日横截面 Percentile Rank (0.0 ~ 1.0)
        rank_label_col = f"rank_label_{self.horizon}d"
        
        date_groups = []
        for date_val, group in df.groupby("trade_date"):
            g = group.copy()
            valid_mask = g[raw_label_col].notna()
            if valid_mask.any():
                g.loc[valid_mask, rank_label_col] = g.loc[valid_mask, raw_label_col].rank(pct=True)
            else:
                g[rank_label_col] = np.nan
            date_groups.append(g)
            
        res_df = pd.concat(date_groups, ignore_index=True)
        logger.info(f"Successfully generated rank label '{rank_label_col}'.")
        return res_df
