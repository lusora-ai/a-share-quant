import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Any, Tuple
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.models.metrics")

def compute_daily_ic(df: pd.DataFrame, score_col: str, label_col: str) -> pd.DataFrame:
    """
    按 trade_date 计算每日 Spearman RankIC
    """
    if df.empty or score_col not in df.columns or label_col not in df.columns:
        logger.warning("Invalid DataFrame or missing columns for IC computation.")
        return pd.DataFrame()
        
    ic_list = []
    for date_val, group in df.groupby("trade_date"):
        valid = group[[score_col, label_col]].dropna()
        if len(valid) >= 5:
            ic, _ = spearmanr(valid[score_col], valid[label_col])
            if not np.isnan(ic):
                ic_list.append({"trade_date": date_val, "rank_ic": ic})
                
    return pd.DataFrame(ic_list)

def compute_ic_stats(ic_df: pd.DataFrame) -> Dict[str, float]:
    """
    计算 IC 统计量: Mean IC, Std IC, ICIR, Positive IC Ratio
    """
    if ic_df.empty or "rank_ic" not in ic_df.columns:
        return {"mean_ic": 0.0, "std_ic": 0.0, "icir": 0.0, "pos_ratio": 0.0}
        
    rank_ic = ic_df["rank_ic"]
    mean_ic = rank_ic.mean()
    std_ic = rank_ic.std()
    icir = mean_ic / (std_ic + 1e-8)
    pos_ratio = (rank_ic > 0).mean()
    
    return {
        "mean_ic": float(mean_ic),
        "std_ic": float(std_ic),
        "icir": float(icir),
        "pos_ratio": float(pos_ratio)
    }

def compute_quantile_performance(
    df: pd.DataFrame,
    score_col: str,
    return_col: str,
    n_quantiles: int = 5
) -> pd.DataFrame:
    """
    分位数分组表现评估 (1 ~ N 组收益分布)
    """
    if df.empty or score_col not in df.columns or return_col not in df.columns:
        return pd.DataFrame()
        
    res = df.copy()
    
    def assign_quantile(group: pd.DataFrame) -> pd.DataFrame:
        valid = group[score_col].dropna()
        if len(valid) >= n_quantiles:
            group["quantile"] = pd.qcut(group[score_col], q=n_quantiles, labels=False, duplicates="drop") + 1
        else:
            group["quantile"] = np.nan
        return group
        
    res = res.groupby("trade_date", group_keys=False).apply(assign_quantile)
    
    quantile_perf = res.groupby(["quantile"])[return_col].agg(["mean", "std", "count"]).reset_index()
    return quantile_perf
