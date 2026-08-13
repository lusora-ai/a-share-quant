import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.labels.executable_5d")

class ExecutableLabel5D:
    """
    正式生产标签生成器 (5-Day Executable Return Label)
    时间语义:
    t 日 15:00 收盘后生成信号 signal_t
    t+1 日开盘价格 open[t+1] 买入执行
    t+5 日收盘价格 close[t+5] 卖出平仓
    可执行收益率 exec_return = close[t+5] / open[t+1] - 1.0
    超额收益率 excess_return = exec_return - benchmark_exec_return
    每天做截面 Percentile Rank (0.0 ~ 1.0)
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
        生成严格匹配真实可执行收益的 5 日超额 percentile rank 标签
        """
        if daily_df.empty:
            logger.warning("Empty DataFrame passed to ExecutableLabel5D.")
            return daily_df
            
        df = daily_df.copy()
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        
        # 兼容 open_adj 与 open，close_adj 与 close
        open_col = "open_adj" if "open_adj" in df.columns else "open"
        close_col = "close_adj" if "close_adj" in df.columns else "close"
        
        logger.info(f"Generating executable 5D return label using {open_col} at t+1 and {close_col} at t+5...")
        
        # 1. 计算 exec_return = close[t+5] / open[t+1] - 1.0
        def compute_stock_exec_return(g: pd.DataFrame) -> pd.DataFrame:
            open_next = g[open_col].shift(-1)
            close_future = g[close_col].shift(-self.horizon)
            g["forward_5d_exec_return"] = (close_future / (open_next + 1e-8)) - 1.0
            return g
            
        group_list = []
        for code, group in df.groupby("ts_code"):
            g_lbl = compute_stock_exec_return(group.copy())
            group_list.append(g_lbl)
            
        df = pd.concat(group_list, ignore_index=True)
        
        # 2. 扣除基准收益 (若未指定，用截面均值作为代理基准)
        if benchmark_df is not None and not benchmark_df.empty:
            bm = benchmark_df.sort_values("trade_date").reset_index(drop=True)
            bm_open_col = "open_adj" if "open_adj" in bm.columns else "open"
            bm_close_col = "close_adj" if "close_adj" in bm.columns else "close"
            bm["bm_exec_return"] = (bm[bm_close_col].shift(-self.horizon) / (bm[bm_open_col].shift(-1) + 1e-8)) - 1.0
            bm_map = bm.set_index("trade_date")["bm_exec_return"].to_dict()
            df["bm_exec_return"] = df["trade_date"].map(bm_map)
        else:
            df["bm_exec_return"] = df.groupby("trade_date")["forward_5d_exec_return"].transform("mean")
            
        df["excess_return_5d"] = df["forward_5d_exec_return"] - df["bm_exec_return"]
        
        # 3. 每日截面 Percentile Rank (0.0 ~ 1.0)
        rank_col = "rank_label_5d"
        date_groups = []
        for date_val, group in df.groupby("trade_date"):
            g = group.copy()
            valid_mask = g["excess_return_5d"].notna()
            if valid_mask.any():
                g.loc[valid_mask, rank_col] = g.loc[valid_mask, "excess_return_5d"].rank(pct=True)
            else:
                g[rank_col] = np.nan
            date_groups.append(g)
            
        res_df = pd.concat(date_groups, ignore_index=True)
        logger.info("Successfully generated executable 5D rank labels.")
        return res_df
