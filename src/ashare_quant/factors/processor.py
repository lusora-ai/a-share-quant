import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from ashare_quant.factors.engine import FACTOR_NAMES
from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.factors.processor")

class FactorProcessor:
    """
    因子横截面处理引擎:
    - 极端值处理 (Winsorization / Clipping)
    - 缺失值填充与横截面标准化 (Z-score / Rank / MinMax)
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config("factors")
        self.winsorize_cfg = self.config.get("winsorize", {})
        self.standardize_cfg = self.config.get("standardize", {})

    def process_cross_section(
        self,
        df: pd.DataFrame,
        factor_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        按 trade_date 对截面上的因子逐一进行 Winsorize 和 Standardize
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to FactorProcessor.")
            return df
            
        factors = factor_cols or [col for col in FACTOR_NAMES if col in df.columns]
        if not factors:
            logger.warning("No factors found to process.")
            return df
            
        res = df.copy()
        lower_q = self.winsorize_cfg.get("lower_quantile", 0.01)
        upper_q = self.winsorize_cfg.get("upper_quantile", 0.99)
        std_method = self.standardize_cfg.get("method", "zscore")
        
        logger.info(f"Processing {len(factors)} factors cross-sectionally | Method: {std_method} | Winsorize: [{lower_q}, {upper_q}]")
        
        def process_date_group(g: pd.DataFrame) -> pd.DataFrame:
            for factor in factors:
                series = g[factor]
                if series.dropna().empty:
                    continue
                    
                # 1. Winsorize 剪裁
                if self.winsorize_cfg.get("enabled", True):
                    l_val = series.quantile(lower_q)
                    u_val = series.quantile(upper_q)
                    series = series.clip(lower=l_val, upper=u_val)
                    
                # 2. 缺失值填充 (截面中位数)
                median_val = series.median()
                series = series.fillna(median_val if not pd.isna(median_val) else 0.0)
                
                # 3. 标准化
                if std_method == "zscore":
                    std_val = series.std()
                    mean_val = series.mean()
                    if std_val > 1e-8:
                        series = (series - mean_val) / std_val
                    else:
                        series = series - mean_val
                elif std_method == "rank":
                    series = series.rank(pct=True)
                elif std_method == "minmax":
                    min_v = series.min()
                    max_v = series.max()
                    if max_v - min_v > 1e-8:
                        series = (series - min_v) / (max_v - min_v)
                        
                g[factor] = series
            return g
            
        date_groups = []
        for date_val, group in res.groupby("trade_date"):
            processed_group = process_date_group(group.copy())
            date_groups.append(processed_group)
            
        processed_df = pd.concat(date_groups, ignore_index=True)
        logger.info("Cross-sectional factor processing completed.")
        return processed_df
