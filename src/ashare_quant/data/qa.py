import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.data.qa")

class DataQAValidator:
    """
    数据质量检查与数据防泄漏校验器
    对 Raw 价格与 Adj 价格进行独立的逻辑校验
    """
    def __init__(self, check_ohlc: bool = True, check_price_positive: bool = True, max_missing_ratio: float = 0.05):
        self.check_ohlc = check_ohlc
        self.check_price_positive = check_price_positive
        self.max_missing_ratio = max_missing_ratio

    def validate_daily_ohlcv(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        验证并清洗日线数据，生成 QA 校验报告
        """
        report = {
            "total_rows_input": len(df),
            "duplicates_removed": 0,
            "ohlc_errors_fixed": 0,
            "negative_price_errors": 0,
            "missing_values_filled": 0,
            "is_valid": True,
            "quality_flag": "OK",
            "errors": []
        }
        
        if df.empty:
            report["is_valid"] = False
            report["quality_flag"] = "DATA_QUALITY_DEGRADED"
            report["errors"].append("Input DataFrame is empty.")
            return df, report
            
        clean_df = df.copy()
        
        # 1. 检查与去重 (ts_code + trade_date 唯一)
        before_dup = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
        report["duplicates_removed"] = before_dup - len(clean_df)
        
        # 2. 验证价格正数 (检查 open_raw 与 open_adj)
        if self.check_price_positive:
            for prefix in ["raw", "adj"]:
                for col_base in ["open", "high", "low", "close"]:
                    col = f"{col_base}_{prefix}"
                    if col in clean_df.columns:
                        invalid_mask = clean_df[col] <= 0
                        invalid_count = invalid_mask.sum()
                        if invalid_count > 0:
                            report["negative_price_errors"] += invalid_count
                            report["errors"].append(f"Found {invalid_count} non-positive prices in column '{col}'.")
                            clean_df = clean_df[~invalid_mask]
                            
        prefixes = ["raw", "adj"]
        if not any(f"open_{p}" in clean_df.columns for p in prefixes) and "open" in clean_df.columns:
            prefixes = [""]

        if self.check_ohlc and not clean_df.empty:
            for prefix in prefixes:
                p_str = f"_{prefix}" if prefix else ""
                open_col, high_col, low_col, close_col = f"open{p_str}", f"high{p_str}", f"low{p_str}", f"close{p_str}"
                if open_col in clean_df.columns and high_col in clean_df.columns:
                    max_oc = clean_df[[open_col, close_col]].max(axis=1)
                    min_oc = clean_df[[open_col, close_col]].min(axis=1)
                    
                    high_invalid = clean_df[high_col] < max_oc - 1e-4
                    low_invalid = clean_df[low_col] > min_oc + 1e-4
                    
                    ohlc_invalid_mask = high_invalid | low_invalid
                    ohlc_invalid_count = ohlc_invalid_mask.sum()
                    
                    if ohlc_invalid_count > 0:
                        report["ohlc_errors_fixed"] += ohlc_invalid_count
                        logger.warning(f"Fixing {ohlc_invalid_count} rows with invalid OHLC relationship in '{open_col}'.")
                        clean_df.loc[high_invalid, high_col] = max_oc[high_invalid]
                        clean_df.loc[low_invalid, low_col] = min_oc[low_invalid]
                        
        # 4. 检查涨跌停标记是否降级
        if "limit_buy" not in clean_df.columns or clean_df["limit_buy"].isna().any():
            report["quality_flag"] = "DATA_QUALITY_DEGRADED"
            clean_df["data_quality_flag"] = "DATA_QUALITY_DEGRADED"
            logger.warning("Limit up/down fields missing or corrupted. Marked DATA_QUALITY_DEGRADED.")
            
        # 5. 缺失值比率检查
        missing_cols = [c for c in ["open_raw", "close_raw", "open_adj", "close_adj", "volume"] if c in clean_df.columns]
        missing_count = clean_df[missing_cols].isna().sum().sum()
        missing_ratio = missing_count / (len(clean_df) * len(missing_cols)) if len(clean_df) > 0 else 0
        if missing_ratio > self.max_missing_ratio:
            report["is_valid"] = False
            report["quality_flag"] = "DATA_QUALITY_DEGRADED"
            report["errors"].append(f"Missing ratio {missing_ratio:.2%} exceeds max threshold {self.max_missing_ratio:.2%}.")
            
        clean_df = clean_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        report["total_rows_output"] = len(clean_df)
        
        return clean_df, report
