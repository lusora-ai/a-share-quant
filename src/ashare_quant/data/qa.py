import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.data.qa")

class DataQAValidator:
    """
    数据质量检查与数据防泄漏校验器
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
            "errors": []
        }
        
        if df.empty:
            report["is_valid"] = False
            report["errors"].append("Input DataFrame is empty.")
            return df, report
            
        clean_df = df.copy()
        
        # 1. 检查与去重 (ts_code + trade_date 唯一)
        before_dup = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
        report["duplicates_removed"] = before_dup - len(clean_df)
        
        # 2. 验证价格正数
        if self.check_price_positive:
            for col in ["open", "high", "low", "close"]:
                if col in clean_df.columns:
                    invalid_mask = clean_df[col] <= 0
                    invalid_count = invalid_mask.sum()
                    if invalid_count > 0:
                        report["negative_price_errors"] += invalid_count
                        report["errors"].append(f"Found {invalid_count} non-positive prices in column '{col}'.")
                        # 过滤掉非法价格行
                        clean_df = clean_df[~invalid_mask]
                        
        # 3. OHLC 逻辑校验: high >= max(open, close) 且 low <= min(open, close)
        if self.check_ohlc and not clean_df.empty:
            max_oc = clean_df[["open", "close"]].max(axis=1)
            min_oc = clean_df[["open", "close"]].min(axis=1)
            
            high_invalid = clean_df["high"] < max_oc - 1e-4
            low_invalid = clean_df["low"] > min_oc + 1e-4
            
            ohlc_invalid_mask = high_invalid | low_invalid
            ohlc_invalid_count = ohlc_invalid_mask.sum()
            
            if ohlc_invalid_count > 0:
                report["ohlc_errors_fixed"] = ohlc_invalid_count
                logger.warning(f"Fixing {ohlc_invalid_count} rows with invalid OHLC relationship.")
                # 修复 OHLC 边界关系
                clean_df.loc[high_invalid, "high"] = max_oc[high_invalid]
                clean_df.loc[low_invalid, "low"] = min_oc[low_invalid]
                
        # 4. 缺失值比率检查
        missing_count = clean_df[["open", "high", "low", "close", "volume"]].isna().sum().sum()
        missing_ratio = missing_count / (len(clean_df) * 5) if len(clean_df) > 0 else 0
        if missing_ratio > self.max_missing_ratio:
            report["is_valid"] = False
            report["errors"].append(f"Missing ratio {missing_ratio:.2%} exceeds max threshold {self.max_missing_ratio:.2%}.")
            
        # 5. 最终按 ts_code 与 trade_date 排序
        clean_df = clean_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        report["total_rows_output"] = len(clean_df)
        
        logger.info(f"QA Validation Completed. Input: {report['total_rows_input']} | Output: {report['total_rows_output']} | Valid: {report['is_valid']}")
        return clean_df, report
