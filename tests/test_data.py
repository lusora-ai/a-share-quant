import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from ashare_quant.data.qa import DataQAValidator
from ashare_quant.data.storage import StorageEngine

def test_qa_validator_ohlc_fix():
    validator = DataQAValidator(check_ohlc=True, check_price_positive=True)
    
    # 构建有 OHLC 逻辑问题的样本数据
    raw_data = pd.DataFrame([
        {"ts_code": "600000.SH", "trade_date": "2024-01-01", "open": 10.0, "high": 9.5, "low": 9.0, "close": 10.5, "volume": 1000}, # high < max(open, close)
        {"ts_code": "600000.SH", "trade_date": "2024-01-02", "open": 10.0, "high": 11.0, "low": 10.2, "close": 9.8, "volume": 1000}, # low > min(open, close)
        {"ts_code": "600000.SH", "trade_date": "2024-01-02", "open": 10.0, "high": 11.0, "low": 9.8, "close": 9.8, "volume": 1000},  # 重复行
    ])
    
    clean_df, report = validator.validate_daily_ohlcv(raw_data)
    
    assert report["duplicates_removed"] == 1
    assert report["ohlc_errors_fixed"] == 1
    assert len(clean_df) == 2
    
    # 验证修复后的 high 与 low
    row0 = clean_df[clean_df["trade_date"] == "2024-01-01"].iloc[0]
    assert row0["high"] >= max(row0["open"], row0["close"])
    
    row1 = clean_df[clean_df["trade_date"] == "2024-01-02"].iloc[0]
    assert row1["low"] <= min(row1["open"], row1["close"])

def test_storage_engine_parquet_and_duckdb():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageEngine(data_dir=tmpdir, db_name="test.duckdb")
        
        df_sample = pd.DataFrame({
            "ts_code": ["600000.SH", "000001.SZ"],
            "trade_date": ["2024-01-01", "2024-01-01"],
            "close": [10.5, 12.3]
        })
        
        # 1. Parquet 存储与读取
        storage.save_parquet(df_sample, "test_dataset", is_processed=True)
        df_loaded = storage.load_parquet("test_dataset", is_processed=True)
        assert len(df_loaded) == 2
        assert set(df_loaded["ts_code"]) == {"600000.SH", "000001.SZ"}
        
        # 2. DuckDB 同步与查询
        storage.sync_to_duckdb("test_table", df_sample, if_exists="replace")
        df_sql = storage.query_duckdb("SELECT * FROM test_table WHERE close > 11.0")
        assert len(df_sql) == 1
        assert df_sql.iloc[0]["ts_code"] == "000001.SZ"
        
        storage.close()
