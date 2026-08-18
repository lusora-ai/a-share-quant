import pandas as pd
import numpy as np
import pytest
from ashare_quant.backtest.engine import BacktestEngine, QlibEngineAdapter, DataSchemaError
from tests.test_factors import generate_mock_daily_data

def test_backtest_engine_initialization():
    engine = BacktestEngine()
    assert engine.trade_unit == 100
    assert engine.top_k == 10
    assert isinstance(engine, QlibEngineAdapter)

def test_p0_raw_price_schema_enforced():
    """
    P0 测试: 验证缺少 raw price 价格字段时抛出 DataSchemaError 异常阻止虚假回测
    """
    mock_df = generate_mock_daily_data(num_stocks=2, num_days=20)
    mock_df_no_raw = mock_df.drop(columns=["open_raw", "close_raw"])
    engine = BacktestEngine()
    with pytest.raises(DataSchemaError, match="DataSchemaError"):
        engine.validate_price_schema(mock_df_no_raw)
        
    # Passes validation with raw prices present
    engine.validate_price_schema(mock_df)
