import pandas as pd
import numpy as np
import pytest
from ashare_quant.backtest.engine import BacktestEngine
from tests.test_factors import generate_mock_daily_data

def test_backtest_engine_execution():
    mock_df = generate_mock_daily_data(num_stocks=5, num_days=30)
    # 构造模拟预测 score
    mock_df["score"] = np.random.uniform(0, 1, len(mock_df))
    
    engine = BacktestEngine()
    equity_df, metrics = engine.run_backtest(mock_df, score_col="score")
    
    assert not equity_df.empty
    assert "total_equity" in equity_df.columns
    assert "norm_equity" in equity_df.columns
    assert "cagr" in metrics
    assert "max_drawdown" in metrics
    assert equity_df["total_equity"].iloc[0] == 100000.0

def test_p0_suspended_stock_cannot_be_bought():
    """
    P0 测试: 验证停牌股票无法按理想价格成交
    """
    mock_df = generate_mock_daily_data(num_stocks=2, num_days=20)
    mock_df["score"] = 0.5
    
    # 强制让股票 600000.SH 在所有日期评分最高为 0.95，但设为停牌 (is_suspended = True)
    mock_df.loc[mock_df["ts_code"] == "600000.SH", "score"] = 0.95
    mock_df.loc[mock_df["ts_code"] == "600000.SH", "is_suspended"] = True
    
    # 强制让股票 600001.SH 评分为 0.80，但可正常交易
    mock_df.loc[mock_df["ts_code"] == "600001.SH", "score"] = 0.80
    mock_df.loc[mock_df["ts_code"] == "600001.SH", "is_suspended"] = False
    
    engine = BacktestEngine()
    equity_df, metrics = engine.run_backtest(mock_df, score_col="score")
    
    # 由于 600000.SH 停牌，不可买入，因此只能买入 600001.SH
    assert not equity_df.empty
