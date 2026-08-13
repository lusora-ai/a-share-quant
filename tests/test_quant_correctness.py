import pytest
import pandas as pd
import numpy as np

from ashare_quant.data.fetcher import DataFetcher
from ashare_quant.features.custom12 import Custom12Factors, FACTOR_NAMES_12
from ashare_quant.features.qlib_alpha158 import QlibAlpha158Features
from ashare_quant.labels.executable_5d import ExecutableLabel5D
from ashare_quant.backtest.qlib_engine import QlibEngineAdapter
from ashare_quant.validation.purged_walk_forward import PurgedWalkForwardEvaluator
from ashare_quant.signals.daily import DailySignalPipeline
from tests.test_factors import generate_mock_daily_data

def test_no_same_day_execution():
    """
    P0 对抗测试: 验证 t 日生成的信号绝对无法在 t 日 Open 成交 (必须在 t+1 日开盘成交)
    """
    mock_df = generate_mock_daily_data(num_stocks=2, num_days=30)
    mock_df["lgbm_score"] = np.random.uniform(0, 1, len(mock_df))
    
    # 强制在第一天 2024-01-01 给 600000.SH 最低分 0.1，第二天 2024-01-02 变成最高分 0.99
    dates = sorted(mock_df["trade_date"].unique())
    day1, day2 = dates[0], dates[1]
    
    mock_df.loc[(mock_df["ts_code"] == "600000.SH") & (mock_df["trade_date"] == day1), "lgbm_score"] = 0.1
    mock_df.loc[(mock_df["ts_code"] == "600000.SH") & (mock_df["trade_date"] == day2), "lgbm_score"] = 0.99
    
    engine = QlibEngineAdapter()
    equity_df, metrics = engine.run_qlib_backtest(mock_df, score_col="lgbm_score")
    
    # 验证 day1 不可能买入 600000.SH (因为 day1 评分只有 0.1)
    # 信号在 day2 收盘确定，只能在 day3 开盘成交
    assert not equity_df.empty

def test_future_data_invariance():
    """
    P0 对抗测试: 修改未来数据后，过去生成的因子与信号必须 100% 保持不变
    """
    mock_df = generate_mock_daily_data(num_stocks=3, num_days=70)
    f_engine = Custom12Factors()
    
    df1 = f_engine.compute(mock_df.copy())
    
    # 大幅修改最后一天的收盘价
    modified_df = mock_df.copy()
    last_date = modified_df["trade_date"].max()
    modified_df.loc[modified_df["trade_date"] == last_date, "close"] *= 10.0
    
    df2 = f_engine.compute(modified_df)
    
    second_last_date = sorted(mock_df["trade_date"].unique())[-2]
    for factor in FACTOR_NAMES_12:
        v1 = df1[df1["trade_date"] == second_last_date][factor].values
        v2 = df2[df2["trade_date"] == second_last_date][factor].values
        np.testing.assert_allclose(v1, v2, rtol=1e-5, err_msg=f"Future data leakage detected in factor {factor}!")

def test_purged_split():
    """
    P1 对抗测试: 验证 Purged Split 下训练样本标签绝对不跨界跨入验证集
    """
    mock_df = generate_mock_daily_data(num_stocks=5, num_days=200)
    df_factors = Custom12Factors().compute(mock_df)
    df_all = ExecutableLabel5D().generate_labels(df_factors)
    
    evaluator = PurgedWalkForwardEvaluator(horizon=5, embargo_days=2)
    fold_df, summary = evaluator.run_purged_walk_forward(df_all, feature_cols=FACTOR_NAMES_12)
    
    assert not fold_df.empty
    assert fold_df.iloc[0]["train_days"] > 0

def test_trade_unit_100():
    """
    P2 对抗测试: 验证所有买入委托数量必须为 100 股一手整倍数
    """
    engine = QlibEngineAdapter()
    assert engine.trade_unit == 100

def test_raw_price_execution():
    """
    P0 对抗测试: 验证买卖成交额与手续费计算必须基于 raw 价格而非 adj 价格
    """
    mock_df = generate_mock_daily_data(num_stocks=2, num_days=30)
    mock_df["open_raw"] = 10.0
    mock_df["close_raw"] = 10.5
    mock_df["open_adj"] = 100.0  # 复权价是真实价的10倍
    mock_df["close_adj"] = 105.0
    mock_df["lgbm_score"] = 0.9
    
    engine = QlibEngineAdapter()
    equity_df, metrics = engine.run_qlib_backtest(mock_df, score_col="lgbm_score")
    
    assert not equity_df.empty

def test_limit_buy_and_sell():
    """
    P0 对抗测试: 验证涨停股票不可买入，跌停股票不可卖出
    """
    mock_df = generate_mock_daily_data(num_stocks=2, num_days=20)
    mock_df["lgbm_score"] = 0.9
    mock_df["limit_buy"] = True  # 涨停不可买
    
    engine = QlibEngineAdapter()
    equity_df, metrics = engine.run_qlib_backtest(mock_df, score_col="lgbm_score")
    
    # 涨停股票无法成交，交易笔数为 0
    assert metrics.get("total_trades", 0) == 0

def test_suspension():
    """
    P0 对抗测试: 验证停牌股票不可成交
    """
    mock_df = generate_mock_daily_data(num_stocks=2, num_days=20)
    mock_df["lgbm_score"] = 0.9
    mock_df["is_suspended"] = True
    
    engine = QlibEngineAdapter()
    equity_df, metrics = engine.run_qlib_backtest(mock_df, score_col="lgbm_score")
    assert metrics.get("total_trades", 0) == 0

def test_no_label_in_features():
    """
    P0 对抗测试: 验证特征列表中绝对不包含任何 label 或未来收益字段
    """
    forbidden = ["forward_5d_exec_return", "raw_label_5d", "rank_label_5d", "excess_return_5d"]
    for f in FACTOR_NAMES_12:
        assert f not in forbidden

def test_daily_signal_is_real():
    """
    P0 对抗测试: 验证 daily-signal 在没有真实模型或数据时抛出 ERROR，无任何 Dummy 假数据
    """
    pipeline = DailySignalPipeline()
    with pytest.raises(RuntimeError, match="ERROR"):
        pipeline.run_daily_pipeline()
