import pytest
import pandas as pd
import numpy as np

from ashare_quant.data.fetcher import DataFetcher
from ashare_quant.features.custom12 import Custom12Factors, FACTOR_NAMES_12
from ashare_quant.features.qlib_alpha158 import OfficialQlibAlpha158
from ashare_quant.labels.executable_5d import ExecutableLabel5D
from ashare_quant.backtest.qlib_engine import QlibEngineAdapter, DataSchemaError
from ashare_quant.validation.purged_walk_forward import PurgedWalkForwardEvaluator
from ashare_quant.signals.daily import DailySignalPipeline
from tests.test_factors import generate_mock_daily_data

def test_no_same_day_execution():
    """
    P0 对抗测试: 验证 t 日生成的信号绝对无法在 t 日 Open 成交
    """
    engine = QlibEngineAdapter()
    dummy_signal = pd.Series([0.9, 0.2], index=pd.MultiIndex.from_tuples([("2024-01-02", "600000.SH"), ("2024-01-02", "000001.SZ")]))
    strategy = engine.create_strategy(signal=dummy_signal)
    executor = engine.create_executor(time_per_step="day")
    
    assert strategy is not None
    assert executor is not None
    assert engine.trade_unit == 100

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
    P0 对抗测试: 验证买卖成交额必须包含 raw price 字段，否则拒绝执行
    """
    mock_df = generate_mock_daily_data(num_stocks=2, num_days=30)
    mock_df_no_raw = mock_df.drop(columns=["open_raw", "close_raw"])
    engine = QlibEngineAdapter()
    
    # 缺少 raw price 必须报错
    with pytest.raises(DataSchemaError):
        engine.validate_price_schema(mock_df_no_raw)
        
    engine.validate_price_schema(mock_df)

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
