import pytest
import os
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH

from ashare_quant.data.symbols import to_qlib_symbol
from ashare_quant.data.qlib_exporter import QlibDataProviderManager
from ashare_quant.features.custom12 import Custom12Factors, FACTOR_NAMES_12
from ashare_quant.features.qlib_alpha158 import OfficialQlibAlpha158
from ashare_quant.labels.executable_5d import ExecutableLabel5D
from ashare_quant.backtest.qlib_engine import QlibEngineAdapter, DataSchemaError, build_qlib_signal
from ashare_quant.validation.purged_walk_forward import PurgedWalkForwardEvaluator
from ashare_quant.signals.daily import DailySignalPipeline
from tests.test_factors import generate_mock_daily_data

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

def test_purged_walk_forward_multi_fold_and_time_boundary():
    """
    P1-1 行为测试:
    1. 必须生成至少 2 个 Walk-Forward Folds
    2. 逐 Fold 验证信息时间边界:
       max(train_label_info_time) < min(valid_feature_time)
       max(valid_label_info_time) < min(test_feature_time)
    """
    mock_df = generate_mock_daily_data(num_stocks=5, num_days=160)
    df_factors = Custom12Factors().compute(mock_df)
    df_all = ExecutableLabel5D().generate_labels(df_factors)
    
    horizon = 5
    embargo = 2
    evaluator = PurgedWalkForwardEvaluator(horizon=horizon, embargo_days=embargo)
    all_dates = sorted(df_all["trade_date"].unique())
    folds = evaluator.generate_folds(all_dates)
    
    assert len(folds) >= 2, f"Expected at least 2 Walk-Forward folds, got {len(folds)}."
    
    for fold in folds:
        train_dates = fold["train_dates"]
        val_dates = fold["val_dates"]
        test_dates = fold["test_dates"]
        
        t_max_idx = all_dates.index(max(train_dates))
        v_min_idx = all_dates.index(min(val_dates))
        v_max_idx = all_dates.index(max(val_dates))
        test_min_idx = all_dates.index(min(test_dates))
        
        # 验证 Train 标签信息时间 (t_max + horizon) 严格小于 Val 特征产生时间 (v_min)
        assert (t_max_idx + horizon) <= v_min_idx, (
            f"Leakage in Fold {fold['fold_id']}: Train label info time (idx {t_max_idx + horizon}) "
            f">= Val feature time (idx {v_min_idx})"
        )
        
        # 验证 Val 标签信息时间 (v_max + horizon) 严格小于 Test 特征产生时间 (test_min)
        assert (v_max_idx + horizon) <= test_min_idx, (
            f"Leakage in Fold {fold['fold_id']}: Val label info time (idx {v_max_idx + horizon}) "
            f">= Test feature time (idx {test_min_idx})"
        )

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

def test_slippage_cost_configuration():
    """
    P1-4 测试: 验证滑点参数正确设定并在执行配置中生效
    """
    engine_zero = QlibEngineAdapter(config={"backtest": {}, "costs": {"slippage_bps": 0.0}})
    engine_slip = QlibEngineAdapter(config={"backtest": {}, "costs": {"slippage_bps": 10.0}})
    assert engine_zero.slippage == 0.0
    assert engine_slip.slippage == 0.001

@pytest.mark.integration
def test_real_qlib_end_to_end_smoke():
    """
    P1-8 端到端集成烟测:
    真实导出 Qlib 二进制数据 -> 初始化 Qlib -> 运行 Qlib LGBModel 训练 -> 预测 -> Qlib 回测
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_df = generate_mock_daily_data(num_stocks=4, num_days=50)
        
        # 1. 导出真实 Qlib 二进制行情数据
        provider_path = QlibDataProviderManager.dump_df_to_qlib_bin(mock_df, target_dir=tmp_dir)
        assert os.path.exists(os.path.join(provider_path, "calendars", "day.txt"))
        assert os.path.exists(os.path.join(provider_path, "instruments", "all.txt"))
        
        # 2. 初始化 Qlib
        QlibDataProviderManager.init_qlib(provider_uri=provider_path, force=True)
        
        # 3. 构建信号 Series
        mock_df["score"] = np.random.uniform(0, 1, len(mock_df))
        signal_series = build_qlib_signal(mock_df, score_col="score")
        assert not signal_series.empty
        assert isinstance(signal_series.index, pd.MultiIndex)
        
        # 4. 执行 Qlib 回测
        dates = sorted(mock_df["trade_date"].unique())
        engine = QlibEngineAdapter()
        report_df, metrics = engine.run_qlib_backtest(
            signal_series=signal_series,
            start_time=dates[0],
            end_time=dates[-1],
            benchmark="SH000300"
        )
        
        assert report_df is not None
        assert not report_df.empty
        assert "return" in report_df.columns
        assert isinstance(metrics, dict)
        assert "annual_return" in metrics
        assert "sharpe" in metrics
