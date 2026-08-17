import pytest
import os
import glob
from unittest.mock import patch
import pandas as pd
import numpy as np

import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.executor import SimulatorExecutor

from ashare_quant.data.symbols import to_qlib_symbol, from_qlib_symbol
from ashare_quant.data.qlib_exporter import QlibDataProviderManager, QlibDataNotReadyError
from ashare_quant.features.qlib_alpha158 import OfficialQlibAlpha158
from ashare_quant.models.qlib_lgbm import OfficialQlibLGBMModel
from ashare_quant.backtest.qlib_engine import (
    QlibEngineAdapter,
    DataSchemaError,
    QlibBacktestError,
    build_qlib_signal
)
from ashare_quant.signals.daily import DailySignalPipeline
from tests.test_factors import generate_mock_daily_data

def test_real_qlib_installed():
    """
    验证真实 microsoft/qlib 已安装并可访问
    """
    assert qlib is not None
    assert hasattr(qlib, "__version__")
    assert isinstance(qlib.__version__, str)

def test_qlib_dependency_exists():
    """
    验证 pyqlib 及其关键核心算子在当前 Python 环境中真实存在
    """
    assert Alpha158 is not None
    assert LGBModel is not None
    assert TopkDropoutStrategy is not None
    assert SimulatorExecutor is not None

def test_symbol_canonical_conversion():
    """
    P0-5 验证 A股 与 Qlib Canonical 证券代码双向严格转换
    """
    assert to_qlib_symbol("600000.SH") == "SH600000"
    assert to_qlib_symbol("000001.SZ") == "SZ000001"
    assert to_qlib_symbol("000300.SH") == "SH000300"
    assert to_qlib_symbol("sh600000") == "SH600000"
    assert to_qlib_symbol("SH600000") == "SH600000"
    
    assert from_qlib_symbol("SH600000") == "600000.SH"
    assert from_qlib_symbol("SZ000001") == "000001.SZ"

def test_alpha158_comes_from_qlib():
    """
    验证官方 Alpha158 模块路径确实直接来自 qlib.contrib.data.handler
    """
    adapter = OfficialQlibAlpha158()
    assert adapter.handler_cls is Alpha158
    assert adapter.handler_cls.__module__.startswith("qlib.")
    
    fields, names = OfficialQlibAlpha158.get_feature_config()
    assert len(names) == 158
    assert "KMID" in names
    assert "KLEN" in names

def test_qlib_model_is_real():
    """
    验证正式 Qlib 模型直接实例化自 qlib.contrib.model.gbdt.LGBModel
    """
    adapter = OfficialQlibLGBMModel()
    assert isinstance(adapter.model, LGBModel)
    assert adapter.model.__class__.__module__.startswith("qlib.")

def test_signal_score_col_is_strictly_used():
    """
    P0-4 验证 build_qlib_signal 严格只使用指定的 score_col，第一列放随机垃圾数绝不干扰信号
    """
    df = pd.DataFrame({
        "garbage_col": [9999.0, 8888.0],
        "ts_code": ["600000.SH", "000001.SZ"],
        "trade_date": ["2024-01-02", "2024-01-02"],
        "model_score": [0.85, 0.12]
    })
    
    signal = build_qlib_signal(df, score_col="model_score")
    assert isinstance(signal, pd.Series)
    assert isinstance(signal.index, pd.MultiIndex)
    assert signal.index.names == ["datetime", "instrument"]
    
    # 验证 MultiIndex 中的 instrument 是 Qlib Canonical 格式
    instruments = signal.index.get_level_values("instrument").tolist()
    assert "SH600000" in instruments
    assert "SZ000001" in instruments
    
    # 验证数值严格来自于 model_score，绝非 garbage_col
    dt = pd.to_datetime("2024-01-02")
    assert signal.loc[(dt, "SH600000")] == 0.85
    assert signal.loc[(dt, "SZ000001")] == 0.12

def test_backtest_failure_raises():
    """
    P0-1 对抗测试：Qlib backtest 发生异常时，必须 raise QlibBacktestError，绝不返回 Fake metrics
    """
    adapter = QlibEngineAdapter()
    dummy_signal = pd.Series(
        [0.85, 0.12],
        index=pd.MultiIndex.from_tuples(
            [(pd.to_datetime("2024-01-02"), "SH600000"), (pd.to_datetime("2024-01-02"), "SZ000001")],
            names=["datetime", "instrument"]
        )
    )
    
    with patch.object(adapter, "create_strategy", return_value="mock_strategy"):
        with patch.object(adapter, "create_executor", return_value="mock_executor"):
            with patch("ashare_quant.data.qlib_exporter.QlibDataProviderManager.init_qlib"):
                with patch("ashare_quant.backtest.qlib_engine.qlib_backtest", side_effect=ValueError("Simulated low-level exchange failure")):
                    with pytest.raises(QlibBacktestError) as excinfo:
                        adapter.run_qlib_backtest(
                            signal_series=dummy_signal,
                            start_time="2024-01-02",
                            end_time="2024-01-03",
                            benchmark="SH000300"
                        )
                    assert "Simulated low-level exchange failure" in str(excinfo.value)

def test_qlib_data_not_ready_raises():
    """
    P0-6 验证数据目录不完整时必须抛出 QlibDataNotReadyError
    """
    with pytest.raises(QlibDataNotReadyError):
        QlibDataProviderManager.check_provider_ready("non_existent_data_directory_12345")

def test_no_demo_daily_signal():
    """
    P0-10 验证 daily-signal 在没有真实模型或数据时直接抛出 RuntimeError
    """
    pipeline = DailySignalPipeline(exp_dir="non_existent_experiments_dir")
    with pytest.raises(RuntimeError, match="ERROR"):
        pipeline.run_daily_pipeline()

def test_adjusted_price_not_used_for_cash_execution():
    """
    验证若数据集中缺失真实成交价 open_raw / close_raw 时，必须抛出 DataSchemaError 阻止执行
    """
    adapter = QlibEngineAdapter()
    invalid_df = pd.DataFrame({
        "open_adj": [10.0, 11.0],
        "close_adj": [10.5, 11.5],
        "ts_code": ["600000.SH", "000001.SZ"],
        "trade_date": ["2024-01-02", "2024-01-02"]
    })
    with pytest.raises(DataSchemaError, match="DataSchemaError"):
        adapter.validate_price_schema(invalid_df)
