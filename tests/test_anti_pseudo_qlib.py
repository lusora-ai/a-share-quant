import pytest
import os
import glob
import pandas as pd
import numpy as np

import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.executor import SimulatorExecutor

from ashare_quant.features.qlib_alpha158 import OfficialQlibAlpha158
from ashare_quant.models.qlib_lgbm import OfficialQlibLGBMModel
from ashare_quant.backtest.qlib_engine import QlibEngineAdapter, DataSchemaError
from ashare_quant.signals.daily import DailySignalPipeline

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

def test_alpha158_comes_from_qlib():
    """
    验证官方 Alpha158 模块路径确实直接来自 qlib.contrib.data.handler
    """
    adapter = OfficialQlibAlpha158()
    assert adapter.handler_cls is Alpha158
    assert adapter.handler_cls.__module__.startswith("qlib.")
    
    # 验证 158 个特征表达式定义来自 Qlib 官方 Alpha158DL
    fields, names = OfficialQlibAlpha158.get_feature_config()
    assert len(names) == 158
    assert "KMID" in names
    assert "KLEN" in names
    assert "ROC60" in names or any("ROC" in n for n in names)

def test_qlib_model_is_real():
    """
    验证正式 Qlib 模型直接实例化自 qlib.contrib.model.gbdt.LGBModel
    """
    adapter = OfficialQlibLGBMModel()
    assert isinstance(adapter.model, LGBModel)
    assert adapter.model.__class__.__module__.startswith("qlib.")

def test_qlib_backtest_is_real():
    """
    验证生产回测引擎适配器真实构建 Qlib TopkDropoutStrategy 与 SimulatorExecutor
    """
    adapter = QlibEngineAdapter()
    
    # 验证交易单位为 100 股一手
    assert adapter.trade_unit == 100
    
    dummy_signal = pd.Series([0.8, 0.5], index=pd.MultiIndex.from_tuples([("2024-01-02", "600000.SH"), ("2024-01-02", "000001.SZ")]))
    strategy = adapter.create_strategy(signal=dummy_signal)
    assert isinstance(strategy, TopkDropoutStrategy)
    assert strategy.__class__.__module__.startswith("qlib.")
    
    executor = adapter.create_executor()
    assert isinstance(executor, SimulatorExecutor)
    assert executor.__class__.__module__.startswith("qlib.")

def test_no_fake_performance_metrics():
    """
    对抗测试：扫描整个 src/ 生产代码路径，严禁出现固定假指标字符串 (如 "cagr": 0.15 或 "sharpe": 1.5)
    """
    src_files = glob.glob("src/ashare_quant/**/*.py", recursive=True)
    forbidden_snippets = [
        '"cagr": 0.15',
        "'cagr': 0.15",
        '"sharpe": 1.5',
        "'sharpe': 1.5",
        '"max_drawdown": -0.05',
        "'max_drawdown': -0.05",
    ]
    for filepath in src_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            for snippet in forbidden_snippets:
                assert snippet not in content, f"Forbidden fake mock metric '{snippet}' detected in production file {filepath}!"

def test_no_demo_daily_signal():
    """
    验证 daily-signal 绝不输出任何假 Demo 股票，缺模型或数据时直接抛出 RuntimeError
    """
    pipeline = DailySignalPipeline()
    with pytest.raises(RuntimeError, match="ERROR"):
        pipeline.run_daily_pipeline()

def test_adjusted_price_not_used_for_cash_execution():
    """
    验证若数据集中缺失真实成交价 open_raw / close_raw 时，必须抛出 DataSchemaError 阻止执行
    """
    adapter = QlibEngineAdapter()
    invalid_df = pd.DataFrame({
        "open_adj": [10.0, 11.0],
        "close_adj": [10.5, 11.5]
    })
    with pytest.raises(DataSchemaError, match="DataSchemaError"):
        adapter.validate_price_schema(invalid_df)
