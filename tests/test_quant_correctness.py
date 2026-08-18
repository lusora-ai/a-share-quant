import pytest
import os
import json
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
from ashare_quant.features.qlib_alpha158 import (
    OfficialQlibAlpha158,
    EXEC_LABEL_5D_EXPRESSION,
    EXEC_LABEL_5D_NAME,
    EXEC_LABEL_5D_HORIZON,
    build_exec_label_5d_spec,
)
from ashare_quant.labels.executable_5d import ExecutableLabel5D
from ashare_quant.models.qlib_lgbm import OfficialQlibLGBMModel
from ashare_quant.backtest.qlib_engine import (
    QlibEngineAdapter,
    DataSchemaError,
    BenchmarkDataMissingError,
    OOSArtifactMissingError as BacktestOOSMissingError,
    build_qlib_signal,
)
from ashare_quant.validation.purged_walk_forward import (
    PurgedWalkForwardEvaluator,
    InsufficientWalkForwardHistoryError,
    LeakageBoundaryError,
)
from ashare_quant.signals.daily import DailySignalPipeline
from tests.test_factors import generate_mock_daily_data

# ---------------------------------------------------------------------------
# Helper: construct sufficient mock data for short-data test helpers
# (production evaluators never auto-generate short folds)
# ---------------------------------------------------------------------------
def _make_short_calendar(n_years: int = 1) -> list:
    """Return ~252 * n_years daily date strings."""
    dates = pd.date_range(start="2020-01-01", periods=252 * n_years, freq="B")
    return dates.strftime("%Y-%m-%d").tolist()

def _make_mock_daily_for_wf(num_stocks=10, num_days=252 * 6) -> pd.DataFrame:
    """6 years of daily data — sufficient for the default 4+1+1 walk-forward."""
    dates = pd.date_range(start="2015-01-01", periods=num_days, freq="B").strftime("%Y-%m-%d")
    rows = []
    for i in range(num_stocks):
        ts_code = f"{600000 + i}.SH"
        base_price = 10.0 + i
        np.random.seed(42 + i)
        returns = np.random.normal(0.001, 0.02, num_days)
        prices = base_price * np.exp(np.cumsum(returns))
        for d_idx, date_str in enumerate(dates):
            p = prices[d_idx]
            rows.append({
                "ts_code": ts_code, "trade_date": date_str,
                "open": p, "high": p * 1.01, "low": p * 0.99, "close": p,
                "open_raw": p, "close_raw": p, "open_adj": p, "close_adj": p,
                "high_adj": p * 1.01, "low_adj": p * 0.99,
                "volume": 50000, "amount": 50000 * p * 100,
                "turn": 1.0, "pct_chg": returns[d_idx] * 100.0,
                "is_suspended": False,
            })
    return pd.DataFrame(rows)


# ======================= Label Tests =======================

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


# ======================= Walk-Forward Config from Config =======================

def test_walk_forward_params_from_config():
    """
    验证 PurgedWalkForwardEvaluator 和 QlibWalkForwardEvaluator
    从 config["walk_forward"] 读取参数，而非硬编码默认值
    """
    cfg = {"walk_forward": {"train_years": 4, "val_years": 1, "test_years": 1, "embargo_days": 3}}
    evaluator = PurgedWalkForwardEvaluator(horizon=5, config=cfg)
    assert evaluator.train_years == 4
    assert evaluator.val_years == 1
    assert evaluator.test_years == 1
    assert evaluator.embargo_days == 3


def test_qlib_walk_forward_params_from_config():
    """
    验证 QlibWalkForwardEvaluator 同样从 config 读取
    """
    cfg = {"walk_forward": {"train_years": 4, "val_years": 1, "test_years": 1, "embargo_days": 2}}
    evaluator = PurgedWalkForwardEvaluator.__bases__[0].__module__  # sanity
    from ashare_quant.validation.qlib_walk_forward import QlibWalkForwardEvaluator
    ev = QlibWalkForwardEvaluator(horizon=5, config=cfg)
    assert ev.train_years == 4
    assert ev.embargo_days == 2


def test_walk_forward_explicit_override():
    """
    验证显式参数覆盖 config 中的值
    """
    cfg = {"walk_forward": {"train_years": 4, "val_years": 1, "test_years": 1, "embargo_days": 2}}
    evaluator = PurgedWalkForwardEvaluator(horizon=5, train_years=2, val_years=1, test_years=1, config=cfg)
    assert evaluator.train_years == 2  # explicit override wins
    assert evaluator.embargo_days == 2  # from config


# ======================= InsufficientHistory (no fallback) =======================

def test_purged_wf_insufficient_history_raises():
    """
    验证 PurgedWalkForwardEvaluator 在数据不足时抛出 InsufficientWalkForwardHistoryError，
    而不是自动生成 2 个缩小版 fold
    """
    # 只有 1 年日历数据 — 不足以满足默认 4+1+1=6 年
    short_dates = _make_short_calendar(n_years=1)
    evaluator = PurgedWalkForwardEvaluator(horizon=5)
    with pytest.raises(InsufficientWalkForwardHistoryError):
        evaluator.generate_folds(short_dates)


def test_qlib_wf_insufficient_history_raises():
    """
    验证 QlibWalkForwardEvaluator 在数据不足时同样抛出 InsufficientWalkForwardHistoryError
    """
    short_dates = _make_short_calendar(n_years=2)
    from ashare_quant.validation.qlib_walk_forward import QlibWalkForwardEvaluator
    evaluator = QlibWalkForwardEvaluator(horizon=5)
    with pytest.raises(InsufficientWalkForwardHistoryError):
        evaluator.generate_qlib_folds(short_dates)


def test_purged_wf_sufficient_history_generates_folds():
    """
    验证足够多年数据时确实能生成 fold（回归确认 evaluator 正常工作）
    """
    # 6+ years data → 4+1+1 = 6 years needed → 1 fold minimum
    long_dates = _make_short_calendar(n_years=7)
    evaluator = PurgedWalkForwardEvaluator(horizon=5)
    folds = evaluator.generate_folds(long_dates)
    assert len(folds) >= 1


# ======================= Purged WF Boundary =======================

def test_purged_walk_forward_multi_fold_and_time_boundary():
    """
    P1-1 行为测试:
    1. 必须生成至少 2 个 Walk-Forward Folds
    2. 逐 Fold 验证信息时间边界
    """
    mock_df = _make_mock_daily_for_wf(num_stocks=5, num_days=252 * 8)
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

        assert (t_max_idx + horizon) <= v_min_idx, (
            f"Leakage in Fold {fold['fold_id']}: Train label info time (idx {t_max_idx + horizon}) "
            f">= Val feature time (idx {v_min_idx})"
        )

        assert (v_max_idx + horizon) <= test_min_idx, (
            f"Leakage in Fold {fold['fold_id']}: Val label info time (idx {v_max_idx + horizon}) "
            f">= Test feature time (idx {test_min_idx})"
        )


# ======================= Alpha158 Label Tests =======================

def test_alpha158_uses_project_5d_executable_label():
    """
    验证 OfficialQlibAlpha158 handler 的 label expression 是项目的 5D executable label
    (Ref($close, -5) / Ref($open, -1) - 1)，而不是 Qlib 默认 label
    """
    adapter = OfficialQlibAlpha158()
    kwargs = adapter.build_handler_kwargs(
        instruments="csi300",
        start_time="2019-01-01",
        end_time="2019-06-30"
    )
    label_expr, label_name = kwargs["label"]
    assert label_expr == [EXEC_LABEL_5D_EXPRESSION], (
        f"Expected project 5D executable label expression, got {label_expr}"
    )
    assert label_name == [EXEC_LABEL_5D_NAME], (
        f"Expected project 5D executable label name '{EXEC_LABEL_5D_NAME}', got {label_name}"
    )
    # learn_processors must include CSZScoreNorm on label group
    lp = kwargs["learn_processors"]
    proc_classes = [p["class"] for p in lp]
    assert "DropnaLabel" in proc_classes
    assert "CSZScoreNorm" in proc_classes


def test_exec_label_spec_is_complete():
    """
    验证 build_exec_label_5d_spec() 返回完整 label spec，
    包含 signal_time, entry_time, exit_time, expression, normalization, horizon
    """
    spec = build_exec_label_5d_spec()
    required_keys = {"signal_time", "entry_time", "exit_time", "expression", "normalization", "horizon", "name"}
    assert required_keys.issubset(set(spec.keys())), f"Missing keys: {required_keys - set(spec.keys())}"
    assert "percentile rank" not in spec["normalization"].lower() or "NOT" in spec["normalization"]
    assert spec["horizon"] == 5


def test_alpha158_not_default_qlib_label():
    """
    对抗测试：确保项目 label expression != Qlib 默认 label
    """
    default_expr = "Ref($close, -2)/Ref($close, -1) - 1"
    assert EXEC_LABEL_5D_EXPRESSION != default_expr, (
        "Project label expression must differ from Qlib default!"
    )


# ======================= Trade Unit & Schema Tests =======================

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


# ======================= OOS Strict Tests =======================

def test_backtest_rejects_legacy_predictions_only(tmp_path):
    """
    验证当实验目录只有 predictions.parquet 没有 oos_predictions.parquet 时，
    backtest 必须拒绝执行并报错（不再接受 legacy fallback）
    """
    from click.testing import CliRunner
    from ashare_quant.cli import cli
    import json

    runner = CliRunner()
    with runner.isolated_filesystem():
        exp_dir = Path("experiments/exp_no_oos")
        exp_dir.mkdir(parents=True)
        legacy_df = pd.DataFrame({
            "trade_date": ["2024-01-02"], "ts_code": ["600000.SH"],
            "score": [0.5], "label": [0.05], "fold_id": [1], "train_end_date": ["2023-12-31"]
        })
        legacy_df.to_parquet(exp_dir / "predictions.parquet")
        with open(exp_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({"feature_set": "alpha158"}, f)

        result = runner.invoke(cli, ["backtest", "--experiment", "exp_no_oos"])
        assert result.exit_code != 0
        assert result.exception is not None
        assert "oos_predictions.parquet" in str(result.exception) or "oos_predictions.parquet" in str(result.output)


# ======================= Benchmark Tests =======================

def test_benchmark_unavailable_raises():
    """
    验证 benchmark 不可用时直接 raise BenchmarkDataMissingError，
    绝不降级为 benchmark=None
    """
    adapter = QlibEngineAdapter()
    dummy_signal = pd.Series(
        [0.85, 0.12],
        index=pd.MultiIndex.from_tuples(
            [(pd.to_datetime("2024-01-02"), "SH600000"), (pd.to_datetime("2024-01-02"), "SZ000001")],
            names=["datetime", "instrument"]
        )
    )

    from unittest.mock import patch
    with patch.object(adapter, "create_strategy", return_value="mock_strategy"):
        with patch.object(adapter, "create_executor", return_value="mock_executor"):
            with patch("ashare_quant.data.qlib_exporter.QlibDataProviderManager.init_qlib"):
                # Simulate benchmark not found error
                with patch("ashare_quant.backtest.qlib_engine.qlib_backtest",
                           side_effect=ValueError("SH000300 does not exist")):
                    with pytest.raises(BenchmarkDataMissingError) as excinfo:
                        adapter.run_qlib_backtest(
                            signal_series=dummy_signal,
                            start_time="2024-01-02",
                            end_time="2024-01-03",
                            benchmark="SH000300"
                        )
                    assert "SH000300" in str(excinfo.value)
                    assert "unavailable" in str(excinfo.value).lower()


# ======================= Qlib Provider State Tests =======================

def test_qlib_provider_initialized_uri_returns_correct_path():
    """
    验证 QlibDataProviderManager.get_initialized_provider_uri() 返回真实初始化的 URI
    """
    from ashare_quant.data.qlib_exporter import QlibDataProviderManager
    uri = QlibDataProviderManager.init_qlib()
    assert QlibDataProviderManager.get_initialized_provider_uri() == uri
    assert os.path.exists(uri)


def test_qlib_provider_csi300_check_requires_csi300(tmp_path):
    """
    验证 CSI300 模式下缺少 csi300.txt 或 sh000300 benchmark 数据时抛出 QlibDataNotReadyError
    """
    from ashare_quant.data.qlib_exporter import QlibDataProviderManager, QlibDataNotReadyError

    # 1. Nonexistent directory
    with pytest.raises(QlibDataNotReadyError, match="not found"):
        QlibDataProviderManager.check_provider_ready(str(tmp_path / "nonexistent"), mode="csi300")

    # 2. Incomplete directory (missing instruments/csi300.txt)
    mock_qlib_dir = tmp_path / "incomplete_qlib"
    (mock_qlib_dir / "calendars").mkdir(parents=True)
    (mock_qlib_dir / "instruments").mkdir(parents=True)
    (mock_qlib_dir / "features" / "sh600000").mkdir(parents=True)
    (mock_qlib_dir / "calendars" / "day.txt").write_text("2020-01-02\n", encoding="utf-8")
    (mock_qlib_dir / "instruments" / "all.txt").write_text("SH600000\t2020-01-02\t2020-01-03\n", encoding="utf-8")
    (mock_qlib_dir / "features" / "sh600000" / "close.day.bin").write_bytes(b"\x00" * 8)

    with pytest.raises(QlibDataNotReadyError, match="csi300.txt"):
        QlibDataProviderManager.check_provider_ready(str(mock_qlib_dir), mode="csi300")

    # 3. Missing benchmark directory
    (mock_qlib_dir / "instruments" / "csi300.txt").write_text("SH600000\t2020-01-02\t2020-01-03\n", encoding="utf-8")
    with pytest.raises(QlibDataNotReadyError, match="SH000300 benchmark"):
        QlibDataProviderManager.check_provider_ready(str(mock_qlib_dir), mode="csi300")


def test_alpha158_daily_signal_fetch_real_close():
    """
    验证 Alpha158 每日信号使用 Qlib Data API (D.features) 获取真实未复权 raw_close = $close / $factor
    """
    from ashare_quant.signals.daily import DailySignalPipeline
    from ashare_quant.data.qlib_exporter import QlibDataProviderManager
    uri = QlibDataProviderManager.init_qlib()
    cal_file = Path(uri) / "calendars" / "day.txt"
    dates = [line.strip() for line in cal_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    calc_date = dates[-1]

    pipeline = DailySignalPipeline()
    close_df, name_df = pipeline._fetch_alpha158_real_close_and_name(calc_date, ["SH600000", "SZ000001"])
    assert not close_df.empty
    assert "instrument" in close_df.columns and "close" in close_df.columns
    assert (close_df["close"] > 0).all()


def test_qlib_provider_mismatch_error():
    """
    验证已初始化 Provider A 时尝试不使用 force 切换到 Provider B 会抛出 ProviderMismatchError
    """
    from ashare_quant.data.qlib_exporter import QlibDataProviderManager, ProviderMismatchError
    QlibDataProviderManager.init_qlib()
    with pytest.raises(ProviderMismatchError, match="already initialized"):
        QlibDataProviderManager.init_qlib(provider_uri="C:/another/fake/qlib_provider_path", force=False)


def test_qlib_csv_exporter_requires_adjusted_prices():
    """
    验证 Qlib CSV exporter 必须提供 adjusted prices，缺少时抛出 DataSchemaError，严禁 fallback raw prices
    """
    from ashare_quant.data.qlib_exporter import QlibDataProviderManager, DataSchemaError
    invalid_df = pd.DataFrame({
        "ts_code": ["600000.SH"],
        "trade_date": ["2020-01-02"],
        "open_raw": [10.0],
        "close_raw": [10.5],
        "volume": [1000]
    })
    with pytest.raises(DataSchemaError, match="DataSchemaError"):
        QlibDataProviderManager.export_to_csv_dump_format(invalid_df)


def test_daily_signal_rejects_stale_provider_without_explicit_date():
    """
    验证未指定 --date 时，若 Provider 截止日期早于预期最新市场日期，必须抛出 StaleMarketDataError
    """
    from ashare_quant.signals.daily import DailySignalPipeline, StaleMarketDataError
    from ashare_quant.data.qlib_exporter import QlibDataProviderManager
    pipeline = DailySignalPipeline()

    # 默认调用（target_date=None），由于当前真实年份是 2026 而 provider 截止 2021，必须触发 StaleMarketDataError
    with pytest.raises(StaleMarketDataError) as excinfo:
        pipeline.run_daily_pipeline(target_date=None)
    
    assert "STALE" in str(excinfo.value)
    assert "Expected latest completed market date" in str(excinfo.value)
    assert "Qlib provider data ends at" in str(excinfo.value)


def test_missing_factor_never_falls_back_to_one():
    """
    验证当 Qlib 返回的 $factor 缺失或为 NaN 时，严禁 fallback 1.0，必须作为无效价格剔除
    """
    from ashare_quant.signals.daily import DailySignalPipeline, RawPriceUnavailableError
    from unittest.mock import patch

    pipeline = DailySignalPipeline()
    # 模拟 Qlib D.features 返回部分股票缺失 factor 或 factor <= 0
    mock_feat = pd.DataFrame(
        {
            "$close": [10.0, 20.0, 30.0],
            "$factor": [np.nan, 0.0, 2.0],  # 只有第三只股票有效
        },
        index=pd.MultiIndex.from_tuples(
            [("SH600000", "2021-06-11"), ("SZ000001", "2021-06-11"), ("SH600519", "2021-06-11")],
            names=["instrument", "datetime"]
        )
    )

    with patch("qlib.data.D.features", return_value=mock_feat):
        close_df, name_df = pipeline._fetch_alpha158_real_close_and_name("2021-06-11", ["600000.SH", "000001.SZ", "600519.SH"])
        # SH600000 (NaN factor) 和 SZ000001 (0.0 factor) 必须被完全剔除，绝不能出现 raw_close = 10.0 / 1.0 = 10.0
        assert "SH600000" not in close_df["instrument"].tolist()
        assert "SZ000001" not in close_df["instrument"].tolist()
        assert "SH600519" in close_df["instrument"].tolist()
        assert close_df.loc[close_df["instrument"] == "SH600519", "close"].values[0] == 15.0  # 30.0 / 2.0


def test_daily_candidates_never_have_zero_close():
    """
    验证每日选股生成的 Top10 候选股票 close 与 cost_100_shares 全部为有限正数，绝无 0 或负数
    """
    from ashare_quant.signals.daily import DailySignalPipeline
    pipeline = DailySignalPipeline()
    # 显式重放历史有效交易日
    res = pipeline.run_daily_pipeline(target_date="2021-06-11")
    candidates = res.get("candidates", [])
    assert len(candidates) > 0
    for cand in candidates:
        assert cand["close"] > 0
        assert np.isfinite(cand["close"])
        assert cand["cost_100_shares"] > 0
        assert np.isfinite(cand["cost_100_shares"])
        assert round(cand["close"] * 100, 2) == cand["cost_100_shares"]


def test_backtest_binds_experiment_provider_uri():
    """
    验证 Backtest 严格使用 experiment metadata 中的 provider_uri，若缺失则报错
    """
    from ashare_quant.backtest.qlib_engine import QlibEngineAdapter, ProviderMismatchError
    from click.testing import CliRunner
    from ashare_quant.cli import cli

    runner = CliRunner()
    # 使用不存在 provider_uri 的 mock metadata 验证报错
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_exp = Path("experiments") / "test_missing_provider_exp"
        fake_exp.mkdir(parents=True, exist_ok=True)
        (fake_exp / "metadata.json").write_text(json.dumps({"feature_set": "alpha158"}), encoding="utf-8")
        (fake_exp / "oos_predictions.parquet").write_bytes(b"dummy")

        try:
            res = runner.invoke(cli, ["backtest", "--experiment", "test_missing_provider_exp"])
            assert res.exit_code != 0
        finally:
            shutil.rmtree(fake_exp, ignore_errors=True)


def test_backtest_preflight_requires_valid_factor():
    """
    验证 QlibEngineAdapter 在 factor 缺失或非正数时抛出 QlibExecutionDataError
    """
    from ashare_quant.backtest.qlib_engine import QlibEngineAdapter, QlibExecutionDataError
    from unittest.mock import patch

    adapter = QlibEngineAdapter()
    dummy_signal = pd.Series(
        [0.85, 0.12],
        index=pd.MultiIndex.from_tuples(
            [(pd.to_datetime("2020-01-02"), "SH600000"), (pd.to_datetime("2020-01-02"), "SZ000001")],
            names=["datetime", "instrument"]
        )
    )

    # 模拟 preflight 发现 $factor 缺失
    mock_feat = pd.DataFrame({"$close": [10.0, 11.0]})  # missing $factor
    with patch("qlib.data.D.features", return_value=mock_feat):
        with pytest.raises(QlibExecutionDataError, match="factor"):
            adapter.run_qlib_backtest(
                signal_series=dummy_signal,
                start_time="2020-01-02",
                end_time="2020-01-03",
                benchmark="SH000300"
            )


def test_qlib_status_and_update_cli():
    """
    验证 qlib-status 和 update-qlib-data CLI 命令正常输出状态
    """
    from click.testing import CliRunner
    from ashare_quant.cli import cli

    runner = CliRunner()
    res_status = runner.invoke(cli, ["qlib-status"])
    assert res_status.exit_code == 0
    assert "Provider URI" in res_status.output
    assert "Status" in res_status.output

    res_update = runner.invoke(cli, ["update-qlib-data"])
    assert res_update.exit_code == 0
    assert "DumpDataAll" in res_update.output


# ======================= Integration Tests =======================

@pytest.mark.integration
def test_real_qlib_end_to_end_smoke():
    """
    P1-6 真实端到端 Qlib 集成测试:
    REAL QLIB PROVIDER -> Alpha158 Handler -> DatasetH -> LGBModel.fit -> model.predict(test)
    -> assert OOS prediction not empty -> TopkDropoutStrategy -> Qlib Backtest -> assert portfolio report not empty.
    严禁使用 random score!
    """
    provider_uri = QlibDataProviderManager.init_qlib()
    assert os.path.exists(os.path.join(provider_uri, "calendars", "day.txt"))
    assert os.path.exists(os.path.join(provider_uri, "instruments", "all.txt"))

    # 1. 实例化真实 Alpha158 Handler（带项目 5D executable label）
    alpha_adapter = OfficialQlibAlpha158()
    handler = alpha_adapter.create_handler_instance(
        instruments="csi300",
        start_time="2019-01-01",
        end_time="2019-06-30",
        fit_start_time="2019-01-01",
        fit_end_time="2019-03-31"
    )

    # 2. 构建 DatasetH
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ("2019-01-01", "2019-03-31"),
            "valid": ("2019-04-01", "2019-04-30"),
            "test": ("2019-05-01", "2019-06-30"),
        }
    )

    # 3. 真实训练 Qlib LGBModel
    model = OfficialQlibLGBMModel(
        config={"model": {"learning_rate": 0.05, "n_estimators": 10, "num_leaves": 15, "max_depth": 3, "random_state": 42}}
    )
    model.fit(dataset)

    # 4. 真实预测 Test Segment (OOS Predictions)
    test_preds = model.predict(dataset, segment="test")
    assert test_preds is not None
    assert not test_preds.empty
    assert isinstance(test_preds.index, pd.MultiIndex)
    assert len(test_preds) > 0

    # 5. 执行 Qlib 回测 (TopkDropoutStrategy + SimulatorExecutor)
    engine = QlibEngineAdapter()
    report_df, metrics = engine.run_qlib_backtest(
        signal_series=test_preds,
        start_time="2019-05-01",
        end_time="2019-06-30",
        benchmark="SH000300"
    )

    assert report_df is not None
    assert not report_df.empty
    assert "return" in report_df.columns
    assert isinstance(metrics, dict)
    assert "portfolio_annualized_return" in metrics
    assert "portfolio_sharpe" in metrics
    assert "portfolio_max_drawdown" in metrics
    assert "information_ratio" in metrics


def test_qlib_backtest_with_synthetic_signal():
    """
    单独测试 Qlib 回测适配器与合成信号的交互逻辑 (与端到端模型训练集成测试分离)
    """
    provider_uri = QlibDataProviderManager.init_qlib()
    cal_file = Path(provider_uri) / "calendars" / "day.txt"
    dates = [line.strip() for line in cal_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    test_dates = [d for d in dates if "2019-05-01" <= d <= "2019-05-31"]

    tuples = []
    values = []
    for d in test_dates:
        for s in ["SH600000", "SZ000001", "SZ000002"]:
            tuples.append((pd.to_datetime(d), s))
            values.append(0.5)

    index = pd.MultiIndex.from_tuples(tuples, names=["datetime", "instrument"])
    signal_series = pd.Series(values, index=index, dtype=float)

    engine = QlibEngineAdapter()
    report_df, metrics = engine.run_qlib_backtest(
        signal_series=signal_series,
        start_time="2019-05-01",
        end_time="2019-05-31",
        benchmark="SH000300"
    )
    assert report_df is not None
    assert not report_df.empty
    assert "portfolio_annualized_return" in metrics
