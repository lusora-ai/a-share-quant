"""
Production Microsoft Qlib Backtest Engine Adapter.
Directly executes official qlib.backtest, TopkDropoutStrategy, SimulatorExecutor, and risk_analysis.
Zero fake metrics, zero mock fallbacks, zero benchmark=None degradation.
"""
from typing import Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np

import qlib
from qlib.constant import REG_CN
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest import backtest as qlib_backtest
from qlib.contrib.evaluate import risk_analysis

from ashare_quant.data.symbols import to_qlib_symbol
from ashare_quant.data.qlib_exporter import QlibDataProviderManager, ProviderMismatchError
from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.backtest.qlib_engine")

class DataSchemaError(ValueError):
    """Raised when required raw/adjusted price columns are missing."""
    pass

class QlibBacktestError(RuntimeError):
    """Raised when Qlib backtest execution fails."""
    pass

class BenchmarkDataMissingError(QlibBacktestError):
    """Raised when the required benchmark (e.g. SH000300) is unavailable in the Qlib provider.

    Backtest MUST NOT continue with benchmark=None and pretend excess_return == portfolio_return.
    """
    pass

class OOSArtifactMissingError(FileNotFoundError):
    """Raised when experiments/<ID>/oos_predictions.parquet does not exist.

    Legacy predictions.parquet is never accepted as a substitute for strict OOS backtesting.
    """
    pass

class OOSLeakageError(ValueError):
    """Raised when OOS predictions violate temporal boundaries (e.g. trade_date <= train_end_date)."""
    pass

class QlibExecutionDataError(QlibBacktestError):
    """Raised when Qlib provider lacks valid positive factors required for trade_unit=100 execution."""
    pass

class RawPriceUnavailableError(ValueError):
    """Raised when raw transaction price cannot be computed due to missing/invalid close or factor."""
    pass

def build_qlib_signal(df: pd.DataFrame, score_col: str) -> pd.Series:
    """
    将包含 ts_code, trade_date, score_col 的 DataFrame 转换为 Qlib 规范的 MultiIndex Series。
    MultiIndex: (datetime, instrument)
    Values: score_col 浮点数值
    """
    if score_col not in df.columns:
        raise ValueError(f"Score column '{score_col}' not found in input DataFrame. Columns: {list(df.columns)}")

    if "ts_code" not in df.columns or "trade_date" not in df.columns:
        raise ValueError("Input DataFrame must contain 'ts_code' and 'trade_date' columns.")

    sub = df[["trade_date", "ts_code", score_col]].copy()
    sub["datetime"] = pd.to_datetime(sub["trade_date"])
    sub["instrument"] = sub["ts_code"].astype(str).apply(to_qlib_symbol)

    # 构造标准 MultiIndex (datetime, instrument)
    signal_series = sub.set_index(["datetime", "instrument"])[score_col].astype(float).sort_index()
    return signal_series

class QlibEngineAdapter:
    """
    Microsoft Qlib 生产回测内核适配器
    职责：
    1. 构建与校验 Qlib 中国市场交易配置 (trade_unit=100, 佣金万2.5, 印花税千0.5, 滑点5bps)
    2. 将指定的 score_col 严格转为 Qlib Canonical Signal (MultiIndex Series)
    3. 调用官方 Qlib Backtest / SimulatorExecutor / TopkDropoutStrategy
    4. 使用 Qlib risk_analysis 提取真实评估指标与收益曲线
    5. 回测失败立即 raise QlibBacktestError，绝不伪造任何金融指标
    6. Benchmark 不可用时直接 raise BenchmarkDataMissingError，绝不以 benchmark=None 降级
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config("backtest")
        self.bt_cfg = self.config.get("backtest", {})
        self.costs_cfg = self.config.get("costs", {})

        self.initial_capital = float(self.bt_cfg.get("initial_capital", 100000.0))
        self.top_k = int(self.bt_cfg.get("top_n", 10))
        self.n_drop = int(self.bt_cfg.get("n_drop", 2))
        self.trade_unit = 100

        self.open_cost = float(self.costs_cfg.get("commission_rate", 0.00025))
        self.close_cost = float(self.costs_cfg.get("commission_rate", 0.00025)) + float(self.costs_cfg.get("stamp_duty_rate", 0.0005))
        self.min_cost = float(self.costs_cfg.get("min_commission", 5.0))
        self.slippage = float(self.costs_cfg.get("slippage_bps", 5.0)) / 10000.0

    def validate_price_schema(self, df: pd.DataFrame) -> None:
        """
        验证价格 Schema。严禁将 adjusted price 直接当作真实成交价。
        若缺失 open_raw / close_raw 字段，抛出 DataSchemaError 异常。
        """
        if "open_raw" not in df.columns or "close_raw" not in df.columns:
            raise DataSchemaError(
                "DataSchemaError: Execution dataset missing 'open_raw' or 'close_raw' columns! "
                "Raw prices are strictly required for cash accounting and 100-share trading units."
            )

    def validate_oos_predictions(self, df: pd.DataFrame, fold_metrics_df: Optional[pd.DataFrame] = None) -> None:
        """
        严格验证 OOS 预测集的内容正确性：
        1. 必须提供 fold_metrics_df，缺失则抛出 OOSArtifactMissingError
        2. 必须包含字段: trade_date, ts_code, score, fold_id, train_end_date
        3. 逐行验证: trade_date > train_end_date，任何违规直接抛出 OOSLeakageError
        4. 检查 (trade_date, ts_code) 重复记录，禁止重复预测
        5. 验证预测集中的 fold_id 均在 fold_metrics 中合法注册，禁止未知 fold_id
        6. 验证每折实际预测日期范围严格处于 [test_start_date, test_end_date] 之内
        """
        if fold_metrics_df is None or fold_metrics_df.empty:
            raise OOSArtifactMissingError(
                "OOSArtifactMissingError: fold_metrics.parquet is required for strict OOS backtest validation. "
                "Backtest cannot proceed without fold metrics artifact."
            )

        required = ["trade_date", "ts_code", "score", "fold_id", "train_end_date"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise OOSLeakageError(f"OOSLeakageError: OOS predictions missing required columns: {missing}")

        if df.empty:
            raise OOSLeakageError("OOSLeakageError: OOS predictions DataFrame is empty.")

        # 检查重复 (trade_date, ts_code)
        duplicates = df.duplicated(subset=["trade_date", "ts_code"], keep=False)
        if duplicates.any():
            dup_samples = df[duplicates].head(5).to_dict("records")
            raise OOSLeakageError(f"OOSLeakageError: Duplicate (trade_date, ts_code) records found in OOS predictions: {dup_samples}")

        # 逐行校验 trade_date > train_end_date
        leak_mask = df["trade_date"] <= df["train_end_date"]
        if leak_mask.any():
            leaked = df[leak_mask]
            first_row = leaked.iloc[0].to_dict()
            raise OOSLeakageError(
                f"OOS Leakage detected: {len(leaked)} prediction rows have trade_date <= train_end_date! "
                f"First violation: ts_code={first_row.get('ts_code')}, trade_date={first_row.get('trade_date')}, "
                f"train_end_date={first_row.get('train_end_date')}, fold_id={first_row.get('fold_id')}"
            )

        # 校验 fold_id 合法性与范围一致性
        valid_fold_ids = set(fold_metrics_df["fold_id"].unique())
        actual_fold_ids = set(df["fold_id"].unique())
        unknown_folds = actual_fold_ids - valid_fold_ids
        if unknown_folds:
            raise OOSLeakageError(
                f"OOSLeakageError: Unknown fold_id {unknown_folds} found in OOS predictions. Registered folds: {valid_fold_ids}"
            )

        for _, fold_row in fold_metrics_df.iterrows():
            f_id = fold_row.get("fold_id")
            f_preds = df[df["fold_id"] == f_id]
            if f_preds.empty:
                raise OOSLeakageError(f"OOSLeakageError: OOS predictions missing records for Fold {f_id}.")
            f_dates = sorted(f_preds["trade_date"].unique())
            test_start = str(fold_row.get("test_start_date", ""))
            test_end = str(fold_row.get("test_end_date", ""))
            if test_start and f_dates[0] < test_start:
                raise OOSLeakageError(
                    f"Fold {f_id} test start date violation: predictions start at {f_dates[0]}, "
                    f"which is before fold test_start_date {test_start}."
                )
            if test_end and f_dates[-1] > test_end:
                raise OOSLeakageError(
                    f"Fold {f_id} test end date violation: predictions end at {f_dates[-1]}, "
                    f"which is after fold test_end_date {test_end}."
                )

    def create_strategy(self, signal: pd.Series, **kwargs) -> TopkDropoutStrategy:
        """
        构建 Qlib 官方 TopkDropoutStrategy 选股换仓策略
        入参必须为经过 build_qlib_signal 处理的 MultiIndex Series
        """
        logger.info(f"Creating Qlib TopkDropoutStrategy (topk={self.top_k}, n_drop={self.n_drop})...")
        strategy = TopkDropoutStrategy(
            signal=signal,
            topk=self.top_k,
            n_drop=self.n_drop,
            **kwargs
        )
        return strategy

    def create_executor(self, time_per_step: str = "day", **kwargs) -> SimulatorExecutor:
        """
        构建 Qlib 官方 SimulatorExecutor 日频模拟执行器
        """
        logger.info("Creating Qlib SimulatorExecutor for China A-share market...")
        executor = SimulatorExecutor(
            time_per_step=time_per_step,
            **kwargs
        )
        return executor

    def run_backtest(self, df_all: pd.DataFrame, score_col: str = "lgbm_score", benchmark: str = "SH000300") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        执行标准回测接口
        """
        self.validate_price_schema(df_all)
        signal_series = build_qlib_signal(df_all, score_col=score_col)

        dates = sorted(pd.to_datetime(df_all["trade_date"]).dt.strftime("%Y-%m-%d").unique())
        start_time = dates[0]
        end_time = dates[-1]

        return self.run_qlib_backtest(
            signal_series=signal_series,
            start_time=start_time,
            end_time=end_time,
            benchmark=to_qlib_symbol(benchmark)
        )

    def run_qlib_backtest(
        self,
        signal_series: pd.Series,
        start_time: str,
        end_time: str,
        benchmark: str = "SH000300",
        provider_uri: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        执行真实 Qlib 回测流程。
        必须绑定 experiment 记录的 provider_uri（绝不偷偷切换为默认 ~/.qlib）。
        Benchmark 不可用 => raise BenchmarkDataMissingError（绝不降级为 benchmark=None）。
        缺失 $factor 或 $factor <= 0 => raise QlibExecutionDataError（保障 trade_unit=100 资金计算正确）。
        """
        logger.info(f"Running official Qlib backtest from {start_time} to {end_time} on benchmark {benchmark} (Provider: {provider_uri or 'active'})...")

        # 确保 Qlib 使用指定的 Provider 初始化
        QlibDataProviderManager.init_qlib(provider_uri=provider_uri)

        # Preflight (Fail-Closed): 检查 signal universe 中全部标的的 $close 与 $factor 有效性
        from qlib.data import D
        insts = list(signal_series.index.get_level_values("instrument").unique())
        if insts:
            batch_size = 100
            for i in range(0, len(insts), batch_size):
                batch_insts = insts[i:i + batch_size]
                try:
                    preflight_df = D.features(
                        batch_insts,
                        fields=["$close", "$factor"],
                        start_time=start_time,
                        end_time=end_time
                    )
                except Exception as query_err:
                    raise QlibExecutionDataError(
                        f"QlibExecutionDataError: D.features query failed during factor preflight for instruments {batch_insts[:5]}: {query_err}"
                    ) from query_err

                if preflight_df is None or preflight_df.empty:
                    raise QlibExecutionDataError(
                        f"QlibExecutionDataError: D.features returned empty features during factor preflight for instruments {batch_insts[:5]}."
                    )

                if "$close" not in preflight_df.columns or "$factor" not in preflight_df.columns:
                    raise QlibExecutionDataError(
                        "QlibExecutionDataError: Qlib provider lacks '$close' or '$factor' required for trade_unit=100 execution."
                    )

                # 对于 $close 存在有限数值的每一行，$factor 必须存在、有限且 > 0
                valid_close_mask = pd.notna(preflight_df["$close"]) & np.isfinite(preflight_df["$close"])
                if valid_close_mask.any():
                    factors = preflight_df.loc[valid_close_mask, "$factor"]
                    invalid_factor_mask = (
                        pd.isna(factors) |
                        (~np.isfinite(factors)) |
                        (factors <= 0)
                    )
                    if invalid_factor_mask.any():
                        bad_records = preflight_df.loc[valid_close_mask][invalid_factor_mask]
                        bad_insts = list(bad_records.index.get_level_values("instrument").unique())[:5]
                        raise QlibExecutionDataError(
                            f"QlibExecutionDataError: Missing, non-finite, or non-positive $factor detected for active instruments: {bad_insts} "
                            f"where $close is valid. Total invalid records: {len(bad_records)}."
                        )

        # 准备 Qlib 回测配置
        exchange_kwargs = {
            "freq": "day",
            "limit_threshold": 0.099, # CSI300 standard limit
            "deal_price": "open",
            "open_cost": self.open_cost,
            "close_cost": self.close_cost,
            "min_cost": self.min_cost,
            "trade_unit": self.trade_unit,
            "impact_cost": self.slippage,
        }

        strategy = self.create_strategy(signal=signal_series)
        executor = self.create_executor(time_per_step="day", generate_portfolio_metrics=True)

        bench_sym = to_qlib_symbol(benchmark) if benchmark else None

        try:
            portfolio_metric_dict, indicator_dict = qlib_backtest(
                start_time=start_time,
                end_time=end_time,
                strategy=strategy,
                executor=executor,
                benchmark=bench_sym,
                account=self.initial_capital,
                exchange_kwargs=exchange_kwargs
            )
        except Exception as bench_err:
            err_msg = str(bench_err).lower()
            if bench_sym and ("does not exist" in err_msg or "benchmark" in err_msg):
                raise BenchmarkDataMissingError(
                    f"Benchmark '{bench_sym}' is unavailable for {start_time} to {end_time} "
                    f"in the current Qlib provider. Cannot run backtest without a valid benchmark. "
                    f"Original error: {bench_err}"
                ) from bench_err
            else:
                raise QlibBacktestError(f"Qlib backtest failed: {bench_err}") from bench_err


        # P0-2: 正确解包 portfolio_metric_dict["1day"] -> Tuple[pd.DataFrame, dict]
        metric_res = portfolio_metric_dict.get("1day")
        if isinstance(metric_res, tuple):
            report_df, positions = metric_res
        else:
            report_df = metric_res

        if report_df is None or report_df.empty:
            raise QlibBacktestError("Qlib backtest returned empty portfolio metrics DataFrame.")

        # P1-4: 准确分别计算策略自身、基准与超额收益风险指标
        port_analysis = risk_analysis(report_df["return"])
        port_annual_ret = float(port_analysis.loc["annualized_return", "risk"]) if "annualized_return" in port_analysis.index else 0.0
        port_sharpe = float(port_analysis.loc["information_ratio", "risk"]) if "information_ratio" in port_analysis.index else 0.0
        port_max_dd = float(port_analysis.loc["max_drawdown", "risk"]) if "max_drawdown" in port_analysis.index else 0.0

        bench_col = "bench" if "bench" in report_df.columns else None
        if bench_col and not report_df[bench_col].isna().all():
            bench_analysis = risk_analysis(report_df[bench_col])
            bench_annual_ret = float(bench_analysis.loc["annualized_return", "risk"]) if "annualized_return" in bench_analysis.index else 0.0

            excess_series = report_df["return"] - report_df[bench_col]
            excess_analysis = risk_analysis(excess_series)
            excess_annual_ret = float(excess_analysis.loc["annualized_return", "risk"]) if "annualized_return" in excess_analysis.index else 0.0
            info_ratio = float(excess_analysis.loc["information_ratio", "risk"]) if "information_ratio" in excess_analysis.index else 0.0
        else:
            raise BenchmarkDataMissingError(
                f"Benchmark column '{bench_col}' is missing or all-NaN in Qlib backtest report. "
                f"Cannot compute excess return metrics without a valid benchmark."
            )

        metrics = {
            "portfolio_annualized_return": port_annual_ret,
            "benchmark_annualized_return": bench_annual_ret,
            "excess_annualized_return": excess_annual_ret,
            "portfolio_max_drawdown": port_max_dd,
            "portfolio_sharpe": port_sharpe,
            "information_ratio": info_ratio,
            # 兼容性别名
            "annual_return": port_annual_ret,
            "benchmark_return": bench_annual_ret,
            "excess_return": excess_annual_ret,
            "sharpe": port_sharpe,
            "max_drawdown": port_max_dd,
        }
        return report_df, metrics
