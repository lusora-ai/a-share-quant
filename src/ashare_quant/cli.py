import sys
import os
import click
import json
import joblib
from pathlib import Path
import pandas as pd

from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config
from ashare_quant.data.processor import DataProcessor
from ashare_quant.features.custom12 import Custom12Factors
from ashare_quant.features.qlib_alpha158 import (
    OfficialQlibAlpha158,
    build_exec_label_5d_spec,
)
from ashare_quant.labels.executable_5d import ExecutableLabel5D
from ashare_quant.models.native_lgbm import NativeLGBMModel
from ashare_quant.models.sklearn_hgb import SklearnHGBModel
from ashare_quant.models.qlib_lgbm import OfficialQlibLGBMModel
from ashare_quant.validation.purged_walk_forward import PurgedWalkForwardEvaluator
from ashare_quant.validation.qlib_walk_forward import QlibWalkForwardEvaluator
from ashare_quant.models.experiment import ExperimentTracker, OOSArtifactMissingError
from ashare_quant.backtest.qlib_engine import (
    QlibEngineAdapter,
    BenchmarkDataMissingError,
    OOSArtifactMissingError as BacktestOOSMissingError,
    build_qlib_signal,
)
from ashare_quant.signals.daily import DailySignalPipeline
from ashare_quant.universe.filter import build_custom12_universe
from ashare_quant.reports.daily_report import DailyReportGenerator

logger = setup_logger("ashare_quant.cli")

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """A股个人量化系统 CLI 入口 - 基于 Qlib 成熟框架的无未来函数量化研究系统"""
    pass

@cli.command("update-data")
@click.option("--start-date", default="2018-01-01", help="数据起始日期 (YYYY-MM-DD)")
@click.option("--end-date", default=None, help="数据截止日期 (YYYY-MM-DD)")
@click.option("--provider", default="akshare", help="数据源 (akshare/baostock)")
def update_data(start_date, end_date, provider):
    """真实更新主数据表、交易日历与行情 (Parquet + DuckDB)"""
    logger.info(f"开始真实增量数据更新 | 数据源: {provider} | 起始: {start_date}")
    processor = DataProcessor()
    try:
        processor.update_stock_master()
        processor.update_trade_calendar(start_date=start_date.replace("-", ""))
        processor.update_daily_data(start_date=start_date, end_date=end_date)
        click.echo("股票主数据、交易日历与行情数据真实更新完成。")
    finally:
        processor.close()

@cli.command("train")
@click.option("--feature-set", default="alpha158", help="特征集 (alpha158 / custom12)")
@click.option("--model", default="qlib_lgbm", help="模型类型 (qlib_lgbm / native_lgbm / sklearn_hgb)")
@click.option("--config", default="configs/model_lgbm.yaml", help="配置文件")
def train(feature_set, model, config):
    """训练选股模型并进行 Purged Walk-forward 评估与实验归档"""
    logger.info(f"开始模型真实训练 pipeline | 特征集: {feature_set} | 模型: {model}")
    cfg = load_config("model_lgbm")
    tracker = ExperimentTracker()

    # walk_forward 参数全部从 config 统一读取
    wf_cfg = cfg.get("walk_forward", {})
    horizon = cfg.get("label", {}).get("horizon", 5)

    if feature_set == "alpha158":
        # P0-3: 校验模型与特征集匹配
        if model != "qlib_lgbm":
            raise ValueError(f"Feature set 'alpha158' requires model 'qlib_lgbm', but got '{model}'.")

        # P0-6: 初始化真实 Qlib CN 数据
        from ashare_quant.data.qlib_exporter import QlibDataProviderManager
        provider_uri = QlibDataProviderManager.init_qlib()

        cal_file = Path(provider_uri) / "calendars" / "day.txt"
        provider_calendar_dates = [line.strip() for line in cal_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        provider_data_end_date = provider_calendar_dates[-1]

        # 分离 Research Calendar (受 walk_forward.start_year/end_year 限制) 和 Production Calendar (完整真实 provider calendar)
        start_year = wf_cfg.get("start_year")
        end_year = wf_cfg.get("end_year")
        research_calendar_dates = provider_calendar_dates
        if start_year is not None:
            research_calendar_dates = [d for d in research_calendar_dates if int(d[:4]) >= int(start_year)]
        if end_year is not None:
            research_calendar_dates = [d for d in research_calendar_dates if int(d[:4]) <= int(end_year)]

        research_start_date = research_calendar_dates[0]
        research_end_date = research_calendar_dates[-1]

        # 实例化官方 Alpha158（显式注入项目 5D executable label）
        from ashare_quant.features.qlib_alpha158 import OfficialQlibAlpha158
        from ashare_quant.validation.qlib_walk_forward import QlibWalkForwardEvaluator
        from qlib.data.dataset import DatasetH

        # Handler 使用完整 Provider Calendar 创建 (保障特征计算与生产模型完整历史可用)
        alpha_adapter = OfficialQlibAlpha158()
        handler = alpha_adapter.create_handler_instance(
            instruments="csi300",
            start_time=provider_calendar_dates[0],
            end_time=provider_calendar_dates[-1]
        )

        # 运行 Qlib Purged Walk-Forward (严格限制在 research_calendar_dates 历史区间)
        evaluator = QlibWalkForwardEvaluator(horizon=horizon, config=cfg)
        fold_df, summary, oos_preds_df = evaluator.run_qlib_walk_forward(handler, research_calendar_dates)

        # Production Model 使用完整 provider_calendar_dates: production_train_end_date = provider_data_end - horizon
        production_train_end_date = provider_calendar_dates[-(horizon + 1)] if len(provider_calendar_dates) > horizon else provider_calendar_dates[-1]
        label_mature_end_date = production_train_end_date
        data_end_date = provider_data_end_date
        train_end_date = production_train_end_date

        # 训练全量生产模型 (仅用于未来 Daily Signal，数据覆盖至最新成熟标签日)
        prod_dataset = DatasetH(handler=handler, segments={"train": (provider_calendar_dates[0], production_train_end_date)})
        prod_model = OfficialQlibLGBMModel(config=cfg)
        prod_model.fit(prod_dataset)
        imp_df = prod_model.get_feature_importance()
        feature_cols = OfficialQlibAlpha158.get_feature_names()

        label_spec = build_exec_label_5d_spec()
        walk_forward_meta = evaluator.walk_forward_config()
        universe_meta = {"instruments": "csi300", "source": "qlib_provider"}

    elif feature_set == "custom12":
        if model == "qlib_lgbm":
            raise ValueError("Feature set 'custom12' does not support 'qlib_lgbm'. Use 'native_lgbm' or 'sklearn_hgb'.")

        # 1. 加载真实行情与 master
        processor = DataProcessor()
        try:
            df_master = processor.storage.load_parquet("stock_master", is_processed=True)
            df_daily = processor.storage.load_parquet("daily_ohlcv", is_processed=True)
        finally:
            processor.close()

        if df_daily is None or df_daily.empty:
            raise RuntimeError("ERROR: Daily market data missing. Please run 'ashare-quant update-data' first.")

        # 2. 共享 universe（train / walk-forward / daily 全部复用同一个定义，均传入 master_df）
        from ashare_quant.features.custom12 import FACTOR_NAMES_12
        df_universe = build_custom12_universe(df_daily, master_df=df_master)

        f_engine = Custom12Factors()
        df_factors = f_engine.compute(df_universe)
        label_engine = ExecutableLabel5D()
        df_all = label_engine.generate_labels(df_factors)

        feature_cols = FACTOR_NAMES_12.copy()
        assert set(feature_cols).issubset(df_all.columns), f"Missing custom12 features: {set(feature_cols) - set(df_all.columns)}"

        all_dates = sorted(df_all["trade_date"].unique())
        provider_data_end_date = all_dates[-1]

        start_year = wf_cfg.get("start_year")
        end_year = wf_cfg.get("end_year")
        if start_year is not None or end_year is not None:
            research_df = df_all[df_all["trade_date"].apply(
                lambda d: (start_year is None or int(d[:4]) >= int(start_year)) and (end_year is None or int(d[:4]) <= int(end_year))
            )].copy()
        else:
            research_df = df_all

        research_dates = sorted(research_df["trade_date"].unique())
        research_start_date = research_dates[0]
        research_end_date = research_dates[-1]

        # 3. 运行 Purged Walk-Forward 评估 (严格限制在 research_df 历史区间)
        evaluator = PurgedWalkForwardEvaluator(horizon=horizon, config=cfg)
        fold_df, summary, oos_preds_df = evaluator.run_purged_walk_forward(
            research_df,
            feature_cols=feature_cols,
            model_type="sklearn_hgb" if model == "sklearn_hgb" else "native_lgbm"
        )

        # 4. 训练全量生产模型 (使用全部 df_all，截止到 production_train_end_date)
        production_train_end_date = str(all_dates[-(horizon + 1)] if len(all_dates) > horizon else all_dates[-1])
        label_mature_end_date = production_train_end_date
        data_end_date = provider_data_end_date
        train_end_date = production_train_end_date

        train_prod_df = df_all[df_all["trade_date"] <= production_train_end_date]
        prod_model = SklearnHGBModel(config=cfg, feature_cols=feature_cols) if model == "sklearn_hgb" else NativeLGBMModel(config=cfg, feature_cols=feature_cols)
        imp_df = prod_model.fit(train_prod_df)
        provider_uri = ""

        label_spec = {
            "name": cfg.get("label", {}).get("custom12", {}).get("name", "rank_label_5d"),
            "expression": cfg.get("label", {}).get("custom12", {}).get("expression", "daily cross-section percentile rank of 5D executable excess return"),
            "horizon": horizon,
            "normalization": "daily cross-section percentile rank",
        }
        walk_forward_meta = evaluator.walk_forward_config()
        universe_meta = {
            "filter": "build_custom12_universe (shared train/WF/daily)",
            "min_list_days": 120,
            "exclude_st": True,
            "exclude_suspended": True,
        }

    else:
        raise ValueError(f"Unknown feature set '{feature_set}'. Allowed: 'alpha158', 'custom12'.")

    # 5. 实验归档 (完整元数据，分开记录 research 和 production 日期)
    exp_id = tracker.create_experiment(
        name=f"{feature_set}_{model}",
        config=cfg,
        feature_set=feature_set,
        model_type=model,
        feature_cols=feature_cols,
        label_spec=label_spec,
        walk_forward_config=walk_forward_meta,
        universe=universe_meta,
        research_start_date=research_start_date,
        research_end_date=research_end_date,
        provider_data_end_date=provider_data_end_date,
        train_end_date=train_end_date,
        data_end_date=data_end_date,
        label_mature_end_date=label_mature_end_date,
        production_train_end_date=production_train_end_date,
        provider_uri=provider_uri
    )
    tracker.log_results(
        exp_id=exp_id,
        metrics=summary,
        fold_metrics=fold_df,
        oos_predictions=oos_preds_df,
        feature_importance=imp_df,
        notes=f"Purged Walk-Forward {len(fold_df)} folds. Mean IC: {summary.get('mean_ic', 0.0):.4f}, ICIR: {summary.get('icir', 0.0):.4f}."
    )

    # 保存生产模型独立文件
    exp_dir = Path("experiments") / exp_id
    model_path = exp_dir / "production_model.joblib"
    joblib.dump(prod_model, model_path)

    click.echo(f"\n=======================================================")
    click.echo(f"模型训练与 Purged Walk-Forward 完成。Experiment ID: {exp_id}")
    click.echo(f"Production Model Artifact: {model_path.resolve()}")
    click.echo(f"OOS Predictions Artifact: {(exp_dir / 'oos_predictions.parquet').resolve()}")
    click.echo(f"Folds Count: {len(fold_df)} | Total OOS Predictions: {len(oos_preds_df)}")
    click.echo(f"Mean IC: {summary.get('mean_ic', 0.0):.4f} | ICIR: {summary.get('icir', 0.0):.4f} | Pos Ratio: {summary.get('pos_ratio', 0.0):.2%}")
    click.echo(f"Walk-Forward Config: {walk_forward_meta}")
    click.echo(f"Research Dates     : {research_start_date} -> {research_end_date}")
    click.echo(f"Provider Data End  : {provider_data_end_date} | Production Train End: {production_train_end_date}")
    click.echo(f"=======================================================\n")

@cli.command("backtest")
@click.option("--experiment", required=True, help="实验 ID (experiment_id)")
@click.option("--config", default="configs/backtest.yaml", help="回测配置文件")
def backtest(experiment, config):
    """执行基于 Qlib 引擎与 A 股真实成交约束（T+1/涨跌停/停牌/滑点）的策略回测 (严格检验 OOS Predictions 内容)"""
    logger.info(f"开始 Qlib 真实策略回测 | 实验ID: {experiment}")

    exp_dir = Path("experiments") / experiment
    if not exp_dir.exists():
        raise RuntimeError(f"ERROR: Experiment directory '{exp_dir}' not found.")

    # 严格只加载 oos_predictions.parquet 与 fold_metrics.parquet
    oos_pred_path = exp_dir / "oos_predictions.parquet"
    if not oos_pred_path.exists():
        raise BacktestOOSMissingError(
            f"OOS prediction artifact not found at '{oos_pred_path}'. "
            f"Only oos_predictions.parquet is allowed in backtest. "
            f"The legacy predictions.parquet is no longer accepted."
        )

    fold_metrics_path = exp_dir / "fold_metrics.parquet"
    if not fold_metrics_path.exists():
        raise BacktestOOSMissingError(
            f"Fold metrics artifact not found at '{fold_metrics_path}'. "
            f"Both oos_predictions.parquet and fold_metrics.parquet are strictly required for backtest."
        )

    oos_preds_df = pd.read_parquet(oos_pred_path)
    if oos_preds_df.empty or "score" not in oos_preds_df.columns:
        raise RuntimeError("ERROR: OOS predictions file is empty or missing 'score'.")

    # 严格检验 OOS 预测集内容 (trade_date > train_end_date, 范围与 fold_metrics 一致)
    fold_metrics_df = pd.read_parquet(fold_metrics_path)
    engine = QlibEngineAdapter()
    engine.validate_oos_predictions(oos_preds_df, fold_metrics_df=fold_metrics_df)

    # 读取实验元数据
    meta_path = exp_dir / "metadata.json"
    feature_set = "alpha158"
    experiment_provider_uri = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            feature_set = metadata.get("feature_set", "alpha158")
            experiment_provider_uri = metadata.get("provider_uri")

    if feature_set == "alpha158":
        # Qlib Canonical 回测: 必须使用该实验训练时绑定的 provider_uri
        if not experiment_provider_uri:
            raise ValueError(
                f"ERROR: Experiment metadata for '{experiment}' is missing 'provider_uri'. "
                f"Backtesting cannot proceed without an explicit provider binding."
            )

        from ashare_quant.backtest.qlib_engine import build_qlib_signal
        from ashare_quant.data.qlib_exporter import QlibDataProviderManager
        QlibDataProviderManager.init_qlib(provider_uri=experiment_provider_uri)

        signal_series = build_qlib_signal(oos_preds_df, score_col="score")
        dates = sorted(oos_preds_df["trade_date"].unique())
        report_df, metrics = engine.run_qlib_backtest(
            signal_series=signal_series,
            start_time=dates[0],
            end_time=dates[-1],
            benchmark="SH000300",
            provider_uri=experiment_provider_uri
        )
    else:
        # Custom12 回测: 补充行情真实成交价格字段
        processor = DataProcessor()
        try:
            df_daily = processor.storage.load_parquet("daily_ohlcv", is_processed=True)
        finally:
            processor.close()

        if df_daily is None or df_daily.empty:
            raise RuntimeError("ERROR: Market data missing in storage for backtesting.")

        merged_df = pd.merge(df_daily, oos_preds_df[["ts_code", "trade_date", "score"]], on=["ts_code", "trade_date"], how="inner")
        engine.validate_price_schema(merged_df)
        report_df, metrics = engine.run_backtest(merged_df, score_col="score", benchmark="SH000300")

    # 保存回测结果到实验目录
    report_df.to_parquet(exp_dir / "backtest_report.parquet")
    with open(exp_dir / "backtest_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    click.echo(f"\n=======================================================")
    click.echo(f"实验 {experiment} 基于 Qlib 引擎的真实回测已运行完成。")
    click.echo(f"Portfolio Annual Return : {metrics.get('portfolio_annualized_return', 0.0):.2%}")
    click.echo(f"Benchmark Annual Return : {metrics.get('benchmark_annualized_return', 0.0):.2%}")
    click.echo(f"Excess Annual Return    : {metrics.get('excess_annualized_return', 0.0):.2%}")
    click.echo(f"Portfolio Sharpe Ratio  : {metrics.get('portfolio_sharpe', 0.0):.4f}")
    click.echo(f"Information Ratio (IR)  : {metrics.get('information_ratio', 0.0):.4f}")
    click.echo(f"Portfolio Max Drawdown  : {metrics.get('portfolio_max_drawdown', 0.0):.2%}")
    click.echo(f"=======================================================\n")


@cli.command("daily-signal")
@click.option("--date", default=None, help="目标计算日期 (默认最新交易日, YYYY-MM-DD)")
def daily_signal(date):
    """每日收盘后生成真实选股 Candidate 清单 (无硬编码 Demo 数据)"""
    logger.info(f"执行真实 Daily Signal 管线 | 指定日期: {date or '最新交易日'}")

    pipeline = DailySignalPipeline()
    res = pipeline.run_daily_pipeline(target_date=date)
    click.echo(f"\n=======================================================")
    click.echo(f"真实每日选股信号计算完成。")
    click.echo(f"Signal Date         : {res.get('signal_date')}")
    click.echo(f"Provider Data End   : {res.get('provider_data_end')}")
    click.echo(f"Production Train End: {res.get('production_train_end')}")
    click.echo(f"Candidate Count     : {res.get('candidates_count', 0)}")
    click.echo(f"-------------------------------------------------------")
    for cand in res.get("candidates", [])[:10]:
        click.echo(f"  #{cand['rank']:<2} {cand['ts_code']:<10} {cand['name']:<10} Score: {cand['score']:.4f} Close: {cand['close']:.2f} Cost100: {cand.get('cost_100_shares', 'N/A')}")
    click.echo(f"Daily Report Path   : {res.get('report_path')}")
    click.echo(f"-------------------------------------------------------")
    click.echo(f"提示: 'ashare-quant update-data' 目前只更新 custom Parquet/DuckDB，不更新 Qlib Provider。")
    click.echo(f"如需查看或更新 Qlib Provider，请使用 'ashare-quant qlib-status' 或 'ashare-quant update-qlib-data'。")
    click.echo(f"=======================================================\n")


@cli.command("qlib-status")
@click.option("--provider-uri", default=None, help="Qlib Provider 路径 (默认 ~/.qlib/qlib_data/cn_data)")
def qlib_status(provider_uri):
    """查看 Qlib Provider 数据状态、新鲜度与就绪情况 (READY / STALE / BROKEN / UNKNOWN)"""
    from ashare_quant.data.qlib_exporter import get_qlib_provider_status
    status_info = get_qlib_provider_status(provider_uri=provider_uri)

    click.echo("\n=======================================================")
    click.echo(f"Microsoft Qlib Provider 数据状态与就绪检查")
    click.echo("-------------------------------------------------------")
    click.echo(f"Provider URI                : {status_info['provider_uri']}")
    click.echo(f"Status                      : {status_info['status']}")
    click.echo(f"Trading Days Range          : {status_info['calendar_start']} -> {status_info['calendar_end']} ({status_info['total_trading_days']} days)")
    click.echo(f"Benchmark (SH000300) Ready  : {'YES' if status_info['benchmark_available'] else 'NO (MISSING)'}")
    click.echo(f"CSI300 Instruments Ready    : {'YES' if status_info['csi300_available'] else 'NO (MISSING)'}")
    cov_str = f"{status_info['factor_coverage_count']}/{status_info['factor_expected_count']} ({status_info['factor_coverage_pct']}%)"
    click.echo(f"Factor Coverage (CSI300)    : {cov_str}")
    if status_info.get("missing_factor_examples"):
        click.echo(f"Missing Factor Examples     : {status_info['missing_factor_examples']}")
    click.echo(f"Expected Latest Market Date : {status_info['expected_latest_market_date']}")
    click.echo(f"Status Message              : {status_info['status_message']}")
    click.echo("=======================================================\n")


@cli.command("update-qlib-data")
@click.option("--qlib-repo", default=None, help="Microsoft Qlib 官方仓库本地克隆路径")
@click.option("--target-dir", default=None, help="目标 Qlib 数据目录 (默认 ~/.qlib/qlib_data/cn_data)")
@click.option("--region", default="cn", help="市场区域 (默认 cn)")
def update_qlib_data(qlib_repo, target_dir, region):
    """更新 Microsoft Qlib 官方二进制数据 (严格调用官方 collector / dump_bin 流程)"""
    import subprocess
    import sys
    from ashare_quant.data.qlib_exporter import get_qlib_provider_status

    target = target_dir or str(Path("~/.qlib/qlib_data/cn_data").expanduser())

    if qlib_repo:
        repo_path = Path(qlib_repo).expanduser().resolve()
        get_data_script = repo_path / "scripts" / "get_data.py"
        dump_bin_script = repo_path / "scripts" / "dump_bin.py"

        if not get_data_script.exists() or not dump_bin_script.exists():
            click.echo(f"\n[ERROR] Invalid Qlib repo path: '{repo_path}'.", err=True)
            click.echo(f"Expected official scripts: '{get_data_script}' and '{dump_bin_script}'.", err=True)
            sys.exit(1)

        click.echo(f"\nExecuting official Qlib get_data script from: {get_data_script}...")
        cmd = [sys.executable, str(get_data_script), "qlib_data", "--target_dir", target, "--region", region]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            click.echo(f"\n[ERROR] Qlib get_data script failed with return code {res.returncode}.", err=True)
            sys.exit(res.returncode)

        # 验证更新后的数据健康度
        status = get_qlib_provider_status(provider_uri=target)
        if status["status"] in ("READY", "STALE"):
            click.echo(f"\n[SUCCESS] Qlib provider data updated successfully. Status: {status['status']}")
            sys.exit(0)
        else:
            click.echo(f"\n[ERROR] Qlib provider update completed but status check failed: {status['status_message']}", err=True)
            sys.exit(1)
    else:
        click.echo("\n=======================================================")
        click.echo("AUTOMATIC UPDATE NOT CONFIGURED")
        click.echo("-------------------------------------------------------")
        click.echo("To automatically execute official Qlib data download, please provide --qlib-repo:")
        click.echo(f"  ashare-quant update-qlib-data --qlib-repo /path/to/microsoft/qlib --target-dir {target} --region {region}")
        click.echo("\nOr manually execute the verified Microsoft Qlib scripts:")
        click.echo(f"1. Download official CN dataset:")
        click.echo(f"   python scripts/get_data.py qlib_data --target_dir {target} --region {region}")
        click.echo(f"\n2. Dump preprocessed CSV into Qlib binary format:")
        click.echo(f"   python scripts/dump_bin.py dump_all --csv_path data/qlib_csv --qlib_dir {target} --include_fields open,high,low,close,volume,factor")
        click.echo("=======================================================\n")
        sys.exit(1)


@cli.command("report")
@click.option("--latest", is_flag=True, help="打开最新的选股日报")
@click.option("--date", default=None, help="指定日期的选股报告 (YYYY-MM-DD)")
def report(latest, date):
    """查看选股日报 (HTML/Markdown)"""
    daily_dir = Path("reports/daily")
    if not daily_dir.exists():
        click.echo("尚无已生成的选股报告。请先运行 'ashare-quant daily-signal' 命令。")
        return

    files = sorted(daily_dir.glob("*.html"))
    if not files:
        click.echo("未找到选股日报 HTML 文件。")
        return

    target_file = files[-1] if latest else (daily_dir / f"{date}.html" if date else files[-1])
    click.echo(f"最新选股报告位置: {target_file.resolve()}")

if __name__ == "__main__":
    cli()
