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
from ashare_quant.features.qlib_alpha158 import OfficialQlibAlpha158
from ashare_quant.labels.executable_5d import ExecutableLabel5D
from ashare_quant.models.native_lgbm import NativeLGBMModel
from ashare_quant.models.sklearn_hgb import SklearnHGBModel
from ashare_quant.models.qlib_lgbm import OfficialQlibLGBMModel
from ashare_quant.validation.purged_walk_forward import PurgedWalkForwardEvaluator
from ashare_quant.models.experiment import ExperimentTracker
from ashare_quant.backtest.qlib_engine import QlibEngineAdapter, build_qlib_signal
from ashare_quant.signals.daily import DailySignalPipeline
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

    if feature_set == "alpha158":
        # P0-3: 校验模型与特征集匹配
        if model != "qlib_lgbm":
            raise ValueError(f"Feature set 'alpha158' requires model 'qlib_lgbm', but got '{model}'.")

        # P0-6: 初始化真实 Qlib CN 数据
        from ashare_quant.data.qlib_exporter import QlibDataProviderManager
        provider_uri = QlibDataProviderManager.init_qlib()
        
        cal_file = Path(provider_uri) / "calendars" / "day.txt"
        calendar_dates = [line.strip() for line in cal_file.read_text(encoding="utf-8").splitlines() if line.strip()]

        # 实例化官方 Alpha158
        from ashare_quant.features.qlib_alpha158 import OfficialQlibAlpha158
        from ashare_quant.validation.qlib_walk_forward import QlibWalkForwardEvaluator
        from qlib.data.dataset import DatasetH
        
        alpha_adapter = OfficialQlibAlpha158()
        handler = alpha_adapter.create_handler_instance(
            instruments="csi300",
            start_time=calendar_dates[0],
            end_time=calendar_dates[-1]
        )

        # 运行 Qlib Purged Walk-Forward
        evaluator = QlibWalkForwardEvaluator(horizon=5, embargo_days=2, train_years=2, val_years=1, test_years=1, config=cfg)
        fold_df, summary, oos_preds_df = evaluator.run_qlib_walk_forward(handler, calendar_dates)

        # 训练全量生产模型 (仅用于未来 Daily Signal)
        prod_dataset = DatasetH(handler=handler, segments={"train": (calendar_dates[0], calendar_dates[-1])})
        prod_model = OfficialQlibLGBMModel(config=cfg)
        prod_model.fit(prod_dataset)
        imp_df = prod_model.get_feature_importance()
        feature_cols = OfficialQlibAlpha158.get_feature_names()
        train_end_date = calendar_dates[-1]

    elif feature_set == "custom12":
        if model == "qlib_lgbm":
            raise ValueError("Feature set 'custom12' does not support 'qlib_lgbm'. Use 'native_lgbm' or 'sklearn_hgb'.")

        # 1. 加载真实行情
        processor = DataProcessor()
        try:
            df_daily = processor.storage.load_parquet("daily_ohlcv", is_processed=True)
        finally:
            processor.close()

        if df_daily is None or df_daily.empty:
            raise RuntimeError("ERROR: Daily market data missing. Please run 'ashare-quant update-data' first.")

        # 2. 计算 Custom12 因子与 ExecutableLabel5D 标签
        from ashare_quant.features.custom12 import FACTOR_NAMES_12
        f_engine = Custom12Factors()
        df_factors = f_engine.compute(df_daily)
        label_engine = ExecutableLabel5D()
        df_all = label_engine.generate_labels(df_factors)

        # P0-2: 严禁自动推断，严格指定 FACTOR_NAMES_12
        feature_cols = FACTOR_NAMES_12.copy()
        assert set(feature_cols).issubset(df_all.columns), f"Missing custom12 features: {set(feature_cols) - set(df_all.columns)}"

        # 3. 运行 Purged Walk-Forward 评估
        evaluator = PurgedWalkForwardEvaluator(horizon=5, embargo_days=2, config=cfg)
        fold_df, summary, oos_preds_df = evaluator.run_purged_walk_forward(
            df_all,
            feature_cols=feature_cols,
            model_type="sklearn_hgb" if model == "sklearn_hgb" else "native_lgbm"
        )

        # 4. 训练全量生产模型 (仅用于未来 Daily Signal)
        prod_model = SklearnHGBModel(config=cfg, feature_cols=feature_cols) if model == "sklearn_hgb" else NativeLGBMModel(config=cfg, feature_cols=feature_cols)
        imp_df = prod_model.fit(df_all)
        train_end_date = str(df_all["trade_date"].max())
        provider_uri = ""

    else:
        raise ValueError(f"Unknown feature set '{feature_set}'. Allowed: 'alpha158', 'custom12'.")

    # 5. 实验归档 (P0-10: 规范化 Artifacts 结构)
    exp_id = tracker.create_experiment(
        name=f"{feature_set}_{model}",
        config=cfg,
        feature_set=feature_set,
        model_type=model,
        feature_cols=feature_cols,
        train_end_date=train_end_date,
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

    # 保存生产模型独立文件 (P0-5 / P0-10)
    exp_dir = Path("experiments") / exp_id
    model_path = exp_dir / "production_model.joblib"
    joblib.dump(prod_model, model_path)
    # 保留 model.joblib 兼容
    joblib.dump(prod_model, exp_dir / "model.joblib")

    click.echo(f"\n=======================================================")
    click.echo(f"模型训练与 Purged Walk-Forward 完成。Experiment ID: {exp_id}")
    click.echo(f"Production Model Artifact: {model_path.resolve()}")
    click.echo(f"OOS Predictions Artifact: {(exp_dir / 'oos_predictions.parquet').resolve()}")
    click.echo(f"Folds Count: {len(fold_df)} | Total OOS Predictions: {len(oos_preds_df)}")
    click.echo(f"Mean IC: {summary.get('mean_ic', 0.0):.4f} | ICIR: {summary.get('icir', 0.0):.4f} | Pos Ratio: {summary.get('pos_ratio', 0.0):.2%}")
    click.echo(f"=======================================================\n")

@cli.command("backtest")
@click.option("--experiment", required=True, help="实验 ID (experiment_id)")
@click.option("--config", default="configs/backtest.yaml", help="回测配置文件")
def backtest(experiment, config):
    """执行基于 Qlib 引擎与 A 股真实成交约束（T+1/涨跌停/停牌/滑点）的策略回测 (只允许使用 OOS Predictions)"""
    logger.info(f"开始 Qlib 真实策略回测 | 实验ID: {experiment}")
    
    exp_dir = Path("experiments") / experiment
    if not exp_dir.exists():
        raise RuntimeError(f"ERROR: Experiment directory '{exp_dir}' not found.")

    # P0-5 & P0-11: 严格只加载 oos_predictions.parquet
    oos_pred_path = exp_dir / "oos_predictions.parquet"
    if not oos_pred_path.exists():
        # 如果历史遗留只有 predictions.parquet 则提示
        legacy_path = exp_dir / "predictions.parquet"
        if legacy_path.exists():
            oos_preds_df = pd.read_parquet(legacy_path)
        else:
            raise RuntimeError(f"ERROR: OOS Prediction artifact not found at '{oos_pred_path}'. Only OOS predictions are allowed in backtest!")
    else:
        oos_preds_df = pd.read_parquet(oos_pred_path)

    if oos_preds_df.empty or "score" not in oos_preds_df.columns:
        raise RuntimeError("ERROR: OOS predictions file is empty or missing 'score'.")

    # 读取实验元数据
    meta_path = exp_dir / "metadata.json"
    feature_set = "alpha158"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            feature_set = metadata.get("feature_set", "alpha158")

    engine = QlibEngineAdapter()

    if feature_set == "alpha158":
        # Qlib Canonical 回测
        from ashare_quant.backtest.qlib_engine import build_qlib_signal
        from ashare_quant.data.qlib_exporter import QlibDataProviderManager
        QlibDataProviderManager.init_qlib()

        signal_series = build_qlib_signal(oos_preds_df, score_col="score")
        dates = sorted(oos_preds_df["trade_date"].unique())
        report_df, metrics = engine.run_qlib_backtest(
            signal_series=signal_series,
            start_time=dates[0],
            end_time=dates[-1],
            benchmark="SH000300"
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
    click.echo(f"真实每日选股信号计算完成。计算日期: {res.get('trade_date')} | 候选股票数量: {res.get('candidates_count', 0)}")
    click.echo(f"-------------------------------------------------------")
    for cand in res.get("candidates", [])[:10]:
        click.echo(f"  #{cand['rank']:<2} {cand['ts_code']:<10} {cand['stock_name']:<10} Score: {cand['score']:.4f}")
    click.echo(f"Daily Report Path: {res.get('report_path')}")
    click.echo(f"=======================================================\n")


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
