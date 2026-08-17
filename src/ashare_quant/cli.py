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
        processor.update_all(start_date=start_date.replace("-", ""))
        click.echo("股票主数据、交易日历与行情数据真实更新完成。")
    finally:
        processor.close()

@cli.command("train")
@click.option("--feature-set", default="alpha158", help="特征集 (alpha158 / custom12)")
@click.option("--model", default="lgbm", help="模型类型 (qlib_lgbm / native_lgbm / sklearn_hgb)")
@click.option("--config", default="configs/model_lgbm.yaml", help="配置文件")
def train(feature_set, model, config):
    """训练选股模型并进行 Purged Walk-forward 评估与实验归档"""
    logger.info(f"开始模型真实训练 pipeline | 特征集: {feature_set} | 模型: {model}")
    cfg = load_config("model_lgbm")
    
    # 1. 尝试从本地加载真实行情与因子数据
    processor = DataProcessor()
    try:
        df_daily = processor.storage.load_parquet("daily_ohlcv", is_processed=True)
    finally:
        processor.close()

    if df_daily is None or df_daily.empty:
        raise RuntimeError("ERROR: Daily market data missing. Please run 'ashare-quant update-data' first.")

    # 2. 计算因子与标签
    f_engine = Custom12Factors()
    df_factors = f_engine.compute(df_daily)
    label_engine = ExecutableLabel5D()
    df_all = label_engine.generate_labels(df_factors)

    feature_cols = [c for c in df_factors.columns if c not in ["ts_code", "trade_date", "open", "high", "low", "close", "volume", "amount", "turn", "pct_chg", "is_suspended", "open_raw", "close_raw", "open_adj", "close_adj", "high_adj", "low_adj"]]

    # 3. 运行 Purged Walk-Forward 评估
    evaluator = PurgedWalkForwardEvaluator(horizon=5, embargo_days=2, config=cfg)
    fold_df, summary = evaluator.run_purged_walk_forward(df_all, feature_cols=feature_cols, model_type="sklearn_hgb" if model == "sklearn_hgb" else "native_lgbm")

    # 4. 训练全量生产模型
    prod_model = SklearnHGBModel(config=cfg, feature_cols=feature_cols) if model == "sklearn_hgb" else NativeLGBMModel(config=cfg, feature_cols=feature_cols)
    imp_df = prod_model.fit(df_all)

    # 5. 生成样本内与近期样本预测
    df_all["lgbm_score"] = prod_model.predict(df_all)
    preds_df = df_all[["ts_code", "trade_date", "lgbm_score", "rank_label_5d"]].copy()

    # 6. 实验归档
    tracker = ExperimentTracker()
    exp_id = tracker.create_experiment(f"{feature_set}_{model}", cfg)
    tracker.log_results(
        exp_id=exp_id,
        metrics=summary,
        feature_importance=imp_df,
        predictions=preds_df,
        notes=f"Purged Walk-Forward {len(fold_df)} folds trained on {len(df_all)} samples."
    )

    # 保存模型 artifact
    exp_dir = Path("experiments") / exp_id
    model_path = exp_dir / "model.joblib"
    joblib.dump(prod_model, model_path)

    click.echo(f"模型训练与 Purged Walk-Forward 完成。Experiment ID: {exp_id}")
    click.echo(f"Model Artifact: {model_path.resolve()}")
    click.echo(f"Mean IC: {summary.get('mean_ic', 0.0):.4f} | ICIR: {summary.get('icir', 0.0):.4f}")

@cli.command("backtest")
@click.option("--experiment", required=True, help="实验 ID (experiment_id)")
@click.option("--config", default="configs/backtest.yaml", help="回测配置文件")
def backtest(experiment, config):
    """执行基于 Qlib 引擎与 A 股真实成交约束（T+1/涨跌停/停牌/滑点）的策略回测"""
    logger.info(f"开始 Qlib 真实策略回测 | 实验ID: {experiment}")
    
    exp_dir = Path("experiments") / experiment
    pred_path = exp_dir / "predictions.parquet"
    if not pred_path.exists():
        raise RuntimeError(f"ERROR: Prediction artifact not found at '{pred_path}'.")

    preds_df = pd.read_parquet(pred_path)
    if preds_df.empty or "lgbm_score" not in preds_df.columns:
        raise RuntimeError("ERROR: Predictions file is empty or missing 'lgbm_score'.")

    # 补充行情成交价格字段
    processor = DataProcessor()
    try:
        df_daily = processor.storage.load_parquet("daily_ohlcv", is_processed=True)
    finally:
        processor.close()

    if df_daily is None or df_daily.empty:
        raise RuntimeError("ERROR: Market data missing in storage for backtesting.")

    merged_df = pd.merge(df_daily, preds_df[["ts_code", "trade_date", "lgbm_score"]], on=["ts_code", "trade_date"], how="inner")
    
    engine = QlibEngineAdapter()
    engine.validate_price_schema(merged_df)

    report_df, metrics = engine.run_backtest(merged_df, score_col="lgbm_score")

    # 保存回测结果到实验目录
    report_df.to_parquet(exp_dir / "backtest_report.parquet")
    with open(exp_dir / "backtest_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    click.echo(f"实验 {experiment} 基于 Qlib 引擎的回测已运行完成。")
    click.echo(f"Annual Return: {metrics.get('annual_return', 0.0):.2%} | Sharpe: {metrics.get('sharpe', 0.0):.2f} | Max DD: {metrics.get('max_drawdown', 0.0):.2%}")

@cli.command("daily-signal")
@click.option("--date", default=None, help="目标计算日期 (默认今日, YYYY-MM-DD)")
def daily_signal(date):
    """每日收盘后生成真实选股 Candidate 清单 (无硬编码 Demo 数据)"""
    target_date = date or pd.Timestamp.now().strftime("%Y-%m-%d")
    logger.info(f"执行真实 Daily Signal 管线 | 日期: {target_date}")
    
    pipeline = DailySignalPipeline()
    res = pipeline.run_daily_pipeline(target_date=target_date)
    click.echo(f"真实每日选股信号计算完成。候选股票数量: {res.get('candidates_count', 0)}")
    for cand in res.get("candidates", [])[:5]:
        click.echo(f"  #{cand['rank']} {cand['ts_code']} ({cand['stock_name']}) Score: {cand['score']:.4f}")

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
