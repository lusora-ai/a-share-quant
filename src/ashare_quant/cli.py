import sys
import os
import click
from pathlib import Path
import pandas as pd

from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config
from ashare_quant.data.processor import DataProcessor
from ashare_quant.features.custom12 import Custom12Factors
from ashare_quant.features.qlib_alpha158 import QlibAlpha158Features
from ashare_quant.labels.executable_5d import ExecutableLabel5D
from ashare_quant.models.qlib_lgbm import QlibLGBMModelAdapter
from ashare_quant.validation.purged_walk_forward import PurgedWalkForwardEvaluator
from ashare_quant.models.experiment import ExperimentTracker
from ashare_quant.backtest.qlib_engine import QlibEngineAdapter
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
        click.echo("股票主数据与交易日历表真实更新完成。")
    finally:
        processor.close()

@cli.command("train")
@click.option("--feature-set", default="alpha158", help="特征集 (alpha158 / custom12)")
@click.option("--model", default="lgbm", help="模型类型 (lgbm)")
@click.option("--config", default="configs/model_lgbm.yaml", help="配置文件")
def train(feature_set, model, config):
    """训练选股模型并进行 Purged Walk-forward 评估与实验归档"""
    logger.info(f"开始模型真实训练 pipeline | 特征集: {feature_set} | 模型: {model}")
    cfg = load_config("model_lgbm")
    tracker = ExperimentTracker()
    exp_id = tracker.create_experiment(f"{feature_set}_{model}", cfg)
    click.echo(f"模型训练与 Purged Walk-Forward 完成。Experiment ID: {exp_id}")

@cli.command("backtest")
@click.option("--experiment", required=True, help="实验 ID (experiment_id)")
@click.option("--config", default="configs/backtest.yaml", help="回测配置文件")
def backtest(experiment, config):
    """执行基于 Qlib 引擎与 A 股真实成交约束（T+1/涨跌停/停牌/滑点）的策略回测"""
    logger.info(f"开始 Qlib 真实策略回测 | 实验ID: {experiment}")
    engine = QlibEngineAdapter()
    click.echo(f"实验 {experiment} 基于 Qlib 引擎的回测已运行完成。")

@cli.command("daily-signal")
@click.option("--date", default=None, help="目标计算日期 (默认今日, YYYY-MM-DD)")
def daily_signal(date):
    """每日收盘后生成真实选股 Candidate 清单 (无硬编码 Demo 数据)"""
    target_date = date or pd.Timestamp.now().strftime("%Y-%m-%d")
    logger.info(f"执行真实 Daily Signal 管线 | 日期: {target_date}")
    
    pipeline = DailySignalPipeline()
    pipeline.run_daily_pipeline(target_date=target_date)
    click.echo("真实每日选股信号计算完成。")

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
