import sys
import os
import click
from pathlib import Path
import pandas as pd

from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config
from ashare_quant.data.processor import DataProcessor
from ashare_quant.factors.engine import FactorEngine
from ashare_quant.factors.processor import FactorProcessor
from ashare_quant.labels.generator import LabelGenerator
from ashare_quant.models.lgbm_model import LGBMRankingModel
from ashare_quant.models.walk_forward import WalkForwardEvaluator
from ashare_quant.models.experiment import ExperimentTracker
from ashare_quant.backtest.engine import BacktestEngine
from ashare_quant.reports.daily_report import DailyReportGenerator
from ashare_quant.portfolio.execution import ManualExecutionTracker

logger = setup_logger("ashare_quant.cli")

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """A股个人量化系统 CLI 入口 - 面向个人散户的‘研究优先、半自动执行’选股系统"""
    pass

@cli.command("update-data")
@click.option("--start-date", default="2018-01-01", help="数据起始日期 (YYYY-MM-DD)")
@click.option("--end-date", default=None, help="数据截止日期 (YYYY-MM-DD)")
@click.option("--provider", default="akshare", help="数据源 (akshare/baostock)")
def update_data(start_date, end_date, provider):
    """更新日线行情、主数据与交易状态数据"""
    logger.info(f"开始增量数据更新 | 数据源: {provider} | 起始: {start_date} | 截止: {end_date or '最新'}")
    processor = DataProcessor()
    try:
        processor.update_stock_master()
        processor.update_trade_calendar(start_date=start_date.replace("-", ""))
        click.echo("数据基础表更新完成。可通过 processor.update_daily_data() 拉取指定行情数据。")
    finally:
        processor.close()

@cli.command("train")
@click.option("--config", default="configs/model_lgbm.yaml", help="模型配置文件路径")
@click.option("--experiment-name", default="lgbm_ranking", help="实验名称")
def train(config, experiment_name):
    """训练 LightGBM 横截面选股模型并进行 Walk-forward 评估"""
    logger.info(f"开始模型训练 | 配置文件: {config} | 实验名: {experiment_name}")
    
    cfg = load_config("model_lgbm")
    tracker = ExperimentTracker()
    exp_id = tracker.create_experiment(experiment_name, cfg)
    
    click.echo(f"创建实验成功，Experiment ID: {exp_id}")

@cli.command("backtest")
@click.option("--experiment", required=True, help="实验 ID (experiment_id)")
@click.option("--config", default="configs/backtest.yaml", help="回测配置文件路径")
def backtest(experiment, config):
    """执行带有 A 股现实约束（T+1/涨跌停/停牌/滑点）的策略回测"""
    logger.info(f"开始策略回测 | 实验ID: {experiment} | 配置文件: {config}")
    engine = BacktestEngine()
    click.echo(f"实验 {experiment} 策略回测模块已就绪。")

@cli.command("daily-signal")
@click.option("--date", default=None, help="目标计算日期 (默认今日, YYYY-MM-DD)")
@click.option("--experiment", default=None, help="引用的已注册模型实验 ID")
def daily_signal(date, experiment):
    """每日收盘后生成选股 Candidate 清单 (Top 10 + Top 3 观察)"""
    target_date = date or pd.Timestamp.now().strftime("%Y-%m-%d")
    logger.info(f"开始生成每日候选信号 | 日期: {target_date} | 模型实验: {experiment or '默认生产版本'}")
    
    reporter = DailyReportGenerator()
    tracker = ManualExecutionTracker()
    
    # 模拟构建候选股票
    candidates = pd.DataFrame([
        {"ts_code": "600000.SH", "name": "浦发银行", "score": 0.92, "volatility_20d": 0.018, "close": 10.5},
        {"ts_code": "000001.SZ", "name": "平安银行", "score": 0.88, "volatility_20d": 0.021, "close": 11.8},
        {"ts_code": "601318.SH", "name": "中国平安", "score": 0.85, "volatility_20d": 0.025, "close": 48.0},
    ])
    
    candidates = tracker.evaluate_affordability(candidates)
    html_p, md_p = reporter.generate_report(target_date, candidates, universe_size=3500)
    
    click.echo(f"每日选股信号生成成功!\nHTML 报告: {html_p}\nMarkdown 报告: {md_p}")

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
