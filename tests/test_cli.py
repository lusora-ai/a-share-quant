from click.testing import CliRunner
from ashare_quant.cli import cli

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "A股个人量化系统 CLI 入口" in result.output

def test_cli_subcommands_help():
    runner = CliRunner()
    commands = ["update-data", "train", "backtest", "daily-signal", "report"]
    for cmd in commands:
        result = runner.invoke(cli, [cmd, "--help"])
        assert result.exit_code == 0
        assert f"Usage: cli {cmd}" in result.output

def test_cli_update_data_behavior(monkeypatch):
    calls = []
    
    def fake_update_stock_master(self):
        calls.append("stock_master")
        return None
        
    def fake_update_trade_calendar(self, start_date="20180101"):
        calls.append(("trade_calendar", start_date))
        return None
        
    def fake_update_daily_data(self, symbols=None, start_date="2018-01-01", end_date=None):
        calls.append(("daily_data", start_date, end_date))
        return None
        
    def fake_close(self):
        calls.append("close")
        
    from ashare_quant.data.processor import DataProcessor
    monkeypatch.setattr(DataProcessor, "update_stock_master", fake_update_stock_master)
    monkeypatch.setattr(DataProcessor, "update_trade_calendar", fake_update_trade_calendar)
    monkeypatch.setattr(DataProcessor, "update_daily_data", fake_update_daily_data)
    monkeypatch.setattr(DataProcessor, "close", fake_close)
    
    runner = CliRunner()
    result = runner.invoke(cli, ["update-data", "--start-date", "2024-01-01", "--end-date", "2024-06-30"])
    
    assert result.exit_code == 0
    assert "stock_master" in calls
    assert ("trade_calendar", "20240101") in calls
    assert ("daily_data", "2024-01-01", "2024-06-30") in calls
    assert "close" in calls

