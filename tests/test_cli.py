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
