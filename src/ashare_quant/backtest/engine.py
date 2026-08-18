"""
Production Backtest Engine Module.
Aliases QlibEngineAdapter as the production BacktestEngine.
"""
from ashare_quant.backtest.qlib_engine import QlibEngineAdapter, DataSchemaError

# Production Backtest Engine is the official QlibEngineAdapter
BacktestEngine = QlibEngineAdapter

__all__ = ["BacktestEngine", "QlibEngineAdapter", "DataSchemaError"]
