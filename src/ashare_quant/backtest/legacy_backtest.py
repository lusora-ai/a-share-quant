"""
Legacy self-developed backtest engine stub.
This engine is permanently disabled in favor of official Microsoft Qlib backtest components.
"""

class LegacyBacktestEngine:
    """
    Deprecated legacy backtest engine.
    Calling this engine will immediately raise NotImplementedError.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Legacy engine is disabled. Use Qlib production engine (QlibEngineAdapter).")

    def run_backtest(self, *args, **kwargs):
        raise NotImplementedError("Legacy engine is disabled. Use Qlib production engine (QlibEngineAdapter).")
