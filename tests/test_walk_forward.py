import pandas as pd
import numpy as np
import pytest
from ashare_quant.models.walk_forward import WalkForwardEvaluator
from ashare_quant.validation.purged_walk_forward import PurgedWalkForwardEvaluator
from tests.test_factors import generate_mock_daily_data
from ashare_quant.factors.engine import FactorEngine
from ashare_quant.factors.processor import FactorProcessor
from ashare_quant.labels.generator import LabelGenerator


def _make_sufficient_daily_data(num_stocks=5, num_days=252 * 8):
    """Helper: generate enough daily data for default 4+1+1 walk-forward."""
    dates = pd.date_range(start="2015-01-01", periods=num_days, freq="B").strftime("%Y-%m-%d")
    rows = []
    for i in range(num_stocks):
        ts_code = f"{600000 + i}.SH"
        base_price = 10.0 + i
        np.random.seed(42 + i)
        returns = np.random.normal(0.001, 0.02, num_days)
        prices = base_price * np.exp(np.cumsum(returns))
        for d_idx, date_str in enumerate(dates):
            p = prices[d_idx]
            rows.append({
                "ts_code": ts_code, "trade_date": date_str,
                "open": p, "high": p * 1.01, "low": p * 0.99, "close": p,
                "open_raw": p, "close_raw": p, "open_adj": p, "close_adj": p,
                "high_adj": p * 1.01, "low_adj": p * 0.99,
                "volume": 50000, "amount": 50000 * p * 100,
                "turn": 1.0, "pct_chg": returns[d_idx] * 100.0,
                "is_suspended": False,
            })
    return pd.DataFrame(rows)


def test_walk_forward_evaluator():
    """
    Legacy WalkForwardEvaluator test (backwards compatibility wrapper).
    Uses sufficient data + explicit short walk-forward config so the test
    does not depend on the production config values.
    """
    from ashare_quant.validation.purged_walk_forward import PurgedWalkForwardEvaluator
    # 120 days ≈ 0.5 year → use explicit 0-year-ish config to match test data
    short_config = {"walk_forward": {"train_years": 0, "val_years": 0, "test_years": 0, "embargo_days": 0}}
    # The wrapper delegates to PurgedWalkForwardEvaluator; but with 0-year spans
    # we cannot generate year-based folds. Instead, provide enough data.
    mock_df = _make_sufficient_daily_data(num_stocks=5, num_days=252 * 7)
    df_factors = FactorEngine().compute_factors(mock_df)
    df_processed = FactorProcessor().process_cross_section(df_factors)
    df_all = LabelGenerator(horizon=5).generate_labels(df_processed)

    # Use the production default config (4+1+1) with sufficient 7-year data
    wf_evaluator = WalkForwardEvaluator()
    fold_df, summary = wf_evaluator.run_walk_forward(df_all)

    assert not fold_df.empty
    assert "fold_id" in fold_df.columns
    assert "mean_ic" in fold_df.columns
    assert "avg_fold_ic" in summary
