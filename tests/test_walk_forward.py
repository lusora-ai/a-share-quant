import pandas as pd
import numpy as np
import pytest
from ashare_quant.models.walk_forward import WalkForwardEvaluator
from tests.test_factors import generate_mock_daily_data
from ashare_quant.factors.engine import FactorEngine
from ashare_quant.factors.processor import FactorProcessor
from ashare_quant.labels.generator import LabelGenerator

def test_walk_forward_evaluator():
    mock_df = generate_mock_daily_data(num_stocks=5, num_days=120)
    df_factors = FactorEngine().compute_factors(mock_df)
    df_processed = FactorProcessor().process_cross_section(df_factors)
    df_all = LabelGenerator(horizon=5).generate_labels(df_processed)
    
    wf_evaluator = WalkForwardEvaluator()
    fold_df, summary = wf_evaluator.run_walk_forward(df_all)
    
    assert not fold_df.empty
    assert "fold" in fold_df.columns
    assert "mean_ic" in fold_df.columns
    assert "avg_fold_ic" in summary
