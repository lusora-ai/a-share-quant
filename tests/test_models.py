import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from ashare_quant.models.lgbm_model import LGBMRankingModel
from ashare_quant.models.experiment import ExperimentTracker
from tests.test_factors import generate_mock_daily_data
from ashare_quant.factors.engine import FactorEngine
from ashare_quant.factors.processor import FactorProcessor
from ashare_quant.labels.generator import LabelGenerator

def test_lgbm_model_training_and_prediction():
    mock_df = generate_mock_daily_data(num_stocks=5, num_days=120)
    df_factors = FactorEngine().compute_factors(mock_df)
    df_processed = FactorProcessor().process_cross_section(df_factors)
    df_all = LabelGenerator(horizon=5).generate_labels(df_processed)
    
    unique_dates = sorted(df_all["trade_date"].unique())
    train_dates = unique_dates[60:100]
    test_dates = unique_dates[100:]
    
    train_df = df_all[df_all["trade_date"].isin(train_dates)]
    test_df = df_all[df_all["trade_date"].isin(test_dates)].copy()
    
    lgbm_model = LGBMRankingModel()
    imp_df = lgbm_model.fit(train_df)
    
    assert not imp_df.empty
    assert "importance_gain" in imp_df.columns
    
    preds = lgbm_model.predict(test_df)
    assert len(preds) == len(test_df)
    assert not preds.isna().all()

def test_experiment_tracker():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(base_exp_dir=tmpdir)
        config = {"model": "lgbm", "random_seed": 42}
        
        exp_id = tracker.create_experiment("test_lgbm", config)
        assert os.path.exists(os.path.join(tmpdir, exp_id))
        assert os.path.exists(os.path.join(tmpdir, exp_id, "config.yaml"))
        assert os.path.exists(os.path.join(tmpdir, exp_id, "metadata.json"))
        
        metrics = {"mean_ic": 0.08, "icir": 1.25}
        imp_df = pd.DataFrame({"feature": ["return_5d"], "importance_gain": [10.5]})
        preds_df = pd.DataFrame({"ts_code": ["600000.SH"], "score": [0.75]})
        
        tracker.log_results(exp_id, metrics=metrics, feature_importance=imp_df, predictions=preds_df, notes="Unit test")
        
        assert os.path.exists(os.path.join(tmpdir, exp_id, "metrics.json"))
        assert os.path.exists(os.path.join(tmpdir, exp_id, "feature_importance.csv"))
        assert os.path.exists(os.path.join(tmpdir, exp_id, "predictions.parquet"))
        assert os.path.exists(os.path.join(tmpdir, exp_id, "notes.md"))
