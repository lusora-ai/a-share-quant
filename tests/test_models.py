import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from ashare_quant.models.native_lgbm import NativeLGBMModel
from ashare_quant.models.sklearn_hgb import SklearnHGBModel
from ashare_quant.models.qlib_lgbm import OfficialQlibLGBMModel
from ashare_quant.models.experiment import ExperimentTracker
from tests.test_factors import generate_mock_daily_data
from ashare_quant.factors.engine import FactorEngine
from ashare_quant.factors.processor import FactorProcessor
from ashare_quant.labels.generator import LabelGenerator

def test_native_lgbm_model_training_and_prediction():
    mock_df = generate_mock_daily_data(num_stocks=5, num_days=120)
    df_factors = FactorEngine().compute_factors(mock_df)
    df_processed = FactorProcessor().process_cross_section(df_factors)
    df_all = LabelGenerator(horizon=5).generate_labels(df_processed)
    
    unique_dates = sorted(df_all["trade_date"].unique())
    train_dates = unique_dates[60:100]
    test_dates = unique_dates[100:]
    
    train_df = df_all[df_all["trade_date"].isin(train_dates)]
    test_df = df_all[df_all["trade_date"].isin(test_dates)].copy()
    
    feature_cols = [c for c in df_factors.columns if c not in ["ts_code", "trade_date", "open", "high", "low", "close", "volume", "amount", "turn", "pct_chg", "is_suspended", "open_raw", "close_raw", "open_adj", "close_adj", "high_adj", "low_adj"]]
    
    lgbm_model = NativeLGBMModel(feature_cols=feature_cols)
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
        
        exp_id = tracker.create_experiment(
            name="test_lgbm",
            config=config,
            feature_set="custom12",
            model_type="native_lgbm",
            feature_cols=["return_5d", "return_20d"],
            train_end_date="2024-06-30"
        )
        assert os.path.exists(os.path.join(tmpdir, exp_id))
        assert os.path.exists(os.path.join(tmpdir, exp_id, "config.yaml"))
        assert os.path.exists(os.path.join(tmpdir, exp_id, "metadata.json"))
        
        metrics = {"mean_ic": 0.08, "icir": 1.25, "pos_ratio": 0.65}
        imp_df = pd.DataFrame({"feature": ["return_5d"], "importance_gain": [10.5]})
        oos_df = pd.DataFrame({
            "trade_date": ["2024-07-01"],
            "ts_code": ["600000.SH"],
            "score": [0.75],
            "label": [0.05],
            "fold_id": [1],
            "train_end_date": ["2024-06-30"]
        })
        fold_metrics = pd.DataFrame({
            "fold_id": [1],
            "mean_ic": [0.08],
            "icir": [1.25]
        })
        
        tracker.log_results(
            exp_id,
            metrics=metrics,
            feature_importance=imp_df,
            oos_predictions=oos_df,
            fold_metrics=fold_metrics,
            notes="Unit test"
        )
        
        assert os.path.exists(os.path.join(tmpdir, exp_id, "metrics.json"))
        assert os.path.exists(os.path.join(tmpdir, exp_id, "fold_metrics.parquet"))
        assert os.path.exists(os.path.join(tmpdir, exp_id, "oos_predictions.parquet"))
        assert os.path.exists(os.path.join(tmpdir, exp_id, "feature_importance.csv"))
        assert os.path.exists(os.path.join(tmpdir, exp_id, "notes.md"))

