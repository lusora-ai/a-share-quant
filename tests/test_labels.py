import pandas as pd
import numpy as np
import pytest
from ashare_quant.labels.generator import LabelGenerator
from ashare_quant.models.baseline import EqualWeightBaseline, RidgeBaseline
from ashare_quant.models.metrics import compute_daily_ic, compute_ic_stats
from tests.test_factors import generate_mock_daily_data
from ashare_quant.factors.engine import FactorEngine, FACTOR_NAMES
from ashare_quant.factors.processor import FactorProcessor

def test_label_generator():
    mock_df = generate_mock_daily_data(num_stocks=5, num_days=30)
    label_gen = LabelGenerator(horizon=5)
    df_labeled = label_gen.generate_labels(mock_df)
    
    assert "rank_label_5d" in df_labeled.columns
    assert "raw_label_5d" in df_labeled.columns
    
    # 验证截面 Percentile Rank 在 [0.0, 1.0] 区间
    valid_ranks = df_labeled["rank_label_5d"].dropna()
    assert (valid_ranks >= 0.0).all()
    assert (valid_ranks <= 1.0).all()

def test_baseline_models_and_ic():
    mock_df = generate_mock_daily_data(num_stocks=5, num_days=100)
    engine = FactorEngine()
    df_factors = engine.compute_factors(mock_df)
    df_processed = FactorProcessor().process_cross_section(df_factors)
    
    label_gen = LabelGenerator(horizon=5)
    df_all = label_gen.generate_labels(df_processed)
    
    # 测试等权基线
    eq_model = EqualWeightBaseline()
    df_all["eq_score"] = eq_model.predict(df_all)
    
    # 测试 Ridge 基线
    unique_dates = sorted(df_all["trade_date"].unique())
    train_dates = unique_dates[60:85]
    test_dates = unique_dates[85:]
    
    train_df = df_all[df_all["trade_date"].isin(train_dates)]
    test_df = df_all[df_all["trade_date"].isin(test_dates)].copy()
    
    ridge_model = RidgeBaseline(alpha=10.0)
    ridge_model.fit(train_df, label_col="rank_label_5d")
    test_df["ridge_score"] = ridge_model.predict(test_df)
    
    # 计算 IC 统计量
    ic_df = compute_daily_ic(test_df, score_col="ridge_score", label_col="rank_label_5d")
    ic_stats = compute_ic_stats(ic_df)
    
    assert "mean_ic" in ic_stats
    assert "icir" in ic_stats
    assert isinstance(ic_stats["mean_ic"], float)

def test_p0_label_not_in_feature_list():
    """
    P0 测试: 严禁将 rank_label 作为特征输入给模型
    """
    features = FACTOR_NAMES
    assert "rank_label_5d" not in features
    assert "raw_label_5d" not in features
