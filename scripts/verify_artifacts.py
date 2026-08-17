import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(".").resolve()))

import pandas as pd
import numpy as np

from ashare_quant.backtest.qlib_engine import QlibEngineAdapter, build_qlib_signal
from ashare_quant.data.qlib_exporter import QlibDataProviderManager
from tests.test_factors import generate_mock_daily_data

def main():
    # 1. 查找最新 experiment
    exp_dirs = sorted([p for p in Path('experiments').iterdir() if p.is_dir()])
    latest_exp = exp_dirs[-1]
    model_path = latest_exp / 'model.joblib'
    pred_path = latest_exp / 'predictions.parquet'
    meta_path = latest_exp / 'metadata.json'

    print("=== 5. Model Artifact Path ===")
    print(str(model_path.resolve()))

    print("\n=== 6. Test Prediction (First 5 rows) ===")
    df_preds = pd.read_parquet(pred_path)
    print(df_preds.head(5).to_string())

    print("\n=== 7. Real Qlib Backtest Portfolio Report (First 5 rows) ===")
    mock_df = generate_mock_daily_data(num_stocks=5, num_days=60)
    mock_df['score'] = np.random.uniform(0, 1, len(mock_df))
    provider_path = QlibDataProviderManager.dump_df_to_qlib_bin(mock_df, target_dir='data/qlib_cn')
    QlibDataProviderManager.init_qlib(provider_uri=provider_path, force=True)

    signal = build_qlib_signal(mock_df, score_col='score')
    dates = sorted(mock_df['trade_date'].unique())

    engine = QlibEngineAdapter()
    report_df, metrics = engine.run_qlib_backtest(
        signal_series=signal,
        start_time=dates[0],
        end_time=dates[-1],
        benchmark='SH000300'
    )

    print(report_df.head(5).to_string())

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    print("\n=== 8. Real Performance & Evaluation Metrics ===")
    print("Mean IC:", f"{meta.get('metrics', {}).get('mean_ic', 0.0):.4f}")
    print("ICIR:", f"{meta.get('metrics', {}).get('icir', 0.0):.4f}")
    print("Annualized Excess Return:", f"{metrics.get('excess_return', 0.0):.2%}")
    print("Information Ratio (Sharpe):", f"{metrics.get('sharpe', 0.0):.4f}")
    print("Max Drawdown:", f"{metrics.get('max_drawdown', 0.0):.2%}")

if __name__ == "__main__":
    main()
