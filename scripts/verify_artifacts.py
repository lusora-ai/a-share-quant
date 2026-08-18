"""
Production Artifacts Verification Script.
Inspects real experiment archives, OOS predictions, and backtest results.
Zero mock data generation, zero imports from tests/.
"""
import os
import sys
import json
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

def main():
    exp_base = Path("experiments")
    if not exp_base.exists():
        print("ERROR: 'experiments/' directory does not exist. Run 'ashare-quant train' first.")
        sys.exit(1)

    exp_dirs = sorted([p for p in exp_base.iterdir() if p.is_dir()])
    if not exp_dirs:
        print("ERROR: No experiment folders found in 'experiments/'. Run 'ashare-quant train' first.")
        sys.exit(1)

    latest_exp = exp_dirs[-1]
    print(f"=== 1. Target Experiment: {latest_exp.name} ===")

    meta_path = latest_exp / "metadata.json"
    metrics_path = latest_exp / "metrics.json"
    oos_pred_path = latest_exp / "oos_predictions.parquet"
    if not oos_pred_path.exists():
        oos_pred_path = latest_exp / "predictions.parquet"
    model_path = latest_exp / "production_model.joblib"
    if not model_path.exists():
        model_path = latest_exp / "model.joblib"
    backtest_report_path = latest_exp / "backtest_report.parquet"
    backtest_metrics_path = latest_exp / "backtest_metrics.json"

    # 1. Metadata Verification
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print("\n=== 2. Metadata (metadata.json) ===")
        print(f"Experiment ID   : {meta.get('experiment_id')}")
        print(f"Feature Set     : {meta.get('feature_set')}")
        print(f"Model Type      : {meta.get('model_type')}")
        print(f"Features Count  : {len(meta.get('feature_cols', []))}")
        print(f"Train End Date  : {meta.get('train_end_date')}")
        print(f"Git Commit SHA  : {meta.get('git_commit_sha')}")
        print(f"Created At      : {meta.get('created_at')}")

    # 2. Production Model Artifact
    print("\n=== 3. Production Model Artifact ===")
    if model_path.exists():
        print(f"Model File Path : {model_path.resolve()} (Size: {model_path.stat().st_size} bytes)")
    else:
        print(f"WARNING: Model file missing at {model_path}")

    # 3. OOS Predictions Verification
    print("\n=== 4. OOS Predictions Sample (First 5 rows) ===")
    if oos_pred_path.exists():
        df_preds = pd.read_parquet(oos_pred_path)
        print(f"Total Rows: {len(df_preds)}, Columns: {list(df_preds.columns)}")
        print(df_preds.head(5).to_string(index=False))
    else:
        print(f"WARNING: OOS Predictions file missing at {oos_pred_path}")

    # 4. Evaluation Metrics Verification
    print("\n=== 5. Walk-Forward Evaluation Metrics (metrics.json) ===")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"Mean IC   : {metrics.get('mean_ic', 0.0):.4f}")
        print(f"ICIR      : {metrics.get('icir', 0.0):.4f}")
        print(f"Pos Ratio : {metrics.get('pos_ratio', 0.0):.2%}")
        print(f"Num Folds : {metrics.get('num_folds', 0)}")
    else:
        print(f"WARNING: metrics.json missing at {metrics_path}")

    # 5. Backtest Report Verification (if run)
    print("\n=== 6. Qlib Backtest Report & Risk Metrics ===")
    if backtest_metrics_path.exists():
        with open(backtest_metrics_path, "r", encoding="utf-8") as f:
            bt_metrics = json.load(f)
        print(f"Portfolio Annual Return  : {bt_metrics.get('portfolio_annualized_return', 0.0):.2%}")
        print(f"Benchmark Annual Return  : {bt_metrics.get('benchmark_annualized_return', 0.0):.2%}")
        print(f"Excess Annual Return     : {bt_metrics.get('excess_annualized_return', 0.0):.2%}")
        print(f"Portfolio Sharpe Ratio   : {bt_metrics.get('portfolio_sharpe', 0.0):.4f}")
        print(f"Information Ratio (IR)   : {bt_metrics.get('information_ratio', 0.0):.4f}")
        print(f"Portfolio Max Drawdown   : {bt_metrics.get('portfolio_max_drawdown', 0.0):.2%}")
    else:
        print("Backtest metrics not generated yet for this experiment. Run 'ashare-quant backtest --experiment <EXP_ID>'")

    if backtest_report_path.exists():
        df_rep = pd.read_parquet(backtest_report_path)
        print("\nBacktest Portfolio Report (First 5 rows):")
        print(df_rep.head(5).to_string())

if __name__ == "__main__":
    main()
