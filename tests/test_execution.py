import os
import tempfile
import pandas as pd
import pytest
from ashare_quant.portfolio.execution import ManualExecutionTracker

def test_affordability_evaluator():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "exec.json")
        tracker = ManualExecutionTracker(log_path=log_file, max_cash=300.0)
        
        candidates = pd.DataFrame([
            {"ts_code": "600000.SH", "name": "高价股", "close": 15.0}, # 100股 = 1500元 > 300元 -> unaffordable
            {"ts_code": "000001.SZ", "name": "低价股", "close": 2.5},  # 100股 = 250元 <= 300元 -> affordable
        ])
        
        evaluated = tracker.evaluate_affordability(candidates)
        
        assert evaluated.iloc[0]["affordability_status"] == "unaffordable"
        assert evaluated.iloc[0]["is_affordable"] == False
        
        assert evaluated.iloc[1]["affordability_status"] == "affordable"
        assert evaluated.iloc[1]["is_affordable"] == True

def test_record_manual_trade():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "exec.json")
        tracker = ManualExecutionTracker(log_path=log_file, max_cash=300.0)
        
        tracker.record_manual_trade(
            trade_date="2024-01-15",
            ts_code="000001.SZ",
            name="平安银行",
            action="BUY",
            shares=100,
            price=2.5,
            model_score=0.88,
            notes="首次实盘买入"
        )
        
        summary_df = tracker.get_execution_summary()
        assert len(summary_df) == 1
        assert summary_df.iloc[0]["ts_code"] == "000001.SZ"
        assert summary_df.iloc[0]["cost"] == 250.0
