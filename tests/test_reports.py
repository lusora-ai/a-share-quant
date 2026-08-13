import os
import tempfile
import pandas as pd
import pytest
from ashare_quant.reports.daily_report import DailyReportGenerator

def test_daily_report_generator():
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = DailyReportGenerator(output_dir=tmpdir)
        
        candidates = pd.DataFrame([
            {"ts_code": "600000.SH", "name": "浦发银行", "score": 0.92, "volatility_20d": 0.018},
            {"ts_code": "000001.SZ", "name": "平安银行", "score": 0.88, "volatility_20d": 0.021},
            {"ts_code": "601318.SH", "name": "中国平安", "score": 0.85, "volatility_20d": 0.025},
        ])
        
        html_path, md_path = generator.generate_report("2024-01-15", candidates, universe_size=3500)
        
        assert os.path.exists(html_path)
        assert os.path.exists(md_path)
        
        with open(html_path, "r", encoding="utf-8") as f:
            html_text = f.read()
            assert "600000.SH" in html_text
            assert "浦发银行" in html_text
            assert "Top 10 研究候选列表" in html_text
            
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
            assert "600000.SH" in md_text
            assert "| 1 | 600000.SH | 浦发银行 |" in md_text
