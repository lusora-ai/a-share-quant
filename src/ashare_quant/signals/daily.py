"""
Production Daily Signal Pipeline.
Executes real end-to-end scoring, ranking, and candidate generation without dummy data.
"""
import os
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from ashare_quant.data.processor import DataProcessor
from ashare_quant.features.custom12 import Custom12Factors
from ashare_quant.reports.daily_report import DailyReportGenerator
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.signals.daily")

class DailySignalPipeline:
    """
    真实生产选股信号管线
    严格校验实验元数据与模型文件，无数据/无模型时直接抛出异常终止
    """
    def __init__(self, exp_dir: str = "experiments"):
        self.exp_dir = Path(exp_dir)

    def find_latest_production_model(self) -> Tuple[str, Dict[str, Any], Any]:
        """
        查找并加载最新已训练完成的生产模型 Artifact
        """
        if not self.exp_dir.exists():
            raise RuntimeError("ERROR: Experiments directory 'experiments/' does not exist. Please run 'ashare-quant train' first!")

        exp_folders = [p for p in self.exp_dir.iterdir() if p.is_dir()]
        if not exp_folders:
            raise RuntimeError("ERROR: No trained experiment folders found in 'experiments/'. Please run 'ashare-quant train' first!")

        # 查找包含 model.pkl / model.joblib 且有 metadata.json 的实验
        valid_exps = []
        for p in sorted(exp_folders, reverse=True):
            meta_file = p / "metadata.json"
            model_file = p / "model.pkl"
            if not model_file.exists():
                model_file = p / "model.joblib"
            if meta_file.exists() and model_file.exists():
                valid_exps.append((p.name, meta_file, model_file))

        if not valid_exps:
            raise RuntimeError("ERROR: No valid trained model artifact (model.pkl/joblib) found in experiments/. Please run 'ashare-quant train' first!")

        exp_id, meta_path, model_path = valid_exps[0]
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        try:
            model = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"ERROR: Failed to deserialize model artifact at '{model_path}': {e}") from e

        logger.info(f"Loaded production model from experiment '{exp_id}' ({model_path.name}).")
        return exp_id, metadata, model

    def run_daily_pipeline(
        self,
        target_date: Optional[str] = None,
        model_artifact_path: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info(f"Executing REAL Daily Signal Pipeline for date: {target_date}...")

        # 1. 验证并加载最新数据快照
        processor = DataProcessor()
        try:
            df_master = processor.storage.load_parquet("stock_master", is_processed=True)
            if df_master is None or df_master.empty:
                raise RuntimeError("ERROR: Stock master data missing. Run 'ashare-quant update-data' first!")

            df_daily = processor.storage.load_parquet("daily_ohlcv", is_processed=True)
            if df_daily is None or df_daily.empty:
                raise RuntimeError("ERROR: Daily market data missing. Run 'ashare-quant update-data' first!")
        finally:
            processor.close()

        # 2. 确定计算日期
        available_dates = sorted(df_daily["trade_date"].unique())
        calc_date = target_date if (target_date and target_date in available_dates) else available_dates[-1]

        # 3. 计算多因子特征
        f_engine = Custom12Factors()
        df_factors = f_engine.compute(df_daily)
        df_latest = df_factors[df_factors["trade_date"] == calc_date].copy()
        if df_latest.empty:
            raise RuntimeError(f"ERROR: No factor data available for calculation date {calc_date}.")

        # 4. 加载生产模型并执行预测打分
        if model_artifact_path and os.path.exists(model_artifact_path):
            exp_id = "custom_path"
            model = joblib.load(model_artifact_path)
            meta = {"model": "custom"}
        else:
            exp_id, meta, model = self.find_latest_production_model()

        # 预测打分
        try:
            scores = model.predict(df_latest)
            df_latest["score"] = scores
        except Exception as e:
            raise RuntimeError(f"ERROR: Model prediction failed: {e}") from e

        # 5. 排序选出 Top10 候选股票
        df_sorted = df_latest.sort_values("score", ascending=False).reset_index(drop=True)
        top10 = df_sorted.head(10)

        candidates = []
        for rank, row in enumerate(top10.to_dict("records"), 1):
            candidates.append({
                "rank": rank,
                "ts_code": row.get("ts_code"),
                "stock_name": row.get("name", row.get("ts_code")),
                "score": float(row.get("score", 0.0)),
                "industry": row.get("industry", "N/A"),
                "close": float(row.get("close", 0.0))
            })

        # 6. 生成每日报告
        report_gen = DailyReportGenerator()
        html_path, md_path = report_gen.generate_report(
            date_str=calc_date,
            df_candidates=df_sorted,
            universe_size=len(df_latest),
            model_version=exp_id
        )

        res = {
            "trade_date": calc_date,
            "model_id": exp_id,
            "data_snapshot_id": f"snapshot_{calc_date}",
            "candidates_count": len(candidates),
            "candidates": candidates,
            "report_path": html_path
        }
        logger.info(f"Real daily signal pipeline completed. {len(candidates)} candidates generated.")
        return res
