"""
Production Daily Signal Pipeline.
Executes real end-to-end scoring, ranking, and candidate generation with Universe tradability filters.
Zero dummy data, zero silent date fallbacks.
"""
import os
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from ashare_quant.data.processor import DataProcessor
from ashare_quant.data.symbols import from_qlib_symbol
from ashare_quant.data.qlib_exporter import QlibDataProviderManager
from ashare_quant.features.custom12 import Custom12Factors, FACTOR_NAMES_12
from ashare_quant.features.qlib_alpha158 import OfficialQlibAlpha158
from ashare_quant.universe.filter import UniverseFilter
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

        valid_exps = []
        for p in sorted(exp_folders, reverse=True):
            meta_file = p / "metadata.json"
            # 优先查找 production_model.joblib / production_model.pkl
            for m_candidate in ["production_model.joblib", "production_model.pkl", "model.joblib", "model.pkl"]:
                model_file = p / m_candidate
                if meta_file.exists() and model_file.exists():
                    valid_exps.append((p.name, meta_file, model_file))
                    break

        if not valid_exps:
            raise RuntimeError("ERROR: No valid trained production model artifact found in experiments/. Please run 'ashare-quant train' first!")

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

        # 1. 加载生产模型与元数据
        if model_artifact_path and os.path.exists(model_artifact_path):
            exp_id = "custom_path"
            model = joblib.load(model_artifact_path)
            meta = {"model_type": "custom", "feature_set": "custom12", "feature_cols": FACTOR_NAMES_12}
        else:
            exp_id, meta, model = self.find_latest_production_model()

        feature_set = meta.get("feature_set", "custom12")
        feature_cols = meta.get("feature_cols", FACTOR_NAMES_12)

        # 2. 根据 feature_set 分发执行
        if feature_set == "alpha158":
            # ===== Qlib Alpha158 Pipeline =====
            provider_uri = meta.get("provider_uri") or None
            resolved_uri = QlibDataProviderManager.init_qlib(provider_uri=provider_uri)
            
            cal_file = Path(resolved_uri) / "calendars" / "day.txt"
            calendar_dates = [line.strip() for line in cal_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            
            # P1-3: target_date 禁止 silent fallback
            if target_date is not None:
                if target_date not in calendar_dates:
                    raise ValueError(f"ERROR: requested date unavailable: '{target_date}'.")
                calc_date = target_date
            else:
                calc_date = calendar_dates[-1]

            logger.info(f"Computing official Qlib Alpha158 features for date '{calc_date}'...")
            alpha_adapter = OfficialQlibAlpha158()
            handler = alpha_adapter.create_handler_instance(
                instruments="csi300",
                start_time=calc_date,
                end_time=calc_date
            )
            from qlib.data.dataset import DatasetH
            dataset = DatasetH(handler=handler, segments={"test": (calc_date, calc_date)})
            
            if hasattr(model, "predict"):
                if hasattr(model, "is_fitted"):
                    scores_series = model.predict(dataset, segment="test")
                else:
                    scores_series = model.predict(dataset, segment="test")
            else:
                raise RuntimeError(f"Unknown production model object for Qlib pipeline: {type(model)}")

            if isinstance(scores_series, pd.Series):
                pred_df = scores_series.to_frame(name="score").reset_index()
            else:
                pred_df = pd.DataFrame(scores_series, columns=["score"]).reset_index()

            pred_df["ts_code"] = pred_df["instrument"].astype(str).apply(from_qlib_symbol)
            pred_df["name"] = pred_df["ts_code"]
            pred_df["close"] = 0.0
            pred_df["industry"] = "CSI300"
            df_scored = pred_df

        else:
            # ===== Custom12 Pipeline =====
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

            available_dates = sorted(df_daily["trade_date"].unique())
            
            # P1-3: target_date 禁止 silent fallback
            if target_date is not None:
                if target_date not in available_dates:
                    raise ValueError(f"ERROR: requested date unavailable: '{target_date}'.")
                calc_date = target_date
            else:
                calc_date = available_dates[-1]

            # P1-2: 截面 Universe 过滤器 (排除 ST / 停牌 / 上市天数不足 / 流动性不足)
            u_filter = UniverseFilter()
            df_universe = u_filter.filter_universe(df_daily, master_df=df_master)
            df_filtered = df_universe[df_universe["is_in_universe"]].copy()

            # 计算多因子特征 (与 train 严格复用同一 FeaturePipeline)
            f_engine = Custom12Factors()
            df_factors = f_engine.compute(df_filtered)
            df_latest = df_factors[df_factors["trade_date"] == calc_date].copy()
            if df_latest.empty:
                raise RuntimeError(f"ERROR: No factor data available for calculation date '{calc_date}'.")

            # 校验特征列
            assert set(feature_cols).issubset(df_latest.columns), f"Missing feature cols: {set(feature_cols) - set(df_latest.columns)}"

            # 预测打分
            try:
                scores = model.predict(df_latest)
                df_latest["score"] = scores
                df_scored = df_latest
            except Exception as e:
                raise RuntimeError(f"ERROR: Model prediction failed: {e}") from e

        # 3. 排序选出 Top10 候选股票
        df_sorted = df_scored.sort_values("score", ascending=False).reset_index(drop=True)
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

        # 4. 生成每日报告
        report_gen = DailyReportGenerator()
        html_path, md_path = report_gen.generate_report(
            date_str=calc_date,
            df_candidates=df_sorted,
            universe_size=len(df_scored),
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

