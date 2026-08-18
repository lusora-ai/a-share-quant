"""
Production Daily Signal Pipeline.
Executes real end-to-end scoring, ranking, and candidate generation with Universe tradability filters.
Zero dummy data, zero silent date fallbacks, zero fake close=0.0.
"""
import os
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from ashare_quant.data.processor import DataProcessor
from ashare_quant.data.processor import DataProcessor
from ashare_quant.data.symbols import from_qlib_symbol, to_qlib_symbol
from ashare_quant.data.qlib_exporter import (
    QlibDataProviderManager,
    get_expected_latest_completed_trade_date,
    MarketCalendarUnavailableError
)
from ashare_quant.features.custom12 import Custom12Factors, FACTOR_NAMES_12
from ashare_quant.features.qlib_alpha158 import OfficialQlibAlpha158
from ashare_quant.universe.filter import build_custom12_universe
from ashare_quant.reports.daily_report import DailyReportGenerator
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.signals.daily")

class StaleMarketDataError(RuntimeError):
    """Raised when requesting a signal date beyond the available provider data or when provider is stale."""
    pass

class RawPriceUnavailableError(ValueError):
    """Raised when raw transaction price cannot be computed due to missing/invalid close or factor."""
    pass

class DailySignalPipeline:
    """
    真实生产选股信号管线
    严格校验实验元数据与模型文件，无数据/无模型/数据过旧时直接抛出异常终止
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

    def _fetch_alpha158_real_close_and_name(
        self,
        calc_date: str,
        instruments: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        使用 Qlib Data API (from qlib.data import D) 读取 $close 和 $factor。
        计算真实未复权成交参考价: raw_close = $close / $factor。
        严格要求 $close 与 $factor 均为有限正数；缺失时直接剔除或报错，严禁 fallback 1.0 或 0.0。
        """
        from qlib.data import D

        q_instruments = [to_qlib_symbol(inst) for inst in instruments]
        try:
            price_df = D.features(
                q_instruments,
                fields=["$close", "$factor"],
                start_time=calc_date,
                end_time=calc_date,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch $close and $factor from Qlib for date '{calc_date}': {e}") from e

        if price_df is None or price_df.empty:
            raise RawPriceUnavailableError(f"Qlib returned empty features for date '{calc_date}'.")

        price_df = price_df.reset_index()
        if "$close" not in price_df.columns or "$factor" not in price_df.columns:
            raise RawPriceUnavailableError(f"Qlib features missing $close or $factor. Columns: {list(price_df.columns)}")

        price_df["instrument"] = price_df["instrument"].astype(str)

        # 严格校验: $close 与 $factor 必须均为有限正数，严禁 fillna(1.0)
        valid_mask = (
            pd.notna(price_df["$close"]) & np.isfinite(price_df["$close"]) & (price_df["$close"] > 0) &
            pd.notna(price_df["$factor"]) & np.isfinite(price_df["$factor"]) & (price_df["$factor"] > 0)
        )
        valid_price_df = price_df[valid_mask].copy()

        if valid_price_df.empty:
            raise RawPriceUnavailableError(
                f"RawPriceUnavailableError: No instruments have valid positive $close and $factor for date '{calc_date}'. "
                f"Raw transaction price cannot be computed."
            )

        valid_price_df["raw_close"] = (valid_price_df["$close"] / valid_price_df["$factor"]).round(2)
        close_df = valid_price_df[["instrument", "raw_close"]].rename(columns={"raw_close": "close"})

        # 证券名称：尝试从项目 stock_master 匹配
        name_df = pd.DataFrame(columns=["instrument", "name"])
        try:
            processor = DataProcessor()
            try:
                master = processor.storage.load_parquet("stock_master", is_processed=True)
            finally:
                processor.close()
            if master is not None and not master.empty and "ts_code" in master.columns and "name" in master.columns:
                master = master.copy()
                master["instrument"] = master["ts_code"].astype(str).apply(to_qlib_symbol)
                master = master[["instrument", "name"]].drop_duplicates(subset=["instrument"])
                name_df = master
        except Exception as e:
            logger.warning(f"Could not load stock_master for name lookup ({e}); names will fallback to ts_code.")

        return close_df, name_df

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
            provider_data_end_date = calendar_dates[-1]

            # CSI300 Point-in-Time Membership 解析
            csi300_file = Path(resolved_uri) / "instruments" / "csi300.txt"
            if not csi300_file.exists():
                raise FileNotFoundError(f"Missing required Qlib CSI300 instruments file: '{csi300_file}'")

            membership_records = []
            for line in csi300_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 3:
                    membership_records.append((parts[0], parts[1], parts[2]))
                elif len(parts) >= 1:
                    membership_records.append((parts[0], "2000-01-01", "2099-12-31"))

            max_membership_end = max((r[2] for r in membership_records), default="N/A")

            # Provider Freshness Guard: 校验 Provider 是否过期
            if target_date is None:
                expected_latest_market_date = get_expected_latest_completed_trade_date()
                if provider_data_end_date < expected_latest_market_date:
                    raise StaleMarketDataError(
                        f"Qlib provider market data is STALE! Expected latest completed market date: '{expected_latest_market_date}', "
                        f"but Qlib provider data ends at: '{provider_data_end_date}'. Daily Signal generation for today is blocked. "
                        f"Please update Qlib provider data, or pass --date <HISTORICAL_DATE> to replay historical signals."
                    )
                # CSI300 Membership Freshness Guard for today
                if max_membership_end < expected_latest_market_date:
                    raise StaleMarketDataError(
                        f"CSI300 instrument membership is stale! Max membership end '{max_membership_end}' "
                        f"is older than expected market date '{expected_latest_market_date}'. "
                        f"Daily Signal generation for today is blocked. Please update Qlib instruments/csi300.txt, "
                        f"or pass --date <HISTORICAL_DATE> to replay historical signals."
                    )
                active_csi300 = set(r[0] for r in membership_records if r[1] <= expected_latest_market_date <= r[2])
                active_count = len(active_csi300)
                if active_count < 280 or active_count > 320:
                    raise StaleMarketDataError(
                        f"CSI300 instrument membership invalid on '{expected_latest_market_date}': {active_count} active constituents "
                        f"(expected 280~320). Please update Qlib instruments/csi300.txt."
                    )
                calc_date = provider_data_end_date
            else:
                # 允许显式指定历史日期重放
                if target_date > provider_data_end_date:
                    raise StaleMarketDataError(
                        f"Requested signal date '{target_date}' exceeds Qlib provider latest date '{provider_data_end_date}'. "
                        f"Qlib provider market data is stale. Please update Qlib data before running daily signal."
                    )
                if target_date not in calendar_dates:
                    raise ValueError(f"ERROR: requested date unavailable in Qlib calendar: '{target_date}'.")

                # CSI300 Membership Check for historical date
                active_csi300 = set(r[0] for r in membership_records if r[1] <= target_date <= r[2])
                active_count = len(active_csi300)
                if active_count < 280 or active_count > 320:
                    raise ValueError(
                        f"CSI300 instrument membership invalid on historical date '{target_date}': {active_count} active constituents "
                        f"(expected 280~320)."
                    )
                calc_date = target_date

            logger.info(f"Computing official Qlib Alpha158 features for date '{calc_date}' (Provider End: {provider_data_end_date})...")
            alpha_adapter = OfficialQlibAlpha158()
            handler = alpha_adapter.create_handler_instance(
                instruments="csi300",
                start_time=calc_date,
                end_time=calc_date
            )
            from qlib.data.dataset import DatasetH
            dataset = DatasetH(handler=handler, segments={"test": (calc_date, calc_date)})

            if hasattr(model, "predict"):
                scores_series = model.predict(dataset, segment="test")
            else:
                raise RuntimeError(f"Unknown production model object for Qlib pipeline: {type(model)}")

            if isinstance(scores_series, pd.Series):
                pred_df = scores_series.to_frame(name="score").reset_index()
            else:
                pred_df = pd.DataFrame(scores_series, columns=["score"]).reset_index()

            pred_df["ts_code"] = pred_df["instrument"].astype(str).apply(from_qlib_symbol)

            # 获取真实未复权 raw close 与证券名称 (通过 Qlib Data API: $close / $factor)
            close_df, name_df = self._fetch_alpha158_real_close_and_name(calc_date, pred_df["ts_code"].tolist())
            
            # 严格使用 inner merge，剔除无法获取有效真实股价的标的，严禁 fillna(0.0)
            pred_df = pred_df.merge(close_df, on="instrument", how="inner")
            pred_df = pred_df[pd.notna(pred_df["close"]) & np.isfinite(pred_df["close"]) & (pred_df["close"] > 0)].copy()

            if pred_df.empty:
                raise RawPriceUnavailableError(
                    f"RawPriceUnavailableError: No scored instruments have valid positive raw close prices for date '{calc_date}'."
                )

            pred_df = pred_df.merge(name_df, on="instrument", how="left")
            pred_df["name"] = pred_df["name"].fillna(pred_df["ts_code"])

            pred_df["industry"] = "CSI300"
            df_scored = pred_df

        else:
            # ===== Custom12 Pipeline (shared universe) =====
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
            custom_data_end_date = available_dates[-1]

            # Freshness Guard: 校验 Custom12 数据是否过期
            if target_date is None:
                expected_latest_market_date = get_expected_latest_completed_trade_date()
                if custom_data_end_date < expected_latest_market_date:
                    raise StaleMarketDataError(
                        f"Daily market data is STALE! Expected latest completed market date: '{expected_latest_market_date}', "
                        f"but data ends at: '{custom_data_end_date}'. Please run 'ashare-quant update-data' first, "
                        f"or pass --date <HISTORICAL_DATE> to replay historical signals."
                    )
                calc_date = custom_data_end_date
            else:
                if target_date > custom_data_end_date:
                    raise StaleMarketDataError(
                        f"Requested signal date '{target_date}' exceeds daily market data end '{custom_data_end_date}'. "
                        f"Please run 'ashare-quant update-data' first."
                    )
                if target_date not in available_dates:
                    raise ValueError(f"ERROR: requested date unavailable: '{target_date}'.")
                calc_date = target_date

            # 共享 universe 过滤 (与 train 严格复用同一函数)
            df_universe = build_custom12_universe(df_daily, master_df=df_master)

            # 计算多因子特征 (与 train 严格复用同一 FeaturePipeline)
            f_engine = Custom12Factors()
            df_factors = f_engine.compute(df_universe)
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

        # 3. 排序选出 Top10 候选股票 (严格要求 close > 0, cost_100_shares > 0)
        df_sorted = df_scored.sort_values("score", ascending=False).reset_index(drop=True)
        # 过滤掉价格无效或 <= 0 的股票
        df_valid_candidates = df_sorted[pd.notna(df_sorted["close"]) & np.isfinite(df_sorted["close"]) & (df_sorted["close"] > 0)].copy()
        if df_valid_candidates.empty:
            raise RawPriceUnavailableError(f"RawPriceUnavailableError: No candidate stocks with valid positive prices available on '{calc_date}'.")
        top10 = df_valid_candidates.head(10)

        candidates = []
        for rank, row in enumerate(top10.to_dict("records"), 1):
            close_val = float(row.get("close"))
            if not np.isfinite(close_val) or close_val <= 0:
                continue
            cost_100 = round(close_val * 100, 2)
            candidates.append({
                "trade_date": calc_date,
                "rank": rank,
                "ts_code": row.get("ts_code"),
                "name": row.get("name", row.get("ts_code")),
                "score": float(row.get("score", 0.0)),
                "industry": row.get("industry", "N/A"),
                "close": close_val,
                "cost_100_shares": cost_100,
                "model_id": exp_id,
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
            "signal_date": calc_date,
            "provider_data_end": provider_data_end_date if feature_set == "alpha158" else custom_data_end_date,
            "production_train_end": meta.get("production_train_end_date") or meta.get("train_end_date", ""),
            "model_id": exp_id,
            "data_snapshot_id": f"snapshot_{calc_date}",
            "candidates_count": len(candidates),
            "candidates": candidates,
            "report_path": html_path
        }
        logger.info(f"Real daily signal pipeline completed. {len(candidates)} candidates generated.")
        return res
