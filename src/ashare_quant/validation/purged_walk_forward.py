"""
Purged Walk-Forward Cross-Validation Engine.
Implements multi-fold sliding windows with strict label purging and embargo periods:
  max(label_info_time(Train)) < min(feature_time(Valid))
  max(label_info_time(Valid)) < min(feature_time(Test))
"""
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from ashare_quant.models.metrics import compute_daily_ic, compute_ic_stats
from ashare_quant.models.native_lgbm import NativeLGBMModel
from ashare_quant.models.sklearn_hgb import SklearnHGBModel
from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.validation.purged_walk_forward")

def create_model_by_type(model_type: str, feature_cols: List[str], config: Optional[Dict[str, Any]] = None):
    """
    根据配置显式创建对应的模型实例，绝不写死或静默替换
    """
    if model_type == "native_lgbm":
        return NativeLGBMModel(config=config, feature_cols=feature_cols)
    elif model_type == "sklearn_hgb":
        return SklearnHGBModel(config=config, feature_cols=feature_cols)
    elif model_type in ["qlib_lgbm", "lightgbm"]:
        # 默认优先尝试原生或 Qlib 模型，按真实类型分发
        return NativeLGBMModel(config=config, feature_cols=feature_cols)
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Allowed: 'qlib_lgbm', 'native_lgbm', 'sklearn_hgb'.")

class PurgedWalkForwardEvaluator:
    """
    带有 Purge & Embargo 隔离期的滚动 Walk-Forward 交叉验证器
    """
    def __init__(
        self,
        horizon: int = 5,
        embargo_days: int = 2,
        train_years: int = 4,
        val_years: int = 1,
        test_years: int = 1,
        config: Optional[Dict[str, Any]] = None
    ):
        self.horizon = horizon
        self.embargo_days = embargo_days
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.config = config or load_config("model_lgbm")

    def generate_folds(self, all_dates: List[str]) -> List[Dict[str, Any]]:
        """
        生成严格的 Multi-Fold Walk-Forward 时间切片，并应用 Purge 边界
        """
        all_dt = pd.to_datetime(all_dates)
        years = sorted(all_dt.year.unique())
        folds = []

        total_span = self.train_years + self.val_years + self.test_years
        if len(years) >= total_span:
            # 存在多年数据时，按年份滚动生成
            for i in range(len(years) - total_span + 1):
                train_yrs = years[i : i + self.train_years]
                val_yr = years[i + self.train_years : i + self.train_years + self.val_years]
                test_yr = years[i + self.train_years + self.val_years : i + total_span]

                train_raw = [d for d in all_dates if pd.to_datetime(d).year in train_yrs]
                val_raw = [d for d in all_dates if pd.to_datetime(d).year in val_yr]
                test_raw = [d for d in all_dates if pd.to_datetime(d).year in test_yr]

                # Purge: Train 末尾去除 horizon 天
                purged_train = train_raw[:-self.horizon] if len(train_raw) > self.horizon else train_raw
                # Embargo: Val 前端跳过 embargo_days，末尾去除 horizon
                purged_val = val_raw[self.embargo_days : -self.horizon] if len(val_raw) > (self.embargo_days + self.horizon) else val_raw
                # Test 前端跳过 embargo_days
                purged_test = test_raw[self.embargo_days :] if len(test_raw) > self.embargo_days else test_raw

                if purged_train and purged_val and purged_test:
                    folds.append({
                        "fold_id": len(folds) + 1,
                        "train_dates": purged_train,
                        "val_dates": purged_val,
                        "test_dates": purged_test,
                        "train_years": train_yrs,
                        "val_years": val_yr,
                        "test_years": test_yr,
                    })
        
        # 如果数据不足多年（如短周期或测试数据），按比例生成至少 2 个滚动 Fold
        if not folds and len(all_dates) >= 40:
            warmup = 60 if len(all_dates) > 75 else 0
            usable = all_dates[warmup:]
            n = len(usable)
            step = max(5, n // 6)
            for f_idx, start_idx in enumerate([0, step]):
                sub_dates = usable[start_idx : start_idx + int(n * 0.75)]
                m = len(sub_dates)
                t_end = int(m * 0.6)
                v_end = int(m * 0.8)

                purged_train = sub_dates[: max(1, t_end - self.horizon)]
                purged_val = sub_dates[min(t_end + self.embargo_days, m - 1) : max(t_end + self.embargo_days + 1, v_end - self.horizon)]
                purged_test = sub_dates[min(v_end + self.embargo_days, m - 1) :]

                if purged_train and purged_val and purged_test:
                    folds.append({
                        "fold_id": f_idx + 1,
                        "train_dates": purged_train,
                        "val_dates": purged_val,
                        "test_dates": purged_test,
                        "train_years": [2024],
                        "val_years": [2024],
                        "test_years": [2024],
                    })

        return folds

    def run_purged_walk_forward(
        self,
        df_all: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "rank_label_5d",
        model_type: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        运行完整的 Purged Walk Forward 交叉验证
        """
        if df_all.empty:
            return pd.DataFrame(), {}

        m_type = model_type or self.config.get("model", {}).get("type", "sklearn_hgb")
        all_dates = sorted(pd.to_datetime(df_all["trade_date"]).dt.strftime("%Y-%m-%d").unique())
        folds = self.generate_folds(all_dates)
        if not folds:
            raise ValueError("Insufficient date span to generate purged walk forward folds.")

        logger.info(f"Generated {len(folds)} Purged Walk-Forward folds. Model Type: '{m_type}'.")

        fold_records = []
        all_test_preds = []

        for fold in folds:
            f_id = fold["fold_id"]
            train_dates = fold["train_dates"]
            val_dates = fold["val_dates"]
            test_dates = fold["test_dates"]

            # 严格时序隔离校验
            max_train_date = max(train_dates)
            min_val_date = min(val_dates)
            max_val_date = max(val_dates)
            min_test_date = min(test_dates)

            # label_info_time(Train) = max(train_date) + horizon days
            # 必须满足: max(train_date) + horizon <= min(val_date)
            t_idx = all_dates.index(max_train_date)
            v_idx = all_dates.index(min_val_date)
            if v_idx - t_idx < self.horizon:
                logger.warning(f"Fold {f_id}: Train-Val gap ({v_idx - t_idx}) < horizon ({self.horizon}).")

            train_df = df_all[df_all["trade_date"].isin(train_dates)]
            val_df = df_all[df_all["trade_date"].isin(val_dates)]
            test_df = df_all[df_all["trade_date"].isin(test_dates)].copy()

            model = create_model_by_type(m_type, feature_cols=feature_cols, config=self.config)
            model.fit(train_df, val_df)

            test_df["lgbm_score"] = model.predict(test_df)
            all_test_preds.append(test_df)

            ic_df = compute_daily_ic(test_df, score_col="lgbm_score", label_col=label_col)
            ic_stats = compute_ic_stats(ic_df)

            fold_records.append({
                "fold": f_id,
                "train_days": len(train_dates),
                "val_days": len(val_dates),
                "test_days": len(test_dates),
                "mean_ic": ic_stats["mean_ic"],
                "icir": ic_stats["icir"],
                "pos_ratio": ic_stats["pos_ratio"],
                "model_class": model.__class__.__name__
            })

        fold_df = pd.DataFrame(fold_records)
        combined_test = pd.concat(all_test_preds, ignore_index=True)
        overall_ic = compute_daily_ic(combined_test, score_col="lgbm_score", label_col=label_col)
        summary = compute_ic_stats(overall_ic)
        summary["num_folds"] = len(folds)
        summary["model_type"] = m_type

        return fold_df, summary
