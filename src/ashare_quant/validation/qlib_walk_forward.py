"""
Official Qlib Walk-Forward Cross-Validation Engine.
Executes multi-fold purged rolling evaluation using Qlib Alpha158 Handler, DatasetH, and LGBModel.
Zero mock data, zero leakage.

Walk-forward parameters (train_years / val_years / test_years / embargo_days) are read
from cfg["walk_forward"] unless explicitly overridden by the caller — the CLI, the
evaluator and the experiment metadata must use identical values.
Production NEVER silently generates reduced folds for short data: insufficient history
raises InsufficientWalkForwardHistoryError.
"""
import os
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel

from ashare_quant.data.symbols import from_qlib_symbol
from ashare_quant.models.metrics import compute_daily_ic, compute_ic_stats
from ashare_quant.models.qlib_lgbm import OfficialQlibLGBMModel
from ashare_quant.validation.purged_walk_forward import (
    LeakageBoundaryError,
    InsufficientWalkForwardHistoryError,
    resolve_walk_forward_config,
)
from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.validation.qlib_walk_forward")

class QlibWalkForwardEvaluator:
    """
    基于 Microsoft Qlib 原生 DatasetH 与 LGBModel 的 Purged Walk-Forward 交叉验证器
    """
    def __init__(
        self,
        horizon: int = 5,
        embargo_days: Optional[int] = None,
        train_years: Optional[int] = None,
        val_years: Optional[int] = None,
        test_years: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or load_config("model_lgbm")
        resolved = resolve_walk_forward_config(
            self.config, train_years, val_years, test_years, embargo_days
        )
        self.horizon = horizon
        self.embargo_days = resolved["embargo_days"]
        self.train_years = resolved["train_years"]
        self.val_years = resolved["val_years"]
        self.test_years = resolved["test_years"]

    def walk_forward_config(self) -> Dict[str, Any]:
        """
        返回当前 evaluator 实际使用的 walk-forward 参数（CLI / metadata 使用完全一致的值）
        """
        return {
            "train_years": self.train_years,
            "val_years": self.val_years,
            "test_years": self.test_years,
            "embargo_days": self.embargo_days,
            "horizon": self.horizon,
        }

    def generate_qlib_folds(self, calendar_dates: List[str]) -> List[Dict[str, Any]]:
        """
        生成严格的 Multi-Fold Walk-Forward 时间切片。

        正式研究数据不足时直接抛出 InsufficientWalkForwardHistoryError，
        绝不自动生成缩小版 Fold 改变实验定义。
        """
        all_dt = pd.to_datetime(calendar_dates)
        years = sorted(all_dt.year.unique())
        folds = []

        total_span = self.train_years + self.val_years + self.test_years
        if len(years) >= total_span:
            for i in range(len(years) - total_span + 1):
                train_yrs = years[i : i + self.train_years]
                val_yr = years[i + self.train_years : i + self.train_years + self.val_years]
                test_yr = years[i + self.train_years + self.val_years : i + total_span]

                train_raw = [d for d in calendar_dates if pd.to_datetime(d).year in train_yrs]
                val_raw = [d for d in calendar_dates if pd.to_datetime(d).year in val_yr]
                test_raw = [d for d in calendar_dates if pd.to_datetime(d).year in test_yr]

                # Purge: Train 末尾去除 horizon 天
                purged_train = train_raw[:-self.horizon] if len(train_raw) > self.horizon else train_raw
                # Embargo: Val 前端跳过 embargo_days，末尾去除 horizon
                purged_val = val_raw[self.embargo_days : -self.horizon] if len(val_raw) > (self.embargo_days + self.horizon) else val_raw
                # Test 前端跳过 embargo_days
                purged_test = test_raw[self.embargo_days :] if len(test_raw) > self.embargo_days else test_raw

                if purged_train and purged_val and purged_test:
                    folds.append({
                        "fold_id": len(folds) + 1,
                        "train_dates": (purged_train[0], purged_train[-1]),
                        "val_dates": (purged_val[0], purged_val[-1]),
                        "test_dates": (purged_test[0], purged_test[-1]),
                        "train_raw_dates": purged_train,
                        "val_raw_dates": purged_val,
                        "test_raw_dates": purged_test,
                    })

        if not folds:
            raise InsufficientWalkForwardHistoryError(
                f"Insufficient history for Qlib Walk-Forward: calendar has {len(years)} year(s) "
                f"({years[0] if years else 'n/a'}..{years[-1] if years else 'n/a'}, {len(calendar_dates)} dates), "
                f"but the configured experiment requires train_years={self.train_years} + "
                f"val_years={self.val_years} + test_years={self.test_years} "
                f"(= {total_span} consecutive years). Refusing to degrade the experiment definition."
            )

        return folds

    def run_qlib_walk_forward(
        self,
        handler: Alpha158,
        calendar_dates: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
        """
        在 Qlib Alpha158 Handler 上运行多折 Purged Walk-Forward 交叉验证
        返回: (fold_metrics_df, summary_dict, oos_predictions_df)
        """
        # 设置 MLFlow 环境变量以避免旧版文件存储报错
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

        folds = self.generate_qlib_folds(calendar_dates)

        logger.info(f"Generated {len(folds)} Qlib Purged Walk-Forward folds.")
        logger.info(f"Walk-Forward config: {self.walk_forward_config()}")

        fold_records = []
        all_oos_preds = []

        for fold in folds:
            f_id = fold["fold_id"]
            train_start, train_end = fold["train_dates"]
            val_start, val_end = fold["val_dates"]
            test_start, test_end = fold["test_dates"]

            # 验证 Leakage 隔离边界
            t_end_idx = calendar_dates.index(train_end)
            v_start_idx = calendar_dates.index(val_start)
            v_end_idx = calendar_dates.index(val_end)
            test_start_idx = calendar_dates.index(test_start)

            if v_start_idx - t_end_idx < self.horizon:
                raise LeakageBoundaryError(
                    f"Fold {f_id} Train-Val gap ({v_start_idx - t_end_idx}) < horizon ({self.horizon})"
                )
            if test_start_idx - v_end_idx < self.horizon:
                raise LeakageBoundaryError(
                    f"Fold {f_id} Val-Test gap ({test_start_idx - v_end_idx}) < horizon ({self.horizon})"
                )

            logger.info(
                f"Fold {f_id}: Train [{train_start} -> {train_end}], "
                f"Val [{val_start} -> {val_end}], Test [{test_start} -> {test_end}]"
            )

            # 构建当前 Fold 的 DatasetH
            dataset = DatasetH(
                handler=handler,
                segments={
                    "train": (train_start, train_end),
                    "valid": (val_start, val_end),
                    "test": (test_start, test_end),
                }
            )

            # 训练 Qlib LGBModel
            model_adapter = OfficialQlibLGBMModel(config=self.config)
            model_adapter.fit(dataset)

            # 预测 Test 划分
            test_preds = model_adapter.predict(dataset, segment="test")
            test_labels = dataset.prepare("test", col_set="label")


            # 统一提取成 DataFrame
            if isinstance(test_preds, pd.Series):
                pred_df = test_preds.to_frame(name="score").reset_index()
            else:
                pred_df = pd.DataFrame(test_preds, columns=["score"]).reset_index()

            # 提取标签
            if isinstance(test_labels, pd.DataFrame):
                lbl_series = test_labels.iloc[:, 0].values
            elif isinstance(test_labels, pd.Series):
                lbl_series = test_labels.values
            else:
                lbl_series = np.zeros(len(pred_df))

            pred_df["label"] = lbl_series
            pred_df["fold_id"] = f_id
            pred_df["train_end_date"] = train_end

            # 映射列名: datetime -> trade_date, instrument -> ts_code
            pred_df["trade_date"] = pd.to_datetime(pred_df["datetime"]).dt.strftime("%Y-%m-%d")
            pred_df["ts_code"] = pred_df["instrument"].astype(str).apply(from_qlib_symbol)

            # 严格验证 OOS 日期晚于训练截止日期
            test_dates = set(pred_df["trade_date"])
            train_dates = set(fold["train_raw_dates"])
            overlap = test_dates.intersection(train_dates)
            if overlap:
                raise LeakageBoundaryError(f"Fold {f_id} test dates overlap with train dates: {overlap}")

            fold_oos = pred_df[["trade_date", "ts_code", "score", "label", "fold_id", "train_end_date"]].copy()
            all_oos_preds.append(fold_oos)

            # 计算 IC
            ic_df = compute_daily_ic(fold_oos, score_col="score", label_col="label")
            ic_stats = compute_ic_stats(ic_df)

            fold_records.append({
                "fold_id": f_id,
                "train_days": len(fold["train_raw_dates"]),
                "val_days": len(fold["val_raw_dates"]),
                "test_days": len(fold["test_raw_dates"]),
                "train_end_date": train_end,
                "test_start_date": test_start,
                "test_end_date": test_end,
                "mean_ic": ic_stats["mean_ic"],
                "icir": ic_stats["icir"],
                "pos_ratio": ic_stats["pos_ratio"],
                "model_class": "OfficialQlibLGBMModel"
            })

        fold_df = pd.DataFrame(fold_records)
        oos_predictions_df = pd.concat(all_oos_preds, ignore_index=True)

        overall_ic = compute_daily_ic(oos_predictions_df, score_col="score", label_col="label")
        summary = compute_ic_stats(overall_ic)
        summary["num_folds"] = len(folds)
        summary["model_type"] = "qlib_lgbm"

        return fold_df, summary, oos_predictions_df
