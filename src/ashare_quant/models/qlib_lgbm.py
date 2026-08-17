"""
Official Microsoft Qlib LGBModel Adapter.
Directly wraps and uses qlib.contrib.model.gbdt.LGBModel and qlib.data.dataset.DatasetH.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

import qlib
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.models.qlib_lgbm")

class OfficialQlibLGBMModel:
    """
    官方 Qlib LGBModel 适配器
    直接调用 microsoft/qlib 的 qlib.contrib.model.gbdt.LGBModel
    配合 Qlib DatasetH 进行标准化的训练、验证与测试
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        self.config = config or load_config("model_lgbm")
        self.model_params = self.config.get("model", {})
        
        loss = "mse" if self.model_params.get("objective") in ["mse", "regression", None] else "binary"
        self.qlib_kwargs = {
            "loss": loss,
            "learning_rate": self.model_params.get("learning_rate", 0.03),
            "num_leaves": self.model_params.get("num_leaves", 31),
            "max_depth": self.model_params.get("max_depth", 5),
            "subsample": self.model_params.get("subsample", 0.8),
            "colsample_bytree": self.model_params.get("colsample_bytree", 0.8),
            "random_state": self.model_params.get("random_state", 42),
            "num_boost_round": self.model_params.get("n_estimators", 300),
            "early_stopping_rounds": 30,
            "verbosity": -1,
        }
        self.qlib_kwargs.update(kwargs)
        
        logger.info("Initializing official qlib.contrib.model.gbdt.LGBModel...")
        self.model = LGBModel(**self.qlib_kwargs)
        self.is_fitted = False

    def fit(self, dataset: DatasetH) -> "OfficialQlibLGBMModel":
        """
        使用 Qlib 官方 DatasetH 训练 Qlib LGBModel
        """
        logger.info(f"Fitting official Qlib LGBModel on DatasetH (segments: {list(dataset.segments.keys()) if hasattr(dataset, 'segments') else 'unknown'})...")
        self.model.fit(dataset)
        self.is_fitted = True
        logger.info("Official Qlib LGBModel fitted successfully.")
        return self

    def predict(self, dataset: DatasetH, segment: str = "test") -> pd.Series:
        """
        使用 Qlib LGBModel 对 DatasetH 指定 segment 进行打分预测
        """
        if not self.is_fitted:
            raise RuntimeError("OfficialQlibLGBMModel has not been fitted yet.")
        logger.info(f"Predicting with official Qlib LGBModel on segment '{segment}'...")
        preds = self.model.predict(dataset, segment=segment)
        if isinstance(preds, np.ndarray):
            index = dataset.prepare(segment=segment, col_set="label").index
            return pd.Series(preds.flatten(), index=index)
        return preds

    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取 Qlib LGBModel 内部 Booster 的特征重要性
        """
        if not self.is_fitted or not hasattr(self.model, "model") or self.model.model is None:
            return pd.DataFrame()
        booster = self.model.model
        importance_gain = booster.feature_importance(importance_type="gain")
        importance_split = booster.feature_importance(importance_type="split")
        feature_names = booster.feature_name()
        return pd.DataFrame({
            "feature": feature_names,
            "importance_gain": importance_gain,
            "importance_split": importance_split
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

# Alias for backward compatibility
QlibLGBMModelAdapter = OfficialQlibLGBMModel

