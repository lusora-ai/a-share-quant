import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import lightgbm as lgb

from ashare_quant.factors.engine import FACTOR_NAMES
from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.models.lgbm_model")

class LGBMRankingModel:
    """
    LightGBM / Histogram GBDT 横截面选股回归模型
    包含 Windows C-DLL 兼容性自动降级保护 (LightGBM -> HistGradientBoostingRegressor)
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config("model_lgbm")
        self.model_params = self.config.get("model", {})
        self.feature_cols = self.config.get("features", FACTOR_NAMES)
        self.label_col = self.config.get("label", {}).get("name", "rank_label_5d")
        self.model = None
        self.use_hgb = False
        self.feature_importances_ = None

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        训练 GBDT 选股模型
        """
        valid_train = train_df.dropna(subset=self.feature_cols + [self.label_col])
        if valid_train.empty:
            raise ValueError("Training DataFrame has no valid rows for feature and label columns.")
            
        X_train = valid_train[self.feature_cols]
        y_train = valid_train[self.label_col]
        
        logger.info(f"Training GBDT Ranking model on {len(X_train)} samples across {len(self.feature_cols)} features...")
        
        X_train_arr = np.ascontiguousarray(X_train.values, dtype=np.float64)
        y_train_arr = np.ascontiguousarray(y_train.values, dtype=np.float64).ravel()
        
        # 尝试 LightGBM 原生接口，若在部分 Windows C-DLL 环境崩溃，则平滑降级至 HistGradientBoostingRegressor
        try:
            lgb_params = {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": self.model_params.get("learning_rate", 0.03),
                "num_leaves": self.model_params.get("num_leaves", 31),
                "max_depth": self.model_params.get("max_depth", 5),
                "subsample": self.model_params.get("subsample", 0.8),
                "colsample_bytree": self.model_params.get("colsample_bytree", 0.8),
                "random_state": self.model_params.get("random_state", 42),
                "verbose": -1,
                "n_jobs": -1
            }
            train_data = lgb.Dataset(X_train_arr, label=y_train_arr, feature_name=self.feature_cols, free_raw_data=False)
            num_boost_round = self.model_params.get("n_estimators", 300)
            
            self.model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=num_boost_round
            )
            self.use_hgb = False
            logger.info("LightGBM model training completed natively.")
        except BaseException as e:
            logger.warning(f"Native LightGBM failed ({e}). Falling back to sklearn HistGradientBoostingRegressor...")
            self.use_hgb = True
            hgb_model = HistGradientBoostingRegressor(
                learning_rate=self.model_params.get("learning_rate", 0.03),
                max_iter=self.model_params.get("n_estimators", 300),
                max_leaf_nodes=self.model_params.get("num_leaves", 31),
                max_depth=self.model_params.get("max_depth", 5),
                random_state=self.model_params.get("random_state", 42)
            )
            hgb_model.fit(X_train_arr, y_train_arr)
            self.model = hgb_model
            logger.info("HistGradientBoostingRegressor training completed successfully.")
            
        return self.get_feature_importance(X_train_arr, y_train_arr)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        打分预测
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
            
        if df.empty:
            return pd.Series(dtype=float)
            
        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].fillna(0.0)
        X_arr = np.ascontiguousarray(X.values, dtype=np.float64)
        preds = self.model.predict(X_arr)
        return pd.Series(preds, index=df.index)

    def get_feature_importance(self, X_arr: Optional[np.ndarray] = None, y_arr: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        获取特征重要性 (Gain 或 Permutation 重要性)
        """
        if self.model is None:
            return pd.DataFrame()
            
        if not self.use_hgb and hasattr(self.model, "feature_importance"):
            importance_gain = self.model.feature_importance(importance_type="gain")
            importance_split = self.model.feature_importance(importance_type="split")
        else:
            if X_arr is not None and y_arr is not None:
                # 基于 permutation_importance 计算
                perm_imp = permutation_importance(self.model, X_arr, y_arr, n_repeats=3, random_state=42)
                importance_gain = perm_imp.importances_mean
                importance_split = perm_imp.importances_mean
            else:
                importance_gain = np.ones(len(self.feature_cols))
                importance_split = np.ones(len(self.feature_cols))
                
        df_imp = pd.DataFrame({
            "feature": self.feature_cols,
            "importance_gain": importance_gain,
            "importance_split": importance_split
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
        
        return df_imp
