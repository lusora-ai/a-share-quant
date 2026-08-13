import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import lightgbm as lgb

from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.models.qlib_lgbm")

class QlibLGBMModelAdapter:
    """
    Qlib 格式 GBDT 选股适配模型
    支持显式模型注册类型:
    1. model_type = 'lgbm' / 'lightgbm': 尝试原生 LightGBM
       若 Windows 原生 C-DLL 崩溃则 FAIL LOUDLY 并明确提示切至 'sklearn_hgb' (严禁静默降级)
    2. model_type = 'sklearn_hgb': 使用 HistGradientBoostingRegressor 运行标准直方图 GBDT
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, feature_cols: Optional[List[str]] = None, model_type: Optional[str] = None):
        self.config = config or load_config("model_lgbm")
        self.model_params = self.config.get("model", {})
        self.model_type = (model_type or self.model_params.get("type", "sklearn_hgb")).lower()
        self.feature_cols = feature_cols or self.config.get("features", [])
        self.label_col = self.config.get("label", {}).get("name", "rank_label_5d")
        self.model = None
        self.is_hgb = False

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        训练模型 (若有 val_df 则开启 Validation/Early Stopping)
        """
        cols = [c for c in self.feature_cols if c in train_df.columns]
        if not cols:
            raise ValueError("No matching feature columns found in training DataFrame.")
            
        valid_train = train_df.dropna(subset=cols + [self.label_col])
        if valid_train.empty:
            raise ValueError(f"Training DataFrame has no valid rows for feature cols and label '{self.label_col}'.")
            
        X_train = valid_train[cols]
        y_train = valid_train[self.label_col]
        
        logger.info(f"Training GBDT Model (type='{self.model_type}') on {len(X_train)} samples across {len(cols)} features...")
        
        X_tr = np.ascontiguousarray(X_train.values, dtype=np.float64)
        y_tr = np.ascontiguousarray(y_train.values, dtype=np.float64).ravel()
        
        if self.model_type in ["sklearn_hgb", "hgb"]:
            self.is_hgb = True
            hgb = HistGradientBoostingRegressor(
                learning_rate=self.model_params.get("learning_rate", 0.03),
                max_iter=self.model_params.get("n_estimators", 300),
                max_leaf_nodes=self.model_params.get("num_leaves", 31),
                max_depth=self.model_params.get("max_depth", 5),
                random_state=self.model_params.get("random_state", 42)
            )
            hgb.fit(X_tr, y_tr)
            self.model = hgb
            logger.info("sklearn_hgb model trained successfully.")
        else:
            # 原生 LightGBM 模式
            self.is_hgb = False
            lgb_model = lgb.LGBMRegressor(
                learning_rate=self.model_params.get("learning_rate", 0.03),
                num_leaves=self.model_params.get("num_leaves", 31),
                max_depth=self.model_params.get("max_depth", 5),
                min_child_samples=1,
                subsample=self.model_params.get("subsample", 0.8),
                colsample_bytree=self.model_params.get("colsample_bytree", 0.8),
                random_state=self.model_params.get("random_state", 42),
                n_estimators=self.model_params.get("n_estimators", 300),
                verbosity=-1,
                n_jobs=-1
            )
            
            try:
                eval_set = None
                callbacks = None
                if val_df is not None and not val_df.empty:
                    valid_val = val_df.dropna(subset=cols + [self.label_col])
                    if len(valid_val) >= 10:
                        X_v = np.ascontiguousarray(valid_val[cols].values, dtype=np.float64)
                        y_v = np.ascontiguousarray(valid_val[self.label_col].values, dtype=np.float64).ravel()
                        eval_set = [(X_v, y_v)]
                        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
                        
                lgb_model.fit(X_tr, y_tr, eval_set=eval_set, callbacks=callbacks)
                self.model = lgb_model
                logger.info("LightGBM model trained successfully.")
            except BaseException as e:
                # 严禁静默降级! Fail Loudly 并给出显式指引
                logger.error(f"FATAL: Native LightGBM failed ({e}). Silent fallback is DISABLED.")
                raise RuntimeError(
                    f"LightGBM Training Failed: {e}. On Windows 64-bit environment, please configure model.type='sklearn_hgb' in configs/model_lgbm.yaml to use HistGradientBoostingRegressor."
                ) from e
                
        return self.get_feature_importance(cols, X_tr, y_tr)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
            
        if df.empty:
            return pd.Series(dtype=float)
            
        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].fillna(0.0)
        X_arr = np.ascontiguousarray(X.values, dtype=np.float64)
        preds = self.model.predict(X_arr)
        return pd.Series(preds, index=df.index)

    def get_feature_importance(self, cols: Optional[List[str]] = None, X_tr: Optional[np.ndarray] = None, y_tr: Optional[np.ndarray] = None) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
            
        features = cols or self.feature_cols
        if not self.is_hgb and hasattr(self.model, "booster_"):
            importance_gain = self.model.booster_.feature_importance(importance_type="gain")
            importance_split = self.model.booster_.feature_importance(importance_type="split")
        else:
            if X_tr is not None and y_tr is not None and len(X_tr) > 0:
                perm_imp = permutation_importance(self.model, X_tr, y_tr, n_repeats=3, random_state=42)
                importance_gain = perm_imp.importances_mean
                importance_split = perm_imp.importances_mean
            else:
                importance_gain = np.ones(len(features))
                importance_split = np.ones(len(features))
                
        df_imp = pd.DataFrame({
            "feature": features,
            "importance_gain": importance_gain,
            "importance_split": importance_split
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
        
        return df_imp
