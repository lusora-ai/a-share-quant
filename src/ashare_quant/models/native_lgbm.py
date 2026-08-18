import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Dict, Any, Optional
from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.models.native_lgbm")

class NativeLGBMModel:
    """
    原生 LightGBM 选股回归模型
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, feature_cols: Optional[List[str]] = None):
        self.config = config or load_config("model_lgbm")
        self.model_params = self.config.get("model", {})
        self.feature_cols = feature_cols or self.config.get("features", [])
        self.label_col = self.config.get("label", {}).get("name", "rank_label_5d")
        self.model: Optional[lgb.LGBMRegressor] = None

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        cols = [c for c in self.feature_cols if c in train_df.columns]
        if not cols:
            raise ValueError("No matching feature columns found in training DataFrame.")
            
        valid_train = train_df.dropna(subset=cols + [self.label_col])
        if valid_train.empty:
            raise ValueError(f"Training DataFrame has no valid rows for features and label '{self.label_col}'.")
            
        X_train = np.ascontiguousarray(valid_train[cols].values, dtype=np.float32)
        y_train = np.ascontiguousarray(valid_train[self.label_col].values, dtype=np.float32)
        
        logger.info(f"Training Native LightGBM Model on {len(X_train)} samples...")
        
        lgb_model = lgb.LGBMRegressor(
            learning_rate=self.model_params.get("learning_rate", 0.03),
            num_leaves=self.model_params.get("num_leaves", 31),
            max_depth=self.model_params.get("max_depth", 5),
            subsample=self.model_params.get("subsample", 0.8),
            colsample_bytree=self.model_params.get("colsample_bytree", 0.8),
            random_state=self.model_params.get("random_state", 42),
            n_estimators=self.model_params.get("n_estimators", 100),
            verbosity=-1,
            n_jobs=-1
        )
        
        eval_set = None
        callbacks = None
        if val_df is not None and not val_df.empty:
            valid_val = val_df.dropna(subset=cols + [self.label_col])
            if len(valid_val) >= 10:
                X_val = np.ascontiguousarray(valid_val[cols].values, dtype=np.float32)
                y_val = np.ascontiguousarray(valid_val[self.label_col].values, dtype=np.float32)
                eval_set = [(X_val, y_val)]
                callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
                
        lgb_model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        self.model = lgb_model
        return self.get_feature_importance(cols)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("NativeLGBMModel has not been trained yet.")
        if df.empty:
            return pd.Series(dtype=float)
        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].fillna(0.0)
        preds = self.model.predict(X)
        return pd.Series(preds, index=df.index)

    def get_feature_importance(self, cols: Optional[List[str]] = None) -> pd.DataFrame:
        if self.model is None or not hasattr(self.model, "booster_"):
            return pd.DataFrame()
        features = cols or self.feature_cols
        gain = self.model.booster_.feature_importance(importance_type="gain")
        split = self.model.booster_.feature_importance(importance_type="split")
        return pd.DataFrame({"feature": features, "importance_gain": gain, "importance_split": split}).sort_values("importance_gain", ascending=False).reset_index(drop=True)
