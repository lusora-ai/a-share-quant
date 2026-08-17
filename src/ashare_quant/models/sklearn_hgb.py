import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from typing import List, Dict, Any, Optional
from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.models.sklearn_hgb")

class SklearnHGBModel:
    """
    Scikit-Learn HistGradientBoostingRegressor 选股回归模型
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, feature_cols: Optional[List[str]] = None):
        self.config = config or load_config("model_lgbm")
        self.model_params = self.config.get("model", {})
        self.feature_cols = feature_cols or self.config.get("features", [])
        self.label_col = self.config.get("label", {}).get("name", "rank_label_5d")
        self.model: Optional[HistGradientBoostingRegressor] = None

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        cols = [c for c in self.feature_cols if c in train_df.columns]
        if not cols:
            raise ValueError("No matching feature columns found in training DataFrame.")
            
        valid_train = train_df.dropna(subset=cols + [self.label_col])
        if valid_train.empty:
            raise ValueError(f"Training DataFrame has no valid rows for features and label '{self.label_col}'.")
            
        X_train = valid_train[cols].values
        y_train = valid_train[self.label_col].values
        
        logger.info(f"Training Sklearn HGB Model on {len(X_train)} samples...")
        
        hgb = HistGradientBoostingRegressor(
            learning_rate=self.model_params.get("learning_rate", 0.03),
            max_iter=self.model_params.get("n_estimators", 300),
            max_leaf_nodes=self.model_params.get("num_leaves", 31),
            max_depth=self.model_params.get("max_depth", 5),
            random_state=self.model_params.get("random_state", 42)
        )
        hgb.fit(X_train, y_train)
        self.model = hgb
        return self.get_feature_importance(cols, X_train, y_train)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("SklearnHGBModel has not been trained yet.")
        if df.empty:
            return pd.Series(dtype=float)
        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].fillna(0.0).values
        preds = self.model.predict(X)
        return pd.Series(preds, index=df.index)

    def get_feature_importance(self, cols: Optional[List[str]] = None, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        features = cols or self.feature_cols
        if X is not None and y is not None and len(X) > 0:
            perm_imp = permutation_importance(self.model, X, y, n_repeats=3, random_state=42)
            gain = perm_imp.importances_mean
        else:
            gain = np.ones(len(features))
        return pd.DataFrame({"feature": features, "importance_gain": gain, "importance_split": gain}).sort_values("importance_gain", ascending=False).reset_index(drop=True)
