import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from scipy.stats import rankdata
from ashare_quant.factors.engine import FACTOR_NAMES
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.models.baseline")

class EqualWeightBaseline:
    """
    等权因子打分基线模型
    Simple average score of standardized baseline factors
    """
    def __init__(self, factor_cols: Optional[List[str]] = None):
        self.factor_cols = factor_cols or FACTOR_NAMES

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        计算截面等权打分
        """
        if df.empty:
            return pd.Series(dtype=float)
            
        cols = [c for c in self.factor_cols if c in df.columns]
        if not cols:
            logger.warning("No matching factor columns for EqualWeightBaseline.")
            return pd.Series(0.0, index=df.index)
            
        score = df[cols].mean(axis=1)
        return score

class RidgeBaseline:
    """
    Ridge 线性回归打分基线模型
    """
    def __init__(self, alpha: float = 1.0, factor_cols: Optional[List[str]] = None):
        self.alpha = alpha
        self.factor_cols = factor_cols or FACTOR_NAMES
        self.weights = None
        self.intercept = 0.0

    def fit(self, df: pd.DataFrame, label_col: str):
        """
        在训练集上拟合 Ridge 回归权重
        """
        valid = df.dropna(subset=self.factor_cols + [label_col])
        if valid.empty:
            logger.warning("No valid samples to fit RidgeBaseline.")
            return
            
        X = valid[self.factor_cols].values
        y = valid[label_col].values
        
        # Ridge Regression closed-form solution: (X^T X + alpha * I)^(-1) X^T y
        n_features = X.shape[1]
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean
        y_mean = y.mean()
        y_centered = y - y_mean
        
        A = X_centered.T @ X_centered + self.alpha * np.eye(n_features)
        b = X_centered.T @ y_centered
        
        self.weights = np.linalg.solve(A, b)
        self.intercept = y_mean - X_mean @ self.weights
        
        logger.info(f"Fitted RidgeBaseline model across {len(valid)} samples. Alpha: {self.alpha}")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.weights is None:
            logger.warning("RidgeBaseline not fitted yet. Returning zeros.")
            return pd.Series(0.0, index=df.index)
            
        cols = [c for c in self.factor_cols if c in df.columns]
        X = df[cols].fillna(0.0).values
        pred = X @ self.weights + self.intercept
        return pd.Series(pred, index=df.index)
