"""
Walk-Forward Model Evaluation Module.
Aliases and delegates to the production PurgedWalkForwardEvaluator.
"""
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from ashare_quant.validation.purged_walk_forward import PurgedWalkForwardEvaluator
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.models.walk_forward")

class WalkForwardEvaluator:
    """
    Standard Walk-Forward Evaluator wrapping PurgedWalkForwardEvaluator.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.evaluator = PurgedWalkForwardEvaluator(config=config)

    def run_walk_forward(
        self,
        df_all: pd.DataFrame,
        label_col: str = "rank_label_5d"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        feature_cols = [
            c for c in df_all.columns
            if c not in ["ts_code", "trade_date", "open", "high", "low", "close", "volume", "amount", "turn", "pct_chg", "is_suspended", "open_raw", "close_raw", "open_adj", "close_adj", "high_adj", "low_adj", label_col, "year"]
        ]
        fold_df, summary = self.evaluator.run_purged_walk_forward(
            df_all=df_all,
            feature_cols=feature_cols,
            label_col=label_col,
            model_type="sklearn_hgb"
        )
        # Adapt keys for backward compatibility
        summary["avg_fold_ic"] = summary.get("mean_ic", 0.0)
        summary["avg_fold_icir"] = summary.get("icir", 0.0)
        return fold_df, summary
