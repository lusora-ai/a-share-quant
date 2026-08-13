import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ashare_quant.models.qlib_lgbm import QlibLGBMModelAdapter
from ashare_quant.models.metrics import compute_daily_ic, compute_ic_stats
from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.validation.purged_walk_forward")

class PurgedWalkForwardEvaluator:
    """
    带有 Purge & Embargo 隔离期的 Purged Walk-Forward 滚动验证器
    严格隔离未来 Label 数据跨界泄漏:
    max(label_info_time(Train)) < min(feature_time(Valid))
    """
    def __init__(self, horizon: int = 5, embargo_days: int = 2, config: Optional[Dict[str, Any]] = None):
        self.horizon = horizon
        self.embargo_days = embargo_days
        self.config = config or load_config("model_lgbm").get("walk_forward", {})

    def run_purged_walk_forward(
        self,
        df_all: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "rank_label_5d"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        进行带有 Purge 剔除隔离的 Walk-Forward 交叉验证
        """
        if df_all.empty:
            return pd.DataFrame(), {}
            
        df = df_all.copy()
        dates = sorted(df["trade_date"].unique())
        
        # 扣除初始 60 天因子 warmup 积累期
        warmup_offset = 60 if len(dates) > 75 else 0
        usable_dates = dates[warmup_offset:]
        n = len(usable_dates)
        
        train_end_idx = int(n * 0.5)
        val_end_idx = int(n * 0.75)
        
        # ---  aplicar Purge 规则 ---
        # 1. Train 集合: 去掉 Train 末尾 5 天 (horizon)，防止 Train 标签跨界进入 Validation
        purged_train_dates = usable_dates[:max(0, train_end_idx - self.horizon)]
        
        # 2. Validation 集合: 从 train_end_idx + embargo_days 开始
        val_start_idx = min(train_end_idx + self.embargo_days, n - 1)
        purged_val_dates = usable_dates[val_start_idx:max(val_start_idx, val_end_idx - self.horizon)]
        
        # 3. Test 集合: 从 val_end_idx + embargo_days 开始
        test_start_idx = min(val_end_idx + self.embargo_days, n - 1)
        test_dates = usable_dates[test_start_idx:]
        
        logger.info(f"Purged Split generated | Train days: {len(purged_train_dates)} | Val days: {len(purged_val_dates)} | Test days: {len(test_dates)}")
        
        train_df = df[df["trade_date"].isin(purged_train_dates)]
        val_df = df[df["trade_date"].isin(purged_val_dates)]
        test_df = df[df["trade_date"].isin(test_dates)].copy()
        
        # 训练 Qlib 模型
        model = QlibLGBMModelAdapter(feature_cols=feature_cols)
        model.fit(train_df, val_df)
        
        # 测试集单次预测
        test_df["lgbm_score"] = model.predict(test_df)
        
        # 评估测试集 RankIC
        ic_df = compute_daily_ic(test_df, score_col="lgbm_score", label_col=label_col)
        ic_stats = compute_ic_stats(ic_df)
        
        fold_results = [{
            "fold": 1,
            "train_days": len(purged_train_dates),
            "val_days": len(purged_val_dates),
            "test_days": len(test_dates),
            "mean_ic": ic_stats["mean_ic"],
            "icir": ic_stats["icir"],
            "pos_ratio": ic_stats["pos_ratio"]
        }]
        
        fold_metrics_df = pd.DataFrame(fold_results)
        return fold_metrics_df, ic_stats
