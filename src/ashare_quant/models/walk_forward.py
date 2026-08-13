import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ashare_quant.models.lgbm_model import LGBMRankingModel
from ashare_quant.models.baseline import EqualWeightBaseline
from ashare_quant.models.metrics import compute_daily_ic, compute_ic_stats
from ashare_quant.backtest.engine import BacktestEngine
from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.models.walk_forward")

class WalkForwardEvaluator:
    """
    Walk-Forward 滚动窗口时间序列验证器
    严禁未来函数与测试集调参，按 Fold 独立保存评估指标
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config("model_lgbm").get("walk_forward", {})
        self.train_years = self.config.get("train_years", 3)
        self.val_years = self.config.get("val_years", 1)
        self.test_years = self.config.get("test_years", 1)
        self.start_year = self.config.get("start_year", 2018)
        self.end_year = self.config.get("end_year", 2025)

    def generate_folds(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        根据数据中的年份划分 Fold 滚动时间窗口
        """
        df["year"] = pd.to_datetime(df["trade_date"]).dt.year
        available_years = sorted(df["year"].unique())
        
        folds = []
        min_year = min(available_years)
        max_year = max(available_years)
        
        current_start = min_year
        fold_idx = 1
        
        while current_start + self.train_years + self.val_years + self.test_years - 1 <= max_year:
            train_end = current_start + self.train_years - 1
            val_start = train_end + 1
            val_end = val_start + self.val_years - 1
            test_start = val_end + 1
            test_end = test_start + self.test_years - 1
            
            folds.append({
                "fold": fold_idx,
                "train_years": list(range(current_start, train_end + 1)),
                "val_years": list(range(val_start, val_end + 1)),
                "test_years": list(range(test_start, test_end + 1)),
            })
            current_start += 1
            fold_idx += 1
            
        logger.info(f"Generated {len(folds)} Walk-Forward folds from {min_year} to {max_year}.")
        return folds

    def run_walk_forward(
        self,
        df_all: pd.DataFrame,
        label_col: str = "rank_label_5d"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        对所有 Fold 逐一执行 训练 -> 验证 -> 测试集单次评估 -> 回测
        """
        if df_all.empty:
            return pd.DataFrame(), {}
            
        df = df_all.copy()
        df["year"] = pd.to_datetime(df["trade_date"]).dt.year
        folds = self.generate_folds(df)
        
        if not folds:
            logger.warning("Data time span insufficient for multi-year Walk-Forward folds. Falling back to single split.")
            # 兼容样本天数较少的测试情境
            dates = sorted(df["trade_date"].unique())
            # 跳过初始 60 天因子 warmup 积累期
            warmup_offset = 60 if len(dates) > 75 else 0
            usable_dates = dates[warmup_offset:]
            n = len(usable_dates)
            
            train_dates = usable_dates[:int(n*0.5)]
            val_dates = usable_dates[int(n*0.5):int(n*0.75)]
            test_dates = usable_dates[int(n*0.75):]
            
            folds = [{
                "fold": 1,
                "train_dates": train_dates,
                "val_dates": val_dates,
                "test_dates": test_dates
            }]
            
        fold_results = []
        all_test_preds = []
        
        for f in folds:
            fold_num = f["fold"]
            logger.info(f"--- Running Walk-Forward Fold {fold_num} ---")
            
            if "train_years" in f:
                train_df = df[df["year"].isin(f["train_years"])]
                val_df = df[df["year"].isin(f["val_years"])]
                test_df = df[df["year"].isin(f["test_years"])].copy()
            else:
                train_df = df[df["trade_date"].isin(f["train_dates"])]
                val_df = df[df["trade_date"].isin(f["val_dates"])]
                test_df = df[df["trade_date"].isin(f["test_dates"])].copy()
                
            # 训练模型
            model = LGBMRankingModel()
            model.fit(train_df, val_df)
            
            # 测试集单次评估
            test_df["lgbm_score"] = model.predict(test_df)
            all_test_preds.append(test_df)
            
            # 计算 Test 集 IC 统计
            ic_df = compute_daily_ic(test_df, score_col="lgbm_score", label_col=label_col)
            ic_stats = compute_ic_stats(ic_df)
            
            # Test 集回测
            bt_engine = BacktestEngine()
            eq_df, bt_metrics = bt_engine.run_backtest(test_df, score_col="lgbm_score")
            
            fold_record = {
                "fold": fold_num,
                "train_period": f.get("train_years") or f"{f['train_dates'][0]}~{f['train_dates'][-1]}",
                "test_period": f.get("test_years") or f"{f['test_dates'][0]}~{f['test_dates'][-1]}",
                "mean_ic": ic_stats["mean_ic"],
                "icir": ic_stats["icir"],
                "cagr": bt_metrics.get("cagr", 0.0),
                "max_drawdown": bt_metrics.get("max_drawdown", 0.0),
                "win_rate": bt_metrics.get("win_rate", 0.0)
            }
            fold_results.append(fold_record)
            logger.info(f"Fold {fold_num} Results: Mean IC: {ic_stats['mean_ic']:.4f} | ICIR: {ic_stats['icir']:.2f} | CAGR: {bt_metrics.get('cagr', 0.0):.2%}")
            
        fold_metrics_df = pd.DataFrame(fold_results)
        full_preds_df = pd.concat(all_test_preds, ignore_index=True)
        
        summary_stats = {
            "avg_fold_ic": float(fold_metrics_df["mean_ic"].mean()),
            "avg_fold_icir": float(fold_metrics_df["icir"].mean()),
            "avg_fold_cagr": float(fold_metrics_df["cagr"].mean()),
            "avg_fold_max_drawdown": float(fold_metrics_df["max_drawdown"].mean())
        }
        
        return fold_metrics_df, summary_stats
