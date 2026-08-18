import os
import json
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.models.experiment")

def get_git_commit_sha() -> str:
    """获取当前 Git Commit SHA，若非 git 环境则返回 'unknown'"""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"

class ExperimentTracker:
    """
    实验元数据与结果跟踪记录器
    保障 100% 可复现性，规范归档每次模型训练、评估、特征重要性及配置参数
    """
    def __init__(self, base_exp_dir: str = "experiments"):
        self.base_dir = Path(base_exp_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        feature_set: str = "custom12",
        model_type: str = "native_lgbm",
        feature_cols: Optional[List[str]] = None,
        label_spec: Optional[Dict[str, Any]] = None,
        train_end_date: Optional[str] = None,
        data_snapshot_id: Optional[str] = None,
        provider_uri: Optional[str] = None
    ) -> str:
        """
        创建一个唯一 experiment_id 并创建归档文件夹，记录完整元数据
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{timestamp_str}_{name}"
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存 config.yaml
        with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)
            
        # 2. 保存 metadata.json (P0-10: 包含全部核心追溯字段)
        meta = {
            "experiment_id": exp_id,
            "created_at": datetime.now().isoformat(),
            "name": name,
            "feature_set": feature_set,
            "model_type": model_type,
            "feature_cols": feature_cols or [],
            "label_spec": label_spec or {},
            "train_end_date": train_end_date or "",
            "data_snapshot_id": data_snapshot_id or f"snapshot_{timestamp_str}",
            "provider_uri": provider_uri or "",
            "git_commit_sha": get_git_commit_sha(),
            "random_seed": config.get("random_seed", 42),
            "python_version": os.sys.version
        }
        with open(exp_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Created new experiment archive directory: {exp_dir}")
        return exp_id

    def log_results(
        self,
        exp_id: str,
        metrics: Dict[str, Any],
        feature_importance: Optional[pd.DataFrame] = None,
        oos_predictions: Optional[pd.DataFrame] = None,
        fold_metrics: Optional[pd.DataFrame] = None,
        predictions: Optional[pd.DataFrame] = None, # alias for backward compatibility
        notes: str = ""
    ):
        """
        写回实验结果 (metrics.json, fold_metrics.parquet, oos_predictions.parquet, feature_importance.csv, notes.md)
        """
        exp_dir = self.base_dir / exp_id
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory {exp_dir} does not exist.")
            
        # 1. 写回 metrics.json
        with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
            
        # 2. 写回 fold_metrics.parquet
        if fold_metrics is not None and not fold_metrics.empty:
            fold_metrics.to_parquet(exp_dir / "fold_metrics.parquet")

        # 3. 写回 oos_predictions.parquet (P0-5: 唯一允许进入回测的预测集)
        preds_to_save = oos_predictions if oos_predictions is not None else predictions
        if preds_to_save is not None and not preds_to_save.empty:
            preds_to_save.to_parquet(exp_dir / "oos_predictions.parquet")
            # 保留 predictions.parquet 兼容性软链/拷贝
            preds_to_save.to_parquet(exp_dir / "predictions.parquet")
            
        # 4. 写回 feature_importance.csv
        if feature_importance is not None and not feature_importance.empty:
            feature_importance.to_csv(exp_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
            
        # 5. 写回 notes.md
        with open(exp_dir / "notes.md", "w", encoding="utf-8") as f:
            f.write(f"# Experiment Notes: {exp_id}\n\n{notes}\n")
            
        logger.info(f"Successfully logged metrics & artifacts for experiment '{exp_id}'.")

