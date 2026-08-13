import os
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.models.experiment")

class ExperimentTracker:
    """
    实验元数据与结果跟踪记录器
    保障 100% 可复现性，规范归档每次模型训练、评估、特征重要性及配置参数
    """
    def __init__(self, base_exp_dir: str = "experiments"):
        self.base_dir = Path(base_exp_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(self, name: str, config: Dict[str, Any]) -> str:
        """
        创建一个唯一 experiment_id 并创建归档文件夹
        例如: 20260813_001_momentum_lgbm
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{timestamp_str}_{name}"
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存 config.yaml
        with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)
            
        # 2. 保存 metadata.json
        meta = {
            "experiment_id": exp_id,
            "created_at": datetime.now().isoformat(),
            "name": name,
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
        predictions: Optional[pd.DataFrame] = None,
        notes: str = ""
    ):
        """
        写回实验结果 (metrics.json, feature_importance.csv, predictions.parquet, notes.md)
        """
        exp_dir = self.base_dir / exp_id
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory {exp_dir} does not exist.")
            
        # 1. 写回 metrics.json
        with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
            
        # 2. 写回 feature_importance.csv
        if feature_importance is not None and not feature_importance.empty:
            feature_importance.to_csv(exp_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
            
        # 3. 写回 predictions.parquet
        if predictions is not None and not predictions.empty:
            predictions.to_parquet(exp_dir / "predictions.parquet")
            
        # 4. 写回 notes.md
        with open(exp_dir / "notes.md", "w", encoding="utf-8") as f:
            f.write(f"# Experiment Notes: {exp_id}\n\n{notes}\n")
            
        logger.info(f"Successfully logged metrics & artifacts for experiment '{exp_id}'.")
