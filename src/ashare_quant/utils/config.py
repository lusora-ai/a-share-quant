import os
from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

def load_config(config_name: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> Dict[str, Any]:
    """
    加载指定的 YAML 配置文件。若传入不带 .yaml 后缀的名字，会自动补全。
    """
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"
        
    config_path = Path(config_dir) / config_name
    
    if not config_path.exists():
        # Fallback to local cwd configs/ directory if not found in package root
        fallback_path = Path.cwd() / "configs" / config_name
        if fallback_path.exists():
            config_path = fallback_path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path} or {fallback_path}")
            
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
        
    return config

def load_all_configs(config_dir: Path = DEFAULT_CONFIG_DIR) -> Dict[str, Dict[str, Any]]:
    """
    一次性加载 configs/ 下的全部配置文件
    """
    configs = {}
    config_files = ["base", "data", "factors", "model_lgbm", "backtest"]
    for name in config_files:
        try:
            configs[name] = load_config(name, config_dir)
        except Exception:
            configs[name] = {}
    return configs
