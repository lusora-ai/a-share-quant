import os
import pytest
from pathlib import Path
from ashare_quant.utils.config import load_config, load_all_configs

def test_load_base_config():
    config = load_config("base")
    assert isinstance(config, dict)
    assert config.get("project_name") == "ashare-quant"
    assert "paths" in config
    assert "logging" in config

def test_load_all_configs():
    configs = load_all_configs()
    assert "base" in configs
    assert "data" in configs
    assert "factors" in configs
    assert "model_lgbm" in configs
    assert "backtest" in configs

def test_load_missing_config():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config_xyz")
