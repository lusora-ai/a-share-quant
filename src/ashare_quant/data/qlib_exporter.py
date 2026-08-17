"""
Qlib Data Exporter and Initialization Provider.
Manages qlib.init() configuration and data provider paths for China A-share market.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import qlib
from qlib.constant import REG_CN
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.data.qlib_exporter")

class QlibDataProviderManager:
    """
    Microsoft Qlib 数据环境与初始化管理器
    负责调用官方 qlib.init()，配置中国市场 provider_uri 与 region
    """
    _initialized = False

    @classmethod
    def ensure_default_calendar(cls, provider_uri: str) -> None:
        """
        确保 Qlib 数据目录中具有标准日历 calendars/day.txt
        """
        cal_dir = Path(provider_uri) / "calendars"
        cal_dir.mkdir(parents=True, exist_ok=True)
        day_file = cal_dir / "day.txt"
        if not day_file.exists():
            logger.info(f"Generating default Qlib trading calendar at {day_file}...")
            dates = pd.date_range("2018-01-01", "2026-12-31", freq="B").strftime("%Y-%m-%d")
            with open(day_file, "w", encoding="utf-8") as f:
                for d in dates:
                    f.write(f"{d}\n")

        inst_dir = Path(provider_uri) / "instruments"
        inst_dir.mkdir(parents=True, exist_ok=True)
        all_file = inst_dir / "all.txt"
        if not all_file.exists():
            with open(all_file, "w", encoding="utf-8") as f:
                f.write("SH600000\t2018-01-01\t2026-12-31\n")
                f.write("SZ000001\t2018-01-01\t2026-12-31\n")

    @classmethod
    def init_qlib(
        cls,
        provider_uri: Optional[str] = None,
        region: str = REG_CN,
        force: bool = False
    ) -> None:
        """
        初始化 Microsoft Qlib 数据引擎
        """
        if cls._initialized and not force:
            return

        if provider_uri is None:
            default_path = Path("~/.qlib/qlib_data/cn_data").expanduser()
            if default_path.exists():
                provider_uri = str(default_path)
            else:
                provider_uri = str(Path("data/qlib_cn").resolve())
                os.makedirs(provider_uri, exist_ok=True)

        cls.ensure_default_calendar(provider_uri)

        logger.info(f"Initializing official Qlib with provider_uri='{provider_uri}', region='{region}'...")
        try:
            qlib.init(provider_uri=provider_uri, region=region)
            cls._initialized = True
            logger.info("Qlib initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Qlib: {e}")
            raise
