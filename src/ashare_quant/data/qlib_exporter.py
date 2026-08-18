"""
Qlib Data Exporter and Initialization Provider.
Manages qlib.init() configuration, data validation, and standard CSV preparation for Qlib official DumpData.
Zero fake binary mocks, zero fake benchmark constants.
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import qlib
from qlib.constant import REG_CN
from ashare_quant.data.symbols import to_qlib_symbol
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.data.qlib_exporter")

class QlibDataNotReadyError(FileNotFoundError):
    """Raised when Qlib provider directory is missing or incomplete."""
    pass

class QlibDataProviderManager:
    """
    Microsoft Qlib 数据环境与初始化管理器
    严格检查真实 Qlib 数据目录完整性，绝不构造假伪数据
    """
    _initialized = False
    _initialized_provider_uri: Optional[str] = None

    @classmethod
    def get_initialized_provider_uri(cls) -> Optional[str]:
        """
        返回当前已成功初始化的 Qlib provider URI。
        如果尚未初始化，返回 None。
        """
        return cls._initialized_provider_uri

    @classmethod
    def check_provider_ready(cls, provider_uri: str, mode: str = "csi300") -> None:
        """
        验证 provider_uri 是否具备合法的 Qlib 数据结构。

        基础验证（所有模式）:
          - calendars/day.txt 存在且非空
          - instruments/all.txt 存在且非空
          - features/ 目录存在且包含真实数据

        CSI300 模式额外验证:
          - instruments/csi300.txt 存在且非空
          - SH000300 benchmark 数据存在（features/sh000300 目录）
          缺任意核心数据直接 FAIL。
        """
        p = Path(provider_uri).expanduser()
        if not p.exists():
            raise QlibDataNotReadyError(
                f"Qlib data directory not found at '{provider_uri}'.\n"
                f"Please download official Qlib CN data via:\n"
                f"  python -c \"from qlib.tests.data import GetData; GetData().qlib_data(target_dir='~/.qlib/qlib_data/cn_data', region='cn')\""
            )

        cal_file = p / "calendars" / "day.txt"
        if not cal_file.exists() or cal_file.stat().st_size == 0:
            raise QlibDataNotReadyError(
                f"Qlib calendar file missing or empty at '{cal_file}'."
            )

        inst_file = p / "instruments" / "all.txt"
        if not inst_file.exists() or inst_file.stat().st_size == 0:
            raise QlibDataNotReadyError(
                f"Qlib instrument file missing or empty at '{inst_file}'."
            )

        feat_dir = p / "features"
        if not feat_dir.exists() or not any(feat_dir.iterdir()):
            raise QlibDataNotReadyError(
                f"Qlib features directory missing or empty at '{feat_dir}'."
            )

        # CSI300 mode: additional strict checks
        if mode == "csi300":
            csi300_file = p / "instruments" / "csi300.txt"
            if not csi300_file.exists() or csi300_file.stat().st_size == 0:
                raise QlibDataNotReadyError(
                    f"CSI300 instrument file missing or empty at '{csi300_file}'. "
                    f"Alpha158 CSI300 mode requires this file."
                )

            benchmark_dir = p / "features" / "sh000300"
            if not benchmark_dir.exists() or not any(benchmark_dir.iterdir()):
                raise QlibDataNotReadyError(
                    f"SH000300 benchmark data directory missing or empty at '{benchmark_dir}'. "
                    f"Benchmark data is required for backtesting."
                )

    @classmethod
    def init_qlib(
        cls,
        provider_uri: Optional[str] = None,
        region: str = REG_CN,
        force: bool = False,
        mode: str = "csi300"
    ) -> str:
        """
        初始化 Microsoft Qlib 数据引擎

        成功初始化后保存真正的 provider URI 到 _initialized_provider_uri。
        如果已经 initialized 且不强制，返回真实已初始化的 provider URI，
        不会自己猜测 ~/.qlib/...。
        """
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

        if cls._initialized and not force:
            if cls._initialized_provider_uri is not None:
                return cls._initialized_provider_uri

        if provider_uri is None:
            # 优先查找本地已存在的官方 Qlib 数据目录
            candidates = [
                Path("~/.qlib/qlib_data/cn_data").expanduser(),
                Path("data/qlib_cn").resolve(),
            ]
            found = None
            for cand in candidates:
                if (cand / "calendars" / "day.txt").exists() and (cand / "instruments" / "all.txt").exists() and (cand / "features").exists():
                    found = str(cand)
                    break
            if found:
                provider_uri = found
            else:
                provider_uri = str(Path("~/.qlib/qlib_data/cn_data").expanduser())

        cls.check_provider_ready(provider_uri, mode=mode)

        logger.info(f"Initializing official Qlib with provider_uri='{provider_uri}', region='{region}', mode='{mode}'...")
        try:
            qlib.init(provider_uri=provider_uri, region=region)
            cls._initialized = True
            cls._initialized_provider_uri = provider_uri
            logger.info("Qlib initialized successfully.")
            return provider_uri
        except Exception as e:
            logger.error(f"Failed to initialize Qlib: {e}")
            raise

    @classmethod
    def export_to_csv_dump_format(cls, df: pd.DataFrame, target_dir: str = "data/qlib_csv") -> str:
        """
        P0-7: 将 A 股日线数据导出为符合 Qlib 官方 dump_bin.py / DumpDataAll 标准格式的 CSV 文件：
        symbol, date, open, high, low, close, volume, factor, change
        降级为 CSV 数据准备适配器，禁止自行二进制序列化
        """
        out_path = Path(target_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        for ts_code, group in df.groupby("ts_code"):
            q_sym = to_qlib_symbol(ts_code)
            sub = group.sort_values("trade_date").copy()

            sub_csv = pd.DataFrame({
                "symbol": q_sym,
                "date": pd.to_datetime(sub["trade_date"]).dt.strftime("%Y-%m-%d"),
                "open": sub["open_raw"] if "open_raw" in sub.columns else sub["open"],
                "high": sub["high_raw"] if "high_raw" in sub.columns else sub["high"],
                "low": sub["low_raw"] if "low_raw" in sub.columns else sub["low"],
                "close": sub["close_raw"] if "close_raw" in sub.columns else sub["close"],
                "volume": sub["volume"],
                "factor": sub["adj_factor"] if "adj_factor" in sub.columns else 1.0,
                "change": sub["pct_chg"] if "pct_chg" in sub.columns else 0.0,
            })

            file_path = out_path / f"{q_sym}.csv"
            sub_csv.to_csv(file_path, index=False)

        logger.info(f"Exported {len(df['ts_code'].unique())} stock CSVs to '{out_path}' for official Qlib dump_bin.")
        return str(out_path)
