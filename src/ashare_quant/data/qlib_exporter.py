"""
Qlib Data Exporter and Initialization Provider.
Manages qlib.init() configuration, data validation, and standard CSV preparation for Qlib official DumpData.
Zero fake binary mocks, zero fake benchmark constants.
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
import qlib
from qlib.constant import REG_CN
from ashare_quant.data.symbols import to_qlib_symbol
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.data.qlib_exporter")

class QlibDataNotReadyError(FileNotFoundError):
    """Raised when Qlib provider directory is missing or incomplete."""
    pass

class ProviderMismatchError(RuntimeError):
    """Raised when attempting to reinitialize Qlib with a different provider URI."""
    pass

class DataSchemaError(ValueError):
    """Raised when required adjusted price columns are missing for export."""
    pass

def get_expected_latest_completed_trade_date(reference_time: Optional[datetime] = None) -> str:
    """
    计算当前预期最新已完成结算的 A 股交易日 (YYYY-MM-DD)。
    如果在交易日 15:30 之前，则为上一个交易日；如果已过 15:30，则为当日。
    """
    now = reference_time or datetime.now()
    # A股 15:00 收盘，15:30 结算完毕
    is_after_market_close = (now.hour > 15) or (now.hour == 15 and now.minute >= 30)

    # 优先从 storage 的 trade_calendar 获取
    try:
        from ashare_quant.data.processor import DataProcessor
        proc = DataProcessor()
        try:
            df_cal = proc.storage.load_parquet("trade_calendar", is_processed=True)
        finally:
            proc.close()
        if df_cal is not None and not df_cal.empty and "trade_date" in df_cal.columns:
            today_str = now.strftime("%Y-%m-%d")
            if is_after_market_close:
                valid_dates = df_cal[df_cal["trade_date"] <= today_str]["trade_date"].tolist()
            else:
                valid_dates = df_cal[df_cal["trade_date"] < today_str]["trade_date"].tolist()
            if valid_dates:
                return str(sorted(valid_dates)[-1])
    except Exception:
        pass

    # Fallback: 使用工作日计算
    if is_after_market_close and now.weekday() < 5:
        return now.strftime("%Y-%m-%d")
    else:
        prev_bday = now - BDay(1)
        return prev_bday.strftime("%Y-%m-%d")

def get_qlib_provider_status(provider_uri: Optional[str] = None) -> Dict[str, Any]:
    """
    检查指定或默认 Qlib Provider 的真实就绪与新鲜度状态 (READY / STALE / BROKEN)
    """
    if provider_uri is None:
        provider_uri = str(Path("~/.qlib/qlib_data/cn_data").expanduser())

    p = Path(provider_uri).expanduser().resolve()
    cal_file = p / "calendars" / "day.txt"
    all_inst_file = p / "instruments" / "all.txt"
    csi300_file = p / "instruments" / "csi300.txt"
    features_dir = p / "features"
    benchmark_dir = features_dir / "sh000300"

    expected_market_date = get_expected_latest_completed_trade_date()

    if not p.exists() or not cal_file.exists() or not all_inst_file.exists() or not features_dir.exists():
        return {
            "provider_uri": str(p),
            "status": "BROKEN",
            "calendar_start": "N/A",
            "calendar_end": "N/A",
            "total_trading_days": 0,
            "benchmark_available": False,
            "csi300_available": False,
            "factor_available": False,
            "expected_latest_market_date": expected_market_date,
            "status_message": f"Provider directory '{p}' is missing or incomplete."
        }

    calendar_dates = [line.strip() for line in cal_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    cal_start = calendar_dates[0] if calendar_dates else "N/A"
    cal_end = calendar_dates[-1] if calendar_dates else "N/A"
    total_days = len(calendar_dates)

    benchmark_avail = benchmark_dir.exists() and (benchmark_dir / "close.day.bin").exists()
    csi300_avail = csi300_file.exists()

    # 检查 factor.day.bin 是否存在
    sample_stocks = [d for d in features_dir.iterdir() if d.is_dir() and not d.name.startswith("sh000300")]
    factor_avail = any((s / "factor.day.bin").exists() for s in sample_stocks[:10])

    if not benchmark_avail or not csi300_avail or not factor_avail or total_days == 0:
        status = "BROKEN"
        msg = "Provider is missing benchmark, csi300 instruments, or factor feature bins."
    elif cal_end < expected_market_date:
        status = "STALE"
        msg = f"Provider data ends at '{cal_end}', older than expected market date '{expected_market_date}'."
    else:
        status = "READY"
        msg = "Provider data is complete, benchmark ready, and fully up-to-date."

    return {
        "provider_uri": str(p),
        "status": status,
        "calendar_start": cal_start,
        "calendar_end": cal_end,
        "total_trading_days": total_days,
        "benchmark_available": benchmark_avail,
        "csi300_available": csi300_avail,
        "factor_available": factor_avail,
        "expected_latest_market_date": expected_market_date,
        "status_message": msg
    }

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
                if provider_uri is not None:
                    target_resolved = str(Path(provider_uri).expanduser().resolve())
                    current_resolved = str(Path(cls._initialized_provider_uri).expanduser().resolve())
                    if target_resolved != current_resolved:
                        raise ProviderMismatchError(
                            f"Qlib is already initialized with provider '{cls._initialized_provider_uri}'. "
                            f"Cannot reinitialize with different provider '{provider_uri}' in the same process without force=True."
                        )
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
        严格使用 adjusted prices (open_adj, high_adj, low_adj, close_adj, adj_factor)；
        若缺少复权价格，直接抛出 DataSchemaError 异常，严禁降级使用 raw price。
        """
        required_cols = ["open_adj", "high_adj", "low_adj", "close_adj", "adj_factor", "volume", "trade_date", "ts_code"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise DataSchemaError(
                f"DataSchemaError: Qlib CSV export requires adjusted price columns: {missing}. "
                f"Raw price fallback is strictly forbidden by Qlib official specification."
            )

        out_path = Path(target_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        for ts_code, group in df.groupby("ts_code"):
            q_sym = to_qlib_symbol(ts_code)
            sub = group.sort_values("trade_date").copy()

            sub_csv = pd.DataFrame({
                "symbol": q_sym,
                "date": pd.to_datetime(sub["trade_date"]).dt.strftime("%Y-%m-%d"),
                "open": sub["open_adj"],
                "high": sub["high_adj"],
                "low": sub["low_adj"],
                "close": sub["close_adj"],
                "volume": sub["volume"],
                "factor": sub["adj_factor"],
                "change": sub["pct_chg"] if "pct_chg" in sub.columns else 0.0,
            })

            file_path = out_path / f"{q_sym}.csv"
            sub_csv.to_csv(file_path, index=False)

        logger.info(f"Exported {len(df['ts_code'].unique())} stock CSVs to '{out_path}' for official Qlib dump_bin.")
        return str(out_path)
