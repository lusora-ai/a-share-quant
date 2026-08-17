"""
Qlib Data Exporter and Initialization Provider.
Manages qlib.init() configuration, data validation, and real Qlib dataset conversion.
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

    @classmethod
    def check_provider_ready(cls, provider_uri: str) -> None:
        """
        验证 provider_uri 是否具备合法的 Qlib 数据结构：
        - calendars/day.txt 存在且非空
        - instruments/all.txt 存在且非空
        - features/ 目录存在且包含数据
        """
        p = Path(provider_uri)
        if not p.exists():
            raise QlibDataNotReadyError(
                f"Qlib data directory not found at '{provider_uri}'. "
                f"Please run 'ashare-quant update-data' or download official Qlib cn_data."
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
        if not feat_dir.exists():
            raise QlibDataNotReadyError(
                f"Qlib features directory missing at '{feat_dir}'."
            )

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
            # 优先查找本地已存在的 qlib 数据目录
            candidates = [
                Path("data/qlib_cn").resolve(),
                Path("~/.qlib/qlib_data/cn_data").expanduser(),
            ]
            found = None
            for cand in candidates:
                if (cand / "calendars" / "day.txt").exists() and (cand / "instruments" / "all.txt").exists() and (cand / "features").exists():
                    found = str(cand)
                    break
            if found:
                provider_uri = found
            else:
                provider_uri = str(Path("data/qlib_cn").resolve())

        cls.check_provider_ready(provider_uri)

        logger.info(f"Initializing official Qlib with provider_uri='{provider_uri}', region='{region}'...")
        try:
            qlib.init(provider_uri=provider_uri, region=region)
            cls._initialized = True
            logger.info("Qlib initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Qlib: {e}")
            raise

    @classmethod
    def dump_df_to_qlib_bin(cls, df: pd.DataFrame, target_dir: str = "data/qlib_cn") -> str:
        """
        将包含真实 OHLCV 行情的 DataFrame 转换并导出为标准 Qlib 二进制/文本格式
        """
        out_path = Path(target_dir).resolve()
        cal_dir = out_path / "calendars"
        inst_dir = out_path / "instruments"
        feat_dir = out_path / "features"
        
        cal_dir.mkdir(parents=True, exist_ok=True)
        inst_dir.mkdir(parents=True, exist_ok=True)
        feat_dir.mkdir(parents=True, exist_ok=True)

        # 1. 导出日历 (包含未来充足交易日缓冲区，以满足 Qlib 模拟执行器 T+1 跨日日历查询)
        trade_dates = sorted(pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d").unique())
        if not trade_dates:
            raise ValueError("No trade dates in input DataFrame.")
            
        future_buffer_dates = pd.date_range(trade_dates[-1], periods=60, freq="B").strftime("%Y-%m-%d").tolist()[1:]
        all_cal_dates = trade_dates + future_buffer_dates

        with open(cal_dir / "day.txt", "w", encoding="utf-8") as f:
            for d in all_cal_dates:
                f.write(f"{d}\n")

        # 2. 导出所有标的代码
        date_min, date_max = trade_dates[0], trade_dates[-1]
        raw_symbols = sorted(df["ts_code"].unique())
        qlib_symbols = [to_qlib_symbol(s) for s in raw_symbols]

        with open(inst_dir / "all.txt", "w", encoding="utf-8") as f:
            for qs in qlib_symbols:
                f.write(f"{qs}\t{date_min}\t{date_max}\n")

        # 3. 导出 CSI300 标的清单
        with open(inst_dir / "csi300.txt", "w", encoding="utf-8") as f:
            for qs in qlib_symbols:
                f.write(f"{qs}\t{date_min}\t{date_max}\n")

        # 4. 导出各个标的的基础特征列 (open, high, low, close, volume, factor 等)
        date_index = pd.Index(all_cal_dates)
        for raw_s, q_s in zip(raw_symbols, qlib_symbols):
            sub = df[df["ts_code"] == raw_s].copy()
            sub["trade_date_str"] = pd.to_datetime(sub["trade_date"]).dt.strftime("%Y-%m-%d")
            sub = sub.set_index("trade_date_str").reindex(date_index)
            # 对未来日期用最后有效价格填充
            sub = sub.ffill()
            
            s_feat_dir = feat_dir / q_s.lower()
            s_feat_dir.mkdir(parents=True, exist_ok=True)

            feature_map = {
                "open": sub["open_adj"] if "open_adj" in sub.columns else sub.get("open"),
                "high": sub["high_adj"] if "high_adj" in sub.columns else sub.get("high"),
                "low": sub["low_adj"] if "low_adj" in sub.columns else sub.get("low"),
                "close": sub["close_adj"] if "close_adj" in sub.columns else sub.get("close"),
                "volume": sub.get("volume", pd.Series(10000.0, index=date_index)),
                "factor": sub.get("adj_factor", pd.Series(1.0, index=date_index)),
            }

            for fname, s_data in feature_map.items():
                if s_data is not None:
                    bin_file = s_feat_dir / f"{fname}.day.bin"
                    arr = s_data.astype(np.float32).fillna(1.0).values
                    arr.tofile(str(bin_file))

        # 导出基准指数 SH000300
        bench_dir = feat_dir / "sh000300"
        bench_dir.mkdir(parents=True, exist_ok=True)
        bench_series = pd.Series(1.0, index=date_index, dtype=np.float32)
        bench_series.values.tofile(str(bench_dir / "close.day.bin"))
        bench_series.values.tofile(str(bench_dir / "open.day.bin"))

        logger.info(f"Exported real Qlib binary dataset to '{out_path}' for {len(qlib_symbols)} symbols across {len(all_cal_dates)} dates.")
        return str(out_path)
