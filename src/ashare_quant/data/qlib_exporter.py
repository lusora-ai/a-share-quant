"""
Qlib Data Exporter and Initialization Provider.
Manages qlib.init() configuration, data validation, and standard CSV preparation for Qlib official DumpData.
Zero fake binary mocks, zero fake benchmark constants.
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
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

class ProviderMismatchError(RuntimeError):
    """Raised when attempting to reinitialize Qlib with a different provider URI."""
    pass

class DataSchemaError(ValueError):
    """Raised when required adjusted price columns are missing for export."""
    pass

class MarketCalendarUnavailableError(RuntimeError):
    """Raised when an authentic market trading calendar covering current date is unavailable."""
    pass

def get_expected_latest_completed_trade_date(reference_time: Optional[datetime] = None) -> str:
    """
    计算当前预期最新已完成结算的 A 股交易日 (YYYY-MM-DD)。
    严格使用 Asia/Shanghai 时区。
    若未过 15:30，预期为上一个交易日；若已过 15:30，预期为当日（若当日为交易日）或此前最近一个交易日。
    禁止使用 pandas BDay 猜测（无法识别中国法定节假日）。
    若本地 cached trade_calendar 未覆盖当前日期附近的真实交易日 schedule，
    且无法从可靠日历数据源拉取到证明覆盖当前日期的真实日历（即包含当前日期之后的已知交易日），
    抛出 MarketCalendarUnavailableError。
    """
    shanghai_tz = ZoneInfo("Asia/Shanghai")
    if reference_time is None:
        now = datetime.now(shanghai_tz)
    else:
        if reference_time.tzinfo is None:
            now = reference_time.replace(tzinfo=shanghai_tz)
        else:
            now = reference_time.astimezone(shanghai_tz)

    today_str = now.strftime("%Y-%m-%d")
    is_after_market_close = (now.hour > 15) or (now.hour == 15 and now.minute >= 30)

    # 尝试从 storage 加载 trade_calendar
    df_cal = None
    try:
        from ashare_quant.data.processor import DataProcessor
        proc = DataProcessor()
        try:
            df_cal = proc.storage.load_parquet("trade_calendar", is_processed=True)
        finally:
            proc.close()
    except Exception:
        pass

    def extract_open_dates(df: Optional[pd.DataFrame]) -> List[str]:
        if df is None or df.empty or "trade_date" not in df.columns:
            return []
        if "is_open" in df.columns:
            valid = df[(df["is_open"] == 1) | (df["is_open"] == True) | (df["is_open"] == "1")]
            return sorted([str(d) for d in valid["trade_date"].dropna().unique()])
        return sorted([str(d) for d in df["trade_date"].dropna().unique()])

    open_dates = extract_open_dates(df_cal)
    has_future_dates = any(d > today_str for d in open_dates)

    # 若本地日历无法证明覆盖 today，尝试从 DataFetcher 动态拉取前后窗口 (today - 45d ~ today + 45d)
    if not has_future_dates:
        try:
            from ashare_quant.data.fetcher import DataFetcher
            fetcher = DataFetcher()
            start_window = (now - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
            end_window = (now + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
            fresh_cal = fetcher.fetch_trade_calendar(start_date=start_window, end_date=end_window)
            fresh_open_dates = extract_open_dates(fresh_cal)
            if any(d > today_str for d in fresh_open_dates):
                open_dates = fresh_open_dates
                has_future_dates = True
        except Exception:
            pass

    # 严格检验日历覆盖范围：禁止猜测交易日
    if not open_dates or not has_future_dates:
        cal_max = max(open_dates) if open_dates else "None"
        raise MarketCalendarUnavailableError(
            f"MarketCalendarUnavailableError: Reliable trading calendar covering current date '{today_str}' is unavailable. "
            f"Calendar maximum known trading date is '{cal_max}', which does not contain future trading dates beyond '{today_str}'. "
            f"Guessing trade dates via weekday/BDay is forbidden."
        )

    # 计算预期最新已完成交易日
    past_dates = [d for d in open_dates if d < today_str]

    if not is_after_market_close:
        # 当前时间 < 15:30: expected_date = 最大 trading_date < today
        if not past_dates:
            raise MarketCalendarUnavailableError(
                f"MarketCalendarUnavailableError: No completed trading dates found before '{today_str}' in market calendar."
            )
        return past_dates[-1]
    else:
        # 当前时间 >= 15:30:
        # 如果 today 是 trading day: expected_date = today
        # 否则: expected_date = 最大 trading_date < today
        if today_str in open_dates:
            return today_str
        else:
            if not past_dates:
                raise MarketCalendarUnavailableError(
                    f"MarketCalendarUnavailableError: No completed trading dates found before '{today_str}' in market calendar."
                )
            return past_dates[-1]

def get_qlib_provider_status(provider_uri: Optional[str] = None) -> Dict[str, Any]:
    """
    检查指定或默认 Qlib Provider 的真实就绪与新鲜度状态 (READY / STALE / BROKEN / UNKNOWN)
    严格检查 CSI300 标的池的 factor coverage，严禁 10 只有 1 只就判断 READY。
    """
    if provider_uri is None:
        provider_uri = str(Path("~/.qlib/qlib_data/cn_data").expanduser())

    p = Path(provider_uri).expanduser().resolve()
    cal_file = p / "calendars" / "day.txt"
    all_inst_file = p / "instruments" / "all.txt"
    csi300_file = p / "instruments" / "csi300.txt"
    features_dir = p / "features"
    benchmark_dir = features_dir / "sh000300"

    # 预期最新交易日 (独立日历检查)
    expected_market_date = None
    calendar_error = None
    try:
        expected_market_date = get_expected_latest_completed_trade_date()
    except MarketCalendarUnavailableError as e:
        calendar_error = str(e)
    except Exception as e:
        calendar_error = f"Calendar check error: {e}"

    if not p.exists() or not cal_file.exists() or not all_inst_file.exists() or not features_dir.exists():
        return {
            "provider_uri": str(p),
            "status": "BROKEN",
            "calendar_start": "N/A",
            "calendar_end": "N/A",
            "total_trading_days": 0,
            "benchmark_available": False,
            "csi300_available": False,
            "csi300_member_count": 0,
            "csi300_membership_max_end": "N/A",
            "csi300_active_count_on_expected_date": 0,
            "csi300_membership_fresh": False,
            "factor_coverage_count": 0,
            "factor_expected_count": 0,
            "factor_coverage_pct": 0.0,
            "missing_factor_examples": [],
            "expected_latest_market_date": expected_market_date or "UNAVAILABLE",
            "status_message": f"Provider directory '{p}' is missing or incomplete."
        }

    calendar_dates = [line.strip() for line in cal_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    cal_start = calendar_dates[0] if calendar_dates else "N/A"
    cal_end = calendar_dates[-1] if calendar_dates else "N/A"
    total_days = len(calendar_dates)

    benchmark_avail = benchmark_dir.exists() and (benchmark_dir / "close.day.bin").exists()
    csi300_avail = csi300_file.exists()

    # 解析 instruments/csi300.txt 的 Point-in-Time membership 数据
    csi300_records = []
    csi300_instruments = []
    if csi300_avail:
        for line in csi300_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 3:
                csi300_records.append((parts[0], parts[1], parts[2]))
                csi300_instruments.append(parts[0])
            elif len(parts) >= 1:
                csi300_records.append((parts[0], "2000-01-01", "2099-12-31"))
                csi300_instruments.append(parts[0])

    unique_csi300_members = list(dict.fromkeys(csi300_instruments))
    csi300_member_count = len(unique_csi300_members)
    membership_max_end = max((r[2] for r in csi300_records), default="N/A")

    active_count_on_expected = 0
    membership_fresh = False
    if expected_market_date and expected_market_date != "UNAVAILABLE" and csi300_records:
        active_members = set(
            r[0] for r in csi300_records if r[1] <= expected_market_date <= r[2]
        )
        active_count_on_expected = len(active_members)
        if membership_max_end != "N/A" and membership_max_end >= expected_market_date and 280 <= active_count_on_expected <= 320:
            membership_fresh = True

    # 严格检查 CSI300 constituent universe 的 factor coverage
    check_universe = unique_csi300_members if unique_csi300_members else (
        [line.strip().split()[0] for line in all_inst_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    )
    stock_universe = [inst for inst in check_universe if not inst.lower().startswith("sh000300")]

    factor_expected_count = len(stock_universe)
    missing_factors = []
    factor_coverage_count = 0
    for inst in stock_universe:
        inst_feat = features_dir / inst.lower()
        if not inst_feat.exists():
            inst_feat = features_dir / inst
        if inst_feat.exists() and (inst_feat / "factor.day.bin").exists():
            factor_coverage_count += 1
        else:
            missing_factors.append(inst)

    factor_coverage_pct = round(factor_coverage_count / factor_expected_count * 100.0, 2) if factor_expected_count > 0 else 0.0

    if not benchmark_avail or not csi300_avail or total_days == 0 or factor_coverage_pct < 100.0 or not membership_fresh:
        status = "BROKEN"
        reasons = []
        if not benchmark_avail:
            reasons.append("Missing benchmark (SH000300)")
        if not csi300_avail:
            reasons.append("Missing csi300.txt")
        if total_days == 0:
            reasons.append("Empty trading calendar")
        if factor_coverage_pct < 100.0:
            reasons.append(f"Incomplete factor coverage: {factor_coverage_count}/{factor_expected_count} ({factor_coverage_pct}%)")
        if csi300_avail and not membership_fresh:
            if expected_market_date is None or expected_market_date == "UNAVAILABLE":
                reasons.append("Market calendar unavailable to determine CSI300 membership freshness")
            elif membership_max_end < expected_market_date:
                reasons.append(f"CSI300 instrument membership is stale (max membership end '{membership_max_end}' < expected market date '{expected_market_date}')")
            elif active_count_on_expected < 280 or active_count_on_expected > 320:
                reasons.append(f"Abnormal active CSI300 constituents count on {expected_market_date}: {active_count_on_expected} (expected 280~320)")
        msg = f"Provider is BROKEN: {', '.join(reasons)}."
    elif calendar_error is not None:
        status = "UNKNOWN"
        msg = f"Market calendar unavailable to determine freshness: {calendar_error}"
    elif expected_market_date and cal_end < expected_market_date:
        status = "STALE"
        msg = f"Provider data ends at '{cal_end}', older than expected market date '{expected_market_date}'."
    else:
        status = "READY"
        msg = f"Provider data is complete ({factor_coverage_count}/{factor_expected_count} factors), benchmark ready, CSI300 membership active ({active_count_on_expected}), and fully up-to-date."

    return {
        "provider_uri": str(p),
        "status": status,
        "calendar_start": cal_start,
        "calendar_end": cal_end,
        "total_trading_days": total_days,
        "benchmark_available": benchmark_avail,
        "csi300_available": csi300_avail,
        "csi300_member_count": csi300_member_count,
        "csi300_membership_max_end": membership_max_end,
        "csi300_active_count_on_expected_date": active_count_on_expected,
        "csi300_membership_fresh": membership_fresh,
        "factor_coverage_count": factor_coverage_count,
        "factor_expected_count": factor_expected_count,
        "factor_coverage_pct": factor_coverage_pct,
        "missing_factor_examples": missing_factors[:5],
        "expected_latest_market_date": expected_market_date or "UNAVAILABLE",
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
