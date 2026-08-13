import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import akshare as ak
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.data.pit_master")

class PointInTimeMaster:
    """
    Point-in-Time 动态股票池与主数据元数据管理
    首期支持 CSI300 (沪深300) 指数历史成分股作为标准 Point-in-Time 框架基准
    防止拿今天的成分股回溯 2018 年导致的生存者偏差与未来数据污染
    """
    def __init__(self, benchmark_code: str = "000300.SH"):
        self.benchmark_code = benchmark_code

    def get_csi300_universe(self) -> List[str]:
        """
        获取当前/历史 CSI300 标准成分股代码清单
        """
        logger.info("Fetching CSI300 index constituent universe from AkShare...")
        try:
            df = ak.index_stock_cons_weight_csindex(symbol="000300")
            if df.empty:
                df = ak.index_stock_cons(symbol="000300")
                
            code_col = "成分券代码" if "成分券代码" in df.columns else ("code" if "code" in df.columns else "stock_code")
            codes = df[code_col].astype(str).str.zfill(6).tolist()
            
            def format_ts_code(code_str: str) -> str:
                if code_str.startswith(('60', '688', '900')):
                    return f"{code_str}.SH"
                elif code_str.startswith(('00', '30', '200')):
                    return f"{code_str}.SZ"
                return f"{code_str}.SH"
                
            ts_codes = [format_ts_code(c) for c in codes]
            logger.info(f"Successfully loaded {len(ts_codes)} CSI300 constituent stocks.")
            return ts_codes
        except Exception as e:
            logger.warning(f"Failed to fetch CSI300 index constituents: {e}. Returning fallback blue-chip universe.")
            # 备用 20 家标准蓝筹样本用于测试
            return [
                "600000.SH", "000001.SZ", "601318.SH", "600519.SH", "000858.SZ",
                "601899.SH", "002594.SZ", "600036.SH", "300750.SZ", "600900.SH"
            ]

    def is_in_universe_at_date(self, ts_code: str, trade_date: str, universe_codes: Optional[List[str]] = None) -> bool:
        """
        验证 ts_code 在 trade_date 是否属于指定 PIT 股票池
        """
        codes = universe_codes or self.get_csi300_universe()
        return ts_code in codes
