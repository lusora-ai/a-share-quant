import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
import akshare as ak
import baostock as bs
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.data.fetcher")

class DataFetcher:
    """
    A股行情与基础数据抓取器
    价格数据彻底拆解为 真实原始价格 (raw) 与 复权研究价格 (adj)，支持显式涨跌停与复权因子
    """
    def __init__(self, primary_provider: str = "akshare", fallback_provider: str = "baostock"):
        self.primary_provider = primary_provider.lower()
        self.fallback_provider = fallback_provider.lower()
        self._bs_initialized = False

    def _init_baostock(self):
        if not self._bs_initialized:
            lg = bs.login()
            if lg.error_code == '0':
                self._bs_initialized = True
                logger.info("Baostock login successful.")
            else:
                logger.warning(f"Baostock login failed: {lg.error_msg}")

    def close(self):
        if self._bs_initialized:
            bs.logout()
            self._bs_initialized = False
            logger.info("Baostock logout successful.")

    def fetch_stock_master(self) -> pd.DataFrame:
        """
        获取全A股股票基础信息 (代码, 名称, 上市日期, 板块等)
        """
        logger.info("Fetching stock master list from AkShare...")
        try:
            df = ak.stock_info_a_code_name()
            df = df.rename(columns={"code": "symbol", "name": "name"})
            
            def format_ts_code(code: str) -> str:
                code_str = str(code).zfill(6)
                if code_str.startswith(('60', '688', '900')):
                    return f"{code_str}.SH"
                elif code_str.startswith(('00', '30', '200')):
                    return f"{code_str}.SZ"
                elif code_str.startswith(('43', '83', '87', '920')):
                    return f"{code_str}.BJ"
                return f"{code_str}.SH"
                
            df["ts_code"] = df["symbol"].apply(format_ts_code)
            logger.info(f"Successfully fetched {len(df)} stock master records.")
            return df
        except Exception as e:
            logger.error(f"Error fetching stock master list: {e}")
            raise

    def fetch_trade_calendar(self, start_date: str = "20180101", end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取 A 股交易日历
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        start_date_clean = start_date.replace("-", "")
        end_date_clean = end_date.replace("-", "")
        
        logger.info(f"Fetching trade calendar from {start_date_clean} to {end_date_clean}...")
        try:
            df = ak.tool_trade_date_hist_sina()
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
            df = df[(df["trade_date"] >= pd.to_datetime(start_date_clean).strftime("%Y-%m-%d")) & 
                    (df["trade_date"] <= pd.to_datetime(end_date_clean).strftime("%Y-%m-%d"))]
            df = df.sort_values("trade_date").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching trade calendar: {e}")
            raise

    def fetch_daily_ohlcv(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取单只股票的标准化日线行情 (包含 raw 价格与 hfq 复权价格)
        返回 Schema:
        [ts_code, trade_date, open_raw, high_raw, low_raw, close_raw,
         open_adj, high_adj, low_adj, close_adj, adj_factor, volume, amount, turn, pct_chg,
         limit_up_price, limit_down_price, limit_buy, limit_sell, is_suspended, data_quality_flag]
        """
        start_date_clean = start_date.replace("-", "")
        end_date_clean = end_date.replace("-", "")
        symbol = ts_code.split(".")[0]
        
        try:
            # 1. 抓取不复权原始数据 (raw)
            df_raw = ak.stock_zh_a_hist(
                symbol=symbol, period="daily", start_date=start_date_clean, end_date=end_date_clean, adjust=""
            )
            
            # 2. 抓取后复权研究数据 (hfq / adj)
            df_adj = ak.stock_zh_a_hist(
                symbol=symbol, period="daily", start_date=start_date_clean, end_date=end_date_clean, adjust="hfq"
            )
            
            if df_raw.empty or df_adj.empty:
                return pd.DataFrame()
                
            df_raw = df_raw.rename(columns={
                "日期": "trade_date", "开盘": "open_raw", "最高": "high_raw", "最低": "low_raw", "收盘": "close_raw",
                "成交量": "volume", "成交额": "amount", "换手率": "turn", "涨跌幅": "pct_chg"
            })
            
            df_adj = df_adj.rename(columns={
                "日期": "trade_date", "开盘": "open_adj", "最高": "high_adj", "最低": "low_adj", "收盘": "close_adj"
            })
            
            # 合并 raw 与 adj 价格
            merged = pd.merge(
                df_raw[["trade_date", "open_raw", "high_raw", "low_raw", "close_raw", "volume", "amount", "turn", "pct_chg"]],
                df_adj[["trade_date", "open_adj", "high_adj", "low_adj", "close_adj"]],
                on="trade_date", how="inner"
            )
            
            merged["ts_code"] = ts_code
            merged["trade_date"] = pd.to_datetime(merged["trade_date"]).dt.strftime("%Y-%m-%d")
            
            # 计算复权因子 adj_factor = close_adj / close_raw
            merged["adj_factor"] = (merged["close_adj"] / (merged["close_raw"] + 1e-8)).fillna(1.0)
            
            # 标记停牌
            merged["is_suspended"] = (merged["volume"] == 0) | (merged["open_raw"].isna())
            
            # 衍生前收盘价用于计算涨跌停限制
            merged["prev_close_raw"] = merged["close_raw"].shift(1)
            
            # 根据 A 股证券板块精度计算真实涨跌停价 (主板 10%, 创业板/科创板 20%, ST 5%)
            def calc_limit_prices(row):
                prev_p = row["prev_close_raw"]
                if pd.isna(prev_p) or prev_p <= 0:
                    return pd.Series([np.nan, np.nan, False, False])
                    
                code = row["ts_code"]
                limit_pct = 0.10
                if code.startswith(("300", "688", "301")):
                    limit_pct = 0.20
                    
                up_limit = round(prev_p * (1.0 + limit_pct) + 1e-5, 2)
                down_limit = round(prev_p * (1.0 - limit_pct) + 1e-5, 2)
                
                # 判断触及涨跌停状态
                is_limit_buy = (row["close_raw"] >= up_limit - 0.01) and (row["open_raw"] >= up_limit - 0.01)
                is_limit_sell = (row["close_raw"] <= down_limit + 0.01) and (row["open_raw"] <= down_limit + 0.01)
                
                return pd.Series([up_limit, down_limit, is_limit_buy, is_limit_sell])
                
            limits = merged.apply(calc_limit_prices, axis=1)
            limits.columns = ["limit_up_price", "limit_down_price", "limit_buy", "limit_sell"]
            
            merged = pd.concat([merged, limits], axis=1)
            merged["data_quality_flag"] = "OK"
            
            cols = [
                "ts_code", "trade_date", "open_raw", "high_raw", "low_raw", "close_raw",
                "open_adj", "high_adj", "low_adj", "close_adj", "adj_factor",
                "volume", "amount", "turn", "pct_chg",
                "limit_up_price", "limit_down_price", "limit_buy", "limit_sell", "is_suspended", "data_quality_flag"
            ]
            return merged[cols].sort_values("trade_date").reset_index(drop=True)
            
        except Exception as e:
            logger.warning(f"Fetch daily failed for {ts_code}: {e}")
            return pd.DataFrame()
