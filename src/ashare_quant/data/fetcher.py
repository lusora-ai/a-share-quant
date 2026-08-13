import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import akshare as ak
import baostock as bs
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.data.fetcher")

class DataFetcher:
    """
    A股行情与基础数据抓取器
    支持 AkShare (主数据源) 与 Baostock (备用数据源)
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
            # 字段标准化: symbol (e.g. 600000), name (e.g. 浦发银行)
            df = df.rename(columns={"code": "symbol", "name": "name"})
            
            # 添加交易所后缀与 standard ts_code (e.g. 600000.SH / 000001.SZ / 300001.SZ / 688001.SH / 830000.BJ)
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
            logger.info(f"Successfully fetched {len(df)} stocks master records.")
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

    def fetch_daily_ohlcv(self, ts_code: str, start_date: str, end_date: str, adjust: str = "hfq") -> pd.DataFrame:
        """
        获取单只股票在指定日期范围内的日线 OHLCV 行情
        返回标准化列: [ts_code, trade_date, open, high, low, close, volume, amount, turn, pct_chg, is_suspended]
        """
        start_date_clean = start_date.replace("-", "")
        end_date_clean = end_date.replace("-", "")
        symbol = ts_code.split(".")[0]
        
        try:
            # 优先调用 AkShare
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date_clean,
                end_date=end_date_clean,
                adjust=adjust
            )
            
            if df.empty:
                return pd.DataFrame()
                
            # 标准化列名
            rename_dict = {
                "日期": "trade_date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turn",
                "涨跌幅": "pct_chg"
            }
            df = df.rename(columns=rename_dict)
            df["ts_code"] = ts_code
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
            
            # 数值类型转换
            num_cols = ["open", "high", "low", "close", "volume", "amount", "turn", "pct_chg"]
            for col in num_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    
            # 标记停牌
            df["is_suspended"] = (df["volume"] == 0) | (df["open"].isna())
            
            # 选择并排序核心列
            cols = ["ts_code", "trade_date", "open", "high", "low", "close", "volume", "amount", "turn", "pct_chg", "is_suspended"]
            present_cols = [c for c in cols if c in df.columns]
            return df[present_cols].sort_values("trade_date").reset_index(drop=True)
            
        except Exception as e:
            logger.warning(f"AkShare fetch daily failed for {ts_code}: {e}. Trying Baostock fallback...")
            return self._fetch_daily_baostock(ts_code, start_date, end_date, adjust)

    def _fetch_daily_baostock(self, ts_code: str, start_date: str, end_date: str, adjust: str = "hfq") -> pd.DataFrame:
        """
        Baostock 备用日线抓取实现
        """
        self._init_baostock()
        if not self._bs_initialized:
            return pd.DataFrame()
            
        # Baostock ts_code 格式: sh.600000 或 sz.000001
        symbol, market = ts_code.split(".")
        bs_code = f"{market.lower()}.{symbol}"
        
        adjust_flag = "1" if adjust == "hfq" else ("2" if adjust == "qfq" else "3")
        
        fields = "date,code,open,high,low,close,volume,amount,turn,pctChg,tradestatus"
        rs = bs.query_history_k_data_plus(
            bs_code, fields,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            frequency="d", adjustflag=adjust_flag
        )
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
            
        if not data_list:
            return pd.DataFrame()
            
        df = pd.DataFrame(data_list, columns=rs.fields)
        df = df.rename(columns={
            "date": "trade_date",
            "pctChg": "pct_chg",
            "tradestatus": "trade_status"
        })
        df["ts_code"] = ts_code
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        
        num_cols = ["open", "high", "low", "close", "volume", "amount", "turn", "pct_chg"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        df["is_suspended"] = df["trade_status"].astype(str) == "0"
        
        cols = ["ts_code", "trade_date", "open", "high", "low", "close", "volume", "amount", "turn", "pct_chg", "is_suspended"]
        return df[cols].sort_values("trade_date").reset_index(drop=True)
