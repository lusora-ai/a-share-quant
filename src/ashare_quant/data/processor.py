import pandas as pd
from typing import Optional, List, Dict, Any
from ashare_quant.data.fetcher import DataFetcher
from ashare_quant.data.storage import StorageEngine
from ashare_quant.data.qa import DataQAValidator
from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.data.processor")

class DataProcessor:
    """
    数据流水线总控器
    负责主数据抓取、历史与增量数据更新、QA 验证与持久化落盘
    """
    def __init__(self, data_config: Optional[Dict[str, Any]] = None):
        self.config = data_config or load_config("data")
        self.fetcher = DataFetcher(
            primary_provider=self.config.get("providers", {}).get("primary", "akshare"),
            fallback_provider=self.config.get("providers", {}).get("fallback", "baostock")
        )
        self.storage = StorageEngine()
        qa_cfg = self.config.get("qa_checks", {})
        self.qa = DataQAValidator(
            check_ohlc=qa_cfg.get("check_ohlc_logic", True),
            check_price_positive=qa_cfg.get("check_price_positive", True),
            max_missing_ratio=qa_cfg.get("max_missing_ratio", 0.05)
        )

    def update_stock_master(self) -> pd.DataFrame:
        """
        更新股票主数据与代码元数据
        """
        df_master = self.fetcher.fetch_stock_master()
        if not df_master.empty:
            self.storage.save_parquet(df_master, "stock_master", is_processed=True)
            self.storage.sync_to_duckdb("stock_master", df_master, if_exists="replace")
        return df_master

    def update_trade_calendar(self, start_date: str = "20180101") -> pd.DataFrame:
        """
        更新交易日历
        """
        df_cal = self.fetcher.fetch_trade_calendar(start_date=start_date)
        if not df_cal.empty:
            self.storage.save_parquet(df_cal, "trade_calendar", is_processed=True)
            self.storage.sync_to_duckdb("trade_calendar", df_cal, if_exists="replace")
        return df_cal

    def update_daily_data(self, symbols: Optional[List[str]] = None, start_date: str = "2018-01-01", end_date: Optional[str] = None):
        """
        更新指定股票池或全市场的日线数据
        """
        if symbols is None:
            df_master = self.storage.load_parquet("stock_master", is_processed=True)
            if df_master.empty:
                df_master = self.update_stock_master()
            symbols = df_master["ts_code"].tolist()
            
        logger.info(f"Starting daily OHLCV data update for {len(symbols)} stocks from {start_date} to {end_date or 'today'}...")
        
        all_daily = []
        for i, ts_code in enumerate(symbols):
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(symbols)} stocks updated.")
            df_stock = self.fetcher.fetch_daily_ohlcv(ts_code, start_date=start_date, end_date=end_date or "")
            if not df_stock.empty:
                all_daily.append(df_stock)
                
        if not all_daily:
            logger.warning("No daily data fetched.")
            return pd.DataFrame()
            
        df_all = pd.concat(all_daily, ignore_index=True)
        
        # 执行 QA 验证与清洗
        clean_df, report = self.qa.validate_daily_ohlcv(df_all)
        
        # 保存持久化
        self.storage.save_parquet(clean_df, "daily_ohlcv", partition_col="ts_code", is_processed=True)
        self.storage.sync_to_duckdb("daily_ohlcv", clean_df, if_exists="replace")
        
        logger.info("Daily data update finished successfully.")
        return clean_df

    def close(self):
        self.fetcher.close()
        self.storage.close()
