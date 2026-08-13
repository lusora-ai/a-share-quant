import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.data.storage")

class StorageEngine:
    """
    数据持久化存储引擎
    - 原始/清洗数据使用 Parquet 分区存储 (按数据类型/年份)
    - 结合 DuckDB 数据库进行高效 SQL 查询、分析与增量更新
    """
    def __init__(self, data_dir: str = "data", db_name: str = "quant_lab.duckdb"):
        self.base_dir = Path(data_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.cache_dir = self.base_dir / "cache"
        self.db_path = self.base_dir / db_name
        
        # 建立目录
        for d in [self.raw_dir, self.processed_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        self.conn = duckdb.connect(str(self.db_path))
        logger.info(f"StorageEngine initialized at {self.base_dir}. DuckDB: {self.db_path}")

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("DuckDB connection closed.")

    def save_parquet(self, df: pd.DataFrame, dataset_name: str, partition_col: Optional[str] = None, is_processed: bool = False):
        """
        保存 DataFrame 为 Parquet 文件
        """
        if df.empty:
            logger.warning(f"Attempted to save empty DataFrame for dataset {dataset_name}.")
            return
            
        target_dir = self.processed_dir if is_processed else self.raw_dir
        dataset_path = target_dir / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        table = pa.Table.from_pandas(df)
        if partition_col and partition_col in df.columns:
            pq.write_to_dataset(table, root_path=str(dataset_path), partition_cols=[partition_col], use_dictionary=True)
        else:
            file_path = dataset_path / f"{dataset_name}.parquet"
            pq.write_table(table, file_path)
            
        logger.info(f"Saved {len(df)} rows to Parquet at {dataset_path}")

    def load_parquet(self, dataset_name: str, is_processed: bool = False) -> pd.DataFrame:
        """
        读取 Parquet 数据集
        """
        target_dir = self.processed_dir if is_processed else self.raw_dir
        dataset_path = target_dir / dataset_name
        
        if not dataset_path.exists():
            logger.warning(f"Parquet dataset path does not exist: {dataset_path}")
            return pd.DataFrame()
            
        try:
            return pd.read_parquet(dataset_path)
        except Exception as e:
            logger.error(f"Failed to load Parquet dataset {dataset_name}: {e}")
            return pd.DataFrame()

    def sync_to_duckdb(self, table_name: str, df: pd.DataFrame, if_exists: str = "replace"):
        """
        同步内存/Parquet 中的 DataFrame 到 DuckDB 数据库表
        """
        if df.empty:
            logger.warning(f"DataFrame for table {table_name} is empty. Skipping DuckDB sync.")
            return
            
        try:
            if if_exists == "replace":
                self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            elif if_exists == "append":
                self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
            logger.info(f"Successfully synced {len(df)} rows into DuckDB table '{table_name}'.")
        except Exception as e:
            logger.error(f"Error syncing to DuckDB table {table_name}: {e}")
            raise

    def query_duckdb(self, sql: str) -> pd.DataFrame:
        """
        执行 SQL 查询并返回 Pandas DataFrame
        """
        try:
            return self.conn.execute(sql).df()
        except Exception as e:
            logger.error(f"DuckDB query error: {sql} | Error: {e}")
            raise
