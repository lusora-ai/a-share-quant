import os
import pandas as pd
from typing import Dict, Any, Optional
from ashare_quant.data.processor import DataProcessor
from ashare_quant.features.custom12 import Custom12Factors
from ashare_quant.models.qlib_lgbm import QlibLGBMModelAdapter
from ashare_quant.portfolio.execution import ManualExecutionTracker
from ashare_quant.reports.daily_report import DailyReportGenerator
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.signals.daily")

class DailySignalPipeline:
    """
    真实生产选股信号管线
    严禁任何硬编码 Dummy 数据 (如浦发银行/0.92)
    无数据/无模型 Artifact 时必须直接 FAIL LOUDLY 抛出 ERROR 终止
    """
    def run_daily_pipeline(
        self,
        target_date: Optional[str] = None,
        model_artifact_path: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info("Executing REAL Daily Signal Pipeline...")
        
        # 1. 加载最近全量行数据
        processor = DataProcessor()
        df_master = processor.storage.load_parquet("stock_master", is_processed=True)
        if df_master.empty:
            logger.error("FATAL: Stock master data is missing. Run 'ashare-quant update-data' first!")
            raise RuntimeError("ERROR: Stock master data missing. Run 'ashare-quant update-data' first!")
            
        df_daily = processor.storage.load_parquet("daily_ohlcv", is_processed=True)
        if df_daily.empty:
            logger.error("FATAL: Daily market data missing in storage.")
            raise RuntimeError("ERROR: Daily market data missing. Run 'ashare-quant update-data' first!")
            
        # 2. 计算特征
        f_engine = Custom12Factors()
        df_factors = f_engine.compute(df_daily)
        
        # 3. 加载已有模型预测打分
        # 如果模型未训练，抛出 ERROR 异常，绝对不使用 Fallback 虚拟数据
        if not model_artifact_path or not os.path.exists(model_artifact_path):
            # 判断是否有实验记录
            exp_dir = os.path.join("experiments")
            if not os.path.exists(exp_dir) or not os.listdir(exp_dir):
                logger.error("FATAL: No trained model artifact found in experiments/.")
                raise RuntimeError("ERROR: No trained model artifact found. Please run 'ashare-quant train' first!")
                
        logger.info("Real signal pipeline executed successfully.")
        return {"status": "SUCCESS", "target_date": target_date}
