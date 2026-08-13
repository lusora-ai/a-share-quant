import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from ashare_quant.utils.logging import setup_logger

logger = setup_logger("ashare_quant.portfolio.execution")

class ManualExecutionTracker:
    """
    300 元小额真实账户手动交易与复盘追踪器
    强制规则: 资金上限 ¥300，整百股下单；买不起的股票标记为 unaffordable，不影响/不修改模型研究股票池
    """
    def __init__(self, log_path: str = "reports/execution_log.json", max_cash: float = 300.0):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_cash = max_cash
        self.records: List[Dict[str, Any]] = self._load_records()

    def _load_records(self) -> List[Dict[str, Any]]:
        if self.log_path.exists():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading execution log: {e}")
                return []
        return []

    def save_records(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.records)} execution records to {self.log_path}")

    def evaluate_affordability(self, candidate_df: pd.DataFrame) -> pd.DataFrame:
        """
        评估候选股票在 300 元实盘本金下的可买性
        若 1 手 (100股) 最新成交价高于当前可用现金，则标记 is_affordable = False
        """
        if candidate_df.empty:
            return candidate_df
            
        df = candidate_df.copy()
        
        def check_row(row):
            price = row.get("close", row.get("open", 0.0))
            min_cost = price * 100.0
            if min_cost > self.max_cash:
                return False, "unaffordable"
            return True, "affordable"
            
        res = df.apply(check_row, axis=1)
        df["is_affordable"] = [r[0] for r in res]
        df["affordability_status"] = [r[1] for r in res]
        return df

    def record_manual_trade(
        self,
        trade_date: str,
        ts_code: str,
        name: str,
        action: str,
        shares: int,
        price: float,
        model_score: float,
        notes: str = ""
    ):
        """
        人工录入实盘成交记录
        """
        cost = shares * price
        record = {
            "trade_id": len(self.records) + 1,
            "trade_date": trade_date,
            "ts_code": ts_code,
            "name": name,
            "action": action.upper(),
            "shares": shares,
            "price": price,
            "cost": cost,
            "model_score": model_score,
            "notes": notes,
            "recorded_at": datetime.now().isoformat()
        }
        self.records.append(record)
        self.save_records()
        logger.info(f"Recorded manual trade: {action} {shares} shares of {name} ({ts_code}) @ ¥{price:.2f}")

    def get_execution_summary(self) -> pd.DataFrame:
        """
        返回实盘交易履约复盘表
        """
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame(self.records)
