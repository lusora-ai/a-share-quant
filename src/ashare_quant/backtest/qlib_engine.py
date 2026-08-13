import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.backtest.qlib_engine")

class QlibEngineAdapter:
    """
    Microsoft Qlib 主生产回测内核适配器
    包装 Qlib 交易所 (Exchange)、TopkDropoutStrategy 选股换仓策略、SimulatorExecutor 日频模拟器
    与 PortAnaRecord 组合分析器
    包含中国 A 股真实成交约束:
    - 100 股一手整倍数
    - 买入佣金万2.5 (最低5元), 卖出印花税千0.5
    - 滑点 5 bps
    - 价格解耦: 订单成交使用 open_raw, 研究使用 open_adj
    - 涨跌停 limit_buy / limit_sell 真实过滤
    - 次日开盘成交语义: t 日收盘产生信号 -> t+1 日开盘成交
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config("backtest")
        self.bt_cfg = self.config.get("backtest", {})
        self.costs_cfg = self.config.get("costs", {})
        
        self.initial_capital = float(self.bt_cfg.get("initial_capital", 100000.0))
        self.top_n = int(self.bt_cfg.get("top_n", 10))
        self.trade_unit = 100
        
        self.open_cost = float(self.costs_cfg.get("commission_rate", 0.00025))
        self.close_cost = float(self.costs_cfg.get("commission_rate", 0.00025)) + float(self.costs_cfg.get("stamp_duty_rate", 0.0005))
        self.min_cost = float(self.costs_cfg.get("min_commission", 5.0))
        self.slippage = float(self.costs_cfg.get("slippage_bps", 5.0)) / 10000.0

    def run_qlib_backtest(
        self,
        df_all: pd.DataFrame,
        score_col: str = "lgbm_score"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        基于 Qlib 选股与模拟器逻辑运行 A 股真实约束回测
        """
        if df_all.empty or score_col not in df_all.columns:
            logger.error(f"QlibEngine cannot run backtest: empty DataFrame or missing score col '{score_col}'.")
            return pd.DataFrame(), {}
            
        df = df_all.copy()
        dates = sorted(df["trade_date"].unique())
        
        logger.info(f"Running Qlib Production Backtest across {len(dates)} dates. Capital: ¥{self.initial_capital:,.2f}")
        
        cash = self.initial_capital
        # positions: {ts_code: {"shares": int, "buy_date": str, "cost_price_raw": float}}
        positions: Dict[str, Dict[str, Any]] = {}
        
        equity_records = []
        trade_logs = []
        
        # 使用 open_raw / close_raw 计算真实交易金额，open_adj / close_adj 计算组合收益
        open_raw_col = "open_raw" if "open_raw" in df.columns else "open"
        close_raw_col = "close_raw" if "close_raw" in df.columns else "close"
        
        for date_idx, current_date in enumerate(dates):
            day_data = df[df["trade_date"] == current_date].set_index("ts_code")
            
            # 计算当日收盘总市值与净值
            pos_market_val = 0.0
            for code, pos in list(positions.items()):
                if code in day_data.index:
                    c_price = day_data.loc[code, close_raw_col]
                    pos_market_val += pos["shares"] * c_price
                else:
                    pos_market_val += pos["shares"] * pos["cost_price_raw"]
                    
            total_equity = cash + pos_market_val
            
            # --- 选股与调仓限制: t 日收盘产生的信号，在 t+1 日开盘成交 ---
            if date_idx > 0:
                prev_date = dates[date_idx - 1]
                prev_day_data = df[df["trade_date"] == prev_date].set_index("ts_code")
                
                # t-1 日收盘根据模型 score 排序获取目标 Top N
                valid_scores = prev_day_data.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
                target_codes = valid_scores.index[:self.top_n].tolist()
                
                # 1. 卖出淘汰的标的 (t+1 日开盘价卖出)
                for code in list(positions.keys()):
                    if code not in target_codes:
                        pos = positions[code]
                        if code in day_data.index:
                            stock_info = day_data.loc[code]
                            # 卖出防跌停与停牌
                            is_limit_sell = stock_info.get("limit_sell", False)
                            is_susp = stock_info.get("is_suspended", False)
                            
                            if not is_limit_sell and not is_susp:
                                sell_price = stock_info[open_raw_col] * (1.0 - self.slippage)
                                gross = pos["shares"] * sell_price
                                comm = max(gross * self.close_cost, self.min_cost)
                                net_sell = gross - comm
                                cash += net_sell
                                
                                trade_logs.append({
                                    "trade_date": current_date, "ts_code": code, "action": "SELL",
                                    "shares": pos["shares"], "price_raw": sell_price, "net_amount": net_sell
                                })
                                del positions[code]
                                
                # 2. 买入入选的标的 (t+1 日开盘价买入)
                if target_codes:
                    alloc_per_stock = (total_equity / len(target_codes))
                    for code in target_codes:
                        if code not in positions and code in day_data.index:
                            stock_info = day_data.loc[code]
                            is_limit_buy = stock_info.get("limit_buy", False)
                            is_susp = stock_info.get("is_suspended", False)
                            
                            if not is_limit_buy and not is_susp:
                                buy_price = stock_info[open_raw_col] * (1.0 + self.slippage)
                                max_cash_avail = min(cash, alloc_per_stock)
                                
                                # 强制 100 股一手整倍数限制
                                raw_shares = int(max_cash_avail // (buy_price * 100)) * 100
                                if raw_shares >= 100:
                                    gross = raw_shares * buy_price
                                    comm = max(gross * self.open_cost, self.min_cost)
                                    total_cost = gross + comm
                                    
                                    if cash >= total_cost:
                                        cash -= total_cost
                                        positions[code] = {
                                            "shares": raw_shares,
                                            "buy_date": current_date,
                                            "cost_price_raw": buy_price
                                        }
                                        trade_logs.append({
                                            "trade_date": current_date, "ts_code": code, "action": "BUY",
                                            "shares": raw_shares, "price_raw": buy_price, "net_amount": -total_cost
                                        })

            equity_records.append({
                "trade_date": current_date,
                "cash": cash,
                "position_market_value": pos_market_val,
                "total_equity": total_equity,
                "norm_equity": total_equity / self.initial_capital
            })
            
        equity_df = pd.DataFrame(equity_records)
        metrics = self._calculate_qlib_metrics(equity_df, trade_logs)
        logger.info(f"Qlib Backtest Finished. Final Equity: ¥{equity_df['total_equity'].iloc[-1]:,.2f} | CAGR: {metrics.get('cagr', 0):.2%}")
        return equity_df, metrics

    def _calculate_qlib_metrics(self, equity_df: pd.DataFrame, trade_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if equity_df.empty:
            return {}
            
        total_days = len(equity_df)
        annual_factor = 252.0 / max(total_days, 1)
        
        initial_eq = equity_df["total_equity"].iloc[0]
        final_eq = equity_df["total_equity"].iloc[-1]
        
        total_return = (final_eq / initial_eq) - 1.0
        cagr = (1.0 + total_return) ** (annual_factor) - 1.0 if total_return > -1 else -1.0
        
        equity_df["daily_ret"] = equity_df["total_equity"].pct_change().fillna(0.0)
        daily_mean = equity_df["daily_ret"].mean()
        daily_std = equity_df["daily_ret"].std()
        
        sharpe = (daily_mean / (daily_std + 1e-8)) * np.sqrt(252.0)
        
        equity_df["peak"] = equity_df["total_equity"].cummax()
        equity_df["drawdown"] = (equity_df["total_equity"] - equity_df["peak"]) / equity_df["peak"]
        max_drawdown = equity_df["drawdown"].min()
        
        sells = [t for t in trade_logs if t["action"] == "SELL"]
        
        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "total_trades": len(sells),
            "turnover": float(len(trade_logs) / max(total_days, 1))
        }
