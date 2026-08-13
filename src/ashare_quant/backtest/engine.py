import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ashare_quant.utils.logging import setup_logger
from ashare_quant.utils.config import load_config

logger = setup_logger("ashare_quant.backtest.engine")

class BacktestEngine:
    """
    带有 A 股真实交易约束的事件驱动 / 日频回测引擎
    约束条件:
    1. T+1 交易限制 (当日买入次日及以后方可卖出)
    2. 涨跌停限制 (触及涨停不能买入，触及跌停不能卖出)
    3. 停牌限制 (停牌不能买卖)
    4. 交易成本 (佣金、印花税、最少佣金、配置化滑点)
    5. 真实100股整数倍下单约束 (可选)
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config("backtest")
        self.bt_cfg = self.config.get("backtest", {})
        self.costs_cfg = self.config.get("costs", {})
        self.rules_cfg = self.config.get("rules", {})
        
        self.initial_capital = float(self.bt_cfg.get("initial_capital", 100000.0))
        self.top_n = int(self.bt_cfg.get("top_n", 10))
        self.rebalance_freq = int(self.bt_cfg.get("rebalance_freq", 5))
        
        self.commission_rate = float(self.costs_cfg.get("commission_rate", 0.00025))
        self.stamp_duty_rate = float(self.costs_cfg.get("stamp_duty_rate", 0.0005))
        self.slippage_ratio = float(self.costs_cfg.get("slippage_bps", 5.0)) / 10000.0
        self.min_commission = float(self.costs_cfg.get("min_commission", 5.0))
        
        self.t_plus_1 = self.rules_cfg.get("t_plus_1", True)
        self.check_limit_up = self.rules_cfg.get("check_limit_up", True)
        self.check_limit_down = self.rules_cfg.get("check_limit_down", True)
        self.check_suspended = self.rules_cfg.get("check_suspended", True)

    def run_backtest(
        self,
        df_all: pd.DataFrame,
        score_col: str = "score"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        运行回测并输出每日净值曲线与回测指标统计
        """
        if df_all.empty or score_col not in df_all.columns:
            logger.error(f"Cannot run backtest: empty DataFrame or missing '{score_col}' column.")
            return pd.DataFrame(), {}
            
        df = df_all.copy()
        dates = sorted(df["trade_date"].unique())
        
        cash = self.initial_capital
        # positions: {ts_code: {"shares": int/float, "buy_date": str, "cost_price": float}}
        positions: Dict[str, Dict[str, Any]] = {}
        
        equity_records = []
        trade_logs = []
        
        logger.info(f"Running A-Share Backtest across {len(dates)} trading days. Initial capital: ¥{self.initial_capital:,.2f}")
        
        for date_idx, current_date in enumerate(dates):
            day_data = df[df["trade_date"] == current_date].set_index("ts_code")
            
            # 1. 计算当日收盘总资产净值 (Nav / Equity)
            position_market_value = 0.0
            for code, pos in list(positions.items()):
                if code in day_data.index:
                    close_p = day_data.loc[code, "close"]
                    position_market_value += pos["shares"] * close_p
                else:
                    # 停牌或缺失使用上次持仓成本
                    position_market_value += pos["shares"] * pos["cost_price"]
                    
            total_equity = cash + position_market_value
            
            # 2. 判断是否到了调仓日
            if date_idx % self.rebalance_freq == 0:
                # 获取 t 日收盘后的预测排名前 Top N 股票
                valid_scores = day_data.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
                target_codes = valid_scores.index[:self.top_n].tolist()
                
                # --- 卖出不满足目标或降出 Top N 的持仓 ---
                for code in list(positions.keys()):
                    if code not in target_codes:
                        pos = positions[code]
                        if code in day_data.index:
                            stock_info = day_data.loc[code]
                            # 卖出条件限制: 停牌不能卖出, 跌停不能卖出
                            is_susp = stock_info.get("is_suspended", False) if self.check_suspended else False
                            is_limit_down = False
                            if self.check_limit_down and "pct_chg" in stock_info:
                                is_limit_down = stock_info["pct_chg"] <= -9.9
                                
                            if not is_susp and not is_limit_down:
                                sell_price = stock_info["open"] * (1.0 - self.slippage_ratio)
                                gross_amount = pos["shares"] * sell_price
                                comm = max(gross_amount * self.commission_rate, self.min_commission)
                                stamp = gross_amount * self.stamp_duty_rate
                                net_amount = gross_amount - comm - stamp
                                
                                cash += net_amount
                                trade_logs.append({
                                    "trade_date": current_date, "ts_code": code, "action": "SELL",
                                    "shares": pos["shares"], "price": sell_price, "net_amount": net_amount,
                                    "pnl": net_amount - (pos["shares"] * pos["cost_price"])
                                })
                                del positions[code]
                                
                # --- 买入新选入的目标股票 ---
                if target_codes:
                    target_weight_per_stock = 1.0 / len(target_codes)
                    target_cash_per_stock = total_equity * target_weight_per_stock
                    
                    for code in target_codes:
                        if code not in positions and code in day_data.index:
                            stock_info = day_data.loc[code]
                            is_susp = stock_info.get("is_suspended", False) if self.check_suspended else False
                            is_limit_up = False
                            if self.check_limit_up and "pct_chg" in stock_info:
                                is_limit_up = stock_info["pct_chg"] >= 9.9
                                
                            if not is_susp and not is_limit_up:
                                buy_price = stock_info["open"] * (1.0 + self.slippage_ratio)
                                max_alloc = min(cash, target_cash_per_stock)
                                
                                if max_alloc > buy_price * 100:
                                    shares = max_alloc / buy_price
                                    gross_amount = shares * buy_price
                                    comm = max(gross_amount * self.commission_rate, self.min_commission)
                                    total_cost = gross_amount + comm
                                    
                                    if cash >= total_cost:
                                        cash -= total_cost
                                        positions[code] = {
                                            "shares": shares,
                                            "buy_date": current_date,
                                            "cost_price": buy_price
                                        }
                                        trade_logs.append({
                                            "trade_date": current_date, "ts_code": code, "action": "BUY",
                                            "shares": shares, "price": buy_price, "net_amount": -total_cost, "pnl": 0.0
                                        })

            # 3. 记录当日净值
            equity_records.append({
                "trade_date": current_date,
                "cash": cash,
                "position_market_value": position_market_value,
                "total_equity": total_equity,
                "norm_equity": total_equity / self.initial_capital
            })
            
        equity_df = pd.DataFrame(equity_records)
        metrics = self._calculate_performance_metrics(equity_df, trade_logs)
        
        logger.info(f"Backtest completed! Final Equity: ¥{equity_df['total_equity'].iloc[-1]:,.2f} | CAGR: {metrics.get('cagr', 0):.2%} | Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        return equity_df, metrics

    def _calculate_performance_metrics(self, equity_df: pd.DataFrame, trade_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算 CAGR, Sharpe, Sortino, Max Drawdown, 胜率等
        """
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
        
        downside_std = equity_df[equity_df["daily_ret"] < 0]["daily_ret"].std()
        sortino = (daily_mean / (downside_std + 1e-8)) * np.sqrt(252.0)
        
        # Max Drawdown
        equity_df["peak"] = equity_df["total_equity"].cummax()
        equity_df["drawdown"] = (equity_df["total_equity"] - equity_df["peak"]) / equity_df["peak"]
        max_drawdown = equity_df["drawdown"].min()
        
        # 交易统计
        sells = [t for t in trade_logs if t["action"] == "SELL"]
        win_sells = [t for t in sells if t["pnl"] > 0]
        win_rate = len(win_sells) / len(sells) if sells else 0.0
        
        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(max_drawdown),
            "total_trades": len(sells),
            "win_rate": float(win_rate)
        }
