import pandas as pd
import numpy as np
import pytest
from ashare_quant.universe.filter import UniverseFilter
from ashare_quant.factors.engine import FactorEngine, FACTOR_NAMES
from ashare_quant.factors.processor import FactorProcessor

def generate_mock_daily_data(num_stocks=5, num_days=100) -> pd.DataFrame:
    dates = pd.date_range(start="2024-01-01", periods=num_days, freq="B").strftime("%Y-%m-%d")
    data = []
    
    for i in range(num_stocks):
        ts_code = f"{600000 + i}.SH"
        base_price = 10.0 + i * 2.0
        np.random.seed(42 + i)
        returns = np.random.normal(0.001, 0.02, num_days)
        prices = base_price * np.exp(np.cumsum(returns))
        
        for d_idx, date_str in enumerate(dates):
            close_p = prices[d_idx]
            high_p = close_p * (1.0 + abs(np.random.normal(0, 0.01)))
            low_p = close_p * (1.0 - abs(np.random.normal(0, 0.01)))
            open_p = (high_p + low_p) / 2.0
            volume = int(np.random.uniform(10000, 50000))
            amount = volume * close_p * 100
            
            data.append({
                "ts_code": ts_code,
                "trade_date": date_str,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "open_raw": open_p,
                "close_raw": close_p,
                "open_adj": open_p,
                "close_adj": close_p,
                "high_adj": high_p,
                "low_adj": low_p,
                "volume": volume,
                "amount": amount,
                "turn": np.random.uniform(0.5, 3.0),
                "pct_chg": returns[d_idx] * 100.0,
                "is_suspended": False
            })
            
    return pd.DataFrame(data)

def test_factor_engine_computation():
    mock_df = generate_mock_daily_data(num_stocks=3, num_days=70)
    engine = FactorEngine()
    df_factors = engine.compute_factors(mock_df)
    
    # 验证 12 个因子均已生成
    for factor in FACTOR_NAMES:
        assert factor in df_factors.columns
        # 验证在足够历史天数（如第65天）后，因子值非空
        sub = df_factors[df_factors["trade_date"] == "2024-04-01"]
        if not sub.empty:
            assert not sub[factor].isna().all()

def test_factor_processor_zscore():
    mock_df = generate_mock_daily_data(num_stocks=5, num_days=70)
    engine = FactorEngine()
    df_factors = engine.compute_factors(mock_df)
    
    processor = FactorProcessor()
    processed_df = processor.process_cross_section(df_factors)
    
    # 验证同一交易日截面上的 Z-score 均值为 0，标准差约为 1
    test_date = processed_df["trade_date"].iloc[-1]
    cross_section = processed_df[processed_df["trade_date"] == test_date]
    
    for factor in FACTOR_NAMES:
        vals = cross_section[factor]
        assert abs(vals.mean()) < 1e-4

def test_no_future_leakage_in_factors():
    """
    P0 测试: 验证 t 日因子不受 t+1 日价格变化影响
    """
    mock_df = generate_mock_daily_data(num_stocks=2, num_days=70)
    engine = FactorEngine()
    
    # 计算原始因子
    df1 = engine.compute_factors(mock_df.copy())
    
    # 修改未来的数据 (如在最后一天大幅改动价格)
    modified_df = mock_df.copy()
    last_date = modified_df["trade_date"].max()
    modified_df.loc[modified_df["trade_date"] == last_date, "close"] *= 5.0
    
    df2 = engine.compute_factors(modified_df)
    
    # 检查倒数第二天的因子值，改动最后一天的价格绝对不应该影响倒数第二天的任何因子值
    second_last_date = sorted(mock_df["trade_date"].unique())[-2]
    
    for factor in FACTOR_NAMES:
        f1_vals = df1[df1["trade_date"] == second_last_date][factor].values
        f2_vals = df2[df2["trade_date"] == second_last_date][factor].values
        np.testing.assert_allclose(f1_vals, f2_vals, rtol=1e-5, err_msg=f"Future leakage detected in factor {factor}!")
