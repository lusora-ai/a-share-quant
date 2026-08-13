# Upstream References & Architectural Adaptations

本文档按项目规则记录 `lusora-ai/a-share-quant` 从成熟开源框架中借鉴、引用与适配的具体组件与设计。

我们遵循 **Reuse / Wrap / Adapt** 原则，绝不重复发明轮子或进行简单 Copy-Paste。

---

## 1. Upstream Source 1: `microsoft/qlib`

| 借鉴/引用 Qlib 组件 | 对应 GitHub URL / 文件路径 | 我方系统中的封装与适配路径 (`ashare-quant`) | 采用的设计与适配原因 |
| :--- | :--- | :--- | :--- |
| **Alpha158 因子集** | [`qlib/contrib/data/handler.py`](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py) | `src/ashare_quant/features/qlib_alpha158.py` | 采用 Qlib 经过充分验证的 158 个 Alpha 量价因子算子库，作为首选成熟上游特征 Benchmark |
| **DatasetH 机制** | [`qlib/data/dataset/__init__.py`](https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/__init__.py) | `src/ashare_quant/models/qlib_lgbm.py` | 采用 Qlib `DatasetH` 机制统一组织 `train` / `valid` / `test` 的特征与标签分流，保证 Early Stopping 生效 |
| **LGBModel** | [`qlib/contrib/model/gbdt.py`](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/gbdt.py) | `src/ashare_quant/models/qlib_lgbm.py` | 适配 Qlib 官方 GBDT 模型引擎，严格取消静默 Fallback，遇到环境问题 Fail Loudly 终止 |
| **Workflow Config** | [`examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml) | `configs/model_lgbm.yaml` | 参照 Qlib 标准实验配置定义 LightGBM 学习率、树深、叶子节点数与调参区间 |
| **Exchange 成交引擎** | [`qlib/backtest/exchange.py`](https://github.com/microsoft/qlib/blob/main/qlib/backtest/exchange.py) | `src/ashare_quant/backtest/qlib_engine.py` | 接入 Qlib 交易所成交引擎，处理中国 A 股 100 股一手整倍数、买卖佣金、印花税、最少佣金与涨跌停/停牌不可成交约束 |
| **TopkDropoutStrategy** | [`qlib/contrib/strategy/signal_strategy.py`](https://github.com/microsoft/qlib/blob/main/qlib/contrib/strategy/signal_strategy.py) | `src/ashare_quant/backtest/qlib_engine.py` | 采用标准的 TopK 选股加 Dropout 换仓策略，作为衡量预测 score 排序稳定性的基准组合策略 |
| **SimulatorExecutor** | [`qlib/backtest/executor.py`](https://github.com/microsoft/qlib/blob/main/qlib/backtest/executor.py) | `src/ashare_quant/backtest/qlib_engine.py` | 采用日频模拟执行器，严格约束 $t$ 日 15:00 收盘信号在 $t+1$ 日开盘价成交 |
| **SignalRecord & SigAnaRecord** | [`qlib/workflow/record_temp.py`](https://github.com/microsoft/qlib/blob/main/qlib/workflow/record_temp.py) | `src/ashare_quant/models/metrics.py` | 采用 Qlib 评估工具计算 RankIC、ICIR、RankICIR 以及 Long-Short 选股能力分析 |
| **PortAnaRecord** | [`qlib/contrib/evaluate.py`](https://github.com/microsoft/qlib/blob/main/qlib/contrib/evaluate.py) | `src/ashare_quant/backtest/qlib_engine.py` | 采用 Qlib 组合评估分析器生成标准化收益率曲线、CAGR、Sharpe、Max Drawdown 与周转率统计 |

---

## 2. Upstream Source 2: `microsoft/RD-Agent`

* **参考设计**：仅参考其 Quant Research 研究闭环工作流：`Hypothesis → Factor → Model → Qlib Evaluation`。
* **隔离边界**：目前系统暂不直接引入 RD-Agent 自动化 LLM 代码生成整体框架，仅在研究分析和文档归因中借鉴其假设驱动思想。

---

## 3. Upstream Source 3: `ricequant/rqalpha`

* **参考设计**：仅作为后期 A 股回测结果交叉验证的独立参考对象（不作为首选主回测框架）。
