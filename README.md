# A股个人量化研究与选股系统 (`ashare-quant` V1.0)

面向个人散户的**“研究优先、半自动执行”** A 股日频量化研究与选股系统。

> **核心原则**：先证明“研究流程正确”，再谈收益。V1 的成功标准不是赚多少钱，而是建立无未来函数、可复现、可解释、可回测、可人工执行的量化闭环。

---

## 一、 系统架构

```text
                 Gemini 3.6 Flash
           （开发 / Debug / 实验分析）
                        │
                        ▼
 ┌──────────────────────────────────────────────┐
 │             a-share-quant Engine             │
 │                                              │
 │  1. Data Layer (AkShare/Baostock → Parquet)  │
 │                      ↓                       │
 │  2. Universe Filter (ST / 新股 / 停牌过滤)    │
 │                      ↓                       │
 │  3. Factor Engine (12个基线因子)              │
 │                      ↓                       │
 │  4. Label Engine (未来5日相对基准超额Rank)   │
 │                      ↓                       │
 │  5. Models (Equal-Weight & LightGBM Rank)    │
 │                      ↓                       │
 │  6. Backtest Engine (T+1 / 涨跌停 / 停牌)     │
 │                      ↓                       │
 │  7. Walk-Forward Evaluator (滚动窗口验证)     │
 │                      ↓                       │
 │  8. Daily Report (HTML/MD 日报, Top10+Top3)   │
 └──────────────────────┬───────────────────────┘
                        │
                        ▼
               用户查看候选与风险信息
                        │
                        ▼
---

## 二、 架构实现状态 (Architecture Status)

| 核心组件 (Component) | 具体实现路径 (Implementation) | 集成状态 (Status) | 权威上游参考 (Upstream Reference) |
| :--- | :--- | :--- | :--- |
| **Qlib Alpha158 因子集** | `qlib.contrib.data.handler.Alpha158` / `Alpha158DL` | ✅ **REAL QLIB CALL** | [microsoft/qlib](https://github.com/microsoft/qlib) |
| **Qlib LGBModel 模型** | `qlib.contrib.model.gbdt.LGBModel` + `DatasetH` | ✅ **REAL QLIB CALL** | [microsoft/qlib](https://github.com/microsoft/qlib) |
| **Qlib 回测引擎** | `TopkDropoutStrategy` + `SimulatorExecutor` | ✅ **REAL QLIB CALL** | [microsoft/qlib](https://github.com/microsoft/qlib) |
| **Qlib 评估指标** | `qlib.contrib.evaluate.risk_analysis` / `PortAnaRecord` | ✅ **REAL QLIB CALL** | [microsoft/qlib](https://github.com/microsoft/qlib) |
| **PIT 动态成分股票池** | `PointInTimeMaster` (CSI300 历史有效成分过滤) | ✅ **PRODUCTION** | Point-in-Time Master |
| **12 个基线量价因子** | `src/ashare_quant/features/custom12.py` | ✅ **IN-HOUSE / ADAPTED** | PIT 截面多因子库 |
| **Purged Walk-Forward** | `PurgedWalkForwardEvaluator` (严格 5 日 Purge 隔离) | ✅ **PRODUCTION** | De Prado AFML 标准 |
| **Legacy 简化回测** | `src/ashare_quant/backtest/legacy_backtest.py` | ❌ **PERMANENTLY DISABLED** | 已由 Qlib 官方回测内核全面接管 |

> **说明与约束边界**：
> 1. **涨跌停限制模式**：当前回测处于 `benchmark_mode: uniform_limit_threshold` (统一以 9.9% 涨跌停逼近 CSI300 主板标的约束)。后续接入全 A 股 PIT 精细化行情数据时将升级为逐股票历史状态表达式 (`limit_buy` / `limit_sell`)。在此之前系统不声称已完整覆盖创业板/科创板 20% 及 ST 5% 的全部复杂动态涨跌停。
> 2. **基准代码**：统一采用 Qlib Canonical Symbol（如 `SH000300`）。

---

## 三、 项目目录结构

```text
a-share-quant/
├── README.md                  # 开发与使用说明
├── pyproject.toml              # 包配置与依赖项
├── .env.example                # 环境变量模版
├── configs/                    # 系统、数据、因子、模型、回测配置文件
├── data/                       # 本地存储 (raw/ processed/ cache/)
├── src/ashare_quant/           # 核心源代码包
├── tests/                      # pytest 自动化测试套件
├── scripts/                    # 运行脚本
├── experiments/                # 实验运行元数据与指标归档
└── reports/                    # 选股日报与评估报告输出
```

---

## 三、 快速开始

### 1. 安装项目 (开发模式)

```powershell
pip install -e .[dev]
```

### 2. 命令行使用指南

```powershell
# 1. 增量更新行情与主数据
python -m ashare_quant.cli update-data --start-date 2018-01-01

# 2. 训练 LightGBM 模型并执行 Walk-forward 评估
python -m ashare_quant.cli train --config configs/model_lgbm.yaml

# 3. 运行带有 A 股现实约束的策略回测
python -m ashare_quant.cli backtest --experiment 20260813_001_momentum_lgbm

# 4. 生成今日研究候选 Top 10 清单
python -m ashare_quant.cli daily-signal

# 5. 查看选股日报
python -m ashare_quant.cli report --latest
```

---

## 四、 自动化测试

运行 pytest 单元测试套件：

```powershell
pytest tests/ -v
```

---

## 五、 开发里程碑 (Milestones)

- [x] **M0: 脚手架与项目架构** (配置、CLI、日志、单元测试、README)
- [ ] **M1: 数据采集层与 QA 校验** (AkShare/Baostock 适配、Parquet/DuckDB 存取)
- [ ] **M2: 股票池与 12 个基线因子** (ST/停牌/新股过滤，截面 Winsorize/Standardize)
- [ ] **M3: 标签生成与基线模型** (5日超额收益 Rank Label，等权与 Ridge 线性基线)
- [ ] **M4: LightGBM 选股模型** (横截面 Rank 回归，特征重要性，实验追踪)
- [ ] **M5: 带有 A 股现实约束的回测引擎** (T+1、涨跌停、停牌、佣金、滑点)
- [ ] **M6: Walk-Forward 滚动窗口验证** (滚动训练/验证/测试切分，无测试集调参)
- [ ] **M7: 每日报告与因子贡献解释** (一键 HTML/Markdown 选股日报)
- [ ] **M8: 真实交易日志与复盘追踪** (300元真实小额体验交易追踪)
