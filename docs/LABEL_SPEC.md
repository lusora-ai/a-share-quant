# 唯一正式标签定义规范 (Label Specification)

为彻底消除模糊定义与未来函数（Look-ahead bias），本文档规范 `a-share-quant` 系统中唯一正式生产标签的计算逻辑与时间语义。

---

## 1. 交易时间节点定义 (Time Semantics)

1. **信号生成时点 ($t$ 日 15:00 收盘)**：
   - 仅能使用截至 $t$ 日 15:00 收盘时**已经确定且已知**的数据（包括历史行情、历史财务及 $t$ 日收盘价）。
   - 特征矩阵 $X_t$ 中**严禁包含任何 $t$ 日收盘以后的数据**，严禁包含 `raw_label` 或 `rank_label`。

2. **订单执行时点 ($t+1$ 日 09:30 开盘)**：
   - 在 $t$ 日 15:00 得到的模型打分预测与选股信号 `signal_t`，**必须在 $t+1$ 交易日开盘价 `open_{t+1}` 执行成交**（并加入配置化滑点）。

3. **订单平仓时点 ($t+6$ 日 09:30 开盘 / $t+5$ 日 15:00 收盘)**：
   - 目标持仓周期为 5 个交易日（在第 5 个交易日结束时平仓）。

---

## 2. 唯一正式标签数学定义

定义 `forward_5d_exec_return` 为实际可执行收益率：

$$\text{exec\_return}(i, t) = \frac{\text{close}_{i, t+5}}{\text{open}_{i, t+1}} - 1.0$$

定义基准收益率（如沪深300指数）：

$$\text{benchmark\_return}(t) = \frac{\text{benchmark\_close}_{t+5}}{\text{benchmark\_open}_{t+1}} - 1.0$$

超额收益率 $\text{excess\_return}(i, t)$：

$$\text{excess\_return}(i, t) = \text{exec\_return}(i, t) - \text{benchmark\_return}(t)$$

每日横截面 Percentile Rank 标签 $\text{rank\_label\_5d}(i, t)$：

$$\text{rank\_label\_5d}(i, t) = \text{CrossSectionalPercentile}(\text{excess\_return}(\cdot, t))$$

数值范围严格归一化至 $[0.0, 1.0]$。

---

## 3. 标签与隔离保护约束 (P0 Requirements)

- **训练与评估限定**：`rank_label_5d` **只能用于模型训练 (y) 和验证集评估**，绝不可作为特征进入特征列表。
- **Purge 隔离期要求**：在滚动窗口 (Walk-Forward) 中，当 Train 集合结束于日期 $T_{train}$ 时，因为 $T_{train}$ 的标签使用了未来 $T_{train}+1 \rightarrow T_{train}+5$ 的价格，必须消除后续 5 个交易日的样本（Purge），保证：
  $$\max(\text{label\_info\_time}(\text{Train})) < \min(\text{feature\_time}(\text{Valid}))$$
