# rule_edge 单品种新闻重搜（去多品种挂载）

> 脚本：`news/code/train/rule_edge_single_sym_search.py`  
> 产出：`news/data/results/rule_dynamic_single/`  
> 过滤：同一 `path` 映射到 **多于 1 个 symbol** 的报告全部丢弃（早评合集、商品指数、席位数据等）

---

## 1. 为什么改

旧 D1 用全量映射时，一篇「南华商品指数：所有板块均上涨 / 下跌」可挂到大量品种，隔日正负翻转会把截面 spread 抬得很高，但**不是可交易的多空逻辑**。多品种新闻是主要污染源。

过滤后：约 **8359 / 86126** 行、**66** 个品种（路径唯一）。

---

## 2. Stage A：固定 2L2S sticky hold

| 信号 | hold | Sharpe | Calmar | 回撤 |
|------|------|--------|--------|------|
| **rule_edge \| no_dianping \| L30** | 20 | **2.45** | **3.71** | −12.2% |
| core_lex \| strategy \| L7 | 10 | 1.80 | 3.61 | −5.4% |
| core_lex \| weekly \| L30 | 14 | 1.80 | 3.48 | −8.9% |
| rule_edge \| no_dianping \| L7（旧信号） | — | 明显弱于 L30 | — | — |

结论：去多品种后，**L30 优于旧 L7**。

---

## 3. Stage B：动态滞回 / 日频（2880×4 信号）

### 方案 S1（首选）

| 项 | 值 |
|----|-----|
| 信号 | **rule_edge \| no_dianping \| L30 \| uniform \| single** |
| 模式 | hysteresis |
| enter / exit | **0.40 / 0.10**（exit 在 0.10~0.25 结果接近，推荐记 0.10） |
| min_hold | **5** |
| spread_hi | **0.55**（低→1L1S，高→2L2S） |
| abs_thr | 0 |

| 指标 | S1 | 旧 D1（多品种污染，不可比） |
|------|----|---------------------------|
| Sharpe | **2.86** | 2.94（虚高风险） |
| Calmar | **9.73** | 10.66 |
| 回撤 | **−4.9%** | −5.6% |
| 空仓 | **27.5%** | 33% |
| 有效天 | 441 | — |

分年 Sharpe（同设定）：**2024 3.16 / 2025 2.55 / 2026 3.37**（三年均正）。

### 同信号下旧 D1 参数（single 后）

`enter0.3/exit0.2/hold7` 在 **L30-single** 上仍可用，但 Calmar/空仓结构略逊于 S1。  
同一过滤下 **L7-single + 旧 D1** 掉到 Sharpe ~1.5、Calmar ~2.9 —— 旧方案依赖多品种噪声。

### 备选

| 方案 | 信号 | 动态要点 | Sharpe | Calmar | 备注 |
|------|------|----------|--------|--------|------|
| S2 | core_lex strategy L7 | enter0.6 hold14 abs0.15 | 2.55 | 5.70 | 更严、样本偏策略报告 |
| sticky | rule_edge L30 | hold20 固定换仓 | 2.45 | 3.71 | 无空仓逻辑，更简单 |

不推荐：daily 高 Calmar + 极高空仓（过拟合形态仍在）。

---

## 4. 产出文件

```
news/data/results/rule_dynamic_single/
  hold4_all.csv / hold4_filtered.csv
  dynamic_all.csv / dynamic_ranked.csv / dynamic_robust.csv
  report.json
```

```bash
python3 news/code/train/rule_edge_single_sym_search.py
# 复用 Stage A：
SKIP_STAGE_A=1 python3 news/code/train/rule_edge_single_sym_search.py
```

辅助：`rule_factor_mine.keep_single_symbol_docs()`。
