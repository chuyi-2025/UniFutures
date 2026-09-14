# rule_edge 动态持仓 / 可变品种数

> ⚠️ **已过时**：本搜索未去掉多品种新闻，截面易被早评/商品指数稿污染。  
> 请改看 [RULE_DYNAMIC_SINGLE.md](./RULE_DYNAMIC_SINGLE.md)。

> 信号固定：`rule_edge | no_dianping | L7 | uniform`  
> 脚本：`news/code/train/rule_edge_dynamic_search.py`  
> 搜索 2880 组动态规则（允许空仓、1~4 品种可变）

---

## 1. 固定部分（不变）

| 项 | 规则 |
|----|------|
| 信号 | rule_edge = 0.5×标题 + 0.5×词表 |
| 研报 | no_dianping（去掉点评） |
| 窗口 | L=7 自然日，等权平均 |

---

## 2. 动态规则（两种模式）

### 2.1 滞回模式 hysteresis（推荐）

```
【开仓】截面 spread = max(score) − min(score) ≥ enter
【持仓】至少 min_hold 天
【平仓/空仓】min_hold 之后 spread < exit → 全平，等待下次开仓
【品种数】spread < spread_hi → 1 多 1 空；否则 2 多 2 空（最多 4 个）
```

### 2.2 日频模式 daily

```
【每天】spread ≥ enter → 按规则选品种；否则空仓
【无 min_hold】信号弱就当天不持仓
```

---

## 3. 推荐方案

### 方案 D1：滞回均衡（首选）

| 参数 | 值 |
|------|-----|
| enter / exit | **0.30 / 0.20** |
| min_hold | **7** 天 |
| spread_hi | **0.55**（低于此 1L1S，高于 2L2S） |
| abs_thr | 0（不过滤单品种强度） |

| 指标 | vs 固定 hold10 |
|------|----------------|
| Sharpe | **2.94** vs 2.25 |
| Calmar | **10.66** vs 4.55 |
| 回撤 | −5.6% vs −6.6% |
| 空仓占比 | **33%** vs 0% |
| 均品种数 | **2.5** vs 4.0 |
| 持仓中位 | 7 天 vs 10 天 |

**分年：** 2024 Sh 4.61 / 2025 Sh 1.97 / 2026 Sh 2.66（三年均正）

### 方案 D2：滞回保守

| 参数 | 值 |
|------|-----|
| enter / exit | **0.40 / 0.25** |
| min_hold | 5 天 |
| abs_thr | **0.15**（\|score\|≥0.15 才入选） |
| spread_hi | 0.45 |

空仓 **46%**，Sharpe 3.39，Calmar 9.76；信号更严、交易更少。

### 不推荐：daily enter0.4

Calmar 16+ 但 **70% 天空仓**、仅 184 有效天，过拟合风险高。

---

## 4. 与固定 hold10 对比

| | 固定 hold10 | **动态 D1** |
|---|------------|------------|
| 弱信号 | 仍持仓 | **空仓** |
| 品种数 | 固定 4 | **1~4 动态** |
| 换仓 | 每 10 天机械换 | spread 弱于 exit 才平 |
| 泛化 | 较稳 | 参数略多，需 OCR 长样本复验 |

---

## 5. 产出

`news/data/results/rule_dynamic/top_robust.csv`

```bash
python3 news/code/train/rule_edge_dynamic_search.py
```
