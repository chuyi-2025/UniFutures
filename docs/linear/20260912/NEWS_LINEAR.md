# 新闻因子线性模型实验

> 数据源：`lab/UniFutures/news` 研报 OCR 情绪（2024-03 ~ 2026-06，65 品种）  
> 回测窗：**2024-03-08 ~ 2026-09-08**（新闻可用区间）  
> 脚本：`news/code/train/build_news_factors.py` + `news/code/backtest/news_linear_eval.py`

---

## 1. 思路（尽量简单）

从已有 `signals_ft_full_ensemble.parquet`（finance_zh 三分类概率）构造 **3 个日频截面因子**，与价格因子 `pos64` 做截面多空（long top 20% / short bottom 20%）。

**无未来函数**：交易日 T 只用 `report_date ∈ [T−L, T)` 的研报（不含当天）。

| 因子 | 公式 | 含义 |
|------|------|------|
| **news_edge** | exp 加权 mean(prob_pos − prob_neg) | 软情绪方向 |
| **news_vote** | exp 加权 mean(position ∈ {−1,0,+1}) | 硬情绪方向 |
| **news_cnt** | log(1 + 窗口内研报数) | 关注度 / 覆盖 |

权重：指数衰减，半衰期 = L/2 自然日（与 lookback 实验 `wavg_exp` 一致）。

---

## 2. 实验方案

| 方案 | 说明 |
|------|------|
| pos64_baseline | 价格 64 日位置趋势（对照） |
| news_edge | 仅用 news_edge 截面 LS |
| news_vote | 仅用 news_vote |
| news_cnt | 仅用 news_cnt |
| pos64+news_edge | pos64 与 news_edge **等权 zscore 混合** |
| ridge_pos64_news | 季 walk-forward Ridge：pos64 + news_edge + news_cnt |

Lookback L ∈ {7, 14, 60} 自然日。

---

## 3. 回测结果

产出：`news/data/results/news_linear/summary.csv`

### 3.1 与 pos64 对照

| 方案 | L | Sharpe | 收益 | 天数 |
|------|---|--------|------|------|
| **pos64_baseline** | — | **0.95** | **+48.6%** | 542 |
| news_edge | 14 | 0.87 | +28.2% | 383 |
| pos64+news_edge | 14 | 0.92 | +35.7% | 383 |
| news_vote | 7 | 0.85 | +26.4% | 351 |
| news_edge | 7 | 0.47 | +11.9% | 351 |
| news_cnt | 7 | 0.56 | +15.7% | 351 |
| news_cnt | 60 | 0.89 | +28.9% | 429 |
| news_edge | 60 | **−0.44** | −12.6% | 429 |
| ridge_pos64_news | 7/14 | −0.80 / −1.60 | — | 样本短，不稳定 |

### 3.2 因子 IC（截面 Spearman，2024-03+）

| 因子 | L | 日均 IC | 有因子天数 |
|------|---|---------|-----------|
| pos_64 | — | **0.021** | 608 |
| news_cnt | 7 | 0.019 | 441 |
| news_vote | 7 | 0.004 | 441 |
| news_edge | 7 | 0.0005 | 441 |
| news_edge | 14 | ~0.01* | 383 |

\* L14 news_edge IC 略高于 L7，与 Sharpe 0.87 一致。

---

## 4. 结论

| 发现 | 说明 |
|------|------|
| **新闻单独弱于 pos64** | 最优 news_edge L14 Sharpe 0.87，仍低于 pos64 0.95 |
| **简单混合未增效** | pos64+news L14 Sharpe 0.92 < pos64 单独 |
| **L 窗口敏感** | L7 vote 好、L14 edge 好、L60 edge 转负 |
| **Ridge 三因子过拟合** | 样本短 + 新闻覆盖仅 24%，季 Ridge 不稳定 |
| **IC 整体偏低** | news_edge 近零 IC；news_cnt 略正（多研报≠多涨） |

**建议（保持简单）：**

1. **暂不纳入 daily 定稿** — 2024 以来新闻因子未稳定超越 pos64。  
2. 若继续探索：优先 **L14 news_edge**（Sharpe 0.87）或 **L60 news_cnt**（Sharpe 0.89，「研报多≠方向」）；加 **min_docs≥3** 过滤后再混合。  
3. 更长样本需等 OCR 回补 2020–2023 研报，或接 wire 新闻。

---

## 5. 与 news 既有实验的关系

| 已有线 | 与本实验区别 |
|--------|-------------|
| `backtest_sentiment.py` | 单品种 long/short/flat，非截面 LS |
| `build_lookback_signals.py` | 文档级 lookback，L60 concat 单品种 Sharpe 高 |
| **本实验** | **截面线性因子**，与 `code/experiment` 价格因子同一框架 |

Lookback 单品种最优（concat L60）median Sharpe ~0.69；截面 news_edge L14 Sharpe 0.87 — 截面聚合后反而更接近可用，但仍不如 pos64。

---

## 6. 运行

```bash
bash lab/UniFutures/news/code/run_news_linear.sh
# 或分步：
python3 news/code/train/build_news_factors.py --lookback 14
python3 news/code/backtest/news_linear_eval.py
```

**依赖数据：**

- `news/data/sentiment/signals_ft_full_ensemble.parquet`
- `news/data/sentiment/news_factors_L{7,14,60}.parquet`（生成）
- 价格面板：`code/experiment/linear_ridge_walkforward.build_panel`

---

## 7. 相关路径

```
news/
├── code/train/build_news_factors.py
├── code/backtest/news_linear_eval.py
├── code/run_news_linear.sh
├── data/sentiment/news_factors_L*.parquet
└── data/results/news_linear/
    ├── summary.csv
    ├── factor_ic.csv
    └── report.json
```

---

## 8. 非训练 / 简单线性方案（新增）

此前 `news_edge/vote/cnt` 依赖 **finance_zh 微调模型**。下面三层由简到繁：

| 层级 | 做法 | 是否训练 |
|------|------|----------|
| **A. 规则** | 标题关键词 + 正负面词表 | **否** |
| **B. 词袋 Ridge** | 汉字 2-gram 计数 + Ridge 回归次日收益 | **是（线性）** |
| **C. 微调 BERT** | 原 news_edge 路线 | 深度模型 |

### 8.1 规则因子（零训练）

| 因子 | 算法 |
|------|------|
| **rule_title** | 文件名/标题：「上涨点评」→+1，「下跌点评」→−1，「偏强/偏弱」→±0.6 |
| **rule_lex** | 正文词表：统计「上涨、去库、提涨…」vs「下跌、累库、承压…」，(多−空)/(多+空+1) |
| **rule_edge** | 0.5×title + 0.5×lex |
| **rule_cnt** | log(1+篇数) |

脚本：`news/code/train/build_rule_factors.py`（扫 1.7 万篇 OCR）

### 8.2 词袋 + Ridge（最简单线性训练）

- 特征：Top 300 汉字词频（`CountVectorizer`）
- 标签：次日主连 `log_ret_1d`
- 训练：2024-03 ~ 2025-06；预测全样本得 `bow_score`
- 脚本：`news/code/train/build_bow_ridge_factors.py`

### 8.3 规则方案回测（2024-03+）

| 因子 | L | Sharpe | 与 pos64 混合 Sharpe |
|------|---|--------|---------------------|
| pos64 | — | 0.95 | — |
| **rule_edge** | **7** | **1.99** | **1.86** |
| rule_lex | 7 | 1.70 | 1.74 |
| rule_title | 7 | 1.16 | 1.34 |
| rule_edge | 14 | 1.22 | 1.22 |
| rule_bow | 14 | −0.74 | −0.95 |

产出：`news/data/results/news_rule/summary.csv`

### 8.4 解读与注意

1. **规则线明显优于微调 emotion 线**（L7 rule_edge Sharpe 1.99 vs news_edge 0.47）——因为「上涨点评/下跌点评」标题和正文词表直接对应方向，不经过黑盒模型。
2. **潜在偏差**：大量研报是**事后点评**（今天涨完了才写「上涨点评」），因子可能在做**短期动量/情绪延续**，需样本外再验证。
3. **BoW Ridge 失败**：300 词线性回归过拟合/词表噪声大，不如规则透明。
4. **下一步若上策略**：优先 **rule_lex + rule_title**（可解释），加 **min_docs** 过滤；与 pos64 混合前做分年稳健性检查。

```bash
python3 news/code/train/build_rule_factors.py --lookback 7
python3 news/code/backtest/news_rule_eval.py
```

---

## 9. 规则因子挖掘（IC/IR 优化）

脚本：`news/code/train/rule_factor_mine.py`（~200 组合 × min_docs）  
详见 **[RULE_FACTOR_MINE.md](./RULE_FACTOR_MINE.md)**

| 因子 | IC | IR | Sharpe | 交易天 | 说明 |
|------|-----|-----|--------|--------|------|
| rule_edge L7 全报告（旧） | 0.028 | 0.124 | 1.99 | — | 基线 |
| **rule_edge\|strategy\|L3\|min3** | **0.070** | **0.273** | 2.76 | 75 | IC 最高，样本少 |
| **rule_edge\|strategy\|L7\|uniform\|min3** | **0.044** | **0.187** | 1.80 | 255 | **推荐均衡** |
| core_lex\|no_dianping\|L7\|min3 | 0.035 | 0.173 | 1.49 | 367 | 广覆盖 |

**关键改动**：仅 **策略月报/季报** + **min_docs≥3** + **短窗口 L3~7**。

产出：
- `news/data/sentiment/rule_factor_best.parquet` — 方案 A（peak IC）
- `news/data/sentiment/rule_factor_balanced.parquet` — 方案 B（均衡）
- `news/data/results/rule_mine/top_ic.csv` / `top_ir.csv`

---

## 10. 4 品种 / 持仓≥7 天（新约束）

旧方案 `strategy L3 min3` 日频 LS **持仓中位仅 1 天、品种 ~11 个**，不符合实盘约束。  
重筛见 **[RULE_HOLD4.md](./RULE_HOLD4.md)**：

| 方案 | 持仓中位 | 品种 | Calmar | Sharpe | IC |
|------|----------|------|--------|--------|-----|
| rule_edge no_dianping L7 hold10 | 10 天 | 4 | **4.55** | 2.25 | 0.024 |
| rule_lex strategy L7 hold14 min2 | 14 天 | 4 | 3.69 | 1.68 | 0.015 |

---

*新闻样本 2024-03 起，结论仅对该区间有效。*
