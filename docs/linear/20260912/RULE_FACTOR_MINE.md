# 规则新闻因子挖掘结果

> 脚本：`news/code/train/rule_factor_mine.py`  
> 搜索 **200+** 组合（报告类型 × 词表 × L × 权重 × min_docs）  
> 样本：2024-03-08 ~ 2026-09，截面 LS 20%

---

## 1. 新增规则特征（相对 rule_edge）

| 特征 | 含义 |
|------|------|
| **fwd_lex** | 「核心观点」里**前瞻词**（有望、预计、供需趋紧…）− 看空词 |
| **core_lex** | 仅 **## 核心观点** / 结论段落的正负面词表 |
| **sd_lex** | 需求利好词 − 供应增加词 |
| **kind 过滤** | `strategy` 策略月报/季报；`no_dianping` 去掉点评；`daily`/`weekly` |

---

## 2. IC/IR 排行榜（Top）

### 2.1 按 IR 排序（稳定性）

| 因子 | IC | IR | ic_t | Sharpe | 交易天 | 说明 |
|------|-----|-----|------|--------|--------|------|
| **rule_edge\|strategy\|L3\|exp\|min3** | **0.070** | **0.273** | 2.36 | 2.76 | 75 | IC 最高，样本少 |
| rule_edge\|strategy\|L3\|uniform\|min3 | 0.066 | 0.260 | 2.25 | 1.94 | 75 | |
| rule_lex\|strategy\|L3\|exp\|min3 | 0.065 | 0.251 | 2.17 | 1.50 | 75 | |
| **rule_edge\|strategy\|L7\|uniform\|min3** | **0.044** | **0.187** | 2.98 | **1.80** | **255** | **推荐：IC/IR 与样本均衡** |
| core_lex\|no_dianping\|L7\|exp\|min3 | 0.035 | 0.173 | 3.31 | 1.49 | 367 | 覆盖更广 |
| rule_lex\|daily\|L3\|uniform | 0.030 | 0.170 | 2.85 | 1.31 | 304 | 日报词表 |

### 2.2 与旧因子对比

| 因子 | IC | IR | 评价 |
|------|-----|-----|------|
| rule_edge L7（全报告） | 0.028 | 0.124 | 一般 |
| rule_lex L7 | 0.029 | 0.133 | 一般 |
| **rule_edge strategy L3 min3** | **0.070** | **0.273** | IC 提升 **2.5×** |
| **rule_edge strategy L7 min3** | **0.044** | **0.187** | IR 提升 **1.5×** |
| pos64 | 0.021 | 0.086 | 价格对照 |

---

## 3. 推荐定稿（纯规则）

### 方案 A：最高 IC（研究/稀疏）

```
【文档】仅 strategy（策略报告/月报/季报/展望）
【信号】rule_edge = 0.5×标题规则 + 0.5×全文词表
【窗口】L=3 自然日，exp 衰减
【过滤】窗口内至少 3 篇研报，否则不交易
```

| 指标 | 数值 |
|------|------|
| IC | **0.070** |
| IR | **0.273** |
| Sharpe | 2.76 |
| 交易天 | 75（覆盖偏少） |

### 方案 B：均衡（推荐跟进）

```
【文档】strategy
【信号】rule_edge
【窗口】L=7，uniform 等权
【过滤】min_docs ≥ 3
```

| 指标 | 数值 |
|------|------|
| IC | **0.044** |
| IR | **0.187** |
| Sharpe | **1.80** |
| 收益 | +32.7% |
| 交易天 | 255 |

### 方案 C：广覆盖

```
【文档】no_dianping（去掉点评）
【信号】core_lex（仅核心观点段词表）
【窗口】L=30，exp
【过滤】无 min_docs
```

| 指标 | 数值 |
|------|------|
| IC | 0.034 |
| IR | 0.157 |
| Sharpe | 1.65 |
| 交易天 | 466 |

---

## 4. 分年（方案 A：strategy L3 min3）

| 年 | 收益 | Sharpe | 天数 |
|----|------|--------|------|
| 2024 | +4.8% | 9.36 | 14 |
| 2025 | +12.0% | 3.71 | 56 |
| 2026 | −4.4% | −11.7 | **5** |

2026 样本极少，方案 A 不稳定；**方案 B 更可靠**。

---

## 5. 关键结论

1. **策略月报 >> 点评/日报**：`strategy` 过滤是 IC 提升最大单一改动。  
2. **短窗口 L3~7** 优于 L30（对策略报告）。  
3. **min_docs≥3** 显著抬 IC/IR（共识过滤）。  
4. **fwd_lex / core_lex** 单独未超过 rule_edge+strategy，但 **core_lex+no_dianping** 覆盖最好。  
5. 最好 IC **0.07** 仍低于机构「强因子」0.05+ 的上沿，但已进入 **可用区间**；IR **0.27** 仍低于 0.5 稳定线。

---

## 6. 文件

| 路径 | 说明 |
|------|------|
| `news/data/sentiment/rule_docs_rich.parquet` | 文档级丰富特征 |
| `news/data/sentiment/rule_factor_best.parquet` | 方案 A 日频因子 |
| `news/data/results/rule_mine/top_ic.csv` | IC 排行 |
| `news/data/results/rule_mine/top_ir.csv` | IR 排行 |

```bash
python3 news/code/train/rule_factor_mine.py
```

---

*下一步：方案 B 接入 1 手/4 品种 blotter，与 pos64 混合回测。*
