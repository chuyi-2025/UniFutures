# 宏观因子 × 大宗期货

拉取世界宏观序列（美元、美债、加息压力代理、失业率、CPI、VIX），对齐国内期货主力收益，搜索**简单规则**（高 Sharpe、控回撤、看 IC/IR）。支持研究口径与 **300 万账户口径**（保证金 / 手续费 / 整数手）。

**方案全文**：[docs/MACRO_COMMODITY_STRATEGY.md](./docs/MACRO_COMMODITY_STRATEGY.md)  
**账户候选方案（A–F）**：[docs/MACRO_ACCOUNT_SCHEMES.md](./docs/MACRO_ACCOUNT_SCHEMES.md)

## 跑通

```bash
cd /home/workspace/lab/UniFutures
python3 macro/code/fetch_macro.py
python3 macro/code/build_macro_factors.py
python3 macro/code/align_prices.py
python3 macro/code/search_macro_strategies.py
# 账户口径（可加 --resume --skip-phase-a）
python3 macro/code/search_macro_account.py --minutes 75
```

## 定稿摘要

### 研究口径（AU/AG/SC 等权 book，无保证金）

| | AU | AG | SC | 等权 book |
|--|----|----|-----|-----------|
| 规则 | DXY z60 long_only | 同 AU（soft） | CPI YoY z60 sign | — |
| Sharpe | 0.83 | 0.50 | 0.98 | **≈1.10** |
| maxDD | −8.9% | −17% | −22% | **≈−12%** |

### 账户口径（300 万，落盘 = 方案 A）

| 方案 | 品种 | Sharpe | maxDD |
|------|------|--------|-------|
| A 美元广度→金（落盘） | AU | **1.32** | −3.7% |
| B 美元偏鹰→杂池 | BU,NR,M,AU | **1.34** | −3.1% |
| D 金银油 | AU,AG,SC | 1.19 | −25.8% |
| F CPI→原油 | SC | 1.13 | −5.4% |

数据：`macro/data/`；研究搜索：`macro/data/results/search/`；账户搜索：`macro/data/results/search_account/`。  
日线镜像：`data/infer/results/daily/linear/macro_account_best_daily.csv`。
