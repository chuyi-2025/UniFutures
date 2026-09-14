# 2026-09-13 多策略定稿

| 文档 | 内容 |
|------|------|
| **[STRATEGY_RANKING.md](./STRATEGY_RANKING.md)** | **六策略排序与指标对照**（夏普高→低） |
| **[STRATEGY_PACK.md](./STRATEGY_PACK.md)** | 量价 / 新闻 / 天气 / 宏观 / 组合：实现说明 + **总指标大表** + **分年指标大表** |
| [calendar/CALENDAR_ROLL_STRATEGY.md](../../../calendar/docs/CALENDAR_ROLL_STRATEGY.md) | **交割月 / 换月** 日历策略（非天气；第五类卫星） |
| [spread/SPREAD_STRATEGY.md](../../../spread/docs/SPREAD_STRATEGY.md) | 价差 & carry；**定稿 carry4**（HC/AU/Y/P） |
| `carry4_daily.csv` | Carry4 组合日线（保证金/手数/点位） |
| `metrics_yearly_all.csv` | 全部分年明细（机器可读） |
| `metrics_summary.csv` | 总区间摘要 |
| `metrics_yearly_overlap_long.csv` | 对齐窗长表 |
| `combo_v3_daily.csv` | 组合 v3 对齐窗日线 |
| `combo_full_daily.csv` | 组合 v3 **全历史**日线（2010→2026） |
| `combo_full_yearly.csv` | 组合全历史分年 |
| `metrics_yearly_full_combo.csv` | 全历史各袖套分年 |
| `meta.json` | 本金与假设 |

**开平仓归因表**（类似 `final_scheme_*_daily`）：[`data/infer/results/daily/combo_v3/`](../../../data/infer/results/daily/combo_v3/)

**定稿摘要**

- 量价：`final_scheme`（100 万，BOOK≥1.18）  
- 新闻：Dual-Gated（仅 Core 有仓时叠加）  
- 天气：**v3**（苹果/棉花/白糖/红枣，Phase3）  
- 宏观：方案 A（`usd_broad_z60`→AU，300 万）  
- 组合：Dual + 宏观降权门控 + 天气 v3 代理  
  - 对齐窗 Sharpe ≈ **3.49**（权益 300 万）  
  - **全历史** Sharpe ≈ **2.00**，累计盈亏约 **143 万**（47.6% / 300 万）
