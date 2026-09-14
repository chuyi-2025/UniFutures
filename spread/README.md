# 价差 / 期限结构（Spread & Carry）

跨品种价差（螺纹-热卷、豆棕、金银比）与近远月 carry，数据来自 Tree-Stock `all_contracts`。

**文档**：[docs/SPREAD_STRATEGY.md](./docs/SPREAD_STRATEGY.md)

## 跑通

```bash
cd /home/workspace/lab/UniFutures

# 1. 拉数据 → 构建面板（默认 2014+）
python3 spread/code/build_panels.py --start 2014-01-01

# 2. 规则搜索（2018+ 回测窗，100 万账户）
python3 spread/code/search_spread_strategies.py
```

## 产出

| 路径 | 内容 |
|------|------|
| `spread/data/panels/*.parquet` | 各价差/ carry 日面板 |
| `spread/data/panels/all_spreads.parquet` | 全量合并 |
| `spread/data/panels/manifest.json` | 数据摘要 |
| `spread/data/results/search/spread_search_best.csv` | 各 spread 最优规则 |
| `spread/data/results/search/spread_search_pass.csv` | Sharpe≥0.8 且 DD>-15% |

## 数据源

```
/home/workspace/lab/Tree-Stock/futures/data/all_contracts/{SYMBOL}/{SYMBOL}{MM}.csv
```

- **跨品种**：主力对主力（`pick_main_fast`）
- **期限结构**：同交易日按到期排序，rank1=近月，rank2=远月
