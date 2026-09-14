# 宏观账户口径候选方案（300 万）

本文档汇总账户约束下搜索得到的主要规则方案。全文背景与流水线见 [MACRO_COMMODITY_STRATEGY.md](./MACRO_COMMODITY_STRATEGY.md) §5–§6。

**约束**：启动资金 300 万；券商保证金；手续费；整数手；每日品种上限；可选单腿保证金帽。  
**性质**：规则参数搜索，**无模型训练**。  
**区间**：约 2018-01 → 2026-09。  
**结果目录**：`macro/data/results/search_account/`

---

## 方案一览

| ID | 名称 | 品种 | Sharpe | maxDD | 落盘 |
|----|------|------|--------|-------|------|
| A | 美元广度 → 黄金 | AU | **1.318** | −3.7% | **是（quality 最优）** |
| B | 美元偏鹰 → 杂池 | BU, NR, M, AU | **1.337** | −3.1% | 否（夏普最高） |
| C | VIX 变化 → 五品种 | CU, EG, SC, CF, AU | 1.243 | −4.2% | 否 |
| D | 美元偏鹰 → 金银油 | AU, AG, SC | 1.191 | −25.8% | 否 |
| E | VIX 水平 → 工业品 | V, I, CF | 1.178 | −7.2% | 否 |
| F | CPI 同比 → 原油 | SC | 1.128 | −5.4% | 否 |

---

## A. `usd_broad_z60` × AU（落盘）

- **逻辑**：贸易加权美元 60 日 z-score ≥ 0.25 时做多沪金，持有 5 日；否则空仓。
- **参数**：`long_only` thr=0.25 dir=+1 hold=5；util=0.35；max_lots=2；margin_cap=20万。
- **表现**：Sharpe 1.318（0.72/1.76）；收益 +35.6%；maxDD −3.7%；有仓 1132 天；均保证金 ~18.6 万；最差日 ~−4.9 万。
- **日线**：`best_account_daily.csv`；镜像 `data/infer/results/daily/linear/macro_account_best_daily.csv`。

---

## B. `usd_hawkish` × BU, NR, M, AU

- **逻辑**：`usd_hawkish = dxy_z60 + hike_proxy_z60` ≥ 1.0 时，在沥青 / 20号胶 / 豆粕 / 黄金中按强度最多选 2 个做多，持有 3 日。
- **参数**：`long_only` thr=1.0 dir=+1 hold=3；util=0.35；max_lots=2；margin_cap=20万；max_names=2。
- **表现**：Sharpe **1.337**（1.11/1.54）；收益 +33.9%；maxDD −3.1%；有仓 975 天。
- **备注**：全场夏普最高；品种池可解释性弱于 A/D。

---

## C. `vix_chg` × CU, EG, SC, CF, AU

- **逻辑**：VIX 日变化绝对值 ≥ 2.0 才交易；方向 −1（大涨偏空、大跌偏多）；持有 15 日。
- **参数**：`sign` thr=2.0 dir=−1 hold=15；util=0.12；max_lots=3；margin_cap=8万；max_names=3。
- **表现**：Sharpe 1.243（1.35/1.14）；收益 +35.1%；maxDD −4.2%；有仓 1307 天；仓位轻。

---

## D. `usd_hawkish` × AU, AG, SC

- **逻辑**：美元偏鹰（阈值 0.25）时在金银油篮子分散做多，持有 3 日。
- **参数**：`long_only` thr=0.25 dir=+1 hold=3；util=0.45；max_lots=3；margin_cap=20万。
- **表现**：Sharpe 1.191；收益 **+123.6%**；maxDD **−25.8%**；均保证金 ~67.7 万；最差日 ~−16.9 万。
- **备注**：经典超级大宗；进攻型，回撤显著大于纯金。

---

## E. `vix_z60` × V, I, CF（不含金）

- **逻辑**：VIX 60 日 z 连续仓位，方向 −1（高波动偏空工业品）；每日调仓。
- **参数**：`cont` dir=−1；util=0.45；max_lots=3；margin_cap=12万。
- **表现**：Sharpe 1.178（1.32/1.02）；收益 +36.2%；maxDD −7.2%；几乎天天有仓。

---

## F. `cpi_yoy_z60` × SC

- **逻辑**：美国 CPI 同比 z 绝对值 ≥ 1.25 时，按同号方向交易原油（dir=+1 → 通胀偏高偏多油），持有 2 日。
- **参数**：`sign` thr=1.25 dir=+1 hold=2；util=0.12；max_lots=2；margin_cap=20万。
- **表现**：Sharpe 1.128（1.63/0.75）；收益 +47.5%；maxDD −5.4%；有仓仅 ~490 天。

---

## 如何重放

```bash
cd /home/workspace/lab/UniFutures
# 质量最优已在 search 结束时导出；其它方案用 BookEngine.run_full + 上表参数
python3 macro/code/search_macro_account.py --minutes 1 --resume --skip-phase-a  # 仅在需续搜时
```

参数表与完整搜索统计另见 `account_search_report.json`、`top50_account.csv`、`robust_account.csv`。
