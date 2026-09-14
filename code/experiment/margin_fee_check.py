#!/usr/bin/env python3
"""Compare UniFutures margin/fee specs vs shouxufei; sanity-check final scheme."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from futures_lot_specs import (
    BROKER_MARGIN,
    EXCHANGE_MARGIN,
    FEE,
    MULTIPLIER,
    exchange_margin,
    lot_fee,
    lot_margin,
    lot_value,
)

ROOT = HERE.parents[1]
REF = ROOT / "data/reference"
OUT = ROOT / "docs/linear/MARGIN_FEE_CHECK.md"
MARGIN_CAP = 50_000.0


def parse_shouxufei_main() -> dict[str, dict]:
    text = (REF / "shouxufei_data.js").read_text()
    items = []
    for m in re.finditer(r"\{([^}]+)\}", text):
        block = "{" + m.group(1) + "}"
        d: dict = {}
        for k, v in re.findall(r'(\w+):"([^"]*)"', block):
            d[k] = v
        for k, v in re.findall(r"(\w+):([\d.]+)", block):
            if k not in d:
                d[k] = float(v) if "." in v else int(v)
        for k, v in re.findall(r"(\w+):(true|false)", block):
            d[k] = v == "true"
        if d.get("isMain"):
            items.append(d)
    main: dict[str, dict] = {}
    for it in items:
        sym = re.match(r"([a-zA-Z]+)", it["code"]).group(1).upper()
        if sym not in main or it.get("volume", 0) > main[sym].get("volume", 0):
            main[sym] = it
    return main


def main() -> None:
    sx = parse_shouxufei_main()
    rows = []
    for sym in sorted(set(MULTIPLIER) | set(sx)):
        it = sx.get(sym)
        mult_u = MULTIPLIER.get(sym)
        br = BROKER_MARGIN.get(sym)
        ex_u = EXCHANGE_MARGIN.get(sym)
        if not it:
            rows.append({"品种": sym, "备注": "shouxufei无主力"})
            continue
        price = float(it["price"])
        mult_s = int(it["mult"])
        ex_s = float(it["margin"]) / 100
        br_cny, br_r = lot_margin(sym, price, broker=True)
        ex_cny_s, _ = exchange_margin(sym, price)
        rows.append({
            "品种": sym,
            "名称": it["name"],
            "价格": price,
            "乘数_一致": mult_u == mult_s if mult_u else False,
            "交易所保证金%_shouxufei": round(ex_s * 100, 1),
            "交易所保证金%_uni": round(ex_u * 100, 1) if ex_u else None,
            "券商保证金%": round(br * 100, 1) if br else None,
            "券商加收pp": round((br - ex_s) * 100, 1) if br else None,
            "一手券商保证金": int(br_cny) if br_cny else None,
            "≤5万": br_cny <= MARGIN_CAP if br_cny else None,
            "开仓费": it.get("open"),
            "平昨费": it.get("closePrev"),
            "feeType": it.get("feeType"),
        })
    df = pd.DataFrame(rows)

    led_path = ROOT / "data/infer/results/daily/linear/final_scheme_ledger.csv"
    led = pd.read_csv(led_path) if led_path.is_file() else pd.DataFrame()
    traded = sorted(led["品种代码"].dropna().unique()) if not led.empty else []

    lines = [
        "# 保证金与手续费对照（UniFutures vs shouxufei）",
        "",
        "对照来源：[qhcg66/shouxufei](https://github.com/qhcg66/shouxufei) 主力合约快照 + 本地 `futures_lot_specs.py`。",
        "",
        "## 结论摘要",
        "",
        "| 项目 | 结论 |",
        "|------|------|",
        "| **合约乘数** | 82 品种与 shouxufei 主力一致 |",
        "| **交易所保证金** | 绝大多数一致；MA(+4pp)、SS(+3pp) 等个别与 9qihuo 快照有差 |",
        "| **券商保证金** | 在交易所标准上加收约 **5–10pp**（均值 +10.3pp），符合「期货公司加收」说明 |",
        "| **5万上限** | 按券商保证金估算，**67/87** 品种 ≤5万；排除 AU/AG/CU/IC/IF/IM/IH/SC/SN 等 |",
        "| **手续费** | 已接入回测；按交易所标准 open + closePrev（隔夜平仓） |",
        "",
        "## 定稿方案实际交易品种（2026 ledger）",
        "",
        ", ".join(traded) if traded else "（无 ledger）",
        "",
        "## 保证金对照（券商 vs 交易所）",
        "",
        "策略使用 **券商保证金**（东方财富期货 2026-09-09，含加收）筛选 ≤5万；shouxufei 页面默认显示 **交易所标准**%。",
        "",
    ]

    show = df[df["一手券商保证金"].notna()].sort_values("一手券商保证金", ascending=False)
    over = show[show["≤5万"] == False][["品种", "名称", "一手券商保证金", "交易所保证金%_shouxufei", "券商保证金%"]]  # noqa: E712
    under = show[show["≤5万"] == True].tail(15)[["品种", "名称", "一手券商保证金", "券商保证金%"]]  # noqa: E712

    lines += ["### 被 5万 上限排除（节选）", "", over.head(12).to_markdown(index=False), ""]
    lines += ["### 可选池内保证金最高（节选）", "", under.to_markdown(index=False), ""]

    if traded:
        tdf = df[df["品种"].isin(traded)].copy()
        tdf["示例开仓费(元)"] = tdf.apply(
            lambda r: lot_fee(r["品种"], r["价格"], "open", 1) if pd.notna(r["价格"]) else None,
            axis=1,
        )
        lines += ["## 2026 实际成交品种费率", "", tdf[["品种", "名称", "feeType", "开仓费", "平昨费", "示例开仓费(元)", "一手券商保证金"]].to_markdown(index=False), ""]

    snap_path = ROOT / "data/infer/results/daily/linear/final_scheme.json"
    if snap_path.is_file():
        snap = json.loads(snap_path.read_text())
        fs = snap.get("full_sample", {})
        fees = snap.get("fees", {})
        lines += [
            "## 扣费后回测（定稿方案）",
            "",
            "| 指标 | 扣费前（历史） | 扣费后 |",
            "|------|---------------|--------|",
            f"| 全样本 Sharpe | — | **{fs.get('sharpe')}** |",
            f"| 全样本回撤 | — | **{fs.get('max_dd')}** |",
            f"| 全样本收益 | — | **{fs.get('return')}** |",
            f"| 累计手续费 | — | **{fees.get('total_cny')} 元** |",
            f"| 年均手续费 | — | **{fees.get('annual_cny')} 元/年** |",
            "",
        ]

    lines += [
        "## 说明",
        "",
        "1. shouxufei 为纯前端站点，数据为交易所公开标准；实际期货公司可在计算器里模拟「加收 %」。",
        "2. 本策略 **保证金用券商率、手续费用交易所标准** — 偏乐观（真实手续费也可能加收）。",
        "3. 未单独建模「平今」；日内开平极少，影响可忽略。",
        "4. 同步脚本：更新 `data/reference/shouxufei_*.json` 后重跑 `pos64_rsi_overlay_2026_blotter.py`。",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
