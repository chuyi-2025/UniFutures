#!/usr/bin/env python3
"""Integer-lot search with broker margin and 1-lot CNY PnL.

Two return definitions:
  1m  — 1,000,000 CNY account; r = log1p(pnl_cny / 1e6)
  rom — return on posted margin; r = log1p(pnl_cny / margin)

Lots {0,1}: cash if vol-scale < t0, else 1 lot per selected name.
Factors: pos64 / RSI / -PVR. Max 6 names.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from blend_book import zscore
from family_rotate import fmt
from futures_lot_specs import BROKER_MARGIN, MULTIPLIER
from linear_ridge_walkforward import EVAL_END, OUT_DIR, TREE, _read_contract_csv, build_panel
from pos64_integer_lot_search import apply_lots, lots_cut, scale_of, slim
from shared import REMOVED_SYMBOLS
from six_name_select import pick_ls_n
from three_factor_24_search import flatten_book, pack
from two_model_rotate import BT_START
from linear_ridge_walkforward import DEAD

CAPITAL = 1_000_000.0
NMAP = {
    "pos64-2": 2, "pos64-4": 4, "pos64-6": 6,
    "rsi-2": 2, "rsi-4": 4, "rsi-6": 6,
    "pvr-2": 2, "pvr-4": 4, "pvr-6": 6,
    "z-4": 4, "z-6": 6,
    "pos64+价量叠2": 4, "pos64+rsi叠2": 4, "rsi+价量叠2": 4, "三因子叠2": 6,
}


def load_close() -> pd.DataFrame:
    frames = []
    for sym_dir in sorted(p for p in TREE.iterdir() if p.is_dir()):
        sym = sym_dir.name.upper()
        if sym in REMOVED_SYMBOLS or sym in DEAD:
            continue
        parts = []
        for csv in sorted(sym_dir.glob("*.csv")):
            try:
                raw = _read_contract_csv(csv)
            except Exception:  # noqa: BLE001
                continue
            parts.append(raw[["date", "code", "close"]])
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df["symbol"] = sym
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def net_picks(*books: list[tuple[str, float]]) -> list[tuple[str, float]]:
    side: dict[str, int] = defaultdict(int)
    for picks in books:
        for s, w in picks:
            side[s] += 1 if w > 0 else -1
    out = []
    for s, v in side.items():
        if v > 0:
            out.append((s, 1.0))
        elif v < 0:
            out.append((s, -1.0))
    return out


def dollar_1lot(day: pd.DataFrame, picks: list[tuple[str, float]]) -> tuple[float, float, float, int]:
    """CNY pnl, margin, notional, n names for ±1 lot each."""
    m = day.drop_duplicates("symbol").set_index("symbol")
    pnl = 0.0
    margin = 0.0
    notional = 0.0
    n = 0
    for s, w in picks:
        if s not in m.index:
            continue
        px = m.loc[s, "close"]
        fr = m.loc[s, "fwd_ret"]
        mult = MULTIPLIER.get(s)
        rate = BROKER_MARGIN.get(s)
        if mult is None or rate is None:
            continue
        if not np.isfinite(px) or not np.isfinite(fr) or px <= 0:
            continue
        sign = 1.0 if w > 0 else -1.0
        val = float(mult) * float(px)
        pnl += sign * val * (np.exp(float(fr)) - 1.0)
        margin += val * float(rate)
        notional += val
        n += 1
    return pnl, margin, notional, n


def to_log(simple: pd.Series) -> pd.Series:
    x = simple.replace([np.inf, -np.inf], np.nan).clip(lower=-0.8, upper=2.0)
    return np.log1p(x)


def ok(r) -> bool:
    cash = r.get("cash") or 0
    return (
        (r.get("sharpe") or 0) >= 1.0
        and (r.get("max_dd") or 0) > -0.10
        and 0.20 - 1e-9 <= cash <= 0.50 + 1e-9
        and (r.get("max_names") or 99) <= 6
    )


def main() -> None:
    print("panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01")).dropna(subset=["fwd_ret", "pos_64", "rsi", "price_volume_ratio"])
    print("closes...", flush=True)
    closes = load_close()
    panel = panel.merge(closes, on=["date", "symbol", "code"], how="left")

    books_picks: dict[str, list] = {k: [] for k in NMAP}
    print("daily picks + dollar...", flush=True)
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        day = day.drop_duplicates("symbol").reset_index(drop=True)
        p64, rsi, pvr = day["pos_64"].to_numpy(), day["rsi"].to_numpy(), -day["price_volume_ratio"].to_numpy()
        z = zscore(p64) + zscore(rsi) + zscore(pvr)
        a1, a2, a3 = pick_ls_n(day, p64, 1), pick_ls_n(day, p64, 2), pick_ls_n(day, p64, 3)
        r1, r2, r3 = pick_ls_n(day, rsi, 1), pick_ls_n(day, rsi, 2), pick_ls_n(day, rsi, 3)
        v1, v2, v3 = pick_ls_n(day, pvr, 1), pick_ls_n(day, pvr, 2), pick_ls_n(day, pvr, 3)
        z2, z3 = pick_ls_n(day, z, 2), pick_ls_n(day, z, 3)
        books_picks["pos64-2"].append((dt, a1, dollar_1lot(day, a1)))
        books_picks["pos64-4"].append((dt, a2, dollar_1lot(day, a2)))
        books_picks["pos64-6"].append((dt, a3, dollar_1lot(day, a3)))
        books_picks["rsi-2"].append((dt, r1, dollar_1lot(day, r1)))
        books_picks["rsi-4"].append((dt, r2, dollar_1lot(day, r2)))
        books_picks["rsi-6"].append((dt, r3, dollar_1lot(day, r3)))
        books_picks["pvr-2"].append((dt, v1, dollar_1lot(day, v1)))
        books_picks["pvr-4"].append((dt, v2, dollar_1lot(day, v2)))
        books_picks["pvr-6"].append((dt, v3, dollar_1lot(day, v3)))
        books_picks["z-4"].append((dt, z2, dollar_1lot(day, z2)))
        books_picks["z-6"].append((dt, z3, dollar_1lot(day, z3)))
        ov = net_picks(a1, v1)
        books_picks["pos64+价量叠2"].append((dt, ov, dollar_1lot(day, ov)))
        ov = net_picks(a1, r1)
        books_picks["pos64+rsi叠2"].append((dt, ov, dollar_1lot(day, ov)))
        ov = net_picks(r1, v1)
        books_picks["rsi+价量叠2"].append((dt, ov, dollar_1lot(day, ov)))
        ov = net_picks(a1, r1, v1)
        books_picks["三因子叠2"].append((dt, ov, dollar_1lot(day, ov)))

    series = {}
    for lab, recs in books_picks.items():
        idx = [d for d, _, _ in recs]
        pnl = pd.Series({d: x[0] for d, _, x in recs}, dtype=float).sort_index()
        mar = pd.Series({d: x[1] for d, _, x in recs}, dtype=float).sort_index()
        notion = pd.Series({d: x[2] for d, _, x in recs}, dtype=float).sort_index()
        nn = pd.Series({d: x[3] for d, _, x in recs}, dtype=float).sort_index()
        r1m = to_log(pnl / CAPITAL)
        rom = to_log(pnl / mar.replace(0, np.nan)).fillna(0.0)
        series[lab] = {"pnl": pnl, "mar": mar, "notion": notion, "nn": nn, "r1m": r1m, "rom": rom}

    rows = []

    def add(tag, lab, r, cash, lots=None):
        info = series[lab]
        nn = info["nn"].reindex(r.index).fillna(0)
        nn = nn.where(~cash.reindex(r.index).fillna(False), 0.0)
        st = pack(r, cash, nn)
        st["scheme"] = f"{tag} | {lab}"
        st["book"] = lab
        inn = ~cash.reindex(r.index).fillna(False)
        mar = info["mar"].reindex(r.index)
        notion = info["notion"].reindex(r.index)
        st["mean_margin"] = fmt(float(mar[inn].mean()) if inn.any() else 0, 0)
        st["p90_margin"] = fmt(float(mar[inn].quantile(0.9)) if inn.any() else 0, 0)
        st["mean_notional"] = fmt(float(notion[inn].mean()) if inn.any() else 0, 0)
        lev = (notion / mar.replace(0, np.nan))
        st["mean_lev"] = fmt(float(lev[inn].mean()) if inn.any() else 0, 2)
        if lots is not None:
            st["mean_lots"] = fmt(float(lots.reindex(r.index).fillna(0).mean()), 3)
        rows.append(st)

    print("grids...", flush=True)
    tgts = (0.06, 0.07, 0.08, 0.10)
    t0s = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80)
    book_thrs = (1.10, 1.14, 1.18, 1.22)

    for lab, info in series.items():
        for tag, raw in (("100万账户", info["r1m"]), ("保证金收益", info["rom"])):
            add(f"{tag} | 1手", lab, raw, raw.abs() < 1e-15)
            for thr in book_thrs:
                live, cash = flatten_book(raw, thr)
                add(f"{tag} | 1手 BOOK≥{thr:.2f}空", lab, live, cash)
            for tgt in tgts:
                sc = scale_of(raw, tgt)
                for t0 in t0s:
                    lots = lots_cut(sc, t0, 9.0)
                    live, cash = apply_lots(raw, lots)
                    add(f"{tag} | 波动{int(tgt*100)}% <{t0:.2f}空 1手", lab, live, cash, lots)

    hits_1m = [r for r in rows if r["scheme"].startswith("100万") and ok(r)]
    hits_rom = [r for r in rows if r["scheme"].startswith("保证金") and ok(r)]
    hits_1m.sort(key=lambda r: (-(r.get("sharpe") or 0), -(r.get("max_dd") or -9), -(r.get("sharpe_2026") or -9)))
    hits_rom.sort(key=lambda r: (-(r.get("sharpe") or 0), -(r.get("max_dd") or -9), -(r.get("sharpe_2026") or -9)))

    def pr(title, lst, n=20):
        print(title, len(lst), flush=True)
        for r in lst[:n]:
            print(
                f"  {r['scheme'][:72]:72} sh={r['sharpe']:.3f} dd={r['max_dd']:.1%} "
                f"cash={r['cash']:.0%} 26={r['sharpe_2026']} neg={r['n_neg_years']} "
                f"保证金≈{r.get('mean_margin')} 杠杆≈{r.get('mean_lev')}",
                flush=True,
            )

    pr("HITS 100万账户 cash20-50% sh>=1 dd>-10%", hits_1m, 25)
    pr("HITS 保证金收益(满杠杆) 同约束", hits_rom, 15)

    near_1m = [
        r for r in rows
        if r["scheme"].startswith("100万")
        and (r.get("sharpe") or 0) >= 0.8
        and (r.get("max_dd") or 0) > -0.15
        and 0.15 <= (r.get("cash") or 0) <= 0.55
        and (r.get("max_names") or 99) <= 6
    ]
    near_1m.sort(key=lambda r: (-(r.get("sharpe") or 0), -(r.get("max_dd") or -9)))
    if not hits_1m:
        pr("NEAR 100万 sh>=0.8 dd>-15% cash15-55%", near_1m, 20)
    else:
        pr("NEAR 100万 (sh>=0.8)", near_1m, 10)

    path = OUT_DIR / "pos64_margin_integer_search.json"
    extra = ("book", "mean_margin", "p90_margin", "mean_notional", "mean_lev", "mean_lots")
    def slim2(r):
        out = slim(r)
        for k in extra:
            if k in r:
                out[k] = r[k]
        return out

    path.write_text(json.dumps({
        "note": "1 lot/name; broker margin snapshot 2026-09-09; close×multiplier×(exp(fwd)-1)",
        "capital": CAPITAL,
        "n": len(rows),
        "hits_1m": [slim2(r) for r in hits_1m[:40]],
        "hits_rom": [slim2(r) for r in hits_rom[:40]],
        "near_1m": [slim2(r) for r in near_1m[:30]],
    }, ensure_ascii=False, indent=2, default=str))
    print("saved", path, "n", len(rows), flush=True)


if __name__ == "__main__":
    main()
