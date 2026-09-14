#!/usr/bin/env python3
"""Search 2/4-name books: Sharpe>1 and max DD < 10%. pos64 / RSI / -PVR + BOOK."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from blend_book import book_ratio, zscore
from family_rotate import fmt, year_pack
from linear_ridge_walkforward import EVAL_END, OUT_DIR, build_panel
from six_name_select import pick_ls_n
from two_model_rotate import BT_START
from two_name_dd_control import flatten_after_loss
from two_name_select import apply_hold

FACTORS = [("pos_64", 1, "pos64"), ("rsi", 1, "rsi"), ("price_volume_ratio", -1, "pvr")]


def combine(books: list[list[tuple[str, float]]]) -> list[tuple[str, float]]:
    agg: dict[str, float] = defaultdict(float)
    n = len(books) or 1
    for b in books:
        for s, w in b:
            agg[s] += w / n
    return list(agg.items())


def agree(books: list[list[tuple[str, float]]], need: int) -> list[tuple[str, float]]:
    side: dict[str, list[int]] = defaultdict(list)
    for b in books:
        for s, w in b:
            side[s].append(1 if w > 0 else -1)
    kept = []
    for s, votes in side.items():
        if len(votes) < need:
            continue
        if all(v == votes[0] for v in votes):
            kept.append((s, votes[0]))
    if not kept:
        return []
    longs = [s for s, d in kept if d > 0]
    shorts = [s for s, d in kept if d < 0]
    out = []
    if longs:
        w = 0.5 / len(longs)
        out += [(s, w) for s in longs]
    if shorts:
        w = 0.5 / len(shorts)
        out += [(s, -w) for s in shorts]
    return out


def vol_scale(pnl: pd.Series, target: float, win: int = 20, cap: float = 2.5) -> pd.Series:
    vol = pnl.rolling(win).std(ddof=1) * np.sqrt(252)
    sc = (target / vol.shift(1)).replace([np.inf, -np.inf], np.nan)
    sc = sc.clip(lower=0.0, upper=cap).fillna(1.0)
    return pnl * sc


def flatten_book(pnl: pd.Series, thr: float) -> tuple[pd.Series, pd.Series]:
    r = book_ratio(pnl)
    cash = (r >= thr).fillna(False)
    out = pnl.copy()
    out[cash] = 0.0
    return out, cash


def pack(pnl: pd.Series, cash: pd.Series, n_names: pd.Series) -> dict:
    st = year_pack(pnl)
    inn = ~cash.reindex(pnl.index).fillna(False)
    nn = n_names.reindex(pnl.index)
    st["cash"] = fmt(float(cash.reindex(pnl.index).fillna(False).mean()), 3)
    st["avg_names"] = fmt(float(nn.mean()), 2)
    st["avg_names_on"] = fmt(float(nn[inn].mean()), 2) if inn.any() else 0.0
    st["p90_names"] = fmt(float(nn.quantile(0.9)), 2)
    st["max_names"] = int(nn.max()) if len(nn) else 0
    sharpes = [r["sharpe"] for r in st["yearly"] if r["sharpe"] is not None]
    st["sharpe_std"] = fmt(float(np.std(sharpes, ddof=1)), 3) if len(sharpes) > 2 else None
    return st


def hit(st: dict) -> bool:
    sh = st.get("sharpe") or 0
    dd = st.get("max_dd") or 0
    mx = st.get("max_names") or 99
    return sh >= 1.0 and dd > -0.10 - 1e-12 and mx <= 4


def main() -> None:
    print("panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01")).dropna(subset=["fwd_ret", "pos_64", "rsi", "price_volume_ratio"])
    w1 = {k: [] for _, _, k in FACTORS}
    w2 = {k: [] for _, _, k in FACTORS}
    wz1, wz2 = [], []
    rec_n = []
    print("daily 1L1S / 2L2S...", flush=True)
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        zsc = None
        b1, b2 = [], []
        for col, sgn, name in FACTORS:
            raw = sgn * day[col].to_numpy()
            p1 = pick_ls_n(day, raw, 1)
            p2 = pick_ls_n(day, raw, 2)
            w1[name].append((dt, p1, apply_hold(day, p1)))
            w2[name].append((dt, p2, apply_hold(day, p2)))
            b1.append(p1)
            b2.append(p2)
            z = sgn * zscore(day[col].to_numpy())
            zsc = z if zsc is None else zsc + z
        wz1.append((dt, pick_ls_n(day, zsc, 1), apply_hold(day, pick_ls_n(day, zsc, 1))))
        wz2.append((dt, pick_ls_n(day, zsc, 2), apply_hold(day, pick_ls_n(day, zsc, 2))))
        rec_n.append(dt)

    def pnl_of(rows) -> pd.Series:
        return pd.Series({d: p for d, _, p in rows}).sort_index()

    def w_of(rows) -> dict:
        return {d: w for d, w, _ in rows}

    idx = pnl_of(w1["rsi"]).index
    books_w = {}
    books_p = {}
    for k in ("pos64", "rsi", "pvr"):
        books_w[f"{k}-2"] = w_of(w1[k])
        books_p[f"{k}-2"] = pnl_of(w1[k]).reindex(idx)
        books_w[f"{k}-4"] = w_of(w2[k])
        books_p[f"{k}-4"] = pnl_of(w2[k]).reindex(idx)
    books_w["z-2"] = {d: w for d, w, _ in wz1}
    books_p["z-2"] = pnl_of(wz1).reindex(idx)
    books_w["z-4"] = {d: w for d, w, _ in wz2}
    books_p["z-4"] = pnl_of(wz2).reindex(idx)

    # overlays of 1L1S pairs (max 4 names) and all-three 1L1S (max 6, skip if max>4 later)
    pair_names = {
        "pos64+rsi 叠2": ("pos64-2", "rsi-2"),
        "pos64+价量 叠2": ("pos64-2", "pvr-2"),
        "RSI+价量 叠2": ("rsi-2", "pvr-2"),
        "三因子叠2": ("pos64-2", "rsi-2", "pvr-2"),
        "pos64+价量 叠4": ("pos64-4", "pvr-4"),
        "pos64+rsi 叠4": ("pos64-4", "rsi-4"),
        "RSI+价量 叠4": ("rsi-4", "pvr-4"),
    }
    # rebuild overlay pnl/weights from stored daily weights
    day_index = list(idx)
    # need original day for apply_hold? overlay pnl = mean of component pnls if same dates
    for lab, keys in pair_names.items():
        parts = [books_p[k] for k in keys]
        books_p[lab] = pd.concat(parts, axis=1).mean(axis=1)
        ww = {}
        for dt in day_index:
            bs = [books_w[k].get(dt, []) for k in keys]
            ww[dt] = combine(bs)
        books_w[lab] = ww

    # 2-of-3 agreement on 1L1S
    agr = []
    agr_w = {}
    # we need panel days again for apply_hold of agreement - use mean of votes via combine of signs
    # reconstruct from weights without panel: pnl from weighted fwd not available.
    # Use: agreement pnl ≈ apply by combining stored 1L1S if we stored fwd in hold already.
    # Compute from weights: if we only have book pnls not per-name, approximate:
    # agreement subset: names in >=2 books same side; pnl unknown without returns.
    # Re-loop is expensive. Store fwd via apply_hold on combined weights — we didn't keep day.
    # Fast path: second pass only for agreement using panel.

    print("agreement pass...", flush=True)
    agr_rows = []
    agr_w = {}
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        bs = []
        for col, sgn, name in FACTORS:
            bs.append(pick_ls_n(day, sgn * day[col].to_numpy(), 1))
        w = agree(bs, 2)
        agr_w[dt] = w
        agr_rows.append((dt, apply_hold(day, w)))
    books_p["两因子同向(1L1S)"] = pd.Series({d: p for d, p in agr_rows}).reindex(idx)
    books_w["两因子同向(1L1S)"] = agr_w

    def nser(wmap) -> pd.Series:
        return pd.Series({dt: len({s for s, w in (wmap.get(dt) or []) if abs(w) > 1e-12}) for dt in idx})

    candidates = [
        "pos64-2", "rsi-2", "pvr-2", "z-2",
        "pos64-4", "rsi-4", "pvr-4", "z-4",
        "pos64+rsi 叠2", "pos64+价量 叠2", "RSI+价量 叠2", "三因子叠2",
        "pos64+价量 叠4", "pos64+rsi 叠4", "RSI+价量 叠4",
        "两因子同向(1L1S)",
    ]

    rows = []
    hits = []
    close = []  # sh>=0.9 and dd>-0.12 and max_names<=4

    def add(name, pnl, cash, nn):
        st = pack(pnl, cash, nn)
        st["scheme"] = name
        rows.append(st)
        if (st.get("max_names") or 99) > 4:
            return
        if hit(st):
            hits.append(st)
        sh, dd = st.get("sharpe") or 0, st.get("max_dd") or 0
        if sh >= 0.9 and dd > -0.12:
            close.append(st)

    print("risk overlays...", flush=True)
    for lab in candidates:
        raw = books_p[lab].astype(float)
        nn = nser(books_w[lab])
        # raw (may have cash 0%)
        add(f"{lab} | 无BOOK", raw, raw.abs() < 1e-12, nn)
        for thr in (1.05, 1.08, 1.10, 1.12, 1.15, 1.18, 1.20, 1.25):
            live, cash = flatten_book(raw, thr)
            add(f"{lab} | BOOK≥{thr:.2f}空", live, cash, nn.where(~cash, 0))
        for tgt in (0.06, 0.08, 0.10, 0.12):
            vs = vol_scale(raw, tgt)
            add(f"{lab} | 波动{int(tgt*100)}%", vs, vs.abs() < 1e-12, nn)
            live, cash = flatten_book(vs, 1.15)
            add(f"{lab} | 波动{int(tgt*100)}%+BOOK≥1.15", live, cash, nn.where(~cash, 0))
            live, cash = flatten_book(raw, 1.15)
            live = vol_scale(live, tgt)
            add(f"{lab} | BOOK≥1.15后波动{int(tgt*100)}%", live, cash, nn.where(~cash, 0))
        cool = flatten_after_loss(raw, -0.02, 5)
        add(f"{lab} | 日亏2%空5日", cool, cool.abs() < 1e-12, nn)
        live, cash = flatten_book(cool, 1.15)
        add(f"{lab} | 日亏2%空5日+BOOK≥1.15", live, cash, nn.where(~cash, 0))

    hits.sort(key=lambda r: (-(r["sharpe"] or 0), -(r["max_dd"] or -9)))
    close.sort(key=lambda r: (-(r["sharpe"] or 0), -(r["max_dd"] or -9)))
    print(f"HITS sharpe>=1 & dd>-10% & names<=4: {len(hits)}", flush=True)
    for r in hits[:25]:
        print(
            f"  {r['scheme']:52} sh={r['sharpe']:.2f} dd={r['max_dd']:.1%} "
            f"names={r['avg_names_on']}/{r['max_names']} cash={r['cash']:.0%} 26={r['sharpe_2026']} neg={r['n_neg_years']}",
            flush=True,
        )
    if not hits:
        print("no exact hits; nearest (sh>=0.9, dd>-12%, names<=4):", flush=True)
        for r in close[:20]:
            print(
                f"  {r['scheme']:52} sh={r['sharpe']:.2f} dd={r['max_dd']:.1%} "
                f"names={r['avg_names_on']}/{r['max_names']} cash={r['cash']:.0%} 26={r['sharpe_2026']} neg={r['n_neg_years']}",
                flush=True,
            )

    slim_keys = (
        "scheme", "return", "sharpe", "max_dd", "n_neg_years", "neg_years", "worst_year_sharpe",
        "sharpe_std", "sharpe_2026", "return_2026", "cash", "avg_names", "avg_names_on", "p90_names", "max_names", "yearly",
    )
    slim = lambda r: {k: r[k] for k in slim_keys if k in r}
    path = OUT_DIR / "three_factor_24_sharpe_dd.json"
    path.write_text(json.dumps({
        "goal": "2 or 4 names; Sharpe>=1; max_dd > -10%",
        "n_schemes": len(rows),
        "hits": [slim(r) for r in hits],
        "close": [slim(r) for r in close[:40]],
    }, ensure_ascii=False, indent=2))
    print("saved", path, "n", len(rows))


if __name__ == "__main__":
    main()
