#!/usr/bin/env python3
"""One champion per family, then simple rotation. Goal: few neg-Sharpe years, low DD, high 2026."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from linear_ridge_walkforward import EVAL_END, OUT_DIR, build_panel, ls_pnl, metrics_from_pnl
from six_name_select import pick_ls_n
from two_model_rotate import BT_START
from two_name_select import apply_hold

CANDIDATES = [
    ("rsi", 1, "趋势"),
    ("pos_64", 1, "趋势"),
    ("ratio_min_30", 1, "趋势"),
    ("lower_shadow", -1, "K线"),
    ("upper_shadow", 1, "K线"),
    ("price_volume_ratio", -1, "流动性"),
    ("liquidity_10", 1, "流动性"),
    ("vol_pos_64", 1, "成交量"),
]


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def yearly_rows(s: pd.Series) -> list[dict]:
    rows = []
    for y, g in s.groupby(s.index.year):
        m = metrics_from_pnl(g)
        rows.append({"year": int(y), "return": fmt(m["return"], 4), "sharpe": fmt(m["sharpe"], 3), "max_dd": fmt(m["max_dd"], 4)})
    return rows


def year_pack(s: pd.Series) -> dict:
    ys = yearly_rows(s)
    sharpes = [r["sharpe"] for r in ys if r["sharpe"] is not None]
    neg = [r for r in ys if (r["sharpe"] or 0) < 0]
    y2026 = next((r for r in ys if r["year"] == 2026), {})
    m = metrics_from_pnl(s)
    return {
        **{k: fmt(m[k], 3 if k == "sharpe" else 4) if k != "days" else m[k] for k in m},
        "n_neg_years": len(neg),
        "neg_years": [r["year"] for r in neg],
        "worst_year_sharpe": fmt(min(sharpes) if sharpes else None, 3),
        "sharpe_2026": y2026.get("sharpe"),
        "return_2026": y2026.get("return"),
        "yearly": ys,
    }


def choose_champions(stats: dict[str, dict]) -> dict[str, str]:
    """One name per family: fewest neg years, then better worst-year, then 2026, then full Sharpe."""
    by_fam: dict[str, list[str]] = {}
    for name, sign, fam in CANDIDATES:
        by_fam.setdefault(fam, []).append(name)

    def key(name: str):
        st = stats[name]
        return (
            st["n_neg_years"],
            -(st["worst_year_sharpe"] or -9),
            -(st["sharpe_2026"] or -9),
            -(st["sharpe"] or -9),
        )

    return {fam: min(names, key=key) for fam, names in by_fam.items()}


def trail_sum(s: pd.Series, win: int) -> pd.Series:
    return s.rolling(win).sum().shift(1)


def rotate_daily(series: dict[str, pd.Series], win: int, margin: float = 0.0) -> tuple[pd.Series, pd.Series]:
    names = list(series)
    trails = pd.DataFrame({k: trail_sum(series[k], win) for k in names})
    idx = trails.dropna(how="all").index
    picks = []
    cur = names[0]
    for dt in idx:
        row = trails.loc[dt]
        if row.notna().sum() == 0:
            picks.append((dt, cur))
            continue
        lead = row.idxmax()
        if (row[lead] - row.get(cur, row[lead])) > margin or cur not in row.index or not np.isfinite(row.get(cur, np.nan)):
            cur = lead
        picks.append((dt, cur))
    pick_s = pd.Series({d: n for d, n in picks})
    pnl = pd.Series({dt: float(series[name].loc[dt]) for dt, name in pick_s.items() if dt in series[name].index})
    return pnl.sort_index(), pick_s


def rotate_quarter(series: dict[str, pd.Series], lookback_days: int = 60) -> tuple[pd.Series, pd.Series]:
    names = list(series)
    idx = series[names[0]].index
    quarters = idx.to_period("Q")
    out = []
    picks = []
    for q in quarters.unique():
        q_idx = idx[quarters == q]
        start = q_idx[0]
        hist_end = start - pd.Timedelta(days=1)
        hist_start = start - pd.Timedelta(days=lookback_days + 20)
        scores = {}
        for n in names:
            h = series[n].loc[(series[n].index >= hist_start) & (series[n].index <= hist_end)]
            scores[n] = float(h.tail(lookback_days).sum()) if len(h) else -np.inf
        pick = max(scores, key=scores.get)
        for dt in q_idx:
            if dt in series[pick].index:
                out.append((dt, float(series[pick].loc[dt])))
                picks.append((dt, pick))
    return pd.Series({d: v for d, v in out}).sort_index(), pd.Series({d: n for d, n in picks})


def ew_pnl(series: dict[str, pd.Series]) -> pd.Series:
    df = pd.DataFrame(series)
    return df.mean(axis=1)


def main() -> None:
    print("loading panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01"))
    need = ["fwd_ret"] + [n for n, _, _ in CANDIDATES]
    panel = panel.dropna(subset=[c for c in need if c in panel.columns])

    def zscore(arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=float)
        m = np.isfinite(x)
        out = np.full_like(x, np.nan, dtype=float)
        if m.sum() < 5:
            return out
        sd = np.nanstd(x, ddof=1)
        if not np.isfinite(sd) or sd < 1e-12:
            return out
        out[m] = (x[m] - np.nanmean(x)) / sd
        return out

    blends = {
        "合成 rsi+价量": [("rsi", 1), ("price_volume_ratio", -1)],
        "合成 rsi+价量+量能": [("rsi", 1), ("price_volume_ratio", -1), ("vol_pos_64", 1)],
    }
    cs: dict[str, list] = {n: [] for n, _, _ in CANDIDATES}
    n6: dict[str, list] = {n: [] for n, _, _ in CANDIDATES}
    for b in blends:
        cs[b], n6[b] = [], []
    print("daily LS + 6-name...", flush=True)
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        for name, sign, _ in CANDIDATES:
            sc = sign * day[name].to_numpy()
            cs[name].append((dt, ls_pnl(day, sc)))
            n6[name].append((dt, apply_hold(day, pick_ls_n(day, sc, 3))))
        for bname, parts in blends.items():
            sc = None
            for col, sgn in parts:
                z = sgn * zscore(day[col].to_numpy())
                sc = z if sc is None else sc + z
            cs[bname].append((dt, ls_pnl(day, sc)))
            n6[bname].append((dt, apply_hold(day, pick_ls_n(day, sc, 3))))

    def to_s(pairs):
        return pd.Series({d: v for d, v in pairs}).sort_index().dropna()

    cs_s = {n: to_s(cs[n]) for n in cs}
    n6_s = {n: to_s(n6[n]) for n in n6}

    print("=== family candidates (CS 20% LS) ===", flush=True)
    cand_stats = {}
    for name, sign, fam in CANDIDATES:
        cand_stats[name] = {"family": fam, "sign": sign, **year_pack(cs_s[name])}
        st = cand_stats[name]
        print(
            f"{fam:4} {name:22} sh={st['sharpe']:.2f} dd={st['max_dd']:.1%} "
            f"neg={st['n_neg_years']} worst={st['worst_year_sharpe']:.2f} 2026={st['sharpe_2026']}",
            flush=True,
        )

    champs = choose_champions(cand_stats)
    print("champions:", champs, flush=True)
    champ_names = list(champs.values())
    # drop duplicate if two families somehow same - won't happen

    books = {"cs20": {n: cs_s[n] for n in champ_names}, "n6": {n: n6_s[n] for n in champ_names}}
    payload_schemes = {}

    for book_name, series in books.items():
        rsi, shad, pvr, vol = (champs[k] for k in ("趋势", "K线", "流动性", "成交量"))
        schemes = {
            "固定 rsi": series[rsi],
            "固定 下影线": series[shad],
            "固定 价量比": series[pvr],
            "固定 量能位置": series[vol],
            "等权 rsi+价量": ew_pnl({rsi: series[rsi], pvr: series[pvr]}),
            "等权 rsi+量能": ew_pnl({rsi: series[rsi], vol: series[vol]}),
            "等权 rsi+下影": ew_pnl({rsi: series[rsi], shad: series[shad]}),
            "等权 三族(无K线)": ew_pnl({rsi: series[rsi], pvr: series[pvr], vol: series[vol]}),
            "等权 三族(无量能)": ew_pnl({rsi: series[rsi], pvr: series[pvr], shad: series[shad]}),
            "等权 四族": ew_pnl(series),
            "轮动 rsi↔价量 20d+3%": rotate_daily({rsi: series[rsi], pvr: series[pvr]}, 20, 0.03)[0],
            "轮动 rsi↔量能 20d+3%": rotate_daily({rsi: series[rsi], vol: series[vol]}, 20, 0.03)[0],
            "轮动 四族 季60d": rotate_quarter(series, 60)[0],
            "轮动 四族 20d+3%": rotate_daily(series, 20, 0.03)[0],
        }
        for bname in blends:
            if bname in (cs_s if book_name == "cs20" else n6_s):
                src = cs_s if book_name == "cs20" else n6_s
                schemes[bname] = src[bname]

        packed = {}
        print(f"\n=== {book_name} ===", flush=True)
        for sn, s in schemes.items():
            packed[sn] = year_pack(s)
            st = packed[sn]
            print(
                f"{sn:22} sh={st['sharpe']:.2f} dd={st['max_dd']:.1%} "
                f"neg={st['n_neg_years']}{st['neg_years']} worst={st['worst_year_sharpe']:.2f} "
                f"2026={st['sharpe_2026']}",
                flush=True,
            )
        payload_schemes[book_name] = packed

    out = {
        "goal": "few neg-Sharpe years, low DD, high 2026 Sharpe, simple",
        "champion_rule": "per family: fewest neg years, then worst-year Sharpe, then 2026, then full Sharpe",
        "champions": champs,
        "candidates": {
            n: {k: cand_stats[n][k] for k in ("family", "sign", "return", "sharpe", "max_dd", "n_neg_years", "neg_years", "worst_year_sharpe", "sharpe_2026")}
            for n, _, _ in CANDIDATES
        },
        "cs20": payload_schemes["cs20"],
        "n6": payload_schemes["n6"],
    }
    path = OUT_DIR / "family_rotate.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
