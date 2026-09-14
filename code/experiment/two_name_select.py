#!/usr/bin/env python3
"""Select at most 2 symbols on top of the quarterly pos64/Ridge rotation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from build_feature.build_xgb_feature import FEATURE_NAMES
from linear_ridge_walkforward import (
    EVAL_END,
    L2,
    MIN_TRAIN_DAYS,
    OUT_DIR,
    SKIP_FEATS,
    build_panel,
    fit_ridge,
    ic_stats,
    metrics_from_pnl,
    predict_ridge,
    rank_ic_daily,
)
from two_model_rotate import BT_START, choose_model, daily_mkt

SECTOR = {}
for s, g in {
    "黑色": ["RB", "HC", "I", "J", "JM", "SF", "SM", "SS"],
    "有色": ["CU", "AL", "NI", "SN", "PB", "ZN", "AO", "AD", "SI"],
    "贵金属": ["AU", "AG"],
    "化工": ["TA", "PP", "L", "V", "MA", "EG", "RU", "BU", "PF", "BR", "SH", "PX", "PL", "PR", "PS", "SA", "UR", "NR"],
    "油脂": ["A", "B", "M", "Y", "P", "OI", "RM"],
    "农产品": ["C", "CS", "CF", "SR", "AP", "JD", "LH", "PK", "CJ", "SP"],
    "股指": ["IC", "IF", "IH", "IM"],
    "国债": ["T", "TF", "TS", "TL"],
    "能源": ["SC", "FU", "LU", "PG", "BZ"],
}.items():
    for x in g:
        SECTOR[x] = s


def sector_of(sym: str) -> str:
    return SECTOR.get(str(sym).upper(), "其他")


def pick_ls(day: pd.DataFrame, score: np.ndarray, diversify: bool) -> list[tuple[str, float]]:
    s = pd.Series(score, index=day.index)
    yok = day["fwd_ret"].notna()
    s = s[s.notna() & yok]
    if len(s) < 2:
        return []
    long_i = s.idxmax()
    short_cands = s.drop(index=long_i).sort_values()
    short_i = short_cands.index[0]
    if diversify:
        ls, ss = day.loc[long_i, "symbol"], day.loc[short_i, "symbol"]
        if sector_of(ls) == sector_of(ss):
            for idx in short_cands.index[1:]:
                if sector_of(day.loc[idx, "symbol"]) != sector_of(ls):
                    short_i = idx
                    break
    return [(day.loc[long_i, "symbol"], 0.5), (day.loc[short_i, "symbol"], -0.5)]


def pick_abs2(day: pd.DataFrame, score: np.ndarray) -> list[tuple[str, float]]:
    s = pd.Series(score, index=day.index)
    s = s[s.notna() & day["fwd_ret"].notna()]
    if len(s) < 2:
        return []
    top = s.abs().nlargest(2)
    w = 0.5
    out = []
    for idx in top.index:
        sign = 1.0 if s.loc[idx] >= 0 else -1.0
        out.append((day.loc[idx, "symbol"], sign * w))
    return out


def apply_hold(day: pd.DataFrame, weights: list[tuple[str, float]]) -> float:
    m = day.set_index("symbol")["fwd_ret"]
    pnl = 0.0
    n = 0
    for sym, w in weights:
        if sym in m.index and np.isfinite(m.loc[sym]):
            pnl += w * float(m.loc[sym])
            n += 1
    return pnl if n else np.nan


def turnover(prev: list[tuple[str, float]] | None, cur: list[tuple[str, float]]) -> float:
    if not prev:
        return 1.0
    a = {s: w for s, w in prev}
    b = {s: w for s, w in cur}
    keys = set(a) | set(b)
    return 0.5 * sum(abs(b.get(k, 0.0) - a.get(k, 0.0)) for k in keys)


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main() -> None:
    print("loading panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01")).dropna(subset=["fwd_ret", "pos_64"])
    feats = [c for c in FEATURE_NAMES if c not in SKIP_FEATS and c in panel.columns]
    mkt = daily_mkt(panel)
    print("precomputing ICs...", flush=True)
    ic_map = {f: rank_ic_daily(panel, f) for f in feats}

    rules = [
        "daily_ls",
        "daily_ls_div",
        "daily_abs2",
        "hold_q_ls",
        "hold_q_abs2",
    ]
    pnls: dict[str, list] = {k: [] for k in rules}
    turns: dict[str, list] = {k: [] for k in rules}
    last_w: dict[str, list | None] = {k: None for k in rules}
    q_summary = []

    quarters = pd.period_range("2010Q3", "2026Q3", freq="Q")
    for q in quarters:
        q_start, q_end = q.start_time, min(q.end_time, EVAL_END)
        test = panel[(panel["date"] >= q_start) & (panel["date"] <= q_end)].copy()
        train = panel[panel["date"] < q_start]
        if test["date"].nunique() < 15 or test["date"].min() < BT_START:
            continue

        pick, _ = choose_model(mkt, q_start)
        if pick == "ridge" and train["date"].nunique() >= MIN_TRAIN_DAYS:
            ranked = []
            for f in feats:
                st = ic_stats(ic_map[f].loc[ic_map[f].index < q_start])
                ranked.append({"feature": f, **st})
            rtab = pd.DataFrame(ranked).sort_values("t", key=lambda s: s.abs(), ascending=False)
            usable = rtab[(rtab["t"].abs() >= 1.5) & rtab["ic"].notna()]
            use_feats = usable["feature"].head(12).tolist() or rtab.head(8)["feature"].tolist()
            mu, sd, beta = fit_ridge(train, use_feats, l2=L2)
            test["score"] = predict_ridge(test, use_feats, mu, sd, beta)
        else:
            pick = "pos64"
            test["score"] = test["pos_64"]

        first = test.sort_values("date").groupby("date").head(1)
        # first day's full cross section
        d0 = test[test["date"] == test["date"].min()]
        q_ls = pick_ls(d0, d0["score"].to_numpy(), False)
        q_abs = pick_abs2(d0, d0["score"].to_numpy())
        q_hold = {"hold_q_ls": q_ls, "hold_q_abs2": q_abs}

        q_pnl = {k: [] for k in rules}
        names0 = {}
        for dt, day in test.groupby("date"):
            sc = day["score"].to_numpy()
            ws = {
                "daily_ls": pick_ls(day, sc, False),
                "daily_ls_div": pick_ls(day, sc, True),
                "daily_abs2": pick_abs2(day, sc),
                "hold_q_ls": q_hold["hold_q_ls"],
                "hold_q_abs2": q_hold["hold_q_abs2"],
            }
            if dt == test["date"].min():
                names0 = {k: [s for s, _ in w] for k, w in ws.items()}
            for k, w in ws.items():
                pnl = apply_hold(day, w)
                q_pnl[k].append((dt, pnl))
                pnls[k].append((dt, pnl))
                turns[k].append(turnover(last_w[k], w))
                last_w[k] = w

        rec = {
            "quarter": str(q),
            "model": pick,
            "n_symbols": int(test["symbol"].nunique()),
            "thin": test["symbol"].nunique() < 20,
            "names": names0,
        }
        for k in rules:
            rec[k] = metrics_from_pnl(pd.Series({d: v for d, v in q_pnl[k]}))["return"]
        q_summary.append(rec)
        print(
            f"{q} {pick:5} ls={rec['daily_ls']:+.1%} div={rec['daily_ls_div']:+.1%} "
            f"abs={rec['daily_abs2']:+.1%} qls={rec['hold_q_ls']:+.1%} qabs={rec['hold_q_abs2']:+.1%} "
            f"names={names0.get('daily_ls')}",
            flush=True,
        )

    def ser(pairs):
        return pd.Series({d: v for d, v in pairs}).sort_index().dropna()

    core_end = pd.Timestamp("2026-06-30")
    full = {}
    core = {}
    tovr = {}
    for k in rules:
        s = ser(pnls[k])
        full[k] = {**metrics_from_pnl(s), "turnover": float(np.mean(turns[k])) if turns[k] else None}
        core[k] = metrics_from_pnl(s[s.index <= core_end])
        tovr[k] = float(np.mean(turns[k])) if turns[k] else None

    payload = {
        "rule": "same quarterly pos64/Ridge switch; then keep at most 2 names",
        "methods": {
            "daily_ls": "each day: strongest long + strongest short, 50/50",
            "daily_ls_div": "same, but short prefers a different sector",
            "daily_abs2": "each day: two largest |score|, sign of score, 50/50",
            "hold_q_ls": "quarter start: 1 long + 1 short, hold 3 months",
            "hold_q_abs2": "quarter start: top-2 |score|, hold 3 months",
        },
        "full": full,
        "through_2026q2": core,
        "turnover": tovr,
        "quarters": [
            {
                "quarter": r["quarter"],
                "model": r["model"],
                "n_symbols": r["n_symbols"],
                "thin": r["thin"],
                "daily_ls": fmt(r["daily_ls"], 4),
                "daily_ls_div": fmt(r["daily_ls_div"], 4),
                "daily_abs2": fmt(r["daily_abs2"], 4),
                "hold_q_ls": fmt(r["hold_q_ls"], 4),
                "hold_q_abs2": fmt(r["hold_q_abs2"], 4),
                "open_ls": r["names"].get("daily_ls"),
                "open_hold": r["names"].get("hold_q_ls"),
            }
            for r in q_summary
        ],
    }
    path = OUT_DIR / "two_name_select.json"
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2))
    print(json.dumps(clean({"full": full, "core": core, "turnover": tovr}), indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
