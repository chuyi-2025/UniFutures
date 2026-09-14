#!/usr/bin/env python3
"""Phase-3 weather search: emphasize recent (2024+) robustness + book search.

Extends phase2 grid, scores legs with:
  full Sharpe, half-sample, IC/IR, AND explicit 2024+ Sharpe.
Avoids CF-style traps (loose thr + weak front half + high activity).
Searches EW books from top legs per symbol.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULT_DIR, SYMBOLS, ensure_dirs  # noqa: E402
from search_weather_strategies import (  # noqa: E402
    factor_series,
    make_position,
    max_dd,
    prepare_features,
    sharpe,
)
from search_weather_phase2 import add_horizons, ic_metrics, run_leg, time_split_sharpe  # noqa: E402

OUT = RESULT_DIR / "search"
RECENT = pd.Timestamp("2024-01-01")


def period_sharpe(pnl: pd.Series, start=None, end=None) -> float | None:
    s = pnl.sort_index()
    if start is not None:
        s = s[s.index >= start]
    if end is not None:
        s = s[s.index < end]
    return sharpe(s)


def score_leg(row: dict) -> float:
    """Higher is better. Heavy weight on recent + half-sample floor."""
    q = 0.0
    q += 1.0 * (row.get("sharpe") or 0)
    q += 1.2 * (row.get("sh_2024p") or 0)  # recent emphasis
    q += 0.35 if row.get("oos_both_pos") else 0.0
    q += 0.45 * (row.get("ir5") or 0) + 0.25 * (row.get("ir1") or 0)
    q += 1.5 * abs(row.get("ic5") or 0) + 0.8 * abs(row.get("ic1") or 0)
    # floors
    if (row.get("sh_first") or 0) < 0.15:
        q -= 0.55
    if (row.get("sh_2024p") or -9) < 0.0:
        q -= 0.8
    if (row.get("sh_2024p") or -9) < 0.3:
        q -= 0.25
    # activity / thr traps
    act = row.get("active") or 0
    thr = row.get("thr") or 0
    mode = row.get("mode") or ""
    if mode in ("sign", "long_only") and thr < 1.0 and act > 400:
        q -= 0.5
    if row.get("max_dd") is not None and row["max_dd"] < -0.20:
        q -= 0.3
    # half disagreement
    a, b = row.get("sh_first"), row.get("sh_second")
    if a is not None and b is not None and a * b < 0:
        q -= 0.5
    return round(q, 4)


def eval_book_from_picks(panel: pd.DataFrame, picks: list[dict]) -> dict:
    parts = []
    for p in picks:
        bt = run_leg(
            panel[panel["symbol"] == p["symbol"]],
            p["factor"],
            p["mode"],
            float(p["thr"]),
            float(p["direction"]),
            int(p["hold"]),
            bool(p["pheno_only"]),
        )
        parts.append(bt[["date", "symbol", "signal", "pnl"]])
    legs = pd.concat(parts, ignore_index=True)
    book = legs.groupby("date")["pnl"].mean().sort_index()
    split = time_split_sharpe(book)
    sh24 = period_sharpe(book, RECENT)
    yearly = []
    for y, g in book.groupby(book.index.year):
        yearly.append({"year": int(y), "sh": sharpe(g), "ret": round(float(g.sum()), 4)})
    q = (split["sh_full"] or 0) + 1.3 * (sh24 or 0)
    if split.get("both_pos"):
        q += 0.35
    if (sh24 or -9) < 0.3:
        q -= 0.6
    return {
        "picks": picks,
        "book": book,
        "legs": legs,
        "sh_full": split["sh_full"],
        "sh_first": split["sh_first"],
        "sh_second": split["sh_second"],
        "oos_both_pos": split["both_pos"],
        "sh_2024p": sh24,
        "ret_full": round(float(book.sum()), 4),
        "ret_2024p": round(float(book[book.index >= RECENT].sum()), 4) if (book.index >= RECENT).any() else 0.0,
        "yearly": yearly,
        "quality": round(q, 4),
        "symbols": ",".join(p["symbol"] for p in picks),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=25.0)
    args = ap.parse_args()
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    t_end = time.time() + args.minutes * 60.0
    t0 = time.time()

    panel = pd.read_parquet(RESULT_DIR / "aligned_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"] >= "2018-01-01"].copy()
    panel = prepare_features(add_horizons(panel))
    print(f"panel rows={len(panel)} symbols={sorted(panel.symbol.unique())}", flush=True)

    factors = [
        "rh_z", "tmean_z", "precip_z", "neg_precip_z", "tmax_z", "tmin_z",
        "tmean_z_ma5", "tmean_z_ma10", "precip_z_ma5", "neg_precip_z_ma5",
        "precip_z_ma10", "dry_hot", "wet_cool", "heat_minus_precip", "frost_score", "gdd_z",
        "drought_spell", "dry_hot_cs", "precip_z_cs",
    ]
    # keep only available
    factors = [f for f in factors if f in ("neg_precip_z", "neg_precip_z_ma5") or f in panel.columns
               or f.replace("neg_", "") in panel.columns or True]
    modes = ["cont", "sign", "long_only"]
    thrs = [0.75, 1.0, 1.25, 1.5, 2.0]
    holds = [1, 2, 3, 5, 8, 10, 15]
    dirs = [1.0, -1.0]
    phenos = [True, False]

    rows: list[dict] = []
    n_try = 0
    print("phase3 per-symbol expanded grid...", flush=True)
    for sym in SYMBOLS:
        if time.time() >= t_end:
            break
        g0 = panel[panel["symbol"] == sym]
        for fname, mode, thr, hold, direction, pheno in product(factors, modes, thrs, holds, dirs, phenos):
            if time.time() >= t_end:
                break
            if mode == "cont":
                if thr != 1.0 or hold != 1:
                    continue
                thr_use, hold_use = 0.0, 1
            else:
                thr_use, hold_use = thr, hold
            n_try += 1
            bt = run_leg(g0, fname, mode, thr_use, direction, hold_use, pheno)
            live = bt.dropna(subset=["fwd_ret"])
            act = int((live["signal"].abs() > 1e-8).sum())
            if act < 40:
                continue
            pnl = live.set_index("date")["pnl"]
            split = time_split_sharpe(pnl)
            sh = split["sh_full"]
            if sh is None or sh < 0.2:
                continue
            sh24 = period_sharpe(pnl, RECENT)
            ic1 = ic_metrics(live["_fac"], live["fwd_ret"], live["date"])
            ic5 = ic_metrics(live["_fac"], live["fwd_ret_5"], live["date"])
            row = {
                "symbol": sym,
                "factor": fname,
                "mode": mode,
                "thr": thr_use,
                "hold": hold_use,
                "direction": direction,
                "pheno_only": pheno,
                "sharpe": round(sh, 3),
                "sh_first": None if split["sh_first"] is None else round(split["sh_first"], 3),
                "sh_second": None if split["sh_second"] is None else round(split["sh_second"], 3),
                "oos_both_pos": split["both_pos"],
                "sh_2024p": None if sh24 is None else round(sh24, 3),
                "ret": round(float(pnl.sum()), 4),
                "ret_2024p": round(float(pnl[pnl.index >= RECENT].sum()), 4),
                "max_dd": None if max_dd(pnl) is None else round(max_dd(pnl), 4),
                "active": act,
                "ic1": ic1["ic"],
                "ir1": ic1["ir"],
                "ic5": ic5["ic"],
                "ir5": ic5["ir"],
            }
            row["quality3"] = score_leg(row)
            rows.append(row)
            if n_try % 2000 == 0:
                best = max((r["quality3"] for r in rows), default=None)
                print(f"  try={n_try} kept={len(rows)} best_q3={best} left={t_end-time.time():.0f}s", flush=True)

    res = pd.DataFrame(rows)
    if res.empty:
        print("empty leg search", flush=True)
        return
    res = res.sort_values(["quality3", "sh_2024p", "sharpe"], ascending=False)
    res.to_csv(OUT / "phase3_per_symbol_schemes.csv", index=False)

    # tiers emphasizing recent
    robust = res[
        (res.sharpe >= 0.6)
        & (res.oos_both_pos)
        & (res.sh_first.fillna(0) >= 0.15)
        & (res.sh_2024p.fillna(-9) >= 0.3)
        & (res.ir5.fillna(-9) >= 0.08)
    ].copy()
    soft = res[
        (res.sharpe >= 0.5)
        & (res.oos_both_pos)
        & (res.sh_2024p.fillna(-9) >= 0.2)
        & ((res.ir1.fillna(-9) >= 0.08) | (res.ir5.fillna(-9) >= 0.08) | (res.sh_2024p.fillna(-9) >= 0.5))
    ].copy()
    robust.to_csv(OUT / "phase3_robust.csv", index=False)
    soft.to_csv(OUT / "phase3_soft.csv", index=False)
    print(f"legs kept={len(res)} robust={len(robust)} soft={len(soft)}", flush=True)

    # top candidates per symbol
    cand_by_sym: dict[str, list[dict]] = {}
    for sym in SYMBOLS:
        pool = robust[robust.symbol == sym]
        if pool.empty:
            pool = soft[soft.symbol == sym]
        if pool.empty:
            pool = res[(res.symbol == sym) & (res.oos_both_pos) & (res.sh_2024p.fillna(-9) > 0)].head(30)
        if pool.empty:
            pool = res[res.symbol == sym].head(20)
        # diversify factors
        picks_sym = []
        seen_fac = set()
        for _, r in pool.iterrows():
            key = (r["factor"], r["mode"], round(float(r["direction"]), 1))
            if key in seen_fac and len(picks_sym) >= 2:
                continue
            seen_fac.add(key)
            picks_sym.append({
                "symbol": sym,
                "factor": str(r["factor"]),
                "mode": str(r["mode"]),
                "thr": float(r["thr"]),
                "hold": int(r["hold"]),
                "direction": float(r["direction"]),
                "pheno_only": bool(r["pheno_only"]),
                "sharpe": float(r["sharpe"]),
                "sh_2024p": None if pd.isna(r["sh_2024p"]) else float(r["sh_2024p"]),
                "quality3": float(r["quality3"]),
            })
            if len(picks_sym) >= 5:
                break
        cand_by_sym[sym] = picks_sym
        print(f"  {sym} candidates={len(picks_sym)} top={picks_sym[0] if picks_sym else None}", flush=True)

    # Book search: single best per symbol + combinations of 2-4 symbols using top-2 each
    print("phase3 book search...", flush=True)
    book_rows = []
    # 1) best-per-symbol EW
    best_each = []
    for sym in SYMBOLS:
        if cand_by_sym.get(sym):
            best_each.append(cand_by_sym[sym][0])
    if best_each:
        ev = eval_book_from_picks(panel, best_each)
        book_rows.append({**{k: ev[k] for k in ev if k not in ("book", "legs", "picks", "yearly")},
                          "label": "best_each_ew", "n_legs": len(best_each),
                          "picks_json": json.dumps(ev["picks"], ensure_ascii=False)})
        best_each_ev = ev
    else:
        best_each_ev = None

    # 2) all combinations of symbol subsets size 2..4, each using top candidate
    top1 = {s: c[0] for s, c in cand_by_sym.items() if c}
    for k in range(2, len(top1) + 1):
        for subset in combinations(sorted(top1.keys()), k):
            if time.time() >= t_end:
                break
            picks = [top1[s] for s in subset]
            ev = eval_book_from_picks(panel, picks)
            book_rows.append({
                **{kk: ev[kk] for kk in ev if kk not in ("book", "legs", "picks", "yearly")},
                "label": "top1_" + "_".join(subset),
                "n_legs": len(picks),
                "picks_json": json.dumps(ev["picks"], ensure_ascii=False),
            })

    # 3) for each symbol, try top-3 alternate on full best_each replacement
    if best_each:
        for i, base_leg in enumerate(best_each):
            sym = base_leg["symbol"]
            for alt in cand_by_sym.get(sym, [])[1:3]:
                if time.time() >= t_end:
                    break
                picks = list(best_each)
                picks[i] = alt
                ev = eval_book_from_picks(panel, picks)
                book_rows.append({
                    **{kk: ev[kk] for kk in ev if kk not in ("book", "legs", "picks", "yearly")},
                    "label": f"swap_{sym}_{alt['factor']}_{alt['mode']}_{alt['thr']}",
                    "n_legs": len(picks),
                    "picks_json": json.dumps(ev["picks"], ensure_ascii=False),
                })

    # 4) random-ish mixes of top-2 per symbol while time remains
    rng = np.random.default_rng(20260913)
    while time.time() < t_end and cand_by_sym:
        syms = [s for s, c in cand_by_sym.items() if c]
        if len(syms) < 2:
            break
        k = int(rng.integers(2, len(syms) + 1))
        chosen = list(rng.choice(syms, size=k, replace=False))
        picks = []
        for s in chosen:
            opts = cand_by_sym[s]
            picks.append(opts[int(rng.integers(0, min(3, len(opts))))])
        ev = eval_book_from_picks(panel, picks)
        book_rows.append({
            **{kk: ev[kk] for kk in ev if kk not in ("book", "legs", "picks", "yearly")},
            "label": "rand_" + "_".join(chosen),
            "n_legs": len(picks),
            "picks_json": json.dumps(ev["picks"], ensure_ascii=False),
        })

    books = pd.DataFrame(book_rows)
    if books.empty:
        print("no books", flush=True)
        return
    books = books.sort_values(["quality", "sh_2024p", "sh_full"], ascending=False)
    books.to_csv(OUT / "phase3_books.csv", index=False)

    # select: require recent sh>=0.5 and full>=0.8 if possible
    hard = books[(books.sh_full.fillna(0) >= 0.8) & (books.sh_2024p.fillna(-9) >= 0.5) & (books.oos_both_pos == True)]
    if hard.empty:
        hard = books[(books.sh_full.fillna(0) >= 0.7) & (books.sh_2024p.fillna(-9) >= 0.4)]
    if hard.empty:
        hard = books
    winner = hard.iloc[0]
    picks = json.loads(winner["picks_json"])
    final = eval_book_from_picks(panel, picks)

    # compare to v2
    v2_path = OUT / "picked_schemes_v2.json"
    v2 = json.loads(v2_path.read_text()) if v2_path.exists() else None

    # save phase3 pick as v3
    pd.DataFrame({"日期": final["book"].index.strftime("%Y-%m-%d"), "pnl": final["book"].values}).to_csv(
        OUT / "picked_book_daily_v3.csv", index=False
    )
    final["legs"].assign(日期=lambda d: d["date"].dt.strftime("%Y-%m-%d")).drop(columns=["date"]).to_csv(
        OUT / "picked_legs_daily_v3.csv", index=False
    )

    # combo impact vs v1/v2
    dual_p = Path("/home/workspace/lab/UniFutures/news/data/results/rule_capacity_dual/audit/dual_gated_v1_proper_book_daily.csv")
    combo_note = {}
    if dual_p.exists():
        dual = pd.read_csv(dual_p)
        dual["日期"] = pd.to_datetime(dual["日期"])
        dual = dual.set_index("日期")
        idx = dual.index
        EQUITY = 3_000_000.0
        W = 500_000.0
        pnl_dual = dual["账户日盈亏"].astype(float)
        dual_m = dual["合计保证金"].astype(float)
        macro = pd.read_csv("/home/workspace/lab/UniFutures/macro/data/results/search_account/best_account_daily.csv")
        macro["日期"] = pd.to_datetime(macro["日期"])
        macro = macro.set_index("日期")
        macro_g = macro["账户日盈亏"].astype(float).reindex(idx).fillna(0) * (0.12 / 0.35)
        macro_g = macro_g * (0.5 + 0.5 * (dual_m.reindex(idx).fillna(0) > 0).astype(float))

        def wx(path):
            w = pd.read_csv(path)
            w["日期"] = pd.to_datetime(w["日期"])
            return w.set_index("日期")["pnl"].reindex(idx).fillna(0) * W

        w1 = wx(OUT / "picked_book_daily.csv") if (OUT / "picked_book_daily.csv").exists() else None
        w2 = wx(OUT / "picked_book_daily_v2.csv") if (OUT / "picked_book_daily_v2.csv").exists() else None
        w3 = final["book"].reindex(idx).fillna(0) * W

        def pack(s):
            r = s / EQUITY
            return {
                "pnl": round(float(s.sum()), 1),
                "ret": round(float(r.sum()), 4),
                "sharpe": None if sharpe(r) is None else round(sharpe(r), 3),
            }

        combo_note = {
            "weather_v3_proxy": pack(w3),
            "combo_v3": pack(pnl_dual + macro_g + w3),
        }
        if w1 is not None:
            combo_note["weather_v1_proxy"] = pack(w1)
            combo_note["combo_v1"] = pack(pnl_dual + macro_g + w1)
        if w2 is not None:
            combo_note["weather_v2_proxy"] = pack(w2)
            combo_note["combo_v2"] = pack(pnl_dual + macro_g + w2)
        combo_note["combo_no_weather"] = pack(pnl_dual + macro_g)

    report = {
        "minutes": args.minutes,
        "elapsed_s": round(time.time() - t0, 1),
        "n_try": n_try,
        "n_legs_kept": int(len(res)),
        "n_robust": int(len(robust)),
        "n_soft": int(len(soft)),
        "n_books": int(len(books)),
        "winner": {
            "label": winner["label"],
            "picks": picks,
            "sh_full": final["sh_full"],
            "sh_2024p": final["sh_2024p"],
            "sh_first": final["sh_first"],
            "sh_second": final["sh_second"],
            "oos_both_pos": final["oos_both_pos"],
            "ret_full": final["ret_full"],
            "ret_2024p": final["ret_2024p"],
            "quality": final["quality"],
            "yearly": final["yearly"],
        },
        "v2_baseline": v2,
        "top_books": books.head(15).drop(columns=["picks_json"], errors="ignore").to_dict(orient="records"),
        "top_legs": res.head(20).to_dict(orient="records"),
        "per_symbol_best": {s: (cand_by_sym[s][0] if cand_by_sym.get(s) else None) for s in SYMBOLS},
        "combo_impact": combo_note,
        "better_than_v2_recent": bool(
            v2 is None
            or (final["sh_2024p"] or -9) > float(v2.get("sh_2024p") or -9) + 0.05
            or (
                abs((final["sh_2024p"] or 0) - float(v2.get("sh_2024p") or 0)) <= 0.05
                and (final["sh_full"] or 0) > float(v2.get("sh_full") or 0) + 0.05
            )
        ),
    }
    (OUT / "phase3_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (OUT / "picked_schemes_v3.json").write_text(
        json.dumps(
            {
                "label": winner["label"],
                "picks": picks,
                "sh_full": final["sh_full"],
                "sh_2024p": final["sh_2024p"],
                "ret_full": final["ret_full"],
                "ret_2024p": final["ret_2024p"],
                "source": "phase3",
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    print("\nWINNER", winner["label"], "sh_full", final["sh_full"], "sh_2024p", final["sh_2024p"], "picks", picks, flush=True)
    print("TOP5 books:", flush=True)
    print(books.head(5)[["label", "symbols", "sh_full", "sh_2024p", "oos_both_pos", "quality"]].to_string(index=False), flush=True)
    if combo_note:
        print("combo_impact", json.dumps(combo_note, ensure_ascii=False), flush=True)
    print(f"done elapsed={time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
