#!/usr/bin/env python3
"""Dual-sleeve & scale-final search — different from pure news books.

A) dual: final_scheme overlay (1lot+50k+BOOK) + news satellite * weight
B) scale: final legs only; lots=1, or agree_lots if news side agrees

Evaluate on combined equity / margin. News half-doc still required for satellite.
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

from blend_book import book_ratio  # noqa: E402
from common import DATA_DIR, RESULTS_ROOT  # noqa: E402
from family_rotate import year_pack  # noqa: E402
from final_scheme_2026_blotter import load_px  # noqa: E402
from linear_ridge_walkforward import build_panel  # noqa: E402
from pos64_4name_2026_blotter import meta_of  # noqa: E402
from pos64_margin_cap_search import attach_margin  # noqa: E402
from pos64_margin_integer_search import CAPITAL, to_log  # noqa: E402
from pos64_rsi_overlay_2026_blotter import (  # noqa: E402
    BOOK_THR,
    MARGIN_CAP,
    pick_overlay,
)
from rule_factor_mine import batch_aggregate, filter_docs, keep_single_symbol_docs  # noqa: E402
from rule_news_alt_search import (  # noqa: E402
    build_news_cache,
    hyst_update,
    lots_fixed1,
    lots_risk,
)
from rule_news_capacity_search import day_spread, dollar_multi, fees_multi  # noqa: E402
from rule_news_quad_search import attach_vol  # noqa: E402
from six_name_select import pick_ls_n  # noqa: E402

OUT = RESULTS_ROOT / "rule_capacity_dual"
BT_START = pd.Timestamp("2024-03-08")
GATE_AVG_MARGIN = 55_000.0
GATE_PNL = 94_000.0
GATE_WORST = -7_000.0
GATE_2026_PNL = 41_431.0
GATE_2026_N5K = 2
GATE_SHARPE = 1.2
GATE_DD = -0.08
GATE_HALF = 0.8


def summarize(pnl: pd.Series, mar: pd.Series) -> dict:
    live = to_log(pnl / CAPITAL)
    bt = live[np.isfinite(pd.to_numeric(live, errors="coerce"))]
    st = year_pack(bt) if len(bt) >= 30 else {}
    on = mar > 0
    y26 = pnl[(pnl.index >= "2026-01-01") & (pnl.index <= "2026-12-31")]
    m26 = mar.reindex(y26.index).fillna(0.0)
    return {
        "sharpe": st.get("sharpe"),
        "return": st.get("return"),
        "max_dd": st.get("max_dd"),
        "pnl": round(float(pnl.sum()), 1),
        "avg_margin_on": round(float(mar[on].mean()), 1) if on.any() else 0.0,
        "worst": round(float(pnl.min()), 1) if len(pnl) else None,
        "n_le5k": int((pnl <= -5000).sum()),
        "y2026_pnl": round(float(y26.sum()), 1) if len(y26) else 0.0,
        "y2026_worst": round(float(y26.min()), 1) if len(y26) else None,
        "y2026_n_le5k": int((y26 <= -5000).sum()) if len(y26) else 0,
        "y2026_avg_mar": round(float(m26[m26 > 0].mean()), 1) if (m26 > 0).any() else 0.0,
        "live": live,
        "pnl_series": pnl,
        "mar_series": mar,
    }


def passes(r: dict) -> bool:
    if r.get("sharpe") is None or r.get("max_dd") is None or r.get("worst") is None:
        return False
    return (
        r["sharpe"] >= GATE_SHARPE
        and r["max_dd"] >= GATE_DD
        and r["avg_margin_on"] >= GATE_AVG_MARGIN
        and r["pnl"] >= GATE_PNL
        and r["worst"] >= GATE_WORST
        and r["y2026_pnl"] >= GATE_2026_PNL
        and r["y2026_n_le5k"] <= GATE_2026_N5K
    )


def run_final_book(panel: pd.DataFrame) -> pd.DataFrame:
    """Rebuild final_scheme-like daily pnl/margin with BOOK (news window)."""
    dates = sorted(panel["date"].unique())
    # precompute raw overlay dollar for BOOK ratio
    raw = {}
    meta_by = {}
    picks_by = {}
    for dt0, day0 in panel.groupby("date"):
        dt = pd.Timestamp(dt0)
        day = day0.drop_duplicates("symbol").reset_index(drop=True)
        d = attach_margin(day)
        d = d[d["lot_margin"].notna() & (d["lot_margin"] <= MARGIN_CAP)].reset_index(drop=True)
        picks = pick_overlay(d, use_close=False) if len(d) >= 2 else []
        picks_by[dt] = picks
        meta = meta_of(day)
        meta_by[dt] = meta
        close = {str(r.symbol): float(r.close) for r in day.itertuples() if np.isfinite(r.close)}
        ret = {str(r.symbol): float(r.fwd_ret) for r in day.itertuples() if np.isfinite(r.fwd_ret)}
        pos = [(s, 1 if w > 0 else -1) for s, w in picks]
        if pos:
            gross, mar, _ = dollar_multi(pos, close, ret)
            raw[dt] = gross
        else:
            raw[dt] = 0.0

    raw_s = pd.Series(raw).sort_index()
    live0 = to_log(raw_s / CAPITAL)
    ratio = book_ratio(live0.dropna()).reindex(live0.index)
    if ratio.notna().any():
        ratio = ratio.ffill(limit=1)

    prev = {}
    rows = []
    for dt in sorted(picks_by):
        picks = picks_by[dt]
        meta = meta_by[dt]
        day = panel[panel["date"] == dt].drop_duplicates("symbol")
        close = {str(r.symbol): float(r.close) for r in day.itertuples() if np.isfinite(r.close)}
        ret = {str(r.symbol): float(r.fwd_ret) for r in day.itertuples() if np.isfinite(r.fwd_ret)}
        b = float(ratio.loc[dt]) if dt in ratio.index and np.isfinite(ratio.loc[dt]) else None
        flat = b is not None and b >= BOOK_THR
        pos = [] if flat or not picks else [(s, 1 if w > 0 else -1) for s, w in picks]
        cur = {s: n for s, n in pos}
        fee = fees_multi(meta, prev, cur)
        prev = cur
        if pos:
            gross, mar, nl = dollar_multi(pos, close, ret)
            pnl = gross - fee
        else:
            mar, nl, pnl = 0.0, 0, (-fee if fee else 0.0)
        rows.append({"date": dt, "pnl": pnl, "mar": mar, "picks": picks, "flat": flat})
    return pd.DataFrame(rows).set_index("date")


def run_news_sat(cache, cfg) -> pd.DataFrame:
    state = None
    days_in = 0
    prev = {}
    rows = []
    for dt in cache["dates"]:
        info = cache["days"][dt]
        d = info["frame"]
        spread = day_spread(d["val"].to_numpy()) if len(d) >= 2 else 0.0
        cand = pick_ls_n(d, d["val"].to_numpy(), cfg["n_each"]) if len(d) >= 2 * cfg["n_each"] else []
        if cfg.get("confirm"):
            ma, cl = info["ma20"], info["close"]
            filt = []
            for s, w in cand:
                if s not in ma or s not in cl:
                    continue
                if w > 0 and cl[s] >= ma[s]:
                    filt.append((s, w))
                elif w < 0 and cl[s] <= ma[s]:
                    filt.append((s, w))
            cand = filt
        state, days_in = hyst_update(
            state, days_in, cand, spread, cfg["enter"], cfg["exit_"], cfg["min_hold"], set(d["symbol"])
        )
        names = state
        if cfg.get("util", 0) <= 0:
            pos = lots_fixed1(names, info["m1"], cfg.get("max_leg_margin"))
        else:
            strength = float(np.clip(spread / cfg["spread_ref"], 0.0, cfg["strength_cap"]))
            pos = lots_risk(
                names, info["m1"], info["risk1"], cfg["util"] * CAPITAL * strength,
                cfg["max_lots"], cfg["max_leg_risk"], cfg["max_leg_margin"],
            )
        cur = {s: n for s, n in pos}
        fee = fees_multi(info["meta"], prev, cur)
        prev = cur
        if pos:
            gross, mar, _ = dollar_multi(pos, info["close"], info["ret"])
            pnl = gross - fee
        else:
            mar, pnl = 0.0, (-fee if fee else 0.0)
        # news side map for scale-final
        side = {s: (1 if n > 0 else -1) for s, n in pos}
        rows.append({"date": dt, "pnl": pnl, "mar": mar, "side": side})
    return pd.DataFrame(rows).set_index("date")


def combine_dual(fin: pd.DataFrame, news: pd.DataFrame, w: float) -> dict:
    idx = fin.index.union(news.index)
    pf = fin["pnl"].reindex(idx).fillna(0.0)
    mf = fin["mar"].reindex(idx).fillna(0.0)
    pn = news["pnl"].reindex(idx).fillna(0.0)
    mn = news["mar"].reindex(idx).fillna(0.0)
    pnl = pf + w * pn
    mar = mf + w * mn
    return summarize(pnl, mar)


def run_scale_final(fin_rows: pd.DataFrame, news: pd.DataFrame, panel: pd.DataFrame, agree_lots: int) -> dict:
    prev = {}
    pnls = {}
    mars = {}
    for dt, row in fin_rows.iterrows():
        picks = row["picks"]
        flat = bool(row["flat"])
        ns = news.loc[dt, "side"] if dt in news.index else {}
        if isinstance(ns, float):
            ns = {}
        day = panel[panel["date"] == dt].drop_duplicates("symbol")
        meta = meta_of(day)
        close = {str(r.symbol): float(r.close) for r in day.itertuples() if np.isfinite(r.close)}
        ret = {str(r.symbol): float(r.fwd_ret) for r in day.itertuples() if np.isfinite(r.fwd_ret)}
        pos = []
        if not flat and picks:
            for s, w in picks:
                side = 1 if w > 0 else -1
                n = agree_lots if ns.get(s) == side else 1
                pos.append((s, side * n))
        cur = {s: n for s, n in pos}
        fee = fees_multi(meta, prev, cur)
        prev = cur
        if pos:
            gross, mar, _ = dollar_multi(pos, close, ret)
            pnl = gross - fee
        else:
            mar, pnl = 0.0, (-fee if fee else 0.0)
        pnls[dt] = pnl
        mars[dt] = mar
    return summarize(pd.Series(pnls).sort_index(), pd.Series(mars).sort_index())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("load...", flush=True)
    docs = pd.read_parquet(DATA_DIR / "rule_docs_rich.parquet")
    docs["report_date"] = pd.to_datetime(docs["report_date"])
    docs = keep_single_symbol_docs(filter_docs(docs, "no_dianping"))
    panel = build_panel(BT_START).dropna(subset=["fwd_ret"])
    panel = panel.merge(load_px(), on=["date", "symbol", "code"], how="left")
    panel = panel[panel["date"] >= BT_START].copy()
    panel = attach_vol(panel)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    fac = batch_aggregate(docs, "rule_edge", 30, dates, "uniform")
    cache = build_news_cache(panel, fac)

    print("final sleeve...", flush=True)
    fin = run_final_book(panel)
    print(
        f"final alone pnl={fin['pnl'].sum():.0f} avg_mar={fin.loc[fin.mar>0,'mar'].mean():.0f} "
        f"worst={fin['pnl'].min():.0f}",
        flush=True,
    )

    sat_cfgs = []
    for confirm, n_each, enter, exit_, util, max_lots in product(
        (True, False), (2, 3), (0.30, 0.40), (0.15,), (0.0, 0.008, 0.010), (1, 2),
    ):
        sat_cfgs.append(dict(
            confirm=confirm, n_each=n_each, enter=enter, exit_=exit_, min_hold=5,
            spread_ref=0.30, strength_cap=1.0, max_leg_margin=50_000.0,
            util=util, max_lots=max_lots, max_leg_risk=3000.0,
        ))
    # V2-like
    sat_cfgs.append(dict(
        confirm=False, n_each=2, enter=0.30, exit_=0.15, min_hold=5,
        spread_ref=0.30, strength_cap=1.2, max_leg_margin=50_000.0,
        util=0.08, max_lots=1, max_leg_risk=5000.0,
    ))

    weights = (0.25, 0.35, 0.5, 0.75, 1.0, 1.25)
    rows = []
    news_cache_runs = {}

    print(f"sats={len(sat_cfgs)} weights={len(weights)}", flush=True)
    for i, scfg in enumerate(sat_cfgs):
        key = tuple(sorted(scfg.items()))
        news = run_news_sat(cache, scfg)
        news_cache_runs[key] = news
        for w in weights:
            r = combine_dual(fin, news, w)
            name = (
                f"dual|conf{int(scfg['confirm'])}|n{scfg['n_each']}|e{scfg['enter']}"
                f"|u{scfg['util']}|max{scfg['max_lots']}|w{w}"
            )
            rows.append({"name": name, "kind": "dual", "w": w, **scfg,
                         **{k: r[k] for k in r if k not in ("live", "pnl_series", "mar_series")},
                         "pass": passes(r)})
        if (i + 1) % 10 == 0:
            print(f"  sat {i+1}/{len(sat_cfgs)}", flush=True)

    # scale-final
    print("scale-final...", flush=True)
    for scfg in sat_cfgs:
        if scfg["util"] not in (0.0, 0.01) and scfg["max_lots"] != 1:
            continue
        key = tuple(sorted(scfg.items()))
        news = news_cache_runs[key]
        for al in (2, 3):
            r = run_scale_final(fin, news, panel, al)
            name = (
                f"scale|conf{int(scfg['confirm'])}|n{scfg['n_each']}|e{scfg['enter']}"
                f"|u{scfg['util']}|agree{al}"
            )
            rows.append({"name": name, "kind": "scale", "w": al, **scfg,
                         **{k: r[k] for k in r if k not in ("live", "pnl_series", "mar_series")},
                         "pass": passes(r)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "all_candidates.csv", index=False)
    ok = df[df["pass"]].sort_values(["y2026_pnl", "pnl"], ascending=False)
    ok.to_csv(OUT / "passed.csv", index=False)
    near = df[
        (df.avg_margin_on >= 50_000)
        & (df.y2026_pnl >= 41_000)
        & (df.worst >= -8_000)
        & (df.sharpe >= 1.2)
    ].sort_values(["worst", "avg_margin_on"], ascending=[False, False])
    near.to_csv(OUT / "near_miss.csv", index=False)
    print(f"pass={len(ok)} near={len(near)} / {len(df)}", flush=True)
    if len(ok):
        print(ok[["name", "pnl", "y2026_pnl", "avg_margin_on", "worst", "sharpe", "y2026_n_le5k"]].head(10).to_string(index=False))
    elif len(near):
        print("near:", near[["name", "pnl", "y2026_pnl", "avg_margin_on", "worst", "sharpe"]].head(10).to_string(index=False))

    # show best by kind
    for kind, g in df.groupby("kind"):
        b = g.sort_values(["pass", "y2026_pnl", "worst"], ascending=[False, False, False]).iloc[0]
        print(
            f"[{kind}] pass_any={g['pass'].any()} y2026={b.y2026_pnl} pnl={b.pnl} "
            f"mar={b.avg_margin_on} worst={b.worst} sh={b.sharpe} :: {b['name']}",
            flush=True,
        )

    report = {"n": len(df), "n_pass": int(len(ok)), "n_near": int(len(near)), "checked": []}
    check = ok.head(8) if len(ok) else near.head(6)
    paths = docs["path"].drop_duplicates().to_numpy()
    for _, row in check.iterrows():
        scfg = {k: row[k] for k in (
            "confirm", "n_each", "enter", "exit_", "min_hold", "spread_ref", "strength_cap",
            "max_leg_margin", "util", "max_lots", "max_leg_risk",
        )}
        for k in ("enter", "exit_", "util", "strength_cap", "spread_ref", "max_leg_risk", "max_leg_margin"):
            scfg[k] = float(scfg[k])
        for k in ("n_each", "min_hold", "max_lots"):
            scfg[k] = int(scfg[k])
        scfg["confirm"] = bool(scfg["confirm"])
        hs = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            keep = set(rng.choice(paths, size=max(1, len(paths) // 2), replace=False))
            sub = docs[docs["path"].isin(keep)]
            fac2 = batch_aggregate(sub, "rule_edge", 30, dates, "uniform")
            cache2 = build_news_cache(panel, fac2)
            news2 = run_news_sat(cache2, scfg)
            if row["kind"] == "dual":
                rr = combine_dual(fin, news2, float(row["w"]))
            else:
                rr = run_scale_final(fin, news2, panel, int(row["w"]))
            hs.append(rr["sharpe"])
        med = float(np.nanmedian([x for x in hs if x is not None]))
        item = {
            "name": row["name"], "pass": bool(row["pass"]),
            "pnl": float(row["pnl"]), "y2026_pnl": float(row["y2026_pnl"]),
            "avg_margin_on": float(row["avg_margin_on"]), "worst": float(row["worst"]),
            "sharpe": float(row["sharpe"]), "half": hs, "half_med": med,
            "half_ok": bool(med >= GATE_HALF),
        }
        report["checked"].append(item)
        print(f"half {row['name']}: {hs} med={med:.3f}", flush=True)
        if item["pass"] and item["half_ok"]:
            report["best"] = item
            # export combined daily for dual using full news
            key = tuple(sorted(scfg.items()))
            news = news_cache_runs[key]
            if row["kind"] == "dual":
                full = combine_dual(fin, news, float(row["w"]))
            else:
                full = run_scale_final(fin, news, panel, int(row["w"]))
            pd.DataFrame({
                "日期": [str(pd.Timestamp(i).date()) for i in full["pnl_series"].index],
                "账户日盈亏": full["pnl_series"].to_numpy(),
                "合计保证金": full["mar_series"].reindex(full["pnl_series"].index).fillna(0).to_numpy(),
                "账户日收益": full["live"].reindex(full["pnl_series"].index).to_numpy(),
            }).to_csv(OUT / "dual_best_daily.csv", index=False)
            y = full["pnl_series"]
            y = y[y.index >= "2026-01-01"]
            m = full["mar_series"].reindex(y.index).fillna(0)
            pd.DataFrame({
                "日期": [str(pd.Timestamp(i).date()) for i in y.index],
                "账户日盈亏": y.to_numpy(),
                "合计保证金": m.to_numpy(),
            }).to_csv(OUT / "dual_best_2026_daily.csv", index=False)
            Path("/home/workspace/lab/UniFutures/data/infer/results/daily/linear").mkdir(parents=True, exist_ok=True)
            pd.read_csv(OUT / "dual_best_2026_daily.csv").to_csv(
                "/home/workspace/lab/UniFutures/data/infer/results/daily/linear/news_dual_best_2026_daily.csv",
                index=False,
            )
            break

    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT, flush=True)
    if report.get("best"):
        print("BEST", report["best"]["name"], flush=True)
    else:
        print("NO full dual/scale pass.", flush=True)


if __name__ == "__main__":
    main()
