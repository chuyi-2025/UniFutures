#!/usr/bin/env python3
"""Rich audit of Dual-Gated-V1: multi-year, generalization, neighborhoods, simplifications.

Outputs under news/data/results/rule_capacity_dual/audit/
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

from common import DATA_DIR, RESULTS_ROOT  # noqa: E402
from family_rotate import year_pack  # noqa: E402
from final_scheme_2026_blotter import load_px  # noqa: E402
from linear_ridge_walkforward import build_panel  # noqa: E402
from pos64_margin_integer_search import CAPITAL, to_log  # noqa: E402
from rule_factor_mine import batch_aggregate, filter_docs, keep_single_symbol_docs  # noqa: E402
from rule_news_alt_search import build_news_cache, hyst_update, lots_fixed1, lots_risk  # noqa: E402
from rule_news_capacity_search import day_spread, dollar_multi, fees_multi  # noqa: E402
from rule_news_dual_search import run_final_book  # noqa: E402
from rule_news_quad_search import attach_vol  # noqa: E402
from six_name_select import pick_ls_n  # noqa: E402

OUT = RESULTS_ROOT / "rule_capacity_dual" / "audit"
BT_START = pd.Timestamp("2024-03-08")
OFFICIAL_2026 = Path(
    "/home/workspace/lab/UniFutures/data/infer/results/daily/linear/final_scheme_2026_daily.csv"
)
LINEAR_DIR = Path("/home/workspace/lab/UniFutures/data/infer/results/daily/linear")

# Baseline Dual-Gated-V1
BASE_NEWS = dict(
    n_each=2, enter=0.40, exit_=0.15, min_hold=5, spread_ref=0.30, strength_cap=1.0,
    max_leg_margin=50_000.0, util=0.01, max_lots=2, max_leg_risk=3000.0,
)
BASE_W = 1.25


def metrics(pnl: pd.Series, mar: pd.Series) -> dict:
    live = to_log(pnl / CAPITAL)
    bt = live[np.isfinite(pd.to_numeric(live, errors="coerce"))]
    st = year_pack(bt) if len(bt) >= 20 else {}
    on = mar > 0
    y26 = pnl[(pnl.index >= "2026-01-01") & (pnl.index <= "2026-12-31")]
    return {
        "sharpe": st.get("sharpe"),
        "return": st.get("return"),
        "max_dd": st.get("max_dd"),
        "pnl": float(pnl.sum()),
        "avg_margin_on": float(mar[on].mean()) if on.any() else 0.0,
        "worst": float(pnl.min()) if len(pnl) else None,
        "n_le5k": int((pnl <= -5000).sum()),
        "n_le7k": int((pnl <= -7000).sum()),
        "cash": float((~on).mean()) if len(mar) else 1.0,
        "y2026_pnl": float(y26.sum()) if len(y26) else 0.0,
        "y2026_worst": float(y26.min()) if len(y26) else None,
        "y2026_n_le5k": int((y26 <= -5000).sum()) if len(y26) else 0,
        "days": int(len(pnl)),
        "active_days": int(on.sum()),
    }


def yearly_table(pnl: pd.Series, mar: pd.Series) -> list[dict]:
    rows = []
    for y, g in pnl.groupby(pnl.index.year):
        m = mar.reindex(g.index).fillna(0.0)
        live = to_log(g / CAPITAL)
        bt = live[np.isfinite(pd.to_numeric(live, errors="coerce"))]
        st = year_pack(bt) if len(bt) >= 10 else {}
        on = m > 0
        rows.append({
            "year": int(y),
            "pnl": float(g.sum()),
            "return": st.get("return"),
            "sharpe": st.get("sharpe"),
            "max_dd": st.get("max_dd"),
            "worst": float(g.min()),
            "n_le5k": int((g <= -5000).sum()),
            "avg_margin_on": float(m[on].mean()) if on.any() else 0.0,
            "cash": float((~on).mean()),
            "days": int(len(g)),
        })
    return rows


def quarterly_table(pnl: pd.Series) -> list[dict]:
    rows = []
    s = pnl.copy()
    s.index = pd.to_datetime(s.index)
    for per, g in s.groupby(s.index.to_period("Q")):
        if len(g) < 14:
            continue
        live = to_log(g / CAPITAL)
        st = year_pack(live)
        rows.append({
            "quarter": str(per),
            "pnl": float(g.sum()),
            "sharpe": st.get("sharpe"),
            "return": st.get("return"),
            "worst": float(g.min()),
            "days": int(len(g)),
        })
    return rows


def news_series(cache: dict, cfg: dict) -> pd.DataFrame:
    state = None
    days_in = 0
    prev: dict = {}
    rows = []
    for dt in cache["dates"]:
        info = cache["days"][dt]
        d = info["frame"]
        spread = day_spread(d["val"].to_numpy()) if len(d) >= 2 else 0.0
        cand = pick_ls_n(d, d["val"].to_numpy(), cfg["n_each"]) if len(d) >= 2 * cfg["n_each"] else []
        state, days_in = hyst_update(
            state, days_in, cand, spread, cfg["enter"], cfg["exit_"], cfg["min_hold"], set(d["symbol"])
        )
        if state and cfg.get("util", 0) > 0:
            strength = float(np.clip(spread / cfg["spread_ref"], 0.0, cfg["strength_cap"]))
            pos = lots_risk(
                state, info["m1"], info["risk1"], cfg["util"] * CAPITAL * strength,
                cfg["max_lots"], cfg["max_leg_risk"], cfg["max_leg_margin"],
            )
        else:
            pos = lots_fixed1(state, info["m1"], cfg.get("max_leg_margin"))
        cur = {s: n for s, n in pos}
        fee = fees_multi(info["meta"], prev, cur)
        prev = cur
        if pos:
            gross, mar, _ = dollar_multi(pos, info["close"], info["ret"])
            pnl = gross - fee
        else:
            mar, pnl = 0.0, (-fee if fee else 0.0)
        rows.append({"date": pd.Timestamp(dt), "pnl_n": pnl, "mar_n": mar})
    return pd.DataFrame(rows).set_index("date")


def gated_combine(fin: pd.DataFrame, news: pd.DataFrame, w: float) -> tuple[pd.Series, pd.Series]:
    """fin columns pnl_f, mar_f; news pnl_n, mar_n."""
    idx = fin.index.union(news.index)
    f = fin.reindex(idx).fillna(0.0)
    n = news.reindex(idx).fillna(0.0)
    mask = f["mar_f"] > 0
    pnl = f["pnl_f"] + np.where(mask, w * n["pnl_n"], 0.0)
    mar = f["mar_f"] + np.where(mask, w * n["mar_n"], 0.0)
    return pd.Series(pnl, index=idx), pd.Series(mar, index=idx)


def load_official_fin() -> pd.DataFrame:
    df = pd.read_csv(OFFICIAL_2026)
    df["date"] = pd.to_datetime(df["日期"])
    return pd.DataFrame({
        "pnl_f": pd.to_numeric(df["账户日盈亏"], errors="coerce").fillna(0.0).to_numpy(),
        "mar_f": pd.to_numeric(df["合计保证金"], errors="coerce").fillna(0.0).to_numpy(),
    }, index=df["date"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"base": {"news": BASE_NEWS, "w": BASE_W}}

    print("docs/panel...", flush=True)
    docs = pd.read_parquet(DATA_DIR / "rule_docs_rich.parquet")
    docs["report_date"] = pd.to_datetime(docs["report_date"])
    docs = keep_single_symbol_docs(filter_docs(docs, "no_dianping"))
    panel = build_panel(BT_START).dropna(subset=["fwd_ret"])
    panel = panel.merge(load_px(), on=["date", "symbol", "code"], how="left")
    panel = panel[panel["date"] >= BT_START].copy()
    panel = attach_vol(panel)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))

    print("news factor L30...", flush=True)
    fac = batch_aggregate(docs, "rule_edge", 30, dates, "uniform")
    cache = build_news_cache(panel, fac)
    news = news_series(cache, BASE_NEWS)

    print("rebuild final book...", flush=True)
    fin_rb = run_final_book(panel).rename(columns={"pnl": "pnl_f", "mar": "mar_f"})[
        ["pnl_f", "mar_f"]
    ]
    fin_off = load_official_fin()

    # --- A. rebuild vs official on 2026 overlap ---
    common = fin_rb.index.intersection(fin_off.index)
    a = fin_rb.loc[common, "pnl_f"]
    b = fin_off.loc[common, "pnl_f"]
    report["rebuild_vs_official_2026"] = {
        "corr_pnl": float(a.corr(b)) if a.std() > 0 and b.std() > 0 else None,
        "pnl_rebuild": float(a.sum()),
        "pnl_official": float(b.sum()),
        "pnl_diff": float(a.sum() - b.sum()),
        "worst_rebuild": float(a.min()),
        "worst_official": float(b.min()),
        "mae": float((a - b).abs().mean()),
    }
    print("rebuild vs official", report["rebuild_vs_official_2026"], flush=True)

    # Prefer official for 2026 slice; rebuilt for multi-year spine
    # Hybrid fin: rebuilt full, overwrite overlapping 2026 dates with official
    # (official may extend a few days past panel fwd_ret coverage)
    fin_hyb = fin_rb.copy()
    ov = fin_off.index.intersection(fin_hyb.index)
    fin_hyb.loc[ov, "pnl_f"] = fin_off.loc[ov, "pnl_f"]
    fin_hyb.loc[ov, "mar_f"] = fin_off.loc[ov, "mar_f"]
    report["hybrid_overwrite_days"] = int(len(ov))
    report["official_only_extra_days"] = [
        str(pd.Timestamp(x).date()) for x in fin_off.index.difference(fin_rb.index)
    ]

    # --- B. baseline curves ---
    variants = {}
    for tag, fin in [
        ("official_2026_only", fin_off),
        ("rebuild_full", fin_rb),
        ("hybrid_official2026", fin_hyb),
    ]:
        # align news
        if tag == "official_2026_only":
            n = news[news.index.year == 2026]
            f = fin
        else:
            n, f = news, fin
        pnl, mar = gated_combine(f, n, BASE_W)
        # also final alone
        fa = f["pnl_f"]
        fm = f["mar_f"]
        variants[tag] = {
            "dual": metrics(pnl, mar),
            "final_alone": metrics(fa, fm),
            "yearly_dual": yearly_table(pnl, mar),
            "yearly_final": yearly_table(fa, fm),
            "quarterly_dual": quarterly_table(pnl),
        }
        # save hybrid daily full
        if tag == "hybrid_official2026":
            pd.DataFrame({
                "日期": [str(pd.Timestamp(i).date()) for i in pnl.index],
                "账户日盈亏": pnl.to_numpy(),
                "合计保证金": mar.to_numpy(),
                "final盈亏": f.reindex(pnl.index)["pnl_f"].fillna(0).to_numpy(),
                "news门控加权盈亏": (pnl - f.reindex(pnl.index)["pnl_f"].fillna(0)).to_numpy(),
            }).to_csv(OUT / "dual_gated_v1_hybrid_daily.csv", index=False)
            y26 = pnl.index.year == 2026
            pd.DataFrame({
                "日期": [str(pd.Timestamp(i).date()) for i in pnl.index[y26]],
                "账户日盈亏": pnl[y26].to_numpy(),
                "合计保证金": mar[y26].to_numpy(),
            }).to_csv(OUT / "dual_gated_v1_2026_daily.csv", index=False)
            pd.DataFrame({
                "日期": [str(pd.Timestamp(i).date()) for i in pnl.index[y26]],
                "账户日盈亏": pnl[y26].to_numpy(),
                "合计保证金": mar[y26].to_numpy(),
            }).to_csv(LINEAR_DIR / "news_dual_gated_v1_2026_daily.csv", index=False)

    report["baselines"] = {
        k: {kk: vv for kk, vv in v.items() if kk not in ()}
        for k, v in variants.items()
    }
    # json-safe: already primitives in metrics

    # time half on hybrid
    pnl_h, mar_h = gated_combine(fin_hyb, news, BASE_W)
    mid = pnl_h.index[len(pnl_h) // 2]
    report["time_split_hybrid"] = {
        "mid": str(pd.Timestamp(mid).date()),
        "first": metrics(pnl_h[pnl_h.index <= mid], mar_h[mar_h.index <= mid]),
        "second": metrics(pnl_h[pnl_h.index > mid], mar_h[mar_h.index > mid]),
    }

    # --- C. half-doc / frac keep ---
    print("half-doc grid...", flush=True)
    paths = docs["path"].drop_duplicates().to_numpy()
    half_rows = []
    for frac in (0.25, 0.50, 0.75):
        shs_2026, shs_full, pnls_2026 = [], [], []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            keep = set(rng.choice(paths, size=max(1, int(len(paths) * frac)), replace=False))
            sub = docs[docs["path"].isin(keep)]
            fac2 = batch_aggregate(sub, "rule_edge", 30, dates, "uniform")
            cache2 = build_news_cache(panel, fac2)
            news2 = news_series(cache2, BASE_NEWS)
            # official 2026
            p26, m26 = gated_combine(fin_off, news2[news2.index.year == 2026], BASE_W)
            live26 = to_log(p26 / CAPITAL)
            sh26 = year_pack(live26)["sharpe"] if len(live26) >= 20 else None
            # hybrid full
            pf, mf = gated_combine(fin_hyb, news2, BASE_W)
            livef = to_log(pf / CAPITAL)
            shf = year_pack(livef[np.isfinite(pd.to_numeric(livef, errors="coerce"))])["sharpe"]
            shs_2026.append(sh26)
            shs_full.append(shf)
            pnls_2026.append(float(p26.sum()))
            print(f"  frac={frac} seed={seed} sh26={sh26} shFull={shf} pnl26={p26.sum():.0f}", flush=True)
        half_rows.append({
            "frac": frac,
            "sharpe_2026": shs_2026,
            "sharpe_full": shs_full,
            "pnl_2026": pnls_2026,
            "med_sh_2026": float(np.nanmedian(shs_2026)),
            "min_sh_2026": float(np.nanmin([x for x in shs_2026 if x is not None])),
            "med_sh_full": float(np.nanmedian(shs_full)),
            "min_sh_full": float(np.nanmin([x for x in shs_full if x is not None])),
            "med_pnl_2026": float(np.nanmedian(pnls_2026)),
            "frac_sh26_gt_1": float(np.mean([1 if x is not None and x >= 1 else 0 for x in shs_2026])),
            "frac_sh26_gt_0.8": float(np.mean([1 if x is not None and x >= 0.8 else 0 for x in shs_2026])),
        })
    report["half_docs"] = half_rows
    pd.DataFrame(half_rows).to_csv(OUT / "half_docs.csv", index=False)

    # --- D. weight neighborhood ---
    print("weight / param neighborhood...", flush=True)
    neigh = []
    for w in [0.75, 1.0, 1.1, 1.25, 1.35, 1.5, 1.75, 2.0]:
        p, m = gated_combine(fin_off, news[news.index.year == 2026], w)
        row = {"kind": "weight", "w": w, **metrics(p, m)}
        row["pass_2026"] = bool(
            row["y2026_pnl"] >= 41431 and row["avg_margin_on"] >= 55000
            and row["worst"] >= -7000 and row["y2026_n_le5k"] <= 2
        )
        neigh.append(row)
    # util / max_lots / enter simplifications on official 2026
    for util, max_lots, enter, max_leg_risk, label in [
        (0.01, 2, 0.40, 3000, "V1"),
        (0.01, 1, 0.40, 3000, "max1"),
        (0.0, 1, 0.40, 3000, "1lot_fixed"),  # util0 -> fixed1 in news_series
        (0.008, 2, 0.40, 3000, "u008"),
        (0.01, 2, 0.30, 3000, "e030"),
        (0.01, 2, 0.40, 2500, "mr2500"),
        (0.01, 2, 0.40, 4000, "mr4000"),
        (0.012, 2, 0.40, 3000, "u012"),
    ]:
        cfg = {**BASE_NEWS, "util": util, "max_lots": max_lots, "enter": enter, "max_leg_risk": float(max_leg_risk)}
        ns = news_series(cache, cfg)
        for w in (1.0, 1.25, 1.5):
            p, m = gated_combine(fin_off, ns[ns.index.year == 2026], w)
            row = {"kind": "cfg", "label": label, "w": w, "util": util, "max_lots": max_lots,
                   "enter": enter, "max_leg_risk": max_leg_risk, **metrics(p, m)}
            row["pass_2026"] = bool(
                row["y2026_pnl"] >= 41431 and row["avg_margin_on"] >= 55000
                and row["worst"] >= -7000 and row["y2026_n_le5k"] <= 2
            )
            neigh.append(row)
    # ungated dual (news always) for contrast
    for w in (0.5, 1.0, 1.25):
        n26 = news[news.index.year == 2026].reindex(fin_off.index).fillna(0)
        pnl = fin_off["pnl_f"] + w * n26["pnl_n"]
        mar = fin_off["mar_f"] + w * n26["mar_n"]
        row = {"kind": "ungated", "w": w, **metrics(pnl, mar)}
        row["pass_2026"] = bool(
            row["y2026_pnl"] >= 41431 and row["avg_margin_on"] >= 55000
            and row["worst"] >= -7000 and row["y2026_n_le5k"] <= 2
        )
        neigh.append(row)

    ndf = pd.DataFrame(neigh)
    ndf.to_csv(OUT / "neighborhood.csv", index=False)
    report["neighborhood_pass_2026"] = ndf[ndf["pass_2026"] == True][
        [c for c in ndf.columns if c in (
            "kind", "label", "w", "util", "max_lots", "enter", "y2026_pnl",
            "avg_margin_on", "worst", "y2026_n_le5k", "sharpe", "pass_2026",
        )]
    ].to_dict(orient="records")

    # --- E. simpler candidates half-doc (top simplifications that pass 2026) ---
    print("simple passers half-doc...", flush=True)
    simple_checks = []
    passed_simple = ndf[(ndf["kind"] == "cfg") & (ndf["pass_2026"] == True)].copy()
    # unique labels at w that pass
    seen = set()
    for _, r in passed_simple.sort_values(["label", "w"]).iterrows():
        key = (r["label"], r["w"])
        if key in seen:
            continue
        seen.add(key)
        cfg = {**BASE_NEWS, "util": float(r["util"]), "max_lots": int(r["max_lots"]),
               "enter": float(r["enter"]), "max_leg_risk": float(r["max_leg_risk"])}
        w = float(r["w"])
        hs = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            keep = set(rng.choice(paths, size=max(1, len(paths) // 2), replace=False))
            sub = docs[docs["path"].isin(keep)]
            fac2 = batch_aggregate(sub, "rule_edge", 30, dates, "uniform")
            cache2 = build_news_cache(panel, fac2)
            news2 = news_series(cache2, cfg)
            p26, _ = gated_combine(fin_off, news2[news2.index.year == 2026], w)
            live = to_log(p26 / CAPITAL)
            hs.append(year_pack(live)["sharpe"] if len(live) >= 20 else None)
        simple_checks.append({
            "label": r["label"], "w": w, "util": r["util"], "max_lots": int(r["max_lots"]),
            "y2026_pnl": float(r["y2026_pnl"]), "avg_margin_on": float(r["avg_margin_on"]),
            "worst": float(r["worst"]), "n5": int(r["y2026_n_le5k"]),
            "half50": hs, "half50_med": float(np.nanmedian([x for x in hs if x is not None])),
        })
        print(f"  simple {r['label']} w={w} half={hs}", flush=True)
        if len(simple_checks) >= 8:
            break
    report["simple_passers_half"] = simple_checks

    # --- F. contribution / overlap diagnostics on official 2026 ---
    n26 = news[news.index.year == 2026].reindex(fin_off.index).fillna(0)
    mask = fin_off["mar_f"] > 0
    both = mask & (n26["mar_n"] > 0)
    report["overlap_2026"] = {
        "final_on_days": int(mask.sum()),
        "news_on_days": int((n26["mar_n"] > 0).sum()),
        "both_on_days": int(both.sum()),
        "gated_news_days": int((mask & (n26["mar_n"] > 0)).sum()),
        "corr_daily_pnl": float(fin_off["pnl_f"].corr(n26["pnl_n"])),
        "corr_on_final_on": float(fin_off.loc[mask, "pnl_f"].corr(n26.loc[mask, "pnl_n"])) if mask.sum() > 5 else None,
        "news_extra_pnl_gated_w1": float((np.where(mask, n26["pnl_n"], 0)).sum()),
        "news_extra_pnl_gated_w125": float((np.where(mask, BASE_W * n26["pnl_n"], 0)).sum()),
        "avg_mar_final_on": float(fin_off.loc[mask, "mar_f"].mean()),
        "avg_mar_dual_on": float((fin_off["mar_f"] + np.where(mask, BASE_W * n26["mar_n"], 0))[
            (fin_off["mar_f"] + np.where(mask, BASE_W * n26["mar_n"], 0)) > 0
        ].mean()),
    }

    # gates checklist
    base_off = variants["official_2026_only"]["dual"]
    report["gates_checklist_official_2026"] = {
        "earn_more_vs_final": bool(base_off["y2026_pnl"] > variants["official_2026_only"]["final_alone"]["y2026_pnl"]),
        "capacity_ge_55k": bool(base_off["avg_margin_on"] >= 55000),
        "worst_ge_m7k": bool(base_off["worst"] >= -7000),
        "n5_le_2": bool(base_off["y2026_n_le5k"] <= 2),
        "half50_med_ge_0.8": bool(half_rows[1]["med_sh_2026"] >= 0.8),
        "metrics": base_off,
        "final_alone": variants["official_2026_only"]["final_alone"],
    }

    # recommend simplest passer
    simp = sorted(simple_checks, key=lambda x: (x["max_lots"], x["util"], abs(x["w"] - 1.0)))
    report["recommended_simple"] = simp[0] if simp else None

    def _py(o):
        if isinstance(o, dict):
            return {k: _py(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_py(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return o

    (OUT / "audit_report.json").write_text(
        json.dumps(_py(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", OUT, flush=True)
    print("gates", report["gates_checklist_official_2026"], flush=True)
    print("recommended_simple", report["recommended_simple"], flush=True)


if __name__ == "__main__":
    main()
