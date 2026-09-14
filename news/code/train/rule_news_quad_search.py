#!/usr/bin/env python3
"""Search news strategies aiming at the quad constraints vs final_scheme:

1) capacity: avg margin when on  > final (~47k)  → gate >= 55k
2) money:    full-window pnl     > final 2024+ (~94k)
3) daily loss: worst day >= -7k, 2026 days<=-5k <= 2
4) generalize: half-doc median Sharpe >= 0.8 on shortlist

Key change vs equal-margin multi-lot: risk / notional / per-leg margin caps
so cheap high-vol names cannot stack 5 lots.
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
from futures_lot_specs import BROKER_MARGIN, MULTIPLIER, lot_fee, lot_margin  # noqa: E402
from linear_ridge_walkforward import build_panel  # noqa: E402
from pos64_4name_2026_blotter import meta_of  # noqa: E402
from pos64_margin_integer_search import CAPITAL, to_log  # noqa: E402
from rule_factor_mine import batch_aggregate, filter_docs, keep_single_symbol_docs  # noqa: E402
from rule_news_capacity_search import (  # noqa: E402
    BT_START,
    day_spread,
    dollar_multi,
    fees_multi,
    one_lot_margin,
)
from six_name_select import pick_ls_n  # noqa: E402

OUT = RESULTS_ROOT / "rule_capacity_quad"
LINEAR_YEARLY = Path(
    "/home/workspace/lab/UniFutures/data/infer/results/daily/linear/final_scheme_yearly.csv"
)
LINEAR_2026 = Path(
    "/home/workspace/lab/UniFutures/data/infer/results/daily/linear/final_scheme_2026_daily.csv"
)

# Gates vs final_scheme
GATE_AVG_MARGIN = 55_000.0
GATE_PNL = 94_000.0
GATE_WORST = -7_000.0
GATE_2026_N5K = 2
GATE_SHARPE = 1.20
GATE_DD = -0.08
GATE_HALF_MED = 0.80


def attach_vol(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["symbol", "date"]).copy()
    p["ret1"] = p.groupby("symbol", sort=False)["close"].pct_change()
    p["vol20"] = p.groupby("symbol", sort=False)["ret1"].transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    # floor vol so sizing does not explode
    p["vol20"] = p["vol20"].clip(lower=0.005, upper=0.08)
    return p


def build_day_cache(panel: pd.DataFrame, fac: pd.DataFrame) -> dict:
    p = panel.merge(fac, on=["date", "symbol"], how="left")
    dates = sorted(p["date"].unique())
    days = {}
    for dt in dates:
        day = p[p["date"] == dt].drop_duplicates("symbol").reset_index(drop=True)
        mask = day["symbol"].isin(MULTIPLIER) & day["symbol"].isin(BROKER_MARGIN)
        uni = day[mask].copy()
        d = uni.dropna(subset=["val", "fwd_ret", "close"]).reset_index(drop=True)
        m1, vol, risk1 = [], [], []
        for sym, px, v in zip(d["symbol"], d["close"], d["vol20"]):
            mar = one_lot_margin(str(sym), float(px))
            m1.append(mar)
            vv = float(v) if np.isfinite(v) else 0.02
            vol.append(vv)
            mult = MULTIPLIER.get(str(sym))
            if mult is None or mar is None:
                risk1.append(None)
            else:
                risk1.append(float(mult) * float(px) * vv)
        d["m1"] = m1
        d["vol20"] = vol
        d["risk1"] = risk1
        d = d[d["m1"].notna() & d["risk1"].notna()].reset_index(drop=True)
        meta = meta_of(day)
        days[pd.Timestamp(dt)] = {
            "frame": d,
            "meta": meta,
            "ret": {str(r.symbol): float(r.fwd_ret) for r in d.itertuples() if np.isfinite(r.fwd_ret)},
            "close": {str(r.symbol): float(r.close) for r in d.itertuples()},
            "m1": {str(r.symbol): float(r.m1) for r in d.itertuples()},
            "risk1": {str(r.symbol): float(r.risk1) for r in d.itertuples()},
        }
    return {"dates": [pd.Timestamp(d) for d in dates], "days": days}


def allocate(
    names: list[tuple[str, float]],
    m1: dict[str, float],
    risk1: dict[str, float],
    *,
    budget_eff: float,
    max_lots: int,
    mode: str,
    max_leg_margin: float | None,
    max_leg_risk: float | None,
) -> list[tuple[str, int]]:
    """mode: margin | risk | hybrid (margin then clip by risk)."""
    if budget_eff <= 0 or not names:
        return []
    legs = []
    for sym, w in names:
        if sym not in m1 or sym not in risk1:
            continue
        legs.append((sym, 1.0 if w > 0 else -1.0, m1[sym], risk1[sym]))
    if not legs:
        return []
    n = len(legs)
    out = []
    for sym, sign, mar, r1 in legs:
        if mode == "risk":
            share = budget_eff / n  # interpret budget as total 1σ risk CNY
            raw = int(np.floor(share / max(r1, 1.0)))
        else:
            share = budget_eff / n  # margin CNY
            raw = int(np.floor(share / mar))
        lots = int(np.clip(raw, 0, max_lots))
        if max_leg_margin is not None and mar > 0:
            lots = min(lots, int(np.floor(max_leg_margin / mar)))
        if mode in ("hybrid", "risk") and max_leg_risk is not None and r1 > 0:
            lots = min(lots, int(np.floor(max_leg_risk / r1)))
        elif mode == "hybrid" and max_leg_risk is not None and r1 > 0:
            lots = min(lots, int(np.floor(max_leg_risk / r1)))
        # margin mode can still apply risk clip if max_leg_risk set
        if mode == "margin" and max_leg_risk is not None and r1 > 0:
            lots = min(lots, int(np.floor(max_leg_risk / r1)))
        lots = int(np.clip(lots, 0, max_lots))
        if lots >= 1:
            out.append((sym, int(sign * lots)))
    return out


def run_cfg(cache: dict, cfg: dict) -> dict:
    mode_sig = cfg["mode"]
    n_each = cfg["n_each"]
    enter, exit_, min_hold = cfg["enter"], cfg["exit_"], cfg["min_hold"]
    util, spread_ref = cfg["util"], cfg["spread_ref"]
    max_lots = cfg["max_lots"]
    scap = cfg["strength_cap"]
    alloc_mode = cfg["alloc"]
    max_leg_margin = cfg.get("max_leg_margin")
    max_leg_risk = cfg.get("max_leg_risk")

    state_names = None
    days_in = 0
    prev: dict[str, int] = {}
    pnls: list[tuple[pd.Timestamp, float]] = []
    margins_on: list[float] = []
    lots_on: list[int] = []

    for dt in cache["dates"]:
        info = cache["days"][dt]
        d = info["frame"]
        spread = day_spread(d["val"].to_numpy()) if len(d) >= 2 else 0.0
        cand = pick_ls_n(d, d["val"].to_numpy(), n_each) if len(d) >= 2 * n_each else []

        if mode_sig == "daily":
            names = cand if (spread >= enter and cand) else None
            days_in = 0
        else:
            if state_names is None:
                if spread >= enter and cand:
                    state_names, days_in = cand, 0
            else:
                days_in += 1
                elig = set(d["symbol"])
                state_names = [(s, w) for s, w in state_names if s in elig]
                if len(state_names) < 2:
                    state_names, days_in = None, 0
                elif days_in >= min_hold and spread < exit_:
                    state_names, days_in = None, 0
                elif days_in >= min_hold and spread >= enter and cand:
                    if {s for s, _ in state_names} != {s for s, _ in cand}:
                        state_names, days_in = cand, 0
            names = state_names

        strength = float(np.clip(spread / spread_ref, 0.0, scap)) if spread_ref > 0 else 0.0
        # risk mode: util * CAPITAL means target total 1σ risk (e.g. 0.008*1e6=8k)
        if alloc_mode == "risk":
            budget_eff = util * CAPITAL * strength
        else:
            budget_eff = util * CAPITAL * strength

        positions = (
            allocate(
                names,
                info["m1"],
                info["risk1"],
                budget_eff=budget_eff,
                max_lots=max_lots,
                mode=alloc_mode,
                max_leg_margin=max_leg_margin,
                max_leg_risk=max_leg_risk,
            )
            if names and budget_eff >= 500
            else []
        )
        cur = {s: n for s, n in positions}
        fee = fees_multi(info["meta"], prev, cur)
        prev = cur
        if not positions:
            pnls.append((dt, -fee if fee else 0.0))
            continue
        gross, margin, nlots = dollar_multi(positions, info["close"], info["ret"])
        pnls.append((dt, gross - fee))
        margins_on.append(margin)
        lots_on.append(nlots)

    pnl = pd.Series({d: v for d, v in pnls}).sort_index()
    live = to_log(pnl / CAPITAL)
    bt = live[np.isfinite(pd.to_numeric(live, errors="coerce"))]
    st = year_pack(bt) if len(bt) >= 30 else {}
    y26 = pnl[(pnl.index >= "2026-01-01") & (pnl.index <= "2026-12-31")]
    return {
        "sharpe": st.get("sharpe"),
        "return": st.get("return"),
        "max_dd": st.get("max_dd"),
        "pnl": round(float(pnl.sum()), 1),
        "avg_margin_on": round(float(np.mean(margins_on)), 1) if margins_on else 0.0,
        "p90_margin_on": round(float(np.quantile(margins_on, 0.9)), 1) if margins_on else 0.0,
        "avg_lots_on": round(float(np.mean(lots_on)), 2) if lots_on else 0.0,
        "cash": round(1.0 - len(margins_on) / max(len(cache["dates"]), 1), 3),
        "worst": round(float(pnl.min()), 1) if len(pnl) else None,
        "n_le5k": int((pnl <= -5000).sum()),
        "n_le8k": int((pnl <= -8000).sum()),
        "y2026_pnl": round(float(y26.sum()), 1) if len(y26) else 0.0,
        "y2026_worst": round(float(y26.min()), 1) if len(y26) else None,
        "y2026_n_le5k": int((y26 <= -5000).sum()) if len(y26) else 0,
        "live": live,
        "pnl_series": pnl,
    }


def passes(r: dict) -> bool:
    if r.get("sharpe") is None or r.get("max_dd") is None:
        return False
    return (
        r["sharpe"] >= GATE_SHARPE
        and r["max_dd"] >= GATE_DD
        and r["avg_margin_on"] >= GATE_AVG_MARGIN
        and r["pnl"] >= GATE_PNL
        and r["worst"] is not None
        and r["worst"] >= GATE_WORST
        and r["y2026_n_le5k"] <= GATE_2026_N5K
    )


def half_doc_sharpes(docs, panel, dates, cfg, n_seed: int = 3) -> list[float]:
    out = []
    paths = docs["path"].drop_duplicates().to_numpy()
    for seed in range(n_seed):
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(paths, size=max(1, len(paths) // 2), replace=False))
        sub = docs[docs["path"].isin(keep)]
        fac = batch_aggregate(sub, cfg["col"], cfg["lb"], dates, "uniform")
        cache = build_day_cache(panel, fac)
        out.append(run_cfg(cache, cfg)["sharpe"])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    linear_pnl = float(
        pd.read_csv(LINEAR_YEARLY).loc[lambda x: x["year"] >= 2024, "pnl_cny"].sum()
    )
    print(f"final 2024+ pnl gate ref={linear_pnl:.0f} (use {GATE_PNL})", flush=True)

    print("docs...", flush=True)
    docs = pd.read_parquet(DATA_DIR / "rule_docs_rich.parquet")
    docs["report_date"] = pd.to_datetime(docs["report_date"])
    docs = keep_single_symbol_docs(filter_docs(docs, "no_dianping"))

    print("panel+vol...", flush=True)
    panel = build_panel(BT_START).dropna(subset=["fwd_ret"])
    panel = panel.merge(load_px(), on=["date", "symbol", "code"], how="left")
    panel = panel[panel["date"] >= BT_START].copy()
    panel = attach_vol(panel)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))

    # signal grid: proven rule_edge L30 + L60 neighbour
    sigs = [("rule_edge", 30), ("rule_edge", 60)]
    day_cache = {}
    for col, lb in sigs:
        key = f"{col}|L{lb}"
        print(f"agg+cache {key}", flush=True)
        fac = batch_aggregate(docs, col, lb, dates, "uniform")
        day_cache[key] = build_day_cache(panel, fac)

    configs = []
    # Focus: proven rule_edge L30 (+ L60 neighbour); risk-aware sizing only
    for sig_key in ("rule_edge|L30", "rule_edge|L60"):
        col, lb_s = sig_key.split("|")
        lb = int(lb_s[1:])
        for mode, n_each, enter, exit_ in product(
            ("hysteresis",),
            (1, 2),
            (0.30, 0.40),
            (0.10, 0.15),
        ):
            if exit_ >= enter:
                continue
            # equal-margin + hard risk/margin clips (main hope for capacity without fat tails)
            for util, max_lots, scap, max_leg_m, max_leg_r in product(
                (0.10, 0.12, 0.15, 0.18, 0.22),
                (2, 3),
                (1.0, 1.2),
                (30_000, 40_000, 50_000),
                (3_000, 4_000, 5_000),
            ):
                configs.append({
                    "sig_key": sig_key, "col": col, "lb": lb,
                    "mode": mode, "n_each": n_each, "enter": enter, "exit_": exit_,
                    "min_hold": 5, "util": util, "spread_ref": 0.30, "max_lots": max_lots,
                    "strength_cap": scap, "alloc": "margin",
                    "max_leg_margin": max_leg_m, "max_leg_risk": max_leg_r,
                })
            # vol risk-budget sizing
            for util, max_lots, scap, max_leg_r in product(
                (0.006, 0.008, 0.010, 0.012),
                (2, 3),
                (1.0, 1.2),
                (3_000, 4_000, 5_000),
            ):
                configs.append({
                    "sig_key": sig_key, "col": col, "lb": lb,
                    "mode": mode, "n_each": n_each, "enter": enter, "exit_": exit_,
                    "min_hold": 5, "util": util, "spread_ref": 0.30, "max_lots": max_lots,
                    "strength_cap": scap, "alloc": "risk",
                    "max_leg_margin": 50_000, "max_leg_risk": max_leg_r,
                })

    # de-dup identical dicts
    uniq = []
    seen = set()
    for c in configs:
        key = tuple(sorted((k, str(v)) for k, v in c.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    configs = uniq
    print(f"n configs={len(configs)}", flush=True)

    rows = []
    for i, cfg in enumerate(configs):
        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(configs)}]", flush=True)
        r = run_cfg(day_cache[cfg["sig_key"]], cfg)
        name = (
            f"{cfg['sig_key']}|{cfg['alloc']}|n{cfg['n_each']}|u{cfg['util']}"
            f"|max{cfg['max_lots']}|sc{cfg['strength_cap']}"
            f"|mm{cfg['max_leg_margin']}|mr{cfg['max_leg_risk']}"
            f"|e{cfg['enter']}/x{cfg['exit_']}"
        )
        row = {"name": name, **{k: cfg[k] for k in cfg if k != "sig_key"}, **{k: r[k] for k in r if k not in ("live", "pnl_series")}}
        row["pass"] = passes(r)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "all_candidates.csv", index=False)
    ok = df[df["pass"]].sort_values(["pnl", "sharpe"], ascending=False)
    ok.to_csv(OUT / "passed.csv", index=False)
    print(f"passed {len(ok)} / {len(df)}", flush=True)

    report: dict = {
        "gates": {
            "avg_margin": GATE_AVG_MARGIN,
            "pnl": GATE_PNL,
            "worst": GATE_WORST,
            "y2026_n_le5k": GATE_2026_N5K,
            "sharpe": GATE_SHARPE,
            "dd": GATE_DD,
            "half_med": GATE_HALF_MED,
            "final_2024plus_pnl": linear_pnl,
        },
        "n_total": int(len(df)),
        "n_pass": int(len(ok)),
    }

    shortlist = ok.head(15) if len(ok) else df.sort_values(
        ["pass", "pnl", "worst"], ascending=[False, False, False]
    ).head(20)
    # if none passed, show near-misses
    near = df[
        (df["avg_margin_on"] >= 45_000)
        & (df["pnl"] >= 80_000)
        & (df["worst"] >= -10_000)
        & (df["sharpe"] >= 1.0)
    ].sort_values(["worst", "pnl"], ascending=[False, False]).head(30)
    near.to_csv(OUT / "near_miss.csv", index=False)
    report["near_miss_top"] = near.drop(columns=[], errors="ignore").head(10).to_dict(orient="records")

    winners = []
    for _, row in shortlist.iterrows():
        if not bool(row.get("pass")):
            break
        cfg = {
            "mode": row["mode"], "n_each": int(row["n_each"]),
            "enter": float(row["enter"]), "exit_": float(row["exit_"]), "min_hold": 5,
            "util": float(row["util"]), "spread_ref": 0.30, "max_lots": int(row["max_lots"]),
            "strength_cap": float(row["strength_cap"]), "alloc": row["alloc"],
            "max_leg_margin": None if pd.isna(row["max_leg_margin"]) else float(row["max_leg_margin"]),
            "max_leg_risk": None if pd.isna(row["max_leg_risk"]) else float(row["max_leg_risk"]),
            "col": row["col"], "lb": int(row["lb"]),
        }
        print(f"half-doc {row['name']} ...", flush=True)
        hs = half_doc_sharpes(docs, panel, dates, cfg, n_seed=3)
        med = float(np.nanmedian([x for x in hs if x is not None]))
        item = {
            "name": row["name"],
            "pnl": float(row["pnl"]),
            "sharpe": float(row["sharpe"]),
            "avg_margin_on": float(row["avg_margin_on"]),
            "worst": float(row["worst"]),
            "y2026_n_le5k": int(row["y2026_n_le5k"]),
            "half_doc": [None if x is None else float(x) for x in hs],
            "half_med": med,
            "half_ok": bool(med >= GATE_HALF_MED),
            "cfg": cfg,
        }
        winners.append(item)
        print(f"  half_med={med:.3f} ok={item['half_ok']}", flush=True)
        if item["half_ok"]:
            # dump daily
            cache = day_cache[f"{cfg['col']}|L{cfg['lb']}"]
            full = run_cfg(cache, cfg)
            daily = pd.DataFrame({
                "日期": [str(pd.Timestamp(i).date()) for i in full["pnl_series"].index],
                "账户日盈亏": full["pnl_series"].to_numpy(),
                "账户日收益": full["live"].reindex(full["pnl_series"].index).to_numpy(),
                "合计保证金": np.nan,
            })
            # rebuild margin/lots lightly for daily file via second pass not needed for gate
            daily.to_csv(OUT / "quad_best_daily.csv", index=False)
            report["best"] = item
            break

    report["passed_half_checked"] = winners
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT, flush=True)
    if report.get("best"):
        b = report["best"]
        print(
            f"BEST OK {b['name']} pnl={b['pnl']} sh={b['sharpe']} "
            f"mar={b['avg_margin_on']} worst={b['worst']} half={b['half_med']}",
            flush=True,
        )
    else:
        print("NO strategy passed all gates including half-doc.", flush=True)
        if len(ok):
            print(f"hard gates passed={len(ok)} but half-doc failed or not checked", flush=True)
        print("near_miss count", len(near), flush=True)


if __name__ == "__main__":
    main()
