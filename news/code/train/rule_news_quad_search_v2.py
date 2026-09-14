#!/usr/bin/env python3
"""Quad search v2 — push toward all four gates with new levers:

- universe: drop high-vol names (vol20 cap)
- signal: min_docs filter; L14/30/60; rule_edge/lex
- hold: longer min_hold
- sizing: risk budget + per-leg risk/margin caps; optional score-weighted risk
- gates: capacity, full pnl, daily loss, AND 2026 pnl > final; then half-doc
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
from futures_lot_specs import BROKER_MARGIN, MULTIPLIER  # noqa: E402
from linear_ridge_walkforward import build_panel  # noqa: E402
from pos64_4name_2026_blotter import meta_of  # noqa: E402
from pos64_margin_integer_search import CAPITAL, to_log  # noqa: E402
from rule_factor_mine import batch_aggregate, filter_docs, keep_single_symbol_docs  # noqa: E402
from rule_news_capacity_search import day_spread, dollar_multi, fees_multi, one_lot_margin  # noqa: E402
from rule_news_quad_search import attach_vol  # noqa: E402
from six_name_select import pick_ls_n  # noqa: E402

OUT = RESULTS_ROOT / "rule_capacity_quad_v2"
BT_START = pd.Timestamp("2024-03-08")
LINEAR_YEARLY = Path(
    "/home/workspace/lab/UniFutures/data/infer/results/daily/linear/final_scheme_yearly.csv"
)
FINAL_2026_PNL = 41_431.0
GATE_AVG_MARGIN = 55_000.0
GATE_PNL = 94_000.0
GATE_WORST = -7_000.0
GATE_2026_PNL = FINAL_2026_PNL
GATE_2026_N5K = 2
GATE_SHARPE = 1.20
GATE_DD = -0.08
GATE_HALF50 = 0.80


def build_day_cache(panel: pd.DataFrame, fac: pd.DataFrame, *, max_vol: float | None) -> dict:
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
            risk1.append(None if mult is None or mar is None else float(mult) * float(px) * vv)
        d["m1"] = m1
        d["vol20"] = vol
        d["risk1"] = risk1
        d = d[d["m1"].notna() & d["risk1"].notna()].reset_index(drop=True)
        if max_vol is not None:
            d = d[d["vol20"] <= max_vol].reset_index(drop=True)
        days[pd.Timestamp(dt)] = {
            "frame": d,
            "meta": meta_of(day),
            "ret": {str(r.symbol): float(r.fwd_ret) for r in d.itertuples() if np.isfinite(r.fwd_ret)},
            "close": {str(r.symbol): float(r.close) for r in d.itertuples()},
            "m1": {str(r.symbol): float(r.m1) for r in d.itertuples()},
            "risk1": {str(r.symbol): float(r.risk1) for r in d.itertuples()},
            "score": {str(r.symbol): float(r.val) for r in d.itertuples() if np.isfinite(r.val)},
        }
    return {"dates": [pd.Timestamp(d) for d in dates], "days": days}


def allocate(
    names: list[tuple[str, float]],
    m1: dict[str, float],
    risk1: dict[str, float],
    *,
    budget_eff: float,
    max_lots: int,
    max_leg_margin: float,
    max_leg_risk: float,
    weighted: bool,
) -> list[tuple[str, int]]:
    if budget_eff <= 0 or not names:
        return []
    legs = []
    for sym, w in names:
        if sym not in m1 or sym not in risk1:
            continue
        legs.append((sym, 1.0 if w > 0 else -1.0, abs(float(w)), m1[sym], risk1[sym]))
    if not legs:
        return []
    if weighted:
        sw = sum(a for _, _, a, _, _ in legs) or 1.0
        weights = [a / sw for _, _, a, _, _ in legs]
    else:
        weights = [1.0 / len(legs)] * len(legs)
    out = []
    for (sym, sign, _, mar, r1), wt in zip(legs, weights):
        share = budget_eff * wt
        raw = int(np.floor(share / max(r1, 1.0)))
        lots = int(np.clip(raw, 0, max_lots))
        if mar > 0:
            lots = min(lots, int(np.floor(max_leg_margin / mar)))
        if r1 > 0:
            lots = min(lots, int(np.floor(max_leg_risk / r1)))
        lots = int(np.clip(lots, 0, max_lots))
        if lots >= 1:
            out.append((sym, int(sign * lots)))
    return out


def run_cfg(cache: dict, cfg: dict) -> dict:
    n_each = cfg["n_each"]
    enter, exit_, min_hold = cfg["enter"], cfg["exit_"], cfg["min_hold"]
    util, spread_ref, scap = cfg["util"], cfg["spread_ref"], cfg["strength_cap"]
    max_lots = cfg["max_lots"]
    min_docs = cfg["min_docs"]
    min_names = cfg["min_names"]
    weighted = cfg["weighted"]

    state = None
    days_in = 0
    prev: dict[str, int] = {}
    pnls = []
    margins_on, lots_on = [], []

    for dt in cache["dates"]:
        info = cache["days"][dt]
        d0 = info["frame"]
        d = d0[d0["n_docs"] >= min_docs] if min_docs > 0 and "n_docs" in d0.columns else d0
        spread = day_spread(d["val"].to_numpy()) if len(d) >= 2 else 0.0
        cand = pick_ls_n(d, d["val"].to_numpy(), n_each) if len(d) >= 2 * n_each else []
        if len(d) < min_names:
            cand = []

        if state is None:
            if spread >= enter and cand:
                state, days_in = cand, 0
        else:
            days_in += 1
            elig = set(d["symbol"])
            state = [(s, w) for s, w in state if s in elig]
            if len(state) < 2:
                state, days_in = None, 0
            elif days_in >= min_hold and spread < exit_:
                state, days_in = None, 0
            elif days_in >= min_hold and spread >= enter and cand:
                if {s for s, _ in state} != {s for s, _ in cand}:
                    state, days_in = cand, 0
        names = state

        strength = float(np.clip(spread / spread_ref, 0.0, scap)) if spread_ref > 0 else 0.0
        budget = util * CAPITAL * strength
        positions = (
            allocate(
                names, info["m1"], info["risk1"],
                budget_eff=budget, max_lots=max_lots,
                max_leg_margin=cfg["max_leg_margin"], max_leg_risk=cfg["max_leg_risk"],
                weighted=weighted,
            )
            if names and budget >= 500
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
        "avg_lots_on": round(float(np.mean(lots_on)), 2) if lots_on else 0.0,
        "cash": round(1.0 - len(margins_on) / max(len(cache["dates"]), 1), 3),
        "worst": round(float(pnl.min()), 1) if len(pnl) else None,
        "n_le5k": int((pnl <= -5000).sum()),
        "y2026_pnl": round(float(y26.sum()), 1) if len(y26) else 0.0,
        "y2026_worst": round(float(y26.min()), 1) if len(y26) else None,
        "y2026_n_le5k": int((y26 <= -5000).sum()) if len(y26) else 0,
        "live": live,
        "pnl_series": pnl,
    }


def passes_hard(r: dict) -> bool:
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


def half_docs(docs, panel, dates, cfg, max_vol, frac=0.5, seeds=3) -> list[float]:
    paths = docs["path"].drop_duplicates().to_numpy()
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(paths, size=max(1, int(len(paths) * frac)), replace=False))
        sub = docs[docs["path"].isin(keep)]
        fac = batch_aggregate(sub, cfg["col"], cfg["lb"], dates, "uniform")
        cache = build_day_cache(panel, fac, max_vol=max_vol)
        out.append(run_cfg(cache, cfg)["sharpe"])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    linear_pnl = float(pd.read_csv(LINEAR_YEARLY).loc[lambda x: x["year"] >= 2024, "pnl_cny"].sum())
    print(f"gates: mar>={GATE_AVG_MARGIN} pnl>={GATE_PNL} worst>={GATE_WORST} "
          f"y2026>={GATE_2026_PNL} final2024+={linear_pnl:.0f}", flush=True)

    docs = pd.read_parquet(DATA_DIR / "rule_docs_rich.parquet")
    docs["report_date"] = pd.to_datetime(docs["report_date"])
    docs = keep_single_symbol_docs(filter_docs(docs, "no_dianping"))
    print(f"docs={len(docs)}", flush=True)

    panel = build_panel(BT_START).dropna(subset=["fwd_ret"])
    panel = panel.merge(load_px(), on=["date", "symbol", "code"], how="left")
    panel = panel[panel["date"] >= BT_START].copy()
    panel = attach_vol(panel)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))

    sigs = [("rule_edge", 30), ("rule_edge", 14)]
    vol_caps = (None, 0.035, 0.028)
    # prebuild caches: sig x vol_cap
    caches: dict[tuple, dict] = {}
    for col, lb in sigs:
        fac = batch_aggregate(docs, col, lb, dates, "uniform")
        for mv in vol_caps:
            key = (col, lb, mv)
            print(f"cache {key}", flush=True)
            caches[key] = build_day_cache(panel, fac, max_vol=mv)

    configs = []
    for col, lb in (("rule_edge", 30), ("rule_edge", 14)):
        for mv in vol_caps:
            for n_each, enter, exit_, min_hold, min_docs in product(
                (2,), (0.30, 0.40), (0.10, 0.15), (5, 10), (0, 2),
            ):
                if exit_ >= enter:
                    continue
                for util, max_lots, scap, mr, weighted in product(
                    (0.008, 0.010, 0.012, 0.014),
                    (2, 3),
                    (1.0, 1.2),
                    (3000, 4000),
                    (False, True),
                ):
                    configs.append({
                        "col": col, "lb": lb, "max_vol": mv,
                        "mode": "hysteresis", "n_each": n_each,
                        "enter": enter, "exit_": exit_, "min_hold": min_hold,
                        "min_docs": min_docs, "min_names": 4,
                        "util": util, "spread_ref": 0.30, "max_lots": max_lots,
                        "strength_cap": scap, "max_leg_margin": 50_000.0,
                        "max_leg_risk": float(mr), "weighted": weighted,
                    })
    for mv in (0.028, 0.035, None):
        for enter, exit_, min_hold, util, max_lots, mr, weighted in product(
            (0.30, 0.40), (0.15,), (5, 10),
            (0.010, 0.012, 0.015), (2, 3), (3000, 4500), (False, True),
        ):
            if exit_ >= enter:
                continue
            configs.append({
                "col": "rule_edge", "lb": 30, "max_vol": mv,
                "mode": "hysteresis", "n_each": 1,
                "enter": enter, "exit_": exit_, "min_hold": min_hold,
                "min_docs": 0, "min_names": 4,
                "util": util, "spread_ref": 0.30, "max_lots": max_lots,
                "strength_cap": 1.0, "max_leg_margin": 50_000.0,
                "max_leg_risk": float(mr), "weighted": weighted,
            })
    print(f"n configs={len(configs)}", flush=True)

    rows = []
    for i, cfg in enumerate(configs):
        if (i + 1) % 300 == 0:
            print(f"[{i+1}/{len(configs)}]", flush=True)
        cache = caches[(cfg["col"], cfg["lb"], cfg["max_vol"])]
        r = run_cfg(cache, cfg)
        name = (
            f"{cfg['col']}|L{cfg['lb']}|vol{cfg['max_vol']}|n{cfg['n_each']}"
            f"|u{cfg['util']}|max{cfg['max_lots']}|sc{cfg['strength_cap']}"
            f"|mr{cfg['max_leg_risk']}|md{cfg['min_docs']}|h{cfg['min_hold']}"
            f"|w{int(cfg['weighted'])}|e{cfg['enter']}/x{cfg['exit_']}"
        )
        row = {"name": name, **cfg, **{k: r[k] for k in r if k not in ("live", "pnl_series")}}
        row["pass_hard"] = passes_hard(r)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "all_candidates.csv", index=False)
    ok = df[df["pass_hard"]].sort_values(["y2026_pnl", "pnl"], ascending=False)
    ok.to_csv(OUT / "passed_hard.csv", index=False)
    print(f"pass_hard {len(ok)}/{len(df)}", flush=True)

    # near: capacity + 2026 beat + soften worst
    near = df[
        (df.avg_margin_on >= 50_000)
        & (df.y2026_pnl >= 35_000)
        & (df.pnl >= 80_000)
        & (df.worst >= -10_000)
        & (df.sharpe >= 1.0)
    ].sort_values(["y2026_pnl", "worst"], ascending=[False, False])
    near.to_csv(OUT / "near_miss.csv", index=False)
    print(f"near_miss {len(near)}", flush=True)

    report = {
        "gates": {
            "avg_margin": GATE_AVG_MARGIN,
            "pnl": GATE_PNL,
            "worst": GATE_WORST,
            "y2026_pnl": GATE_2026_PNL,
            "y2026_n_le5k": GATE_2026_N5K,
            "half50": GATE_HALF50,
        },
        "n_total": int(len(df)),
        "n_pass_hard": int(len(ok)),
        "n_near": int(len(near)),
    }

    winners = []
    check_rows = ok.head(20) if len(ok) else near.head(12)
    for _, row in check_rows.iterrows():
        cfg = {k: row[k] for k in (
            "col", "lb", "n_each", "enter", "exit_", "min_hold", "min_docs", "min_names",
            "util", "spread_ref", "max_lots", "strength_cap", "max_leg_margin",
            "max_leg_risk", "weighted", "mode",
        )}
        cfg["exit_"] = float(row["exit_"])
        mv = None if pd.isna(row["max_vol"]) else float(row["max_vol"])
        print(f"half-doc {row['name']} ...", flush=True)
        h50 = half_docs(docs, panel, dates, cfg, mv, frac=0.5, seeds=3)
        h75 = half_docs(docs, panel, dates, cfg, mv, frac=0.75, seeds=3)
        med50 = float(np.nanmedian([x for x in h50 if x is not None]))
        med75 = float(np.nanmedian([x for x in h75 if x is not None]))
        item = {
            "name": row["name"],
            "pass_hard": bool(row.get("pass_hard", False)),
            "pnl": float(row["pnl"]),
            "y2026_pnl": float(row["y2026_pnl"]),
            "sharpe": float(row["sharpe"]),
            "avg_margin_on": float(row["avg_margin_on"]),
            "worst": float(row["worst"]),
            "half50": [None if x is None else float(x) for x in h50],
            "half75": [None if x is None else float(x) for x in h75],
            "half50_med": med50,
            "half75_med": med75,
            "half_ok": bool(med50 >= GATE_HALF50),
            "cfg": {**cfg, "max_vol": mv},
        }
        winners.append(item)
        print(f"  half50_med={med50:.3f} half75_med={med75:.3f} ok={item['half_ok']}", flush=True)
        if item["pass_hard"] and item["half_ok"]:
            report["best"] = item
            # export daily pnl
            cache = caches[(cfg["col"], cfg["lb"], mv)]
            full = run_cfg(cache, cfg)
            pd.DataFrame({
                "日期": [str(pd.Timestamp(i).date()) for i in full["pnl_series"].index],
                "账户日盈亏": full["pnl_series"].to_numpy(),
                "账户日收益": full["live"].reindex(full["pnl_series"].index).to_numpy(),
            }).to_csv(OUT / "quad_v2_best_daily.csv", index=False)
            break

    report["checked"] = winners
    if "best" not in report and winners:
        # best effort: hard pass with highest half50, or near with best 2026
        hard = [w for w in winners if w["pass_hard"]]
        report["best_effort"] = max(hard, key=lambda w: w["half50_med"]) if hard else winners[0]

    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT, flush=True)
    if report.get("best"):
        b = report["best"]
        print(f"BEST ALL-GATES {b['name']} pnl={b['pnl']} y2026={b['y2026_pnl']} half50={b['half50_med']}", flush=True)
    else:
        print("NO full quad pass.", flush=True)
        be = report.get("best_effort")
        if be:
            print(f"best_effort {be['name']} hard={be.get('pass_hard')} half50={be['half50_med']} y2026={be['y2026_pnl']}", flush=True)


if __name__ == "__main__":
    main()
