#!/usr/bin/env python3
"""Heterogeneous news strategy search (not another util/max_lots grid).

Families:
  A wide1:   news LS with n_each=3..5, FIXED 1 lot/name (capacity via breadth)
  B wide_risk: same breadth but risk-budget lots capped hard
  C confirm: news LS + price trend confirm (close vs MA20)
  D weekly:  rebalance only every K days (news L30 risk/1lot)
  E resonate: final_scheme overlay legs; size 1, or 2 if news agrees on side

Gates vs final_scheme (same as quad):
  avg_margin>=55k, full pnl>=94k, worst>=-7k, y2026_pnl>=41431,
  y2026_n_le5k<=2, sharpe>=1.2, dd>=-8%, then half-doc50 med>=0.8
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
from pos64_margin_cap_search import attach_margin  # noqa: E402
from pos64_margin_integer_search import CAPITAL, net_picks, to_log  # noqa: E402
from pos64_rsi_overlay_2026_blotter import MARGIN_CAP, pick_overlay  # noqa: E402
from rule_factor_mine import batch_aggregate, filter_docs, keep_single_symbol_docs  # noqa: E402
from rule_news_capacity_search import day_spread, dollar_multi, fees_multi, one_lot_margin  # noqa: E402
from rule_news_quad_search import attach_vol  # noqa: E402
from six_name_select import pick_ls_n  # noqa: E402

OUT = RESULTS_ROOT / "rule_capacity_alt"
BT_START = pd.Timestamp("2024-03-08")

GATE_AVG_MARGIN = 55_000.0
GATE_PNL = 94_000.0
GATE_WORST = -7_000.0
GATE_2026_PNL = 41_431.0
GATE_2026_N5K = 2
GATE_SHARPE = 1.2
GATE_DD = -0.08
GATE_HALF = 0.8


def build_news_cache(panel: pd.DataFrame, fac: pd.DataFrame) -> dict:
    p = panel.merge(fac, on=["date", "symbol"], how="left")
    # MA20 for confirm family
    p = p.sort_values(["symbol", "date"])
    p["ma20"] = p.groupby("symbol", sort=False)["close"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    days = {}
    for dt, day0 in p.groupby("date"):
        day = day0.drop_duplicates("symbol").reset_index(drop=True)
        mask = day["symbol"].isin(MULTIPLIER) & day["symbol"].isin(BROKER_MARGIN)
        d = day[mask].dropna(subset=["val", "fwd_ret", "close"]).reset_index(drop=True)
        m1, risk1 = [], []
        for sym, px, v in zip(d["symbol"], d["close"], d["vol20"]):
            mar = one_lot_margin(str(sym), float(px))
            m1.append(mar)
            vv = float(v) if np.isfinite(v) else 0.02
            mult = MULTIPLIER.get(str(sym))
            risk1.append(None if (mar is None or mult is None) else float(mult) * float(px) * vv)
        d["m1"] = m1
        d["risk1"] = risk1
        d = d[d["m1"].notna() & d["risk1"].notna()].reset_index(drop=True)
        days[pd.Timestamp(dt)] = {
            "frame": d,
            "meta": meta_of(day),
            "ret": {str(r.symbol): float(r.fwd_ret) for r in d.itertuples() if np.isfinite(r.fwd_ret)},
            "close": {str(r.symbol): float(r.close) for r in d.itertuples()},
            "m1": {str(r.symbol): float(r.m1) for r in d.itertuples()},
            "risk1": {str(r.symbol): float(r.risk1) for r in d.itertuples()},
            "ma20": {
                str(r.symbol): float(r.ma20)
                for r in d.itertuples()
                if np.isfinite(getattr(r, "ma20", np.nan))
            },
        }
    return {"dates": [pd.Timestamp(d) for d in sorted(days)], "days": days}


def hyst_update(state, days_in, cand, spread, enter, exit_, min_hold, elig):
    if state is None:
        if spread >= enter and cand:
            return cand, 0
        return None, 0
    days_in += 1
    state = [(s, w) for s, w in state if s in elig]
    if len(state) < 2:
        return None, 0
    if days_in >= min_hold and spread < exit_:
        return None, 0
    if days_in >= min_hold and spread >= enter and cand:
        if {s for s, _ in state} != {s for s, _ in cand}:
            return cand, 0
    return state, days_in


def lots_fixed1(names, m1, max_leg_margin=None):
    out = []
    for sym, w in names or []:
        if sym not in m1:
            continue
        if max_leg_margin is not None and m1[sym] > max_leg_margin:
            continue
        out.append((sym, 1 if w > 0 else -1))
    return out


def lots_risk(names, m1, risk1, budget, max_lots, max_leg_risk, max_leg_margin):
    if not names or budget < 500:
        return []
    legs = [(s, 1 if w > 0 else -1, m1[s], risk1[s]) for s, w in names if s in m1 and s in risk1]
    if not legs:
        return []
    share = budget / len(legs)
    out = []
    for sym, sign, mar, r1 in legs:
        n = int(np.floor(share / max(r1, 1.0)))
        n = int(np.clip(n, 0, max_lots))
        n = min(n, int(np.floor(max_leg_margin / mar))) if mar > 0 else 0
        n = min(n, int(np.floor(max_leg_risk / r1))) if r1 > 0 else 0
        n = int(np.clip(n, 0, max_lots))
        if n >= 1:
            out.append((sym, sign * n))
    return out


def summarize(pnl: pd.Series, margins_on, lots_on, n_dates) -> dict:
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
        "cash": round(1.0 - len(margins_on) / max(n_dates, 1), 3),
        "worst": round(float(pnl.min()), 1) if len(pnl) else None,
        "n_le5k": int((pnl <= -5000).sum()),
        "y2026_pnl": round(float(y26.sum()), 1) if len(y26) else 0.0,
        "y2026_worst": round(float(y26.min()), 1) if len(y26) else None,
        "y2026_n_le5k": int((y26 <= -5000).sum()) if len(y26) else 0,
        "live": live,
        "pnl_series": pnl,
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


def run_news_family(cache, cfg) -> dict:
    """Families: wide1, wide_risk, confirm, weekly."""
    fam = cfg["family"]
    n_each = cfg["n_each"]
    enter, exit_, min_hold = cfg["enter"], cfg["exit_"], cfg["min_hold"]
    rebal_k = cfg.get("rebal_k", 1)
    use_confirm = fam == "confirm"
    state = None
    days_in = 0
    prev = {}
    pnls = []
    mars, lots = [], []
    frozen = None
    since_rebal = 999

    for dt in cache["dates"]:
        info = cache["days"][dt]
        d = info["frame"]
        spread = day_spread(d["val"].to_numpy()) if len(d) >= 2 else 0.0
        cand = pick_ls_n(d, d["val"].to_numpy(), n_each) if len(d) >= 2 * n_each else []
        if use_confirm and cand:
            ma = info["ma20"]
            cl = info["close"]
            filt = []
            for s, w in cand:
                if s not in ma or s not in cl:
                    continue
                if w > 0 and cl[s] >= ma[s]:
                    filt.append((s, w))
                elif w < 0 and cl[s] <= ma[s]:
                    filt.append((s, w))
            cand = filt

        state, days_in = hyst_update(state, days_in, cand, spread, enter, exit_, min_hold, set(d["symbol"]))
        names = state

        since_rebal += 1
        if fam == "weekly":
            if frozen is None or since_rebal >= rebal_k:
                frozen = names
                since_rebal = 0
            names = frozen

        strength = float(np.clip(spread / cfg["spread_ref"], 0.0, cfg["strength_cap"]))
        if fam == "wide1" or (fam == "confirm" and cfg.get("util", 0) <= 0) or (
            fam == "weekly" and cfg.get("util", 0) <= 0
        ):
            positions = lots_fixed1(names, info["m1"], cfg.get("max_leg_margin"))
        else:
            budget = cfg["util"] * CAPITAL * strength
            positions = lots_risk(
                names, info["m1"], info["risk1"], budget,
                cfg["max_lots"], cfg["max_leg_risk"], cfg["max_leg_margin"],
            )

        cur = {s: n for s, n in positions}
        fee = fees_multi(info["meta"], prev, cur)
        prev = cur
        if not positions:
            pnls.append((dt, -fee if fee else 0.0))
            continue
        gross, mar, nl = dollar_multi(positions, info["close"], info["ret"])
        pnls.append((dt, gross - fee))
        mars.append(mar)
        lots.append(nl)

    pnl = pd.Series(dict(pnls)).sort_index()
    return summarize(pnl, mars, lots, len(cache["dates"]))


def run_resonate(panel: pd.DataFrame, news_cache: dict, cfg: dict) -> dict:
    """Final overlay legs; 2 lots if news score agrees on side, else 1 (or 0 if require_agree)."""
    # index news scores by date,symbol
    news_side = {}
    for dt in news_cache["dates"]:
        d = news_cache["days"][dt]["frame"]
        if len(d) < 4:
            news_side[dt] = {}
            continue
        picks = pick_ls_n(d, d["val"].to_numpy(), cfg["news_n"])
        news_side[dt] = {s: (1 if w > 0 else -1) for s, w in picks}

    prev = {}
    pnls = []
    mars, lots = [], []
    dates = sorted(panel["date"].unique())
    for dt0, day0 in panel.groupby("date"):
        dt = pd.Timestamp(dt0)
        day = day0.drop_duplicates("symbol").reset_index(drop=True)
        if "pos_64" not in day.columns or "rsi" not in day.columns:
            pnls.append((dt, 0.0))
            continue
        # eligible like final
        d = attach_margin(day)
        d = d[d["lot_margin"].notna() & (d["lot_margin"] <= MARGIN_CAP)].reset_index(drop=True)
        if len(d) < 2:
            pnls.append((dt, 0.0))
            continue
        picks = pick_overlay(d, use_close=False)
        ns = news_side.get(dt, {})
        positions = []
        for s, w in picks:
            side = 1 if w > 0 else -1
            n = 1
            if s in ns and ns[s] == side:
                n = cfg["agree_lots"]
            elif cfg.get("require_agree"):
                continue
            positions.append((s, side * n))

        # meta/ret/close from day
        meta = meta_of(day)
        close = {str(r.symbol): float(r.close) for r in day.itertuples() if np.isfinite(r.close)}
        ret = {str(r.symbol): float(r.fwd_ret) for r in day.itertuples() if np.isfinite(r.fwd_ret)}
        cur = {s: n for s, n in positions}
        fee = fees_multi(meta, prev, cur)
        prev = cur
        if not positions:
            pnls.append((dt, -fee if fee else 0.0))
            continue
        gross, mar, nl = dollar_multi(positions, close, ret)
        pnls.append((dt, gross - fee))
        mars.append(mar)
        lots.append(nl)

    pnl = pd.Series(dict(pnls)).sort_index()
    pnl = pnl[pnl.index >= BT_START]
    return summarize(pnl, mars, lots, len(pnl))


def half_doc(docs, panel, dates, cfg, family_runner, seeds=3):
    paths = docs["path"].drop_duplicates().to_numpy()
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(paths, size=max(1, len(paths) // 2), replace=False))
        sub = docs[docs["path"].isin(keep)]
        fac = batch_aggregate(sub, "rule_edge", 30, dates, "uniform")
        cache = build_news_cache(panel, fac)
        if cfg["family"] == "resonate":
            out.append(run_resonate(panel, cache, cfg)["sharpe"])
        else:
            out.append(family_runner(cache, cfg)["sharpe"])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("docs/panel...", flush=True)
    docs = pd.read_parquet(DATA_DIR / "rule_docs_rich.parquet")
    docs["report_date"] = pd.to_datetime(docs["report_date"])
    docs = keep_single_symbol_docs(filter_docs(docs, "no_dianping"))

    panel = build_panel(BT_START).dropna(subset=["fwd_ret"])
    panel = panel.merge(load_px(), on=["date", "symbol", "code"], how="left")
    panel = panel[panel["date"] >= BT_START].copy()
    panel = attach_vol(panel)
    # need pos64/rsi on panel for resonate — merge from hist features via blotter path
    # build_panel already has pos_64/rsi for hist range
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    print("agg news L30...", flush=True)
    fac = batch_aggregate(docs, "rule_edge", 30, dates, "uniform")
    cache = build_news_cache(panel, fac)
    print(f"panel cols has pos_64={('pos_64' in panel.columns)} rsi={('rsi' in panel.columns)}", flush=True)

    configs = []
    # A wide1
    for n_each, enter, exit_, min_hold, cap in product(
        (3, 4, 5), (0.25, 0.30, 0.40), (0.10, 0.15), (5, 10), (None, 50_000.0),
    ):
        if exit_ >= enter:
            continue
        configs.append({
            "family": "wide1", "n_each": n_each, "enter": enter, "exit_": exit_,
            "min_hold": min_hold, "spread_ref": 0.30, "strength_cap": 1.0,
            "max_leg_margin": cap, "util": 0.0, "max_lots": 1, "max_leg_risk": 0.0,
        })
    # B wide_risk
    for n_each, enter, exit_, util, max_lots, mr in product(
        (3, 4, 5), (0.30, 0.40), (0.15,), (0.008, 0.010, 0.012), (1, 2), (2500, 3500),
    ):
        configs.append({
            "family": "wide_risk", "n_each": n_each, "enter": enter, "exit_": exit_,
            "min_hold": 5, "spread_ref": 0.30, "strength_cap": 1.0,
            "max_leg_margin": 50_000.0, "util": util, "max_lots": max_lots, "max_leg_risk": float(mr),
        })
    # C confirm (2-3 names)
    for n_each, enter, exit_, util, max_lots in product(
        (2, 3), (0.30, 0.40), (0.15,), (0.0, 0.010), (1, 2),
    ):
        configs.append({
            "family": "confirm", "n_each": n_each, "enter": enter, "exit_": exit_,
            "min_hold": 5, "spread_ref": 0.30, "strength_cap": 1.0,
            "max_leg_margin": 50_000.0, "util": util, "max_lots": max_lots, "max_leg_risk": 3500.0,
        })
    # D weekly
    for n_each, rebal_k, util, max_lots in product((2, 3, 4), (5, 10), (0.0, 0.010), (1, 2)):
        configs.append({
            "family": "weekly", "n_each": n_each, "enter": 0.30, "exit_": 0.15,
            "min_hold": 5, "rebal_k": rebal_k, "spread_ref": 0.30, "strength_cap": 1.0,
            "max_leg_margin": 50_000.0, "util": util, "max_lots": max_lots, "max_leg_risk": 3500.0,
        })
    # E resonate
    for news_n, agree_lots, require_agree in product((1, 2, 3), (2, 3), (False, True)):
        configs.append({
            "family": "resonate", "news_n": news_n, "agree_lots": agree_lots,
            "require_agree": require_agree,
            # placeholders for naming
            "n_each": news_n, "enter": 0.0, "exit_": 0.0, "min_hold": 0,
            "spread_ref": 0.3, "strength_cap": 1.0, "max_leg_margin": 50_000.0,
            "util": 0.0, "max_lots": agree_lots, "max_leg_risk": 0.0,
        })

    print(f"n configs={len(configs)}", flush=True)
    rows = []
    for i, cfg in enumerate(configs):
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(configs)}]", flush=True)
        if cfg["family"] == "resonate":
            r = run_resonate(panel, cache, cfg)
        elif cfg["family"] == "confirm" and cfg["util"] <= 0:
            # confirm + fixed 1 lot
            cfg2 = {**cfg, "family": "wide1"}
            # monkey: use confirm filter via family confirm with util0 -> treat as wide1 path inside run
            r = run_news_family(cache, {**cfg, "util": 0.0, "max_lots": 1})
        else:
            r = run_news_family(cache, cfg)
        name = (
            f"{cfg['family']}|n{cfg.get('n_each')}|e{cfg.get('enter')}/x{cfg.get('exit_')}"
            f"|h{cfg.get('min_hold')}|u{cfg.get('util')}|max{cfg.get('max_lots')}"
            f"|cap{cfg.get('max_leg_margin')}|rk{cfg.get('rebal_k','')}"
            f"|al{cfg.get('agree_lots','')}|ra{cfg.get('require_agree','')}"
        )
        row = {"name": name, **{k: cfg[k] for k in cfg}, **{k: r[k] for k in r if k not in ("live", "pnl_series")}}
        row["pass"] = passes(r)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "all_candidates.csv", index=False)
    ok = df[df["pass"]].sort_values(["y2026_pnl", "pnl"], ascending=False)
    ok.to_csv(OUT / "passed.csv", index=False)
    near = df[
        (df.avg_margin_on >= 50_000)
        & (df.y2026_pnl >= 30_000)
        & (df.worst >= -10_000)
        & (df.sharpe >= 1.0)
        & (df.pnl >= 70_000)
    ].sort_values(["y2026_pnl", "worst"], ascending=[False, False])
    near.to_csv(OUT / "near_miss.csv", index=False)
    print(f"pass={len(ok)} near={len(near)} / {len(df)}", flush=True)

    # per-family best
    for fam, g in df.groupby("family"):
        best = g.sort_values(["pass", "y2026_pnl", "worst"], ascending=[False, False, False]).iloc[0]
        print(
            f"  [{fam}] pass_any={g['pass'].any()} best_y2026={best['y2026_pnl']} "
            f"pnl={best['pnl']} mar={best['avg_margin_on']} worst={best['worst']} sh={best['sharpe']}",
            flush=True,
        )

    report = {"n": len(df), "n_pass": int(len(ok)), "n_near": int(len(near)), "checked": []}
    check = ok.head(10) if len(ok) else near.head(8)
    for _, row in check.iterrows():
        cfg = {k: row[k] for k in row.index if k in (
            "family", "n_each", "enter", "exit_", "min_hold", "spread_ref", "strength_cap",
            "max_leg_margin", "util", "max_lots", "max_leg_risk", "rebal_k",
            "news_n", "agree_lots", "require_agree",
        )}
        # fix types
        for k in ("enter", "exit_", "util", "strength_cap", "spread_ref", "max_leg_risk"):
            if k in cfg and pd.notna(cfg[k]):
                cfg[k] = float(cfg[k])
        for k in ("n_each", "min_hold", "max_lots", "rebal_k", "news_n", "agree_lots"):
            if k in cfg and pd.notna(cfg[k]):
                cfg[k] = int(cfg[k])
        if "max_leg_margin" in cfg and pd.notna(cfg["max_leg_margin"]):
            cfg["max_leg_margin"] = float(cfg["max_leg_margin"])
        else:
            cfg["max_leg_margin"] = None
        if "require_agree" in cfg and pd.notna(cfg["require_agree"]):
            cfg["require_agree"] = bool(cfg["require_agree"])

        print(f"half {row['name']}...", flush=True)
        hs = half_doc(docs, panel, dates, cfg, run_news_family, seeds=3)
        med = float(np.nanmedian([x for x in hs if x is not None]))
        item = {
            "name": row["name"], "pass": bool(row["pass"]),
            "pnl": float(row["pnl"]), "y2026_pnl": float(row["y2026_pnl"]),
            "avg_margin_on": float(row["avg_margin_on"]), "worst": float(row["worst"]),
            "sharpe": float(row["sharpe"]), "half": [float(x) if x is not None else None for x in hs],
            "half_med": med, "half_ok": bool(med >= GATE_HALF), "cfg": cfg,
        }
        report["checked"].append(item)
        print(f"  half_med={med:.3f} ok={item['half_ok']}", flush=True)
        if item["pass"] and item["half_ok"]:
            report["best"] = item
            break

    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT, flush=True)
    if report.get("best"):
        print("BEST", report["best"]["name"], flush=True)
    else:
        print("NO full pass among alt families.", flush=True)


if __name__ == "__main__":
    main()
