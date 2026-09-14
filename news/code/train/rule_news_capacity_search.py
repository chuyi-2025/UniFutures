#!/usr/bin/env python3
"""News large-capacity search: margin-budget full book + signal-strength adaptive lots.

Capital 1e6. No per-name 50k cap. Fees = shouxufei open + close_prev.
Signal: single-symbol no_dianping rule_edge/rule_lex, L in {14,30,60}.
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
from six_name_select import pick_ls_n  # noqa: E402

BT_START = pd.Timestamp("2024-03-08")
OUT = RESULTS_ROOT / "rule_capacity"
LINEAR_YEARLY = Path(
    "/home/workspace/lab/UniFutures/data/infer/results/daily/linear/final_scheme_yearly.csv"
)
LINEAR_2026 = Path(
    "/home/workspace/lab/UniFutures/data/infer/results/daily/linear/final_scheme_2026_daily.csv"
)

COLS = ("rule_edge", "rule_lex")
LBS = (14, 30, 60)
UTILS = (0.40, 0.60, 0.80)
MAX_LOTS_GRID = (5, 8, 12)
N_EACH_GRID = (1, 2)
SPREAD_REF_GRID = (0.30, 0.50)
MODES = ("hysteresis", "daily")
ENTER_EXIT_HOLD = (
    (0.30, 0.15, 5),
    (0.40, 0.10, 5),
    (0.50, 0.20, 5),
)


def one_lot_margin(sym: str, px: float) -> float | None:
    mar, _ = lot_margin(sym, px, broker=True)
    if mar is None or not np.isfinite(mar) or mar <= 0:
        return None
    return float(mar)


def build_day_cache(panel: pd.DataFrame, fac: pd.DataFrame) -> dict:
    p = panel.merge(fac, on=["date", "symbol"], how="left")
    dates = sorted(p["date"].unique())
    days = {}
    for dt in dates:
        day = p[p["date"] == dt].drop_duplicates("symbol").reset_index(drop=True)
        mask = day["symbol"].isin(MULTIPLIER) & day["symbol"].isin(BROKER_MARGIN)
        uni = day[mask].copy()
        d = uni.dropna(subset=["val", "fwd_ret", "close"]).reset_index(drop=True)
        margins = []
        for sym, px in zip(d["symbol"], d["close"]):
            margins.append(one_lot_margin(str(sym), float(px)))
        d["m1"] = margins
        d = d[d["m1"].notna()].reset_index(drop=True)
        meta = meta_of(day)
        ret = {
            str(r.symbol): float(r.fwd_ret)
            for r in d.itertuples()
            if np.isfinite(r.fwd_ret)
        }
        close = {str(r.symbol): float(r.close) for r in d.itertuples()}
        m1 = {str(r.symbol): float(r.m1) for r in d.itertuples()}
        days[pd.Timestamp(dt)] = {
            "frame": d,
            "meta": meta,
            "ret": ret,
            "close": close,
            "m1": m1,
        }
    return {"dates": [pd.Timestamp(d) for d in dates], "days": days}


def day_spread(vals: np.ndarray) -> float:
    if len(vals) < 2:
        return 0.0
    return float(np.nanmax(vals) - np.nanmin(vals))


def allocate_lots(
    names: list[tuple[str, float]],
    m1: dict[str, float],
    *,
    budget_eff: float,
    max_lots: int,
) -> list[tuple[str, int]]:
    if budget_eff <= 0 or not names:
        return []
    legs = []
    for sym, w in names:
        if sym not in m1:
            continue
        legs.append((sym, 1.0 if w > 0 else -1.0, m1[sym]))
    if not legs:
        return []
    share = budget_eff / len(legs)
    out = []
    for sym, sign, mar in legs:
        n = int(np.floor(share / mar))
        n = int(np.clip(n, 0, max_lots))
        if n >= 1:
            out.append((sym, int(sign * n)))
    return out


def dollar_multi(
    positions: list[tuple[str, int]],
    close: dict[str, float],
    ret: dict[str, float],
) -> tuple[float, float, int]:
    pnl = 0.0
    margin = 0.0
    nlots = 0
    for sym, lots in positions:
        if lots == 0 or sym not in close or sym not in ret:
            continue
        px, fr = close[sym], ret[sym]
        mult, rate = MULTIPLIER.get(sym), BROKER_MARGIN.get(sym)
        if mult is None or rate is None or not np.isfinite(px) or not np.isfinite(fr) or px <= 0:
            continue
        val = float(mult) * float(px)
        n = abs(int(lots))
        sign = 1.0 if lots > 0 else -1.0
        pnl += sign * n * val * (np.exp(float(fr)) - 1.0)
        margin += n * val * float(rate)
        nlots += n
    return pnl, margin, nlots


def fees_multi(meta: dict, prev: dict[str, int], cur: dict[str, int]) -> float:
    fee = 0.0
    for sym, prev_n in prev.items():
        cur_n = cur.get(sym, 0)
        if cur_n == 0 or (prev_n > 0) != (cur_n > 0):
            close_lots = abs(prev_n)
        else:
            close_lots = max(0, abs(prev_n) - abs(cur_n))
        if close_lots <= 0:
            continue
        info = meta.get(sym, {})
        px = info.get("next_open")
        if px is None or not np.isfinite(px):
            px = info.get("close")
        fee += lot_fee(sym, px, "close_prev", close_lots)
    for sym, cur_n in cur.items():
        prev_n = prev.get(sym, 0)
        if cur_n == 0:
            continue
        if prev_n == 0 or (prev_n > 0) != (cur_n > 0):
            open_lots = abs(cur_n)
        else:
            open_lots = max(0, abs(cur_n) - abs(prev_n))
        if open_lots <= 0:
            continue
        info = meta.get(sym, {})
        px = info.get("next_open")
        if px is None or not np.isfinite(px):
            px = info.get("close")
        fee += lot_fee(sym, px, "open", open_lots)
    return fee


def run_config(cache: dict, cfg: dict) -> dict:
    mode = cfg["mode"]
    n_each = cfg["n_each"]
    enter, exit_, min_hold = cfg["enter"], cfg["exit_"], cfg["min_hold"]
    util, spread_ref, max_lots = cfg["util"], cfg["spread_ref"], cfg["max_lots"]

    state_names: list[tuple[str, float]] | None = None
    days_in = 0
    prev_pos: dict[str, int] = {}
    pnls: list[tuple[pd.Timestamp, float]] = []
    margins_on: list[float] = []
    lots_on: list[int] = []
    empty_days = 0
    fee_total = 0.0

    for dt in cache["dates"]:
        info = cache["days"][dt]
        d = info["frame"]
        spread = day_spread(d["val"].to_numpy()) if len(d) >= 2 else 0.0
        cand = pick_ls_n(d, d["val"].to_numpy(), n_each) if len(d) >= 2 * n_each else []

        if mode == "daily":
            names = cand if (spread >= enter and cand) else None
            days_in = 0
        else:
            if state_names is None:
                if spread >= enter and cand:
                    state_names = cand
                    days_in = 0
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

        strength = float(np.clip(spread / spread_ref, 0.0, 1.5)) if spread_ref > 0 else 0.0
        budget_eff = util * CAPITAL * strength
        positions = (
            allocate_lots(names, info["m1"], budget_eff=budget_eff, max_lots=max_lots)
            if names and budget_eff >= 1000
            else []
        )
        cur_pos = {s: n for s, n in positions}
        day_fee = fees_multi(info["meta"], prev_pos, cur_pos)
        fee_total += day_fee
        prev_pos = cur_pos

        if not positions:
            empty_days += 1
            pnls.append((dt, 0.0))
            continue

        gross, margin, nlots = dollar_multi(positions, info["close"], info["ret"])
        pnls.append((dt, gross - day_fee))
        margins_on.append(margin)
        lots_on.append(nlots)

    pnl_cny = pd.Series({d: v for d, v in pnls}).sort_index()
    live = to_log(pnl_cny / CAPITAL)
    bt = live[np.isfinite(pd.to_numeric(live, errors="coerce"))]
    st = year_pack(bt) if len(bt) >= 30 else {
        "return": None, "sharpe": None, "max_dd": None, "days": len(bt),
    }
    n_neg = 0
    if len(bt):
        for _, g in bt.groupby(bt.index.year):
            sh = year_pack(g)["sharpe"]
            if sh is not None and np.isfinite(sh) and sh < 0:
                n_neg += 1

    total_days = len(cache["dates"])
    return {
        "return": st.get("return"),
        "sharpe": st.get("sharpe"),
        "max_dd": st.get("max_dd"),
        "days": int(st.get("days") or len(bt)),
        "n_neg_years": int(n_neg),
        "cash": round(empty_days / total_days, 3) if total_days else 1.0,
        "pnl_cny": round(float(pnl_cny.sum()), 1),
        "fee_total": round(fee_total, 1),
        "avg_margin_on": round(float(np.mean(margins_on)), 1) if margins_on else 0.0,
        "p90_margin_on": round(float(np.quantile(margins_on, 0.9)), 1) if margins_on else 0.0,
        "avg_lots_on": round(float(np.mean(lots_on)), 2) if lots_on else 0.0,
        "active_days": int(len(margins_on)),
        "live": live,
        "pnl_series": pnl_cny,
    }


def main() -> None:
    print("docs...", flush=True)
    docs = pd.read_parquet(DATA_DIR / "rule_docs_rich.parquet")
    docs["report_date"] = pd.to_datetime(docs["report_date"])
    docs = keep_single_symbol_docs(filter_docs(docs, "no_dianping"))
    print(f"single+no_dp docs={len(docs)}", flush=True)

    print("panel...", flush=True)
    panel = build_panel(BT_START).dropna(subset=["fwd_ret"])
    panel = panel.merge(load_px(), on=["date", "symbol", "code"], how="left")
    panel = panel[panel["date"] >= BT_START].copy()
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))

    fac_cache: dict[str, pd.DataFrame] = {}
    day_cache: dict[str, dict] = {}
    for col, lb in product(COLS, LBS):
        key = f"{col}|L{lb}"
        print(f"agg+cache {key}", flush=True)
        fac = batch_aggregate(docs, col, lb, dates, "uniform")
        fac_cache[key] = fac
        day_cache[key] = build_day_cache(panel, fac)

    configs: list[dict] = []
    for sig_key in day_cache:
        col, lb_s = sig_key.split("|")
        lb = int(lb_s[1:])
        for mode, n_each, util, spread_ref, max_lots in product(
            MODES, N_EACH_GRID, UTILS, SPREAD_REF_GRID, MAX_LOTS_GRID
        ):
            for enter, exit_, min_hold in ENTER_EXIT_HOLD:
                if exit_ >= enter:
                    continue
                configs.append({
                    "sig_key": sig_key, "col": col, "lb": lb, "mode": mode,
                    "n_each": n_each, "enter": enter, "exit_": exit_, "min_hold": min_hold,
                    "util": util, "spread_ref": spread_ref, "max_lots": max_lots,
                })
    print(f"n configs={len(configs)}", flush=True)

    rows = []
    best_live = None
    best_pnl = None
    best_row = None
    for i, cfg in enumerate(configs):
        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(configs)}]", flush=True)
        r = run_config(day_cache[cfg["sig_key"]], cfg)
        name = (
            f"{cfg['sig_key']}|{cfg['mode']}|n{cfg['n_each']}|u{cfg['util']}"
            f"|ref{cfg['spread_ref']}|max{cfg['max_lots']}"
            f"|e{cfg['enter']}/x{cfg['exit_']}/h{cfg['min_hold']}"
        )
        row = {
            "name": name,
            **{k: cfg[k] for k in (
                "col", "lb", "mode", "n_each", "enter", "exit_", "min_hold",
                "util", "spread_ref", "max_lots",
            )},
            **{k: r[k] for k in r if k not in ("live", "pnl_series")},
        }
        rows.append(row)
        if best_row is None or (
            (row["avg_margin_on"] or 0) >= 300000
            and (row["sharpe"] or -9) >= 1.0
            and (row["pnl_cny"] or 0) > (best_row["pnl_cny"] or 0)
        ):
            # keep series for eventual top; finalize after filter
            pass

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "all_candidates.csv", index=False)

    ok = df[
        (df["sharpe"].fillna(-9) >= 1.0)
        & (df["n_neg_years"] <= 1)
        & (df["max_dd"].fillna(-9) >= -0.15)
        & (df["avg_margin_on"].fillna(0) >= 300_000)
        & (df["active_days"] >= 80)
    ].copy().sort_values(["pnl_cny", "sharpe"], ascending=False)
    ok.to_csv(OUT / "top_robust.csv", index=False)

    loose = df[
        (df["sharpe"].fillna(-9) >= 0.7)
        & (df["avg_margin_on"].fillna(0) >= 200_000)
        & (df["max_dd"].fillna(-9) >= -0.20)
        & (df["active_days"] >= 60)
    ].copy().sort_values(["pnl_cny", "sharpe"], ascending=False)
    loose.to_csv(OUT / "top_loose.csv", index=False)

    print(f"\npass robust={len(ok)}/{len(df)} loose={len(loose)}", flush=True)
    show = ok.head(15) if len(ok) else loose.head(15)
    cols = [
        "name", "sharpe", "return", "max_dd", "pnl_cny", "avg_margin_on",
        "p90_margin_on", "avg_lots_on", "cash", "n_neg_years", "active_days",
    ]
    if len(show):
        print(show[cols].to_string(index=False), flush=True)
    else:
        print(df.sort_values("pnl_cny", ascending=False)[cols].head(15).to_string(index=False), flush=True)
        show = df.sort_values(["sharpe", "pnl_cny"], ascending=False).head(5)

    linear_pnl = None
    if LINEAR_YEARLY.is_file():
        y = pd.read_csv(LINEAR_YEARLY)
        linear_pnl = float(y.loc[y["year"] >= 2024, "pnl_cny"].sum())
        print(f"daily linear 2024+ pnl≈{linear_pnl:.0f}", flush=True)

    report: dict = {
        "n_total": int(len(df)),
        "n_robust": int(len(ok)),
        "n_loose": int(len(loose)),
        "linear_pnl_2024plus": linear_pnl,
        "top15": show[cols].to_dict(orient="records") if len(show) else [],
    }

    if len(show):
        best = show.iloc[0].to_dict()
        cfg = {
            "mode": best["mode"],
            "n_each": int(best["n_each"]),
            "enter": float(best["enter"]),
            "exit_": float(best["exit_"]),
            "min_hold": int(best["min_hold"]),
            "util": float(best["util"]),
            "spread_ref": float(best["spread_ref"]),
            "max_lots": int(best["max_lots"]),
        }
        sig = f"{best['col']}|L{int(best['lb'])}"
        full = run_config(day_cache[sig], cfg)
        abl = run_config(day_cache[sig], {**cfg, "max_lots": 1, "util": min(0.4, cfg["util"])})

        # half-doc
        half_sh = []
        paths = docs["path"].drop_duplicates().to_numpy()
        for seed in range(3):
            rng = np.random.default_rng(seed)
            keep = set(rng.choice(paths, size=max(1, len(paths) // 2), replace=False))
            sub = docs[docs["path"].isin(keep)]
            fac = batch_aggregate(sub, best["col"], int(best["lb"]), dates, "uniform")
            cache = build_day_cache(panel, fac)
            hs = run_config(cache, cfg)["sharpe"]
            half_sh.append(hs)
            print(f"half-doc seed={seed} sharpe={hs}", flush=True)

        corr = None
        if LINEAR_2026.is_file():
            fs = pd.read_csv(LINEAR_2026)
            fs["date"] = pd.to_datetime(fs["日期"])
            fs["r"] = pd.to_numeric(fs["账户日收益"], errors="coerce").fillna(0.0)
            live = full["live"]
            common = live.index.intersection(fs["date"])
            if len(common) >= 30:
                a = live.reindex(common).fillna(0.0)
                b = pd.Series(fs["r"].values, index=fs["date"]).reindex(common).fillna(0.0)
                corr = float(a.corr(b)) if a.std() > 0 and b.std() > 0 else None

        years = []
        live = full["live"]
        for y, g in live.groupby(live.index.year):
            gg = g[np.isfinite(pd.to_numeric(g, errors="coerce"))]
            if len(gg) >= 10:
                years.append({"year": int(y), **year_pack(gg)})

        def _py(v):
            if isinstance(v, (np.bool_, bool)):
                return bool(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating, float)):
                return float(v) if np.isfinite(v) else None
            return v

        report["best"] = {
            "name": best["name"],
            **{k: _py(best[k]) for k in best if k != "name"},
            "full": {k: _py(full[k]) for k in full if k not in ("live", "pnl_series")},
            "ablation_1lot": {k: _py(abl[k]) for k in abl if k not in ("live", "pnl_series")},
            "half_doc_sharpe": [_py(x) for x in half_sh],
            "half_doc_median": _py(float(np.nanmedian([x for x in half_sh if x is not None]))),
            "corr_final_2026": _py(corr) if corr is not None else None,
            "years": years,
        }
        pd.DataFrame({
            "日期": [str(pd.Timestamp(i).date()) for i in full["pnl_series"].index],
            "账户日盈亏": full["pnl_series"].to_numpy(),
            "账户日收益": full["live"].reindex(full["pnl_series"].index).to_numpy(),
        }).to_csv(OUT / "best_daily.csv", index=False)

    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT, flush=True)
    if "best" in report:
        b = report["best"]
        print(
            f"BEST {b['name']}: sh={b.get('sharpe')} pnl={b.get('pnl_cny')} "
            f"avg_margin={b.get('avg_margin_on')} half_med={b.get('half_doc_median')} "
            f"corr2026={b.get('corr_final_2026')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
