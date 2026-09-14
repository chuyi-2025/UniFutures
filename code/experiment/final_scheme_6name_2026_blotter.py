#!/usr/bin/env python3
"""2026 blotter: one 6-name book (3L3S) on z(pos64)+z(RSI)+z(-PVR), BOOK>=1.2 flatten."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from blend_book import book_ratio, zscore
from family_rotate import fmt, year_pack
from final_scheme_2026_blotter import CN, _append_pos, load_px, pick_ls_close
from final_scheme_stress import BOOK_THR
from linear_ridge_walkforward import OUT_DIR, build_panel
from six_name_select import pick_ls_n
from two_model_rotate import BT_START

PARTS = [("pos_64", 1), ("rsi", 1), ("price_volume_ratio", -1)]
N_EACH = 3


def blend_score(day: pd.DataFrame) -> np.ndarray:
    sc = None
    for col, sgn in PARTS:
        z = sgn * zscore(day[col].to_numpy())
        sc = z if sc is None else sc + z
    return sc if sc is not None else np.full(len(day), np.nan)


def picks_to_w(picks: list[tuple[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for s, wt in picks:
        out[s] = out.get(s, 0.0) + wt
    return out


def main() -> None:
    print("panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01"))
    print("prices...", flush=True)
    px = load_px()
    panel = panel.merge(px, on=["date", "symbol", "code"], how="left")
    full = panel.dropna(subset=["pos_64", "rsi", "price_volume_ratio"]).copy()
    hist = full.dropna(subset=["fwd_ret"])

    recs = []
    pos_rows = []
    print("daily 3L3S...", flush=True)
    for dt, day in hist.groupby("date"):
        if dt < BT_START:
            continue
        sc = blend_score(day)
        picks = pick_ls_n(day, sc, N_EACH)
        w = picks_to_w(picks)
        m = day.drop_duplicates("symbol").set_index("symbol")
        pnl = 0.0
        n = 0
        for s, wt in w.items():
            if s in m.index and np.isfinite(m.loc[s, "fwd_ret"]):
                pnl += wt * float(m.loc[s, "fwd_ret"])
                n += 1
        recs.append({"date": dt, "w": w, "picks": picks, "score": sc, "pnl": pnl if n else np.nan, "n_cs": int(day["symbol"].nunique())})
        _append_pos(pos_rows, dt, {"blend": picks}, m)

    last_full = pd.Timestamp(full["date"].max())
    last_hist = pd.Timestamp(recs[-1]["date"]) if recs else None
    if last_hist is None or last_full > last_hist:
        day = full[full["date"] == last_full]
        sc = blend_score(day)
        picks = pick_ls_close(day, sc, N_EACH)
        w = picks_to_w(picks)
        m = day.drop_duplicates("symbol").set_index("symbol")
        recs.append({"date": last_full, "w": w, "picks": picks, "score": sc, "pnl": np.nan, "n_cs": int(day["symbol"].nunique())})
        _append_pos(pos_rows, last_full, {"blend": picks}, m)

    pnl = pd.Series({r["date"]: r["pnl"] for r in recs}).sort_index()
    ratio = book_ratio(pnl.dropna()).reindex(pnl.index)
    if ratio.notna().any():
        ratio = ratio.ffill(limit=1)
    on = ~(ratio.reindex(pnl.index) >= BOOK_THR)
    on = on.fillna(True)
    live = pnl.copy()
    live[~on] = 0.0

    full_bt = live[np.isfinite(pd.to_numeric(live, errors="coerce"))]
    y2026 = live[(live.index >= "2026-01-01") & (live.index <= "2026-12-31")]
    y2026_bt = y2026[np.isfinite(pd.to_numeric(y2026, errors="coerce"))]
    st_all = year_pack(full_bt)
    st_26 = year_pack(y2026_bt)
    cash_26 = float((~on.reindex(y2026.index).fillna(False)).mean())
    print(
        f"all ret={st_all['return']:+.1%} sh={st_all['sharpe']:.2f} dd={st_all['max_dd']:.1%} | "
        f"2026 ret={st_26['return']:+.1%} sh={st_26['sharpe']:.2f} dd={st_26['max_dd']:.1%} cash={cash_26:.0%}",
        flush=True,
    )

    pos = pd.DataFrame(pos_rows)
    pos["in_market"] = pos["date"].map(lambda d: bool(on.loc[d]) if d in on.index else True)
    pos["book"] = pos["date"].map(lambda d: float(ratio.loc[d]) if d in ratio.index and np.isfinite(ratio.loc[d]) else None)
    pos["account_pnl"] = pos["date"].map(lambda d: float(live.loc[d]) if d in live.index else None)
    pos["nav_weight"] = pos["leg_weight"]  # already ±1/6
    pos["lots"] = np.where(pos["in_market"], np.sign(pos["leg_weight"]).astype(int), 0)

    daily = []
    for r in recs:
        dt = r["date"]
        inn = bool(on.loc[dt]) if dt in on.index else True
        b = float(ratio.loc[dt]) if dt in ratio.index and np.isfinite(ratio.loc[dt]) else None
        longs, shorts = [], []
        if inn:
            for s, wt in sorted(r["w"].items(), key=lambda x: -x[1]):
                (longs if wt > 0 else shorts).append(s)
        n_names = len(longs) + len(shorts)
        daily.append({
            "date": str(pd.Timestamp(dt).date()),
            "book": fmt(b, 3) if b is not None else None,
            "cash": (not inn),
            "pnl": fmt(float(live.loc[dt]), 5) if dt in live.index and np.isfinite(live.loc[dt]) else None,
            "n_cs": r["n_cs"],
            "longs": ",".join(longs),
            "shorts": ",".join(shorts),
            "n_names": n_names,
        })

    ddf = pd.DataFrame(daily)
    ddf["date"] = pd.to_datetime(ddf["date"])
    d2026 = ddf[(ddf["date"] >= "2026-01-01") & (ddf["date"] <= "2026-12-31")].copy()
    p2026 = pos[(pos["date"] >= "2026-01-01") & (pos["date"] <= "2026-12-31")].copy()
    p2026["date"] = pd.to_datetime(p2026["date"]).dt.strftime("%Y-%m-%d")
    d2026["date"] = d2026["date"].dt.strftime("%Y-%m-%d")

    live2026 = y2026.copy()
    live2026.index = pd.to_datetime(live2026.index)
    months = []
    for per, g in live2026.groupby(live2026.index.to_period("M")):
        gg = g[np.isfinite(pd.to_numeric(g, errors="coerce"))]
        m = year_pack(gg) if len(gg) else {"return": 0.0, "sharpe": 0.0, "max_dd": 0.0}
        months.append({
            "month": str(per),
            "return": fmt(m["return"], 4),
            "sharpe": fmt(m["sharpe"], 3),
            "max_dd": fmt(m["max_dd"], 4),
            "cash": fmt(float((~on.reindex(g.index).fillna(False)).mean()), 3),
            "days": int(len(g)),
        })

    last = d2026.iloc[-1].to_dict() if not d2026.empty else {}
    last_dt = last.get("date")
    last_pos = p2026[p2026["date"] == last_dt].copy() if last_dt else p2026.iloc[0:0]

    chk = d2026[d2026["date"].isin(["2026-02-27", "2026-03-02", "2026-03-03", "2026-03-10"])]
    print(chk.to_string(index=False), flush=True)

    d_path = OUT_DIR / "final_scheme_6name_2026_daily.csv"
    p_path = OUT_DIR / "final_scheme_6name_2026_positions.csv"
    d2026.to_csv(d_path, index=False)
    cols = [
        "date", "in_market", "book", "leg", "symbol", "name_cn", "code", "side",
        "lots", "nav_weight", "pos64", "rsi", "pvr", "close", "next_date", "next_open", "fwd_ret", "account_pnl",
    ]
    p2026[[c for c in cols if c in p2026.columns]].to_csv(p_path, index=False)

    snap = {
        "scheme": "z(pos64)+z(RSI)+z(-PVR) → 3L3S (max 6 names), flatten if BOOK>=1.2",
        "full_sample": {k: st_all[k] for k in ("return", "sharpe", "max_dd", "days", "n_neg_years")},
        "year_2026": {**{k: st_26[k] for k in ("return", "sharpe", "max_dd", "days")}, "cash": fmt(cash_26, 3)},
        "months": months,
        "last_day": last,
        "last_positions": last_pos.replace({np.nan: None}).to_dict(orient="records"),
        "files": {"daily": str(d_path), "positions": str(p_path)},
    }
    jpath = OUT_DIR / "final_scheme_6name_2026.json"
    jpath.write_text(json.dumps(snap, ensure_ascii=False, indent=2, default=str))
    print(f"last {last_dt} cash={last.get('cash')} book={last.get('book')} longs={last.get('longs')} shorts={last.get('shorts')} n={last.get('n_names')}")
    print(f"saved {d_path} {p_path} {jpath}")


if __name__ == "__main__":
    main()
