#!/usr/bin/env python3
"""2026 blotter for EW pos64/RSI/-PVR + BOOK>=1.2 flatten."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from blend_book import book_ratio
from family_rotate import fmt, year_pack
from final_scheme_stress import BOOK_THR, LEGS, combine_w
from infer.config import SYMBOL_CN
from linear_ridge_walkforward import DEAD, OUT_DIR, TREE, _read_contract_csv, build_panel
from shared import REMOVED_SYMBOLS
from six_name_select import pick_ls_n
from two_model_rotate import BT_START


def pick_ls_close(day: pd.DataFrame, score: np.ndarray, n_each: int) -> list[tuple[str, float]]:
    s = pd.Series(score, index=day.index)
    ok = s.notna()
    if "close" in day.columns:
        ok = ok & day["close"].notna()
    s = s[ok]
    if len(s) < 2:
        return []
    k = min(n_each, len(s) // 2)
    if k < 1:
        return []
    longs = s.nlargest(k)
    shorts = s.drop(index=longs.index).nsmallest(k)
    w = 0.5 / k
    out = [(day.loc[i, "symbol"], w) for i in longs.index]
    out += [(day.loc[i, "symbol"], -w) for i in shorts.index]
    return out

CN = dict(SYMBOL_CN)
try:
    sys.path.insert(0, str(HERE.parents[1] / "news/code/train"))
    from symbol_map import SYMBOL_ALIASES  # type: ignore
    for alias, sym in SYMBOL_ALIASES.items():
        CN.setdefault(sym, alias)
except Exception:  # noqa: BLE001
    pass


def load_px() -> pd.DataFrame:
    frames = []
    for sym_dir in sorted(p for p in TREE.iterdir() if p.is_dir()):
        sym = sym_dir.name.upper()
        if sym in REMOVED_SYMBOLS or sym in DEAD:
            continue
        parts = []
        for csv in sorted(sym_dir.glob("*.csv")):
            try:
                raw = _read_contract_csv(csv)
            except Exception:  # noqa: BLE001
                continue
            for c in ("open", "high", "low"):
                if c in raw.columns:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce")
            parts.append(raw[[c for c in ("date", "code", "open", "high", "low", "close") if c in raw.columns]])
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True).sort_values(["code", "date"])
        g = df.groupby("code", sort=False)
        df["next_date"] = g["date"].shift(-1)
        df["next_open"] = g["open"].shift(-1)
        df["next_close"] = g["close"].shift(-1)
        df["symbol"] = sym
        frames.append(df[["date", "symbol", "code", "open", "high", "low", "close", "next_date", "next_open", "next_close"]])
    return pd.concat(frames, ignore_index=True)


def _append_pos(pos_rows: list, dt, picks: dict, m: pd.DataFrame) -> None:
    for leg, lst in picks.items():
        for s, wt in lst:
            if s not in m.index:
                continue
            r = m.loc[s]
            pos_rows.append({
                "date": dt,
                "leg": leg,
                "symbol": s,
                "name_cn": CN.get(s, s),
                "code": str(r.get("code", "")),
                "side": "多" if wt > 0 else "空",
                "leg_weight": wt,
                "pos64": float(r["pos_64"]) if np.isfinite(r.get("pos_64", np.nan)) else None,
                "rsi": float(r["rsi"]) if np.isfinite(r.get("rsi", np.nan)) else None,
                "pvr": float(r["price_volume_ratio"]) if np.isfinite(r.get("price_volume_ratio", np.nan)) else None,
                "close": float(r["close"]) if "close" in r and np.isfinite(r["close"]) else None,
                "next_open": float(r["next_open"]) if "next_open" in r and np.isfinite(r.get("next_open", np.nan)) else None,
                "next_close": float(r["next_close"]) if "next_close" in r and np.isfinite(r.get("next_close", np.nan)) else None,
                "next_date": str(pd.Timestamp(r["next_date"]).date()) if "next_date" in r and pd.notna(r.get("next_date")) else None,
                "fwd_ret": float(r["fwd_ret"]) if np.isfinite(r.get("fwd_ret", np.nan)) else None,
            })


def main() -> None:
    print("panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01"))
    print("prices...", flush=True)
    px = load_px()
    panel = panel.merge(px, on=["date", "symbol", "code"], how="left")
    full = panel.dropna(subset=["pos_64", "rsi", "price_volume_ratio"]).copy()
    panel = full.dropna(subset=["fwd_ret"])

    recs = []
    pos_rows = []
    print("daily books...", flush=True)
    for dt, day in panel.groupby("date"):
        if dt < BT_START:
            continue
        books = {}
        picks = {}
        for col, sgn, name in LEGS:
            w = pick_ls_n(day, sgn * day[col].to_numpy(), 3)
            books[name] = w
            picks[name] = w
        w = combine_w([books[n] for n in ("pos64", "rsi", "pvr")])
        m = day.drop_duplicates("symbol").set_index("symbol")
        pnl = 0.0
        n = 0
        for s, wt in w.items():
            if s in m.index and np.isfinite(m.loc[s, "fwd_ret"]):
                pnl += wt * float(m.loc[s, "fwd_ret"])
                n += 1
        recs.append({"date": dt, "w": w, "picks": picks, "pnl": pnl if n else np.nan, "n_cs": int(day["symbol"].nunique())})

        _append_pos(pos_rows, dt, picks, m)

    # Last calendar day may lack fwd_ret (no next close yet) — still emit a live signal.
    last_full = pd.Timestamp(full["date"].max())
    last_hist = pd.Timestamp(recs[-1]["date"]) if recs else None
    if last_hist is None or last_full > last_hist:
        day = full[full["date"] == last_full]
        picks = {}
        books = {}
        for col, sgn, name in LEGS:
            w = pick_ls_close(day, sgn * day[col].to_numpy(), 3)
            books[name] = w
            picks[name] = w
        w = combine_w([books[n] for n in ("pos64", "rsi", "pvr")])
        m = day.drop_duplicates("symbol").set_index("symbol")
        recs.append({"date": last_full, "w": w, "picks": picks, "pnl": np.nan, "n_cs": int(day["symbol"].nunique())})
        _append_pos(pos_rows, last_full, picks, m)

    pnl = pd.Series({r["date"]: r["pnl"] for r in recs}).sort_index()
    ratio = book_ratio(pnl.dropna())
    ratio = ratio.reindex(pnl.index)
    if ratio.notna().any():
        # trailing signal day only
        ratio = ratio.ffill(limit=1)
    on = ~(ratio.reindex(pnl.index) >= BOOK_THR)
    on = on.fillna(True)
    live = pnl.copy()
    live[~on] = 0.0

    y2026 = live[(live.index >= "2026-01-01") & (live.index <= "2026-12-31")]
    y2026_bt = y2026[np.isfinite(pd.to_numeric(y2026, errors="coerce"))]
    stats = year_pack(y2026_bt)
    print(f"2026 ret={stats['return']:+.1%} sh={stats['sharpe']:.2f} dd={stats['max_dd']:.1%} cash={float((~on.reindex(y2026.index).fillna(False)).mean()):.0%}", flush=True)

    pos = pd.DataFrame(pos_rows)
    pos["in_market"] = pos["date"].map(lambda d: bool(on.loc[d]) if d in on.index else True)
    pos["book"] = pos["date"].map(lambda d: float(ratio.loc[d]) if d in ratio.index and np.isfinite(ratio.loc[d]) else None)
    pos["account_pnl"] = pos["date"].map(lambda d: float(live.loc[d]) if d in live.index else None)
    pos["nav_weight"] = pos["leg_weight"] / 3.0  # each leg is 1/3 capital
    # lots: 1 lot per selected slot; 3 legs × 6 names = 18 lots when in market
    pos["lots"] = np.where(pos["in_market"], np.sign(pos["leg_weight"]).astype(int), 0)
    pos["lots_6budget"] = np.where(pos["in_market"], np.sign(pos["leg_weight"]) * (6 / 18), 0.0)

    daily = []
    for r in recs:
        dt = r["date"]
        inn = bool(on.loc[dt]) if dt in on.index else True
        b = float(ratio.loc[dt]) if dt in ratio.index and np.isfinite(ratio.loc[dt]) else None
        longs, shorts = [], []
        if inn:
            agg = defaultdict(float)
            for s, wt in r["w"].items():
                agg[s] += wt / 3.0
            for s, wt in sorted(agg.items(), key=lambda x: -abs(x[1])):
                (longs if wt > 0 else shorts).append(s)
        daily.append({
            "date": str(pd.Timestamp(dt).date()),
            "book": fmt(b, 3) if b is not None else None,
            "cash": (not inn),
            "pnl": fmt(float(live.loc[dt]), 5) if dt in live.index and np.isfinite(live.loc[dt]) else None,
            "n_cs": r["n_cs"],
            "longs": ",".join(longs),
            "shorts": ",".join(shorts),
            "n_names": len(longs) + len(shorts),
        })

    ddf = pd.DataFrame(daily)
    ddf["date"] = pd.to_datetime(ddf["date"])
    d2026 = ddf[(ddf["date"] >= "2026-01-01") & (ddf["date"] <= "2026-12-31")].copy()
    p2026 = pos[(pos["date"] >= "2026-01-01") & (pos["date"] <= "2026-12-31")].copy()
    p2026["date"] = pd.to_datetime(p2026["date"]).dt.strftime("%Y-%m-%d")
    d2026["date"] = d2026["date"].dt.strftime("%Y-%m-%d")

    # monthly
    live2026 = y2026.copy()
    live2026.index = pd.to_datetime(live2026.index)
    months = []
    for per, g in live2026.groupby(live2026.index.to_period("M")):
        m = year_pack(g[np.isfinite(pd.to_numeric(g, errors="coerce"))])
        cash_m = float((~on.reindex(g.index).fillna(False)).mean())
        months.append({
            "month": str(per),
            "return": fmt(m["return"], 4),
            "sharpe": fmt(m["sharpe"], 3),
            "max_dd": fmt(m["max_dd"], 4),
            "cash": fmt(cash_m, 3),
            "days": int(len(g)),
        })

    last = d2026.iloc[-1].to_dict() if not d2026.empty else {}
    last_dt = last.get("date")
    last_pos = p2026[p2026["date"] == last_dt].copy() if last_dt else p2026.iloc[0:0]

    last_ts = pd.Timestamp(last_dt) if last_dt else None
    uni_last = []
    if last_ts is not None:
        day_last = full[full["date"] == last_ts].drop_duplicates("symbol")
        for _, r in day_last.iterrows():
            uni_last.append({
                "symbol": r["symbol"],
                "name_cn": CN.get(r["symbol"], r["symbol"]),
                "code": str(r.get("code", "")),
                "pos64": fmt(float(r["pos_64"]), 4) if np.isfinite(r.get("pos_64", np.nan)) else None,
                "rsi": fmt(float(r["rsi"]), 2) if np.isfinite(r.get("rsi", np.nan)) else None,
                "pvr": fmt(float(r["price_volume_ratio"]), 4) if np.isfinite(r.get("price_volume_ratio", np.nan)) else None,
                "close": fmt(float(r["close"]), 2) if "close" in r and np.isfinite(r["close"]) else None,
            })

    ew_net = []
    if not last_pos.empty:
        g = last_pos.groupby(["symbol", "name_cn", "code"], as_index=False).agg(
            net_lots=("lots", "sum"),
            net_nav=("nav_weight", "sum"),
            close=("close", "first"),
            next_open=("next_open", "first"),
        )
        g = g.sort_values("net_lots", key=lambda s: -s.abs())
        for _, r in g.iterrows():
            ew_net.append({
                "symbol": r["symbol"],
                "name_cn": r["name_cn"],
                "code": r["code"],
                "side": "多" if r["net_lots"] > 0 else ("空" if r["net_lots"] < 0 else "净平"),
                "net_lots": int(r["net_lots"]),
                "nav_weight": fmt(float(r["net_nav"]), 4),
                "signal_close": fmt(float(r["close"]), 2) if pd.notna(r["close"]) else None,
                "fill_next_open": fmt(float(r["next_open"]), 2) if pd.notna(r["next_open"]) else None,
            })

    d_path = OUT_DIR / "final_scheme_2026_daily.csv"
    p_path = OUT_DIR / "final_scheme_2026_positions.csv"
    d2026.to_csv(d_path, index=False)
    cols = [
        "date", "in_market", "book", "leg", "symbol", "name_cn", "code", "side",
        "lots", "nav_weight", "pos64", "rsi", "pvr", "close", "next_date", "next_open", "next_close", "fwd_ret", "account_pnl",
    ]
    p2026[[c for c in cols if c in p2026.columns]].to_csv(p_path, index=False)

    snap = {
        "scheme": "EW pos64 + RSI + (-price/volume), flatten if BOOK>=1.2",
        "year_2026": {**{k: stats[k] for k in ("return", "sharpe", "max_dd", "days")}, "cash": fmt(float((~on.reindex(y2026.index).fillna(False)).mean()), 3)},
        "months": months,
        "last_day": last,
        "universe_last": uni_last,
        "ew_net_last": ew_net,
        "last_positions": last_pos.replace({np.nan: None}).to_dict(orient="records"),
        "daily_2026": d2026.replace({np.nan: None}).to_dict(orient="records"),
        "files": {"daily": str(d_path), "positions": str(p_path)},
    }
    jpath = OUT_DIR / "final_scheme_2026.json"
    jpath.write_text(json.dumps(snap, ensure_ascii=False, indent=2, default=str))
    print(f"last {last_dt} cash={last.get('cash')} book={last.get('book')} longs={last.get('longs')} shorts={last.get('shorts')}")
    print(f"saved {d_path} {p_path} {jpath}")


if __name__ == "__main__":
    main()
