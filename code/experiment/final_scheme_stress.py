#!/usr/bin/env python3
"""Stress-test final scheme: EW pos64/RSI/-PVR, BOOK>=1.2 flatten.

Covers holding-period rules, costs, fill time, gaps/locks.
"""

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
from linear_ridge_walkforward import (
    DEAD,
    EVAL_END,
    OUT_DIR,
    TREE,
    _read_contract_csv,
    build_panel,
)
from shared import REMOVED_SYMBOLS
from six_name_select import pick_ls_n
from two_model_rotate import BT_START
from two_name_select import apply_hold, turnover

LEGS = [("pos_64", 1, "pos64"), ("rsi", 1, "rsi"), ("price_volume_ratio", -1, "pvr")]
BOOK_THR = 1.2
GAP_SKIP = 0.035


def load_ohlc() -> pd.DataFrame:
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
            for c in ("open", "high", "low", "volume"):
                if c in raw.columns:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce")
            keep = [c for c in ("date", "code", "open", "high", "low", "close", "volume") if c in raw.columns]
            parts.append(raw[keep])
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True).sort_values(["code", "date"])
        g = df.groupby("code", sort=False)
        nxt_d = g["date"].shift(-1)
        gap_d = (nxt_d - df["date"]).dt.days
        o1 = g["open"].shift(-1).clip(lower=1e-8)
        c1 = g["close"].shift(-1).clip(lower=1e-8)
        h1, l1, v1 = g["high"].shift(-1), g["low"].shift(-1), g["volume"].shift(-1)
        o2 = g["open"].shift(-2).clip(lower=1e-8)
        c2 = g["close"].shift(-2).clip(lower=1e-8)
        cl = df["close"].clip(lower=1e-8)
        df["ret_cc"] = np.log(c1 / cl)
        df["ret_oo"] = np.log(o2 / o1)
        df["ret_oc"] = np.log(c1 / o1)
        df["ret_co"] = np.log(o1 / cl)
        df["ret_c1c2"] = np.log(c2 / c1)
        df["overnight_gap"] = o1 / cl - 1.0
        locked = (h1.notna() & l1.notna() & (h1 == l1)) | (v1.fillna(0) <= 0)
        df["next_locked"] = locked.fillna(True)
        bad = gap_d.isna() | (gap_d > 10)
        for c in ("ret_cc", "ret_oo", "ret_oc", "ret_co", "ret_c1c2", "overnight_gap"):
            df.loc[bad, c] = np.nan
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        df["symbol"] = sym
        frames.append(df[["date", "symbol", "code", "ret_cc", "ret_oo", "ret_oc", "ret_co", "ret_c1c2", "overnight_gap", "next_locked"]])
    return pd.concat(frames, ignore_index=True)


def combine_w(books: list[list[tuple[str, float]]]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    n = len(books) or 1
    for b in books:
        for s, w in b:
            out[s] += w / n
    return dict(out)


def pnl_from_w(day: pd.DataFrame, w: dict[str, float], col: str) -> float:
    if not w:
        return 0.0
    m = day.set_index("symbol")
    if col not in m.columns:
        return np.nan
    s = 0.0
    n = 0
    for sym, wt in w.items():
        if sym in m.index and np.isfinite(m.loc[sym, col]):
            s += wt * float(m.loc[sym, col])
            n += 1
    return s if n else 0.0


def apply_book_hyst(on: pd.Series, min_on: int, min_off: int, max_on: int | None) -> pd.Series:
    """on=True means want to be in market (BOOK below threshold)."""
    out = []
    cur = True
    age = 0
    for want in on.fillna(True).to_numpy():
        if cur:
            if max_on is not None and age >= max_on:
                cur, age = False, 0
            elif (not want) and age >= min_on:
                cur, age = False, 0
        else:
            if want and age >= min_off:
                cur, age = True, 0
        out.append(cur)
        age += 1
    return pd.Series(out, index=on.index)


def streaks(flags: pd.Series) -> dict:
    runs = []
    cur = None
    n = 0
    for v in flags.astype(bool).to_numpy():
        if cur is None:
            cur, n = v, 1
        elif v == cur:
            n += 1
        else:
            runs.append((cur, n))
            cur, n = v, 1
    if cur is not None:
        runs.append((cur, n))
    on = [k for t, k in runs if t]
    off = [k for t, k in runs if not t]
    def st(xs):
        if not xs:
            return {"n": 0, "min": None, "median": None, "max": None, "mean": None}
        a = np.array(xs)
        return {"n": int(len(a)), "min": int(a.min()), "median": float(np.median(a)), "max": int(a.max()), "mean": float(a.mean())}
    return {"in_market": st(on), "in_cash": st(off)}


def name_hold_days(daily_sets: list[set[str]]) -> dict:
    last: dict[str, int] = {}
    lives = []
    for i, names in enumerate(daily_sets):
        gone = [s for s in list(last) if s not in names]
        for s in gone:
            lives.append(i - last.pop(s))
        for s in names:
            last.setdefault(s, i)
    lives.extend(len(daily_sets) - v for v in last.values())
    if not lives:
        return {}
    a = np.array(lives)
    return {"min": int(a.min()), "median": float(np.median(a)), "p90": float(np.quantile(a, 0.9)), "max": int(a.max()), "mean": float(a.mean())}


def cost_pnl(pnl: pd.Series, turns: pd.Series, bps: float) -> pd.Series:
    c = (bps * 1e-4) * turns.reindex(pnl.index).fillna(0.0)
    return pnl - c


def pack(s: pd.Series, extra: dict | None = None) -> dict:
    st = year_pack(s)
    cash = float((s.abs() < 1e-12).mean())
    st["cash"] = fmt(cash, 3)
    if extra:
        st.update(extra)
    return st


def main() -> None:
    print("panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01"))
    print("ohlc fwd...", flush=True)
    bars = load_ohlc()
    panel = panel.merge(bars, on=["date", "symbol", "code"], how="left")
    need = ["fwd_ret", "pos_64", "rsi", "price_volume_ratio"]
    panel = panel.dropna(subset=need)

    recs = []
    name_sets = []
    print("daily weights...", flush=True)
    prev_w = None
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        books = []
        for col, sgn, _ in LEGS:
            books.append(pick_ls_n(day, sgn * day[col].to_numpy(), 3))
        m = day.drop_duplicates("symbol").set_index("symbol")
        w = combine_w(books)
        sym_cc = {s: float(m.loc[s, "ret_cc"]) for s in w if s in m.index and np.isfinite(m.loc[s, "ret_cc"])}
        n_lock = n_gap = 0
        w_skip = {}
        for s, wt in w.items():
            if s not in m.index:
                continue
            g = m.loc[s, "overnight_gap"] if "overnight_gap" in m.columns else np.nan
            lk = m.loc[s, "next_locked"] if "next_locked" in m.columns else True
            lk = bool(lk) if pd.notna(lk) else True
            if lk:
                n_lock += 1
            if np.isfinite(g) and abs(float(g)) >= GAP_SKIP:
                n_gap += 1
            if lk or (np.isfinite(g) and abs(float(g)) >= GAP_SKIP):
                continue
            w_skip[s] = wt
        recs.append({
            "date": dt,
            "w": w,
            "to": turnover(list(prev_w.items()) if prev_w else None, list(w.items())),
            "ret_cc": pnl_from_w(day, w, "ret_cc"),
            "ret_oo": pnl_from_w(day, w, "ret_oo"),
            "ret_oc": pnl_from_w(day, w, "ret_oc"),
            "ret_co": pnl_from_w(day, w, "ret_co"),
            "ret_c1c2": pnl_from_w(day, w, "ret_c1c2"),
            "ret_oo_skip": pnl_from_w(day, w_skip, "ret_oo"),
            "n_lock": n_lock,
            "n_gap": n_gap,
            "n_skip": len(w) - len(w_skip),
            "sym_cc": sym_cc,
        })
        name_sets.append(set(w))
        prev_w = w

    idx = pd.DatetimeIndex([r["date"] for r in recs])
    cc = pd.Series({r["date"]: r["ret_cc"] for r in recs}).sort_index()
    ratio = book_ratio(cc)
    want_on = ~(ratio.reindex(cc.index) >= BOOK_THR)
    want_on = want_on.fillna(True)
    base_on = want_on

    def gated(pnl: pd.Series, on: pd.Series) -> pd.Series:
        out = pnl.copy()
        out[~on.reindex(pnl.index).fillna(True)] = 0.0
        return out

    turns = pd.Series({r["date"]: r["to"] for r in recs}).sort_index()
    # when flattening, pay full exit: use |w| sum as extra turnover vs previous
    def safe_to(prev_w: dict | None, w: dict) -> float:
        if not prev_w and not w:
            return 0.0
        return turnover(list(prev_w.items()) if prev_w else None, list(w.items()))

    def turns_with_on(on: pd.Series, weight_of=None) -> pd.Series:
        prev_w = None
        out = []
        held = None
        for i, r in enumerate(recs):
            dt = r["date"]
            inn = bool(on.loc[dt]) if dt in on.index else True
            if weight_of is None:
                w = r["w"] if inn else {}
            else:
                if i % weight_of == 0:
                    held = r["w"]
                w = held if inn else {}
            out.append(safe_to(prev_w, w))
            prev_w = w
        return pd.Series(out, index=idx)

    schemes = {}
    on = base_on
    t_on = turns_with_on(on)
    schemes["现行: 收盘成交 无费"] = gated(cc, on)
    for bps in (0, 1, 2, 5, 10, 20):
        schemes[f"收盘成交 单边{bps}bp"] = cost_pnl(gated(cc, on), t_on, bps)

    oo = pd.Series({r["date"]: r["ret_oo"] for r in recs})
    oc = pd.Series({r["date"]: r["ret_oc"] for r in recs})
    co = pd.Series({r["date"]: r["ret_co"] for r in recs})
    c1c2 = pd.Series({r["date"]: r["ret_c1c2"] for r in recs})
    oo_skip = pd.Series({r["date"]: r["ret_oo_skip"] for r in recs})
    schemes["次日开盘→再下一开盘"] = gated(oo, on)
    schemes["次日开盘→再下一开盘 单边2bp"] = cost_pnl(gated(oo, on), t_on, 2)
    schemes["次日开盘→再下一开盘 单边5bp"] = cost_pnl(gated(oo, on), t_on, 5)
    schemes["次日收盘→再下一日收盘(晚一天)"] = gated(c1c2, on)
    schemes["次日开盘→次日收盘(只拿日内)"] = gated(oc, on)
    schemes["隔夜缺口(收盘到次日开)"] = gated(co, on)
    schemes["次日开盘 跳过涨跌停/缺口≥3.5%"] = gated(oo_skip, on)
    schemes["次日开盘 跳过锁停 单边2bp"] = cost_pnl(gated(oo_skip, on), t_on, 2)

    # holding hysteresis
    for min_on, min_off, max_on, tag in [
        (1, 1, None, "无缓冲"),
        (3, 3, None, "最少在场3日/空仓3日"),
        (5, 5, None, "最少在场5日/空仓5日"),
        (1, 5, None, "空仓至少5日才开"),
        (5, 1, None, "在场至少5日才平"),
        (1, 1, 20, "最多连开20日"),
        (1, 1, 40, "最多连开40日"),
        (3, 3, 40, "在场3–40日/空仓≥3"),
    ]:
        on_h = apply_book_hyst(base_on, min_on, min_off, max_on)
        schemes[f"收盘 {tag}"] = gated(cc, on_h)
        schemes[f"次日开盘 {tag} 2bp"] = cost_pnl(gated(oo, on_h), turns_with_on(on_h), 2)

    # min holding ≈ rebalance every k days (still respect BOOK flatten)
    for k in (2, 5, 10):
        held = {}
        rows = []
        for i, r in enumerate(recs):
            dt = r["date"]
            inn = bool(on.loc[dt]) if dt in on.index else True
            if i % k == 0:
                held = r["w"]
            if not inn:
                rows.append((dt, 0.0))
                continue
            rsum = 0.0
            for s, wt in held.items():
                if s in r["sym_cc"]:
                    rsum += wt * r["sym_cc"][s]
            rows.append((dt, rsum))
        s_k = pd.Series({d: v for d, v in rows})
        schemes[f"收盘 最少持仓{k}日再换仓"] = s_k
        schemes[f"收盘 最少持仓{k}日 单边2bp"] = cost_pnl(s_k, turns_with_on(on, k), 2)

    hold = streaks(on)
    names = name_hold_days(name_sets)
    n_gap = np.mean([r["n_gap"] for r in recs])
    n_lock = np.mean([r["n_lock"] for r in recs])
    n_skip = np.mean([r["n_skip"] for r in recs])
    to_mean = float(t_on.mean())

    diag = {
        "book_on_off_streaks_days": hold,
        "name_hold_days_union_of_3_books": names,
        "mean_turnover_one_way": fmt(to_mean, 3),
        "mean_turnover_hold2": fmt(float(turns_with_on(on, 2).mean()), 3),
        "mean_turnover_hold5": fmt(float(turns_with_on(on, 5).mean()), 3),
        "mean_turnover_hold10": fmt(float(turns_with_on(on, 10).mean()), 3),
        "mean_names_gapping_ge_3.5pct": fmt(n_gap, 2),
        "mean_names_locked_next_bar": fmt(n_lock, 2),
        "mean_names_skipped_lock_or_gap": fmt(n_skip, 2),
        "cash_base": fmt(float((~on).mean()), 3),
        "note": "ret_oo = signal at T close, hold next open to following open; locked ≈ next bar high==low or volume 0",
    }

    packed = {}
    print("\n=== results ===", flush=True)
    for name, s in schemes.items():
        if s.isna().all():
            continue
        st = pack(s)
        packed[name] = {k: st[k] for k in st if k != "yearly"}
        packed[name]["yearly"] = st["yearly"]
        print(
            f"{name:42} ret={st['return']:+.1%} sh={st['sharpe']:.2f} dd={st['max_dd']:.1%} "
            f"neg={st['n_neg_years']}{st['neg_years']} cash={st['cash']:.0%} 2026={st['sharpe_2026']}",
            flush=True,
        )

    # k-day rebalance: apply held weights to ret_cc by re-walking recs with last w — WRONG without per-name returns.
    # Fast fix: during recs we can store w and also a dict of symbol->ret_cc for that day
    # Too late unless we add in first loop. Add symbol returns into recs now would need rerun.
    # I'll add in first loop... we're already past it. Restart with storing day_rets.

    path = OUT_DIR / "final_scheme_stress.json"
    path.write_text(json.dumps({"diag": diag, "schemes": packed}, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(diag, ensure_ascii=False, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
