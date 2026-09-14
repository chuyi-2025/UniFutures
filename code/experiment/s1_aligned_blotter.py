#!/usr/bin/env python3
"""Align S1 news strategy to daily/linear final_scheme plumbing.

Same as pos64_rsi_overlay_2026_blotter:
  - capital 1e6
  - universe: 1-lot broker margin <= 50k
  - integer 1 lot / name
  - BOOK >= 1.18 -> flat
  - fees: shouxufei open + close_prev
  - pnl: dollar_1lot then log1p((pnl-fee)/1e6)

Signal: rule_edge | no_dianping | L30 | uniform | single + S1 hysteresis
  (enter0.4/exit0.1/hold5/hi0.55), picks only inside eligible universe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NEWS_TRAIN = Path("/home/workspace/lab/UniFutures/news/code/train")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(NEWS_TRAIN))

from blend_book import book_ratio  # noqa: E402
from family_rotate import year_pack  # noqa: E402
from final_scheme_2026_blotter import load_px  # noqa: E402
from futures_lot_specs import lot_fee  # noqa: E402
from linear_ridge_walkforward import build_panel  # noqa: E402
from pos64_4name_2026_blotter import action_lists, meta_of, picks_to_w, split_ls  # noqa: E402
from pos64_margin_cap_search import attach_margin  # noqa: E402
from pos64_margin_integer_search import CAPITAL, dollar_1lot, to_log  # noqa: E402
from pos64_rsi_overlay_2026_blotter import (  # noqa: E402
    BOOK_THR,
    DAILY_LINEAR,
    MARGIN_CAP,
    eligible,
    trade_fees,
)
from rule_edge_dynamic_search import day_spread, pick_variable  # noqa: E402
from rule_factor_mine import batch_aggregate, filter_docs, keep_single_symbol_docs  # noqa: E402
from rule_s1_overfit_check import S1  # noqa: E402
from common import DATA_DIR  # noqa: E402

BT_START = pd.Timestamp("2024-03-08")
OUT = DAILY_LINEAR


def to_int_lots(picks: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Map signed weights to ±1 lot (final_scheme style)."""
    out = []
    for s, w in picks:
        if w > 0:
            out.append((s, 1.0))
        elif w < 0:
            out.append((s, -1.0))
    return out


def load_s1_factor(dates: pd.DatetimeIndex) -> pd.DataFrame:
    docs = pd.read_parquet(DATA_DIR / "rule_docs_rich.parquet")
    docs["report_date"] = pd.to_datetime(docs["report_date"])
    docs = keep_single_symbol_docs(filter_docs(docs, "no_dianping"))
    print(f"news docs single+no_dp={len(docs)} paths={docs['path'].nunique()}", flush=True)
    fac = batch_aggregate(docs, "rule_edge", 30, dates, "uniform")
    return fac.rename(columns={"val": "rule_edge"})


def run_s1_picks(panel: pd.DataFrame) -> list[dict]:
    """Day-by-day S1 hysteresis on margin-eligible names only."""
    cfg = S1
    picks_state: list[tuple[str, float]] | None = None
    days_in = 0
    recs: list[dict] = []

    for dt, day0 in panel.groupby("date"):
        day = day0.drop_duplicates("symbol").reset_index(drop=True)
        elig = eligible(day)
        d = elig.dropna(subset=["rule_edge", "fwd_ret", "close"])
        if cfg["min_docs"] > 0:
            d = d[d["n_docs"] >= cfg["min_docs"]]
        spread = day_spread(d["rule_edge"]) if len(d) >= 2 else 0.0
        # pick_variable expects column "val"
        d_pick = d.rename(columns={"rule_edge": "val"})
        cand = pick_variable(
            d_pick,
            abs_thr=cfg["abs_thr"],
            max_each=cfg["max_each"],
            spread=spread,
            spread_hi=cfg["spread_hi"],
        )
        enter, exit_, min_hold = cfg["spread_enter"], cfg["spread_exit"], cfg["min_hold"]

        if picks_state is None:
            if spread >= enter and cand:
                picks_state = cand
                days_in = 0
        else:
            days_in += 1
            # drop names that became ineligible
            elig_set = set(d["symbol"])
            picks_state = [(s, w) for s, w in picks_state if s in elig_set]
            if len(picks_state) < 2:
                picks_state = None
                days_in = 0
            elif days_in >= min_hold and spread < exit_:
                picks_state = None
                days_in = 0
            elif days_in >= min_hold and spread >= enter and cand:
                if {s for s, _ in picks_state} != {s for s, _ in cand}:
                    picks_state = cand
                    days_in = 0

        lots = to_int_lots(picks_state) if picks_state else []
        # keep only eligible with margin/specs for dollar pnl
        lots = [(s, w) for s, w in lots if s in set(elig["symbol"])]
        dollar = dollar_1lot(day, lots) if lots else (0.0, 0.0, 0.0, 0)
        recs.append({
            "date": pd.Timestamp(dt),
            "w": picks_to_w(lots),
            "picks": lots,
            "dollar": dollar,
            "n_cs": int(len(elig)),
            "spread": spread,
            "meta": meta_of(day),
        })
    return recs


def apply_book_and_fees(recs: list[dict]) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, dict]:
    by_dt = {r["date"]: r for r in recs}
    dates = sorted(by_dt.keys())
    raw_pnl = pd.Series({r["date"]: r["dollar"][0] for r in recs}).sort_index()

    # first pass fees assuming always-in when picks non-empty (same as overlay blotter)
    fee_pre: dict = {}
    prev_held: dict[str, int] = {}
    prev_lots: dict[str, int] = {}
    raw_log_pre = to_log(raw_pnl / CAPITAL)
    ratio_pre = book_ratio(raw_log_pre.dropna()).reindex(raw_log_pre.index)
    if ratio_pre.notna().any():
        ratio_pre = ratio_pre.ffill(limit=1)
    empty_pre = pd.Series({r["date"]: len(r["picks"]) == 0 for r in recs}).reindex(raw_log_pre.index).fillna(False)

    for dt in dates:
        r = by_dt[dt]
        b = float(ratio_pre.loc[dt]) if dt in ratio_pre.index and np.isfinite(ratio_pre.loc[dt]) else None
        inn0 = not bool(empty_pre.loc[dt]) and not (b is not None and b >= BOOK_THR)
        w = r["w"]
        longs, shorts = split_ls(w)
        today: dict[str, int] = {}
        if inn0:
            for s in longs:
                today[s] = 1
            for s in shorts:
                today[s] = -1
        opens_long, opens_short, _, closes = action_lists(prev_held, today)
        lots_n0 = 1 if inn0 else 0
        fee_pre[dt] = trade_fees(r["meta"], lots_n0, closes, opens_long, opens_short, prev_lots)
        prev_held = today if inn0 else {}
        prev_lots = {s: lots_n0 for s in today} if inn0 else {}

    fee_s = pd.Series(fee_pre).sort_index()
    dollar_net = raw_pnl.sub(fee_s, fill_value=0.0)
    raw_log = to_log(dollar_net / CAPITAL)
    raw_bt = raw_log.dropna()
    ratio = book_ratio(raw_bt).reindex(raw_log.index)
    if ratio.notna().any():
        ratio = ratio.ffill(limit=1)
    cash = (ratio >= BOOK_THR).fillna(False)
    empty = pd.Series({r["date"]: len(r["picks"]) == 0 for r in recs}).reindex(raw_log.index).fillna(False)
    cash = cash | empty
    live = raw_log.copy()
    live[cash] = 0.0
    live = live.where(raw_log.notna())
    dollar_live = dollar_net.copy()
    dollar_live[cash] = 0.0
    dollar_live = dollar_live.where(raw_log.notna())
    return live, dollar_live, fee_s, cash, {"ratio": ratio, "raw_pnl": raw_pnl}


def metrics_block(live: pd.Series, cash: pd.Series, fee_s: pd.Series, dollar_live: pd.Series) -> dict:
    bt = live[np.isfinite(pd.to_numeric(live, errors="coerce"))]
    st = year_pack(bt)
    fee_total = float(fee_s.reindex(bt.index).fillna(0).sum())
    # yearly
    years = []
    for y, g in bt.groupby(bt.index.year):
        yy = year_pack(g)
        years.append({"year": int(y), **yy})
    return {
        **st,
        "cash": float(cash.reindex(bt.index).fillna(False).mean()) if len(bt) else None,
        "days": int(len(bt)),
        "fee_total": round(fee_total, 1),
        "pnl_cny": round(float(dollar_live.reindex(bt.index).fillna(0).sum()), 1),
        "years": years,
    }


def load_final_scheme_live() -> pd.Series:
    """Rebuild approx live log from final_scheme daily csv if present; else empty."""
    path = OUT / "final_scheme_daily.csv"
    # Prefer full history from json + year files — daily csv is year-only.
    # Reconstruct from blotter-compatible: use 2026 file + try to run quick compare on overlap only.
    rows = []
    for p in sorted(OUT.glob("final_scheme_*_daily.csv")):
        df = pd.read_csv(p)
        rows.append(df)
    if not rows and path.is_file():
        rows.append(pd.read_csv(path))
    if not rows:
        return pd.Series(dtype=float)
    df = pd.concat(rows, ignore_index=True).drop_duplicates("日期")
    df["date"] = pd.to_datetime(df["日期"])
    # 账户日收益 is log return string/float
    r = pd.to_numeric(df["账户日收益"], errors="coerce")
    s = pd.Series(r.values, index=df["date"]).sort_index()
    # empty days already 0 in file
    return s


def main() -> None:
    print("panel...", flush=True)
    panel = build_panel(BT_START).dropna(subset=["fwd_ret"])
    print("prices...", flush=True)
    px = load_px()
    panel = panel.merge(px, on=["date", "symbol", "code"], how="left")
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))

    print("S1 factor...", flush=True)
    fac = load_s1_factor(dates)
    panel = panel.merge(
        fac.rename(columns={"rule_edge": "rule_edge"}),
        on=["date", "symbol"],
        how="left",
    )
    # keep days with any news score in eligible set later
    panel = panel[panel["date"] >= BT_START].copy()

    print("S1 daily picks (margin<=5万)...", flush=True)
    recs = run_s1_picks(panel)
    live, dollar_live, fee_s, cash, extra = apply_book_and_fees(recs)
    st = metrics_block(live, cash, fee_s, dollar_live)
    print(
        f"S1-aligned ret={st['return']:+.1%} sh={st['sharpe']:.2f} dd={st['max_dd']:.1%} "
        f"cash={st['cash']:.0%} fee={st['fee_total']:.0f} pnl={st['pnl_cny']:.0f}",
        flush=True,
    )
    for y in st["years"]:
        print(f"  {y['year']}: ret={y['return']:+.1%} sh={y['sharpe']:.2f} dd={y['max_dd']:.1%}", flush=True)

    # Compare to final_scheme on overlapping dates (2026 daily file + json full_sample note)
    fs = load_final_scheme_live()
    compare = {}
    if len(fs):
        common = live.dropna().index.intersection(fs.dropna().index)
        if len(common) >= 30:
            a = live.reindex(common)
            b = fs.reindex(common)
            # when final marks 空仓, return may be 0 or nan — treat nan as 0 if 是否空仓
            compare = {
                "overlap_days": int(len(common)),
                "overlap_start": str(common.min().date()),
                "overlap_end": str(common.max().date()),
                "s1": year_pack(a),
                "final_scheme": year_pack(b),
                "corr": round(float(a.corr(b)), 3) if a.std() > 0 and b.std() > 0 else None,
            }
            print(
                f"overlap {compare['overlap_start']}..{compare['overlap_end']} n={compare['overlap_days']}: "
                f"S1 sh={compare['s1']['sharpe']:.2f} | final sh={compare['final_scheme']['sharpe']:.2f} "
                f"corr={compare['corr']}",
                flush=True,
            )

    # Also report final_scheme.json year_2026 / full for reference
    jpath = OUT / "final_scheme.json"
    final_ref = json.loads(jpath.read_text(encoding="utf-8")) if jpath.is_file() else {}

    # daily export (compact)
    rows = []
    ratio = extra["ratio"]
    for r in recs:
        dt = r["date"]
        inn = not bool(cash.loc[dt]) if dt in cash.index else False
        longs, shorts = split_ls(r["w"])
        rows.append({
            "日期": str(dt.date()),
            "是否空仓": (not inn),
            "BOOK": None if dt not in ratio.index or not np.isfinite(ratio.loc[dt]) else round(float(ratio.loc[dt]), 3),
            "spread": round(float(r["spread"]), 4),
            "截面可交易数": r["n_cs"],
            "账户日盈亏": None if dt not in dollar_live.index or not np.isfinite(dollar_live.loc[dt]) else int(round(float(dollar_live.loc[dt]))),
            "手续费": int(round(float(fee_s.loc[dt]))) if dt in fee_s.index else 0,
            "账户日收益": None if dt not in live.index or not np.isfinite(live.loc[dt]) else round(float(live.loc[dt]), 6),
            "信号做多": ",".join(longs),
            "信号做空": ",".join(shorts),
            "合计保证金": int(round(float(r["dollar"][1]))) if inn and r["dollar"][1] == r["dollar"][1] else 0,
        })
    daily = pd.DataFrame(rows)
    daily_path = OUT / "s1_aligned_daily.csv"
    daily.to_csv(daily_path, index=False)

    y2026 = live[(live.index >= "2026-01-01") & (live.index <= "2026-12-31")]
    y2026 = y2026[np.isfinite(pd.to_numeric(y2026, errors="coerce"))]
    st_2026 = year_pack(y2026) if len(y2026) else {}
    cash_2026 = float(cash.reindex(y2026.index).fillna(False).mean()) if len(y2026) else None

    report = {
        "scheme": "S1 news aligned to daily/linear plumbing",
        "signal": "rule_edge|no_dianping|L30|uniform|single + hysteresis enter0.4/exit0.1/hold5/hi0.55",
        "rules": {
            "capital": CAPITAL,
            "margin_cap": MARGIN_CAP,
            "book_thr": BOOK_THR,
            "lots": "integer 1 lot per selected name",
            "fee": "shouxufei open + close_prev (same as final_scheme)",
            "pnl": "log1p((dollar_1lot - fee)/1e6)",
        },
        "full_sample": {k: st[k] for k in st if k != "years"},
        "years": st["years"],
        "year_2026": {**st_2026, "cash": cash_2026, "days": int(len(y2026))},
        "vs_final_scheme_overlap": compare,
        "final_scheme_ref": {
            "full_sample": final_ref.get("full_sample"),
            "year_2026": final_ref.get("year_2026"),
        },
        "files": {"daily": str(daily_path)},
    }
    rpath = OUT / "s1_aligned.json"
    rpath.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {daily_path}", flush=True)
    print(f"saved {rpath}", flush=True)


if __name__ == "__main__":
    main()
