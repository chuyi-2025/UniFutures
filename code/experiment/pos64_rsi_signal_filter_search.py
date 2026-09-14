#!/usr/bin/env python3
"""Signal-quality filters for pos64+RSI overlay (final_scheme family).

Addresses high-turnover band trading: daily LS flips, weak cross-section days.
Reuses final_scheme rules: ≤5万 margin, 1 lot, BOOK≥1.18 flat.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from blend_book import book_ratio
from family_rotate import fmt, year_pack
from final_scheme_2026_blotter import load_px, pick_ls_close
from pos64_margin_cap_search import attach_margin
from pos64_margin_integer_search import CAPITAL, dollar_1lot, net_picks, to_log
from pos64_rsi_overlay_2026_blotter import build_scheme_panel, eligible, pick_overlay
from six_name_select import pick_ls_n, turnover
from two_model_rotate import BT_START

MARGIN_CAP = 50_000.0
BOOK_THR = 1.18
N_EACH = 1
OUT = HERE.parent.parent / "data/infer/results/linear_factor/signal_filter_search.json"


@dataclass
class FilterCfg:
    name: str
    min_hold: int = 0
    rebal_k: int = 1
    cs_gap: float = 0.0
    dual_confirm: bool = False
    flip_cooldown: int = 0
    rsi_extreme: float = 0.0


def pick_dual_confirm(day: pd.DataFrame) -> list[tuple[str, float]]:
    d = eligible(day)
    if len(d) < 2:
        return []
    a = pick_ls_n(d, d["pos_64"].to_numpy(), N_EACH)
    r = pick_ls_n(d, d["rsi"].to_numpy(), N_EACH)
    longs = {s for s, w in a if w > 0} & {s for s, w in r if w > 0}
    shorts = {s for s, w in a if w < 0} & {s for s, w in r if w < 0}
    out: list[tuple[str, float]] = []
    for s in longs:
        out.append((s, 1.0))
    for s in shorts:
        out.append((s, -1.0))
    return out


def cs_gap_ok(day: pd.DataFrame, gap_thr: float) -> bool:
    if gap_thr <= 0:
        return True
    d = eligible(day)
    if len(d) < 2:
        return False
    p = d["pos_64"].dropna()
    if len(p) < 2:
        return False
    return float(p.max() - p.min()) >= gap_thr


def rsi_extreme_ok(day: pd.DataFrame, thr: float, picks: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if thr <= 0 or not picks:
        return picks
    m = day.set_index("symbol")["rsi"]
    out = []
    for s, w in picks:
        if s not in m.index or not np.isfinite(m.loc[s]):
            continue
        rv = float(m.loc[s])
        if w > 0 and rv >= thr:
            out.append((s, w))
        elif w < 0 and rv <= (100.0 - thr):
            out.append((s, w))
    return out


def apply_filter_cfg(day: pd.DataFrame, cfg: FilterCfg, use_close: bool) -> list[tuple[str, float]]:
    if not cs_gap_ok(day, cfg.cs_gap):
        return []
    if cfg.dual_confirm:
        picks = pick_dual_confirm(day)
    else:
        picks = pick_overlay(day, use_close=use_close)
    picks = rsi_extreme_ok(day, cfg.rsi_extreme, picks)
    return picks


def simulate(cfg: FilterCfg, recs: list[dict]) -> dict:
    by_dt = {r["date"]: r for r in recs}
    dates = sorted(by_dt.keys())

    prev_target: dict[str, int] = {}
    prev_picks: list[tuple[str, float]] = []
    days_in = 0
    rebal_counter = 0
    flip_block: dict[str, tuple[int, int]] = {}  # sym -> (days_left, blocked_side +1/-1)
    raw_pnl: dict[pd.Timestamp, float] = {}
    turn_sum = 0.0
    turn_n = 0
    trade_days = 0

    for dt in dates:
        r = by_dt[dt]
        day = r["day"]
        new_picks = apply_filter_cfg(day, cfg, use_close=False)

        # flip cooldown: drop picks that violate cooldown
        if cfg.flip_cooldown > 0 and new_picks:
            cleaned = []
            for s, w in new_picks:
                side = 1 if w > 0 else -1
                blk = flip_block.get(s)
                if blk and blk[0] > 0 and blk[1] != side:
                    continue
                cleaned.append((s, w))
            new_picks = cleaned

        # min_hold sticky
        if cfg.min_hold > 0 and prev_picks and days_in < cfg.min_hold:
            picks = prev_picks
        elif cfg.rebal_k > 1 and prev_picks and (rebal_counter % cfg.rebal_k) != 0:
            picks = prev_picks
        else:
            picks = new_picks
            if picks != prev_picks:
                days_in = 0
            rebal_counter += 1

        pnl, _, _, n = dollar_1lot(day, picks)
        raw_pnl[dt] = float(pnl) if np.isfinite(pnl) else 0.0
        if n > 0:
            trade_days += 1
        if prev_picks:
            turn_sum += turnover(prev_picks, picks)
            turn_n += 1

        # update flip cooldown on closes / side changes
        if cfg.flip_cooldown > 0:
            prev_set = {s: (1 if w > 0 else -1) for s, w in prev_picks}
            new_set = {s: (1 if w > 0 else -1) for s, w in picks}
            for s in set(prev_set) - set(new_set):
                flip_block[s] = (cfg.flip_cooldown, prev_set[s])
            for s in set(prev_set) & set(new_set):
                if prev_set[s] != new_set[s]:
                    flip_block[s] = (cfg.flip_cooldown, prev_set[s])
            for s in list(flip_block):
                dleft, _ = flip_block[s]
                if s in new_set and new_set[s] == flip_block[s][1]:
                    flip_block.pop(s, None)
                elif dleft <= 1:
                    flip_block.pop(s, None)
                else:
                    flip_block[s] = (dleft - 1, flip_block[s][1])

        if picks == prev_picks and picks:
            days_in += 1
        prev_picks = picks
        prev_target = {s: (1 if w > 0 else -1) for s, w in picks}

    s_pnl = pd.Series(raw_pnl).sort_index()
    raw_log = to_log(s_pnl / CAPITAL)
    ratio = book_ratio(raw_log.dropna()).reindex(raw_log.index)
    if ratio.notna().any():
        ratio = ratio.ffill(limit=1)
    cash = (ratio >= BOOK_THR).fillna(False)
    empty = s_pnl.index.to_series().map(lambda d: len(apply_filter_cfg(by_dt[d]["day"], cfg, False)) == 0)
    cash = cash | empty.reindex(raw_log.index).fillna(False)
    live = raw_log.copy()
    live[cash] = 0.0
    live = live.where(raw_log.notna())

    st = year_pack(live.dropna())
    y2026 = live[(live.index >= "2026-01-01") & (live.index <= "2026-12-31")].dropna()
    st2026 = year_pack(y2026) if len(y2026) else {}

    return {
        "name": cfg.name,
        "sharpe": st.get("sharpe"),
        "return": st.get("return"),
        "max_dd": st.get("max_dd"),
        "cash_pct": st.get("cash_pct"),
        "neg_years": st.get("neg_years"),
        "sharpe_2026": st2026.get("sharpe"),
        "return_2026": st2026.get("return"),
        "avg_turnover": round(turn_sum / turn_n, 3) if turn_n else None,
        "trade_day_pct": round(trade_days / len(dates), 3) if dates else None,
        "days": len(live.dropna()),
    }


def main() -> None:
    print("panel...", flush=True)
    panel = build_scheme_panel(pd.Timestamp("2010-01-01"))
    px = load_px()
    panel = panel.merge(px, on=["date", "symbol", "code"], how="left")
    hist = panel.dropna(subset=["pos_64", "rsi", "fwd_ret"]).copy()

    recs: list[dict] = []
    for dt, day in hist.groupby("date"):
        if dt < BT_START:
            continue
        day = day.drop_duplicates("symbol").reset_index(drop=True)
        recs.append({"date": dt, "day": day})

    configs = [
        FilterCfg("baseline"),
        FilterCfg("dual_confirm", dual_confirm=True),
        FilterCfg("min_hold3", min_hold=3),
        FilterCfg("min_hold5", min_hold=5),
        FilterCfg("min_hold10", min_hold=10),
        FilterCfg("rebal5", rebal_k=5),
        FilterCfg("rebal10", rebal_k=10),
        FilterCfg("cs_gap0.30", cs_gap=0.30),
        FilterCfg("cs_gap0.40", cs_gap=0.40),
        FilterCfg("cs_gap0.50", cs_gap=0.50),
        FilterCfg("flip_cd3", flip_cooldown=3),
        FilterCfg("flip_cd5", flip_cooldown=5),
        FilterCfg("rsi_ext30", rsi_extreme=30.0),
        FilterCfg("rsi_ext25", rsi_extreme=25.0),
        FilterCfg("dual+hold5", dual_confirm=True, min_hold=5),
        FilterCfg("dual+gap0.40", dual_confirm=True, cs_gap=0.40),
        FilterCfg("dual+rebal5", dual_confirm=True, rebal_k=5),
        FilterCfg("gap0.40+hold5", cs_gap=0.40, min_hold=5),
        FilterCfg("dual+gap0.40+hold5", dual_confirm=True, cs_gap=0.40, min_hold=5),
    ]

    rows = []
    for cfg in configs:
        print(f"  {cfg.name}...", flush=True)
        rows.append(simulate(cfg, recs))

    df = pd.DataFrame(rows).sort_values(["sharpe", "sharpe_2026"], ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"configs": rows, "ranked": df.to_dict(orient="records")}, indent=2, ensure_ascii=False))

    print("\n=== Signal filter search (pos64+RSI, BOOK≥1.18, ≤5万) ===")
    cols = ["name", "sharpe", "return", "max_dd", "cash_pct", "neg_years", "sharpe_2026", "return_2026", "avg_turnover", "trade_day_pct"]
    print(df[cols].to_string(index=False))
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
