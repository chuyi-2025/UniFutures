#!/usr/bin/env python3
"""News capacity daily blotter — columns aligned with final_scheme_2026_daily.csv.

Variants:
  V2 (推荐): util=0.08, max_lots=1, strength_cap=1.2 — 抑制单日大亏
  V1: util=0.20, max_lots=5, strength_cap=1.5 — 20 万档
  u10: util=0.10, max_lots=5, strength_cap=1.5 — 旧 10 万档（多手肥尾）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

from common import DATA_DIR  # noqa: E402
from family_rotate import fmt, year_pack  # noqa: E402
from final_scheme_2026_blotter import load_px  # noqa: E402
from futures_lot_specs import lot_value, lot_margin  # noqa: E402
from linear_ridge_walkforward import build_panel  # noqa: E402
from pos64_4name_2026_blotter import cn  # noqa: E402
from pos64_margin_integer_search import CAPITAL, to_log  # noqa: E402
from pos64_rsi_overlay_2026_blotter import fmt_px, join_names, yuan  # noqa: E402
from rule_factor_mine import batch_aggregate, filter_docs, keep_single_symbol_docs  # noqa: E402
from rule_news_capacity_search import (  # noqa: E402
    BT_START,
    OUT,
    allocate_lots,
    build_day_cache,
    day_spread,
    dollar_multi,
    fees_multi,
)
from six_name_select import pick_ls_n  # noqa: E402

# tag, util, max_lots, strength_cap
VARIANTS = (
    ("v2", 0.08, 1, 1.2),   # 推荐：单品种≤1手，压单日尾部
    ("v1", 0.20, 5, 1.5),
    ("u10", 0.10, 5, 1.5),
)
BASE_CFG = {
    "mode": "hysteresis",
    "n_each": 2,
    "enter": 0.30,
    "exit_": 0.15,
    "min_hold": 5,
    "spread_ref": 0.30,
}
COL, LB = "rule_edge", 30
YEAR = 2026
COLS_OUT = [
    "日期", "是否空仓", "BOOK", "手数", "截面品种数", "账户日盈亏", "手续费", "账户日收益",
    "做多", "做空", "平仓", "持仓", "操作说明", "信号做多", "信号做空",
    "合计合约价值", "合计保证金", "组合杠杆", "spread", "strength",
]


def name_px(sym: str, lots: int, meta: dict, *, fill: bool, side: int | None = None) -> str:
    info = meta.get(sym, {})
    px = info.get("next_open") if fill else info.get("close")
    pending = False
    if fill and (px is None or not np.isfinite(px)):
        px = info.get("close")
        pending = True
    base = f"{cn(sym)} {int(lots)}手"
    if side is not None:
        base += f"({'多' if side > 0 else '空'})"
    pxs = fmt_px(px)
    if not pxs:
        return base
    if pending:
        return f"{base}@{pxs}(收盘)"
    return f"{base}@{pxs}"


def action_lists_lots(prev: dict[str, int], today: dict[str, int]):
    opens_long, opens_short, holds, closes = [], [], [], []
    for s, lots in today.items():
        side = 1 if lots > 0 else -1
        prev_n = prev.get(s)
        if prev_n is not None and (prev_n > 0) == (lots > 0):
            holds.append((s, side))
        else:
            if prev_n is not None:
                closes.append((s, 1 if prev_n > 0 else -1))
            (opens_long if lots > 0 else opens_short).append(s)
    for s, prev_n in prev.items():
        if s not in today:
            closes.append((s, 1 if prev_n > 0 else -1))
    return opens_long, opens_short, holds, closes


def instruction(
    inn: bool,
    *,
    spread: float,
    strength: float,
    opens_long: list[str],
    opens_short: list[str],
    holds: list[tuple[str, int]],
    closes: list[tuple[str, int]],
    meta: dict,
    prev_lots: dict[str, int],
    cur_lots: dict[str, int],
) -> str:
    parts: list[str] = []
    tag = f"spread={spread:.3f} strength={strength:.2f}"
    if not inn:
        why = f"空仓（{tag}）"
        if closes:
            parts.append(why)
            parts.append(
                "平仓"
                + "、".join(name_px(s, abs(prev_lots.get(s, 1)), meta, fill=True) for s, _ in closes)
            )
        else:
            parts.append(why)
        return " ".join(parts)
    if closes:
        parts.append(
            "平仓"
            + "、".join(name_px(s, abs(prev_lots.get(s, 1)), meta, fill=True) for s, _ in closes)
        )
    for s in opens_long:
        parts.append("做多" + name_px(s, abs(cur_lots[s]), meta, fill=True))
    for s in opens_short:
        parts.append("做空" + name_px(s, abs(cur_lots[s]), meta, fill=True))
    for s, side in holds:
        parts.append("持仓" + name_px(s, abs(cur_lots[s]), meta, fill=False, side=side))
    return " ".join(parts) if parts else f"空仓（{tag}）"


def money_multi(sym: str, lots: int, price: float | None) -> tuple[int, int]:
    val = lot_value(sym, price)
    mar, _ = lot_margin(sym, price, broker=True)
    n = abs(int(lots))
    return yuan(val * n) or 0, yuan((mar or 0) * n) or 0


def run_blotter(cache: dict, cfg: dict) -> pd.DataFrame:
    mode = cfg["mode"]
    n_each = cfg["n_each"]
    enter, exit_, min_hold = cfg["enter"], cfg["exit_"], cfg["min_hold"]
    util, spread_ref, max_lots = cfg["util"], cfg["spread_ref"], cfg["max_lots"]
    strength_cap = float(cfg.get("strength_cap", 1.5))

    state_names: list[tuple[str, float]] | None = None
    days_in = 0
    prev_pos: dict[str, int] = {}
    rows = []

    for dt in cache["dates"]:
        info = cache["days"][dt]
        d = info["frame"]
        meta = info["meta"]
        n_cs = int(len(d))
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

        strength = float(np.clip(spread / spread_ref, 0.0, strength_cap)) if spread_ref > 0 else 0.0
        budget_eff = util * CAPITAL * strength
        positions = (
            allocate_lots(names, info["m1"], budget_eff=budget_eff, max_lots=max_lots)
            if names and budget_eff >= 1000
            else []
        )
        cur_pos = {s: n for s, n in positions}
        day_fee = fees_multi(meta, prev_pos, cur_pos)

        if positions:
            gross, margin, nlots = dollar_multi(positions, info["close"], info["ret"])
            pnl_cny = gross - day_fee
        else:
            margin, nlots, pnl_cny = 0.0, 0, 0.0 - day_fee if day_fee else 0.0
            # closing fees already netted; empty day with closes still pays fee
            if day_fee:
                pnl_cny = -day_fee

        pnl_log = float(to_log(pd.Series([pnl_cny / CAPITAL])).iloc[0]) if np.isfinite(pnl_cny) else 0.0

        sig_long = [cn(s) for s, w in (cand or []) if w > 0]
        sig_short = [cn(s) for s, w in (cand or []) if w < 0]
        # held-name signal if in market via hysteresis but cand differs
        if names:
            held_long = [cn(s) for s, w in names if w > 0]
            held_short = [cn(s) for s, w in names if w < 0]
        else:
            held_long, held_short = [], []

        inn = bool(positions)
        opens_long, opens_short, holds, closes = action_lists_lots(prev_pos, cur_pos)
        note = instruction(
            inn,
            spread=spread,
            strength=strength,
            opens_long=opens_long,
            opens_short=opens_short,
            holds=holds,
            closes=closes,
            meta=meta,
            prev_lots=prev_pos,
            cur_lots=cur_pos,
        )

        tot_val = tot_mar = 0
        if inn:
            for s, n in cur_pos.items():
                px = meta.get(s, {}).get("close")
                v, m = money_multi(s, n, px)
                tot_val += v
                tot_mar += m

        rows.append({
            "日期": str(pd.Timestamp(dt).date()),
            "是否空仓": (not inn),
            "BOOK": "",  # V1 不用 BOOK
            "手数": int(nlots) if inn else 0,
            "截面品种数": n_cs,
            "账户日盈亏": yuan(pnl_cny) if pnl_cny is not None else 0,
            "手续费": yuan(day_fee) if day_fee else 0,
            "账户日收益": fmt(pnl_log, 5),
            "做多": join_names([name_px(s, abs(cur_pos[s]), meta, fill=True) for s in opens_long]) if inn else "",
            "做空": join_names([name_px(s, abs(cur_pos[s]), meta, fill=True) for s in opens_short]) if inn else "",
            "平仓": join_names([name_px(s, abs(prev_pos.get(s, 1)), meta, fill=True) for s, _ in closes]),
            "持仓": join_names(
                [name_px(s, abs(cur_pos[s]), meta, fill=False, side=side) for s, side in holds]
            ) if inn else "",
            "操作说明": note,
            "信号做多": join_names(held_long if held_long else sig_long),
            "信号做空": join_names(held_short if held_short else sig_short),
            "合计合约价值": tot_val if inn else 0,
            "合计保证金": tot_mar if inn else 0,
            "组合杠杆": fmt(tot_val / tot_mar, 2) if inn and tot_mar else None,
            "spread": fmt(spread, 4),
            "strength": fmt(strength, 3),
        })
        prev_pos = cur_pos

    return pd.DataFrame(rows)


def summarize(ddf: pd.DataFrame) -> dict:
    s = pd.to_numeric(ddf["账户日盈亏"], errors="coerce").fillna(0.0)
    live = to_log(s / CAPITAL)
    live.index = pd.to_datetime(ddf["日期"])
    bt = live[np.isfinite(pd.to_numeric(live, errors="coerce"))]
    st = year_pack(bt) if len(bt) >= 30 else {}
    on = ddf[~ddf["是否空仓"].astype(bool)]
    y = s.copy()
    y.index = live.index
    y26 = y[(y.index >= "2026-01-01") & (y.index <= "2026-12-31")]
    return {
        "sharpe": st.get("sharpe"),
        "return": st.get("return"),
        "max_dd": st.get("max_dd"),
        "pnl": round(float(s.sum()), 1),
        "avg_margin_on": round(float(pd.to_numeric(on["合计保证金"], errors="coerce").mean()), 1) if len(on) else 0,
        "avg_lots_on": round(float(pd.to_numeric(on["手数"], errors="coerce").mean()), 2) if len(on) else 0,
        "y2026_pnl": round(float(y26.sum()), 1),
        "y2026_worst": round(float(y26.min()), 1) if len(y26) else None,
        "y2026_n_le5k": int((y26 <= -5000).sum()),
        "y2026_n_le8k": int((y26 <= -8000).sum()),
    }


def main() -> None:
    print("docs...", flush=True)
    docs = pd.read_parquet(DATA_DIR / "rule_docs_rich.parquet")
    docs["report_date"] = pd.to_datetime(docs["report_date"])
    docs = keep_single_symbol_docs(filter_docs(docs, "no_dianping"))
    print(f"docs={len(docs)}", flush=True)

    print("panel...", flush=True)
    panel = build_panel(BT_START).dropna(subset=["fwd_ret"])
    panel = panel.merge(load_px(), on=["date", "symbol", "code"], how="left")
    panel = panel[panel["date"] >= BT_START].copy()
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))

    print(f"agg {COL} L{LB}...", flush=True)
    fac = batch_aggregate(docs, COL, LB, dates, "uniform")
    cache = build_day_cache(panel, fac)

    OUT.mkdir(parents=True, exist_ok=True)
    linear_dir = Path("/home/workspace/lab/UniFutures/data/infer/results/daily/linear")
    linear_dir.mkdir(parents=True, exist_ok=True)
    y0, y1 = f"{YEAR}-01-01", f"{YEAR}-12-31"

    for tag, util, max_lots, scap in VARIANTS:
        cfg = {**BASE_CFG, "util": util, "max_lots": max_lots, "strength_cap": scap}
        print(f"blotter {tag} util={util} max_lots={max_lots} scap={scap} ...", flush=True)
        ddf = run_blotter(cache, cfg)
        ddf["日期_dt"] = pd.to_datetime(ddf["日期"])
        ydf = ddf[(ddf["日期_dt"] >= y0) & (ddf["日期_dt"] <= y1)].drop(columns=["日期_dt"])
        full_out = ddf.drop(columns=["日期_dt"])

        year_path = OUT / f"news_capacity_{tag}_{YEAR}_daily.csv"
        full_path = OUT / f"news_capacity_{tag}_daily.csv"
        mirror = linear_dir / f"news_capacity_{tag}_{YEAR}_daily.csv"
        ydf[COLS_OUT].to_csv(year_path, index=False)
        full_out[COLS_OUT].to_csv(full_path, index=False)
        ydf[COLS_OUT].to_csv(mirror, index=False)

        st = summarize(full_out)
        on = ydf[~ydf["是否空仓"].astype(bool)]
        print(
            f"  [{tag}] sh={st['sharpe']} pnl={st['pnl']} dd={st['max_dd']} "
            f"avg_mar={st['avg_margin_on']} avg_lots={st['avg_lots_on']} "
            f"2026_worst={st['y2026_worst']} n<=-5k={st['y2026_n_le5k']}",
            flush=True,
        )
        print(
            f"  2026 active={len(on)} pnl={ydf['账户日盈亏'].sum()} "
            f"avg_lots={on['手数'].mean():.1f} avg_margin={on['合计保证金'].mean():.0f}",
            flush=True,
        )
        print("  saved", mirror, flush=True)


if __name__ == "__main__":
    main()
