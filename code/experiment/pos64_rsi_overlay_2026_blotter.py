#!/usr/bin/env python3
"""2026 blotter: pos64+RSI 各 1 多 1 空 overlay, 一手保证金≤5万, 1手, BOOK≥1.18 全平.

Integer 1 lot. Signal after close; fill next open. Last day has no next_open.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from blend_book import book_ratio
from family_rotate import fmt, year_pack
from final_scheme_2026_blotter import load_px, pick_ls_close
from futures_lot_specs import lot_fee, lot_margin, lot_value
from linear_ridge_walkforward import (
    DEAD,
    OUT_DIR,
    TREE,
    load_fwd_ret,
    load_hist_features,
    pick_main_fast,
)

DAILY_LINEAR = OUT_DIR.parent / "daily" / "linear"
from pos64_4name_2026_blotter import (
    CN,  # noqa: F401
    action_lists,
    cn,
    meta_of,
    picks_to_w,
    split_ls,
)
from pos64_margin_cap_search import attach_margin
from pos64_margin_integer_search import CAPITAL, dollar_1lot, net_picks, to_log
from shared import REMOVED_SYMBOLS, WINDOW
from six_name_select import pick_ls_n
from two_model_rotate import BT_START

MARGIN_CAP = 50_000.0
BOOK_THR = 1.18
N_EACH = 1


def yuan(x: float | None) -> int | None:
    if x is None or not np.isfinite(x):
        return None
    return int(round(float(x)))


def fmt_px(px: float | None) -> str | None:
    if px is None or not np.isfinite(px):
        return None
    return f"{float(px):.2f}".rstrip("0").rstrip(".")


def contract_ym(code: str | None) -> str:
    """Extract YYMM from main-contract code like RU2603 / BR2606."""
    if not code or not isinstance(code, str):
        return ""
    digits = "".join(ch for ch in code if ch.isdigit())
    return digits[-4:] if len(digits) >= 4 else digits


def cn_contract(sym: str, meta: dict | None = None) -> str:
    """Display name with delivery month, e.g. 橡胶2603."""
    name = cn(sym)
    code = None
    if meta is not None:
        code = meta.get(sym, {}).get("code")
    ym = contract_ym(code)
    return f"{name}{ym}" if ym else name


def name_px(sym: str, lots: int, meta: dict, *, fill: bool, side: int | None = None) -> str:
    info = meta.get(sym, {})
    px = info.get("next_open") if fill else info.get("close")
    pending = False
    if fill and (px is None or not np.isfinite(px)):
        px = info.get("close")
        pending = True
    base = f"{cn_contract(sym, meta)} {int(lots)}手"
    if side is not None:
        base += f"({'多' if side > 0 else '空'})"
    pxs = fmt_px(px)
    if not pxs:
        return base
    if pending:
        return f"{base}@{pxs}(收盘)"
    return f"{base}@{pxs}"


def money_of(sym: str, price: float | None, inn: bool) -> dict:
    val = lot_value(sym, price)
    mar, rate = lot_margin(sym, price, broker=True)
    if not inn:
        return {"合约价值": yuan(val), "保证金": 0, "保证金率": fmt(rate, 3) if rate else None}
    return {
        "合约价值": yuan(val),
        "保证金": yuan(mar),
        "保证金率": fmt(rate, 3) if rate else None,
    }


def eligible(day: pd.DataFrame) -> pd.DataFrame:
    d = attach_margin(day)
    return d[d["lot_margin"].notna() & (d["lot_margin"] <= MARGIN_CAP)].reset_index(drop=True)


def pick_overlay(day: pd.DataFrame, *, use_close: bool) -> list[tuple[str, float]]:
    d = eligible(day)
    if len(d) < 2:
        return []
    picker = pick_ls_close if use_close else pick_ls_n
    a = picker(d, d["pos_64"].to_numpy(), N_EACH)
    r = picker(d, d["rsi"].to_numpy(), N_EACH)
    return net_picks(a, r)


def instruction(
    inn: bool,
    opens_long: list[str],
    opens_short: list[str],
    holds: list[tuple[str, int]],
    closes: list[tuple[str, int]],
    meta: dict,
    prev_lots: dict[str, int],
    book: float | None,
) -> str:
    parts: list[str] = []
    book_s = f"BOOK={book:.3f}" if book is not None else "BOOK=NA"
    if not inn:
        why = f"空仓（{book_s}≥{BOOK_THR:.2f}全平）" if book is not None and book >= BOOK_THR else f"空仓（{book_s}）"
        if closes:
            parts.append(why)
            parts.append("平仓" + "、".join(name_px(s, prev_lots.get(s, 1), meta, fill=True) for s, _ in closes))
        else:
            parts.append(why)
        return " ".join(parts)
    if closes:
        parts.append("平仓" + "、".join(name_px(s, prev_lots.get(s, 1), meta, fill=True) for s, _ in closes))
    for s in opens_long:
        parts.append("做多" + name_px(s, 1, meta, fill=True))
    for s in opens_short:
        parts.append("做空" + name_px(s, 1, meta, fill=True))
    for s, side in holds:
        parts.append("持仓" + name_px(s, 1, meta, fill=False, side=side))
    return " ".join(parts) if parts else "空仓"


def join_names(items: list[str]) -> str:
    return "、".join(items)


def trade_fees(
    meta: dict,
    lots_n: int,
    closes: list[tuple[str, int]],
    opens_long: list[str],
    opens_short: list[str],
    prev_lots: dict[str, int],
) -> float:
    fee = 0.0
    for s, _ in closes:
        info = meta.get(s, {})
        px = info.get("next_open")
        if px is None or not np.isfinite(px):
            px = info.get("close")
        fee += lot_fee(s, px, "close_prev", prev_lots.get(s, 1))
    for s in opens_long + opens_short:
        info = meta.get(s, {})
        px = info.get("next_open")
        if px is None or not np.isfinite(px):
            px = info.get("close")
        fee += lot_fee(s, px, "open", lots_n)
    return fee


def _read_px_csv(csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv, encoding="utf-8")
    raw.columns = [str(c).replace("\ufeff", "") for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["code"] = raw["code"].astype(str)
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    return raw.dropna(subset=["date", "close"])


def pos64_rsi(close: pd.Series) -> pd.DataFrame:
    c = close.astype(float)
    wmax, wmin = c.rolling(WINDOW).max(), c.rolling(WINDOW).min()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0).rolling(14).mean()) + 1e-8
    rsi = 100 - 100 / (1 + gain / loss)
    return pd.DataFrame({
        "pos_64": (c - wmin) / (wmax - wmin + 1e-5),
        "rsi": rsi,
    }, index=close.index)


def build_pos64_rsi_tail(min_date: pd.Timestamp) -> pd.DataFrame:
    """pos64/RSI from Tree-Stock contracts so the live book is not stuck at EVAL_END."""
    chunks: list[pd.DataFrame] = []
    n_ok, n_fail = 0, 0
    for sym_dir in sorted(p for p in TREE.iterdir() if p.is_dir()):
        sym = sym_dir.name.upper()
        if sym in REMOVED_SYMBOLS or sym in DEAD:
            continue
        for csv in sorted(sym_dir.glob("*.csv")):
            try:
                raw = _read_px_csv(csv)
                if raw.empty or raw["date"].max() < min_date:
                    continue
                raw = raw.sort_values(["code", "date"])
                for _, seg in raw.groupby("code", sort=False):
                    if len(seg) < WINDOW:
                        continue
                    feat = pos64_rsi(seg["close"])
                    part = seg[["date", "code"]].copy()
                    part["pos_64"] = feat["pos_64"].to_numpy()
                    part["rsi"] = feat["rsi"].to_numpy()
                    part = part[(part["date"] >= min_date) & part["pos_64"].notna() & part["rsi"].notna()]
                    if part.empty:
                        continue
                    part["symbol"] = sym
                    chunks.append(part)
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                print(f"  tail skip {sym}/{csv.name}: {exc}", flush=True)
                continue
    print(f"  tail files ok={n_ok} fail={n_fail} chunks={len(chunks)}", flush=True)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def build_scheme_panel(hist_start: pd.Timestamp) -> pd.DataFrame:
    print("loading historical features...", flush=True)
    hist = load_hist_features()
    print(f"  hist rows={len(hist)} last={hist['date'].max().date()}", flush=True)
    keep_from = pd.Timestamp(hist["date"].max()) + pd.Timedelta(days=1)
    print(f"live pos64/rsi from Tree-Stock since {keep_from.date()}...", flush=True)
    tail = build_pos64_rsi_tail(keep_from)
    feat = pd.concat([hist, tail], ignore_index=True) if not tail.empty else hist
    feat = feat.drop_duplicates(["date", "symbol", "code"], keep="last")

    print("picking main contracts...", flush=True)
    mains = []
    for _, grp in feat.groupby("symbol"):
        p = pick_main_fast(grp)
        if not p.empty:
            mains.append(p)
    main = pd.concat(mains, ignore_index=True)

    print("loading forward returns...", flush=True)
    rets = load_fwd_ret()
    panel = main.merge(rets, on=["date", "symbol", "code"], how="left")
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    panel = panel[panel["date"] >= hist_start]
    print(
        f"panel rows={len(panel)} symbols={panel['symbol'].nunique()} "
        f"{panel['date'].min().date()} -> {panel['date'].max().date()} "
        f"fwd_ret coverage={panel['fwd_ret'].notna().mean():.2%}",
        flush=True,
    )
    return panel


def main() -> None:
    print("panel...", flush=True)
    panel = build_scheme_panel(pd.Timestamp("2010-01-01"))
    print("prices...", flush=True)
    px = load_px()
    panel = panel.merge(px, on=["date", "symbol", "code"], how="left")
    full = panel.dropna(subset=["pos_64", "rsi"]).copy()
    hist = full.dropna(subset=["fwd_ret"])

    recs: list[dict] = []
    print("daily pos64+RSI overlay cap 5万...", flush=True)
    for dt, day in hist.groupby("date"):
        if dt < BT_START:
            continue
        day = day.drop_duplicates("symbol").reset_index(drop=True)
        picks = pick_overlay(day, use_close=False)
        recs.append({
            "date": dt,
            "w": picks_to_w(picks),
            "picks": picks,
            "dollar": dollar_1lot(day, picks),
            "n_cs": int(len(eligible(day))),
            "meta": meta_of(day),
        })

    last_full = pd.Timestamp(full["date"].max())
    last_hist = pd.Timestamp(recs[-1]["date"]) if recs else None
    if last_hist is None or last_full > last_hist:
        day = full[full["date"] == last_full].drop_duplicates("symbol").reset_index(drop=True)
        picks = pick_overlay(day, use_close=True)
        recs.append({
            "date": last_full,
            "w": picks_to_w(picks),
            "picks": picks,
            "dollar": (np.nan, np.nan, np.nan, len(picks)),
            "n_cs": int(len(eligible(day))),
            "meta": meta_of(day),
        })

    by_dt = {r["date"]: r for r in recs}
    dates_pre = sorted(by_dt.keys())
    lots_pre: dict = {}
    cash_pre: dict = {}
    fee_pre: dict = {}
    prev_held_pre: dict[str, int] = {}
    prev_lots_pre: dict[str, int] = {}
    raw_pnl = pd.Series({r["date"]: r["dollar"][0] for r in recs}).sort_index()
    raw_log_pre = to_log(raw_pnl / CAPITAL)
    ratio_pre = book_ratio(raw_log_pre.dropna()).reindex(raw_log_pre.index)
    if ratio_pre.notna().any():
        ratio_pre = ratio_pre.ffill(limit=1)
    empty_pre = pd.Series({r["date"]: len(r["picks"]) == 0 for r in recs}).reindex(raw_log_pre.index).fillna(False)
    for dt in dates_pre:
        r = by_dt[dt]
        b = float(ratio_pre.loc[dt]) if dt in ratio_pre.index and np.isfinite(ratio_pre.loc[dt]) else None
        inn0 = not (empty_pre.loc[dt] if dt in empty_pre.index else False)
        inn0 = inn0 and not (b is not None and b >= BOOK_THR)
        w = r["w"]
        meta = r["meta"]
        longs, shorts = split_ls(w)
        today: dict[str, int] = {}
        if inn0:
            for s in longs:
                today[s] = 1
            for s in shorts:
                today[s] = -1
        opens_long, opens_short, _, closes = action_lists(prev_held_pre, today)
        lots_n0 = 1 if inn0 else 0
        fee_pre[dt] = trade_fees(meta, lots_n0, closes, opens_long, opens_short, prev_lots_pre)
        prev_held_pre = today if inn0 else {}
        prev_lots_pre = {s: lots_n0 for s in today} if inn0 else {}
        lots_pre[dt] = lots_n0
        cash_pre[dt] = not inn0

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
    lots = (~cash).astype(int)
    lots = lots.where(raw_log.notna() | (raw_log.index == last_full), 0)
    lots = lots.fillna(0).astype(int)
    lots[cash] = 0

    full_bt = live[np.isfinite(pd.to_numeric(live, errors="coerce"))]
    year = int(pd.Timestamp(last_full).year)
    y0, y1 = f"{year}-01-01", f"{year}-12-31"
    y_cur = live[(live.index >= y0) & (live.index <= y1)]
    y_cur_bt = y_cur[np.isfinite(pd.to_numeric(y_cur, errors="coerce"))]
    st_all = year_pack(full_bt)
    st_cur = year_pack(y_cur_bt)
    cash_all = float(cash.reindex(full_bt.index).fillna(False).mean())
    cash_cur = float(cash.reindex(y_cur.index).fillna(False).mean())
    print(
        f"all ret={st_all['return']:+.1%} sh={st_all['sharpe']:.2f} dd={st_all['max_dd']:.1%} "
        f"cash={cash_all:.0%} neg={st_all['n_neg_years']} | "
        f"{year} ret={st_cur['return']:+.1%} sh={st_cur['sharpe']:.2f} dd={st_cur['max_dd']:.1%} cash={cash_cur:.0%}",
        flush=True,
    )

    dates = list(raw_log.index)
    prev_held: dict[str, int] = {}
    prev_lots: dict[str, int] = {}
    daily_rows = []
    ledger_rows = []

    for dt in dates:
        r = by_dt[dt]
        lots_n = int(lots.loc[dt]) if dt in lots.index else 0
        inn = lots_n > 0
        b = float(ratio.loc[dt]) if dt in ratio.index and np.isfinite(ratio.loc[dt]) else None
        pnl_log = float(live.loc[dt]) if dt in live.index and np.isfinite(live.loc[dt]) else None
        pnl_cny = float(dollar_live.loc[dt]) if dt in dollar_live.index and np.isfinite(dollar_live.loc[dt]) else None
        w = r["w"]
        meta = r["meta"]
        longs, shorts = split_ls(w)
        today: dict[str, int] = {}
        if inn:
            for s in longs:
                today[s] = 1
            for s in shorts:
                today[s] = -1
        opens_long, opens_short, holds, closes = action_lists(prev_held, today)
        day_fee = trade_fees(meta, lots_n, closes, opens_long, opens_short, prev_lots)
        note = instruction(inn, opens_long, opens_short, holds, closes, meta, prev_lots, b)

        tot_val = 0
        tot_mar = 0
        if inn:
            for s in today:
                info = meta.get(s, {})
                m = money_of(s, info.get("close"), True)
                tot_val += m["合约价值"] or 0
                tot_mar += m["保证金"] or 0

        row = {
            "日期": str(pd.Timestamp(dt).date()),
            "是否空仓": (not inn),
            "BOOK": fmt(b, 3),
            "手数": lots_n,
            "截面品种数": r["n_cs"],
            "账户日盈亏": yuan(pnl_cny) if pnl_cny is not None else None,
            "手续费": yuan(day_fee) if day_fee else 0,
            "账户日收益": fmt(pnl_log, 5),
            "做多": join_names([name_px(s, lots_n, meta, fill=True) for s in opens_long]) if inn else "",
            "做空": join_names([name_px(s, lots_n, meta, fill=True) for s in opens_short]) if inn else "",
            "平仓": join_names([name_px(s, prev_lots.get(s, 1), meta, fill=True) for s, _ in closes]),
            "持仓": join_names([name_px(s, lots_n, meta, fill=False, side=side) for s, side in holds]) if inn else "",
            "操作说明": note,
            "信号做多": join_names([cn_contract(s, meta) for s in longs]),
            "信号做空": join_names([cn_contract(s, meta) for s in shorts]),
            "合计合约价值": tot_val if inn else 0,
            "合计保证金": tot_mar if inn else 0,
            "组合杠杆": fmt(tot_val / tot_mar, 2) if inn and tot_mar else None,
        }
        daily_rows.append(row)

        def add_ledger(action: str, sym: str, side: int, n: int, leg_fee: float = 0.0):
            info = meta.get(sym, {})
            px = info.get("close")
            fill_px = info.get("next_open")
            m = money_of(sym, px, action not in ("平仓", "空仓") and inn)
            mark = fill_px if action in ("做多", "做空", "平仓") else px
            pending = action in ("做多", "做空", "平仓") and (fill_px is None or not np.isfinite(fill_px))
            ledger_rows.append({
                "日期": str(pd.Timestamp(dt).date()),
                "动作": action,
                "方向": "多" if side > 0 else "空",
                "品种代码": sym,
                "品种中文": cn(sym),
                "合约": info.get("code"),
                "手数": 0 if action in ("平仓", "空仓") else int(n),
                "点位": fmt_px(px if pending else mark),
                "点位说明": "收盘待成交" if pending else ("次日开盘" if action in ("做多", "做空", "平仓") else "收盘"),
                "收盘": fmt_px(px),
                "次日开盘": fmt_px(fill_px),
                "合约价值": m["合约价值"],
                "保证金": m["保证金"] if action not in ("平仓", "空仓") else 0,
                "保证金率": m["保证金率"],
                "BOOK": fmt(b, 3),
                "是否空仓": (not inn),
                "账户日盈亏": yuan(pnl_cny) if pnl_cny is not None else None,
                "手续费": yuan(leg_fee) if leg_fee else 0,
                "合计保证金": tot_mar if inn else 0,
                "操作说明": note,
            })

        if not inn:
            if closes:
                for s, side in closes:
                    info = meta.get(s, {})
                    fpx = info.get("next_open") or info.get("close")
                    add_ledger("平仓", s, side, prev_lots.get(s, 1), lot_fee(s, fpx, "close_prev", prev_lots.get(s, 1)))
            else:
                ledger_rows.append({
                    "日期": str(pd.Timestamp(dt).date()),
                    "动作": "空仓",
                    "方向": "",
                    "品种代码": "",
                    "品种中文": "",
                    "合约": "",
                    "手数": 0,
                    "点位": None,
                    "点位说明": "",
                    "收盘": None,
                    "次日开盘": None,
                    "合约价值": None,
                    "保证金": 0,
                    "保证金率": None,
                    "BOOK": fmt(b, 3),
                    "是否空仓": True,
                    "账户日盈亏": yuan(pnl_cny) if pnl_cny is not None else None,
                    "合计保证金": 0,
                    "操作说明": note,
                })
        else:
            for s, side in closes:
                info = meta.get(s, {})
                fpx = info.get("next_open") or info.get("close")
                add_ledger("平仓", s, side, prev_lots.get(s, 1), lot_fee(s, fpx, "close_prev", prev_lots.get(s, 1)))
            for s in opens_long:
                info = meta.get(s, {})
                fpx = info.get("next_open") or info.get("close")
                add_ledger("做多", s, 1, lots_n, lot_fee(s, fpx, "open", lots_n))
            for s in opens_short:
                info = meta.get(s, {})
                fpx = info.get("next_open") or info.get("close")
                add_ledger("做空", s, -1, lots_n, lot_fee(s, fpx, "open", lots_n))
            for s, side in holds:
                add_ledger("持仓", s, side, lots_n)

        prev_held = today
        prev_lots = {s: lots_n for s in today}

    ddf = pd.DataFrame(daily_rows)
    ldf = pd.DataFrame(ledger_rows)
    ddf["日期"] = pd.to_datetime(ddf["日期"])
    ldf["日期"] = pd.to_datetime(ldf["日期"])
    d_year = ddf[(ddf["日期"] >= y0) & (ddf["日期"] <= y1)].copy()
    l_year = ldf[(ldf["日期"] >= y0) & (ldf["日期"] <= y1)].copy()
    d_year["日期"] = d_year["日期"].dt.strftime("%Y-%m-%d")
    l_year["日期"] = l_year["日期"].dt.strftime("%Y-%m-%d")

    live_year = live[(live.index >= y0) & (live.index <= y1)]
    live_year.index = pd.to_datetime(live_year.index)
    months = []
    for per, g in live_year.groupby(live_year.index.to_period("M")):
        gg = g[np.isfinite(pd.to_numeric(g, errors="coerce"))]
        m = year_pack(gg) if len(gg) else {"return": 0.0, "sharpe": 0.0, "max_dd": 0.0}
        months.append({
            "month": str(per),
            "return": fmt(m["return"], 4),
            "sharpe": fmt(m["sharpe"], 3),
            "max_dd": fmt(m["max_dd"], 4),
            "cash": fmt(float(cash.reindex(g.index).fillna(False).mean()), 3),
            "days": int(len(g)),
        })

    last = d_year.iloc[-1].to_dict() if not d_year.empty else {}
    last_dt = last.get("日期")

    DAILY_LINEAR.mkdir(parents=True, exist_ok=True)
    d_path = DAILY_LINEAR / f"final_scheme_{year}_daily.csv"
    p_path = DAILY_LINEAR / f"final_scheme_{year}_ledger.csv"
    d_stable = DAILY_LINEAR / "final_scheme_daily.csv"
    p_stable = DAILY_LINEAR / "final_scheme_ledger.csv"
    latest_path = DAILY_LINEAR / "final_scheme_latest.csv"
    d_year.to_csv(d_path, index=False)
    l_year.to_csv(p_path, index=False)
    d_year.to_csv(d_stable, index=False)
    l_year.to_csv(p_stable, index=False)
    # full-history daily (all years) for multi-sleeve combo backtests
    d_all = ddf.copy()
    d_all["日期"] = pd.to_datetime(d_all["日期"]).dt.strftime("%Y-%m-%d")
    d_all.to_csv(DAILY_LINEAR / "final_scheme_all_daily.csv", index=False)
    if not d_year.empty:
        d_year.tail(1).to_csv(latest_path, index=False)

    snap = {
        "scheme": "pos64+RSI 各1多1空叠加；一手券商保证金>5万的品种不选；在场每品种1手；BOOK≥1.18全平",
        "capital": CAPITAL,
        "rules": {
            "universe": "当日收盘估算一手券商保证金≤5万（东方财富期货2026-09-09快照，含期货公司加收）",
            "signal": "pos64 截面1多1空 + RSI 截面1多1空，同向合并、反向抵消，每品种最多1手",
            "fill": "收盘后下单，次日开盘成交；最后一日次日开盘未知则点位用收盘并标注待成交",
            "lots": "整数1手。BOOK=策略近20日收益波动/过去252日该波动中位数（已滞后1日）；BOOK≥1.18 全平",
            "pnl": "账户100万；日收益=log1p((盈亏-手续费)/100万)；盈亏=方向×价格×交易单位×涨跌",
            "fee": "交易所标准手续费（shouxufei）；开仓=open，隔夜平仓=closePrev；平今未单独建模",
        },
        "fees": {
            "total_cny": yuan(float(fee_s.sum())),
            "mean_daily_cny": yuan(float(fee_s.mean())),
            "annual_cny": yuan(float(fee_s.sum()) / max(len(fee_s) / 252, 1)),
            "source": "data/reference/shouxufei_fees.json (github.com/qhcg66/shouxufei)",
        },
        "full_sample": {
            **{k: st_all[k] for k in ("return", "sharpe", "max_dd", "days", "n_neg_years", "neg_years", "worst_year_sharpe")},
            "cash": fmt(cash_all, 3),
        },
        f"year_{year}": {**{k: st_cur[k] for k in ("return", "sharpe", "max_dd", "days")}, "cash": fmt(cash_cur, 3)},
        "months": months,
        "last_day": {k: last.get(k) for k in (
            "日期", "是否空仓", "BOOK", "手数", "做多", "做空", "平仓", "持仓", "操作说明",
            "合计合约价值", "合计保证金", "组合杠杆",
        )},
        "files": {
            "daily": str(d_path),
            "daily_stable": str(d_stable),
            "ledger": str(p_path),
            "latest": str(latest_path),
        },
    }
    jpath = DAILY_LINEAR / f"final_scheme_{year}.json"
    j_stable = DAILY_LINEAR / "final_scheme.json"
    text = json.dumps(snap, ensure_ascii=False, indent=2, default=str)
    jpath.write_text(text)
    j_stable.write_text(text)
    print(f"last {last_dt} cash={last.get('是否空仓')} book={last.get('BOOK')} "
          f"保证金={last.get('合计保证金')} 操作={last.get('操作说明')}")
    print(f"saved {d_path} {p_path} {latest_path} {jpath}")


if __name__ == "__main__":
    main()
