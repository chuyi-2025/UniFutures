#!/usr/bin/env python3
"""2026 blotter: pos64 3L3S (6 names), vol-target 6%, cash if scale<0.7 else 1 lot.

Integer lots only. Signal after close; fill next open.
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
from futures_lot_specs import lot_margin, lot_value
from linear_ridge_walkforward import OUT_DIR, build_panel
from pos64_4name_2026_blotter import (
    CN,  # noqa: F401  — side-effect fills names
    action_lists,
    cn,
    meta_of,
    name_lots,
    picks_to_w,
    split_ls,
    vol_stats,
)
from six_name_select import pick_ls_n
from two_model_rotate import BT_START
from two_name_select import apply_hold

VOL_TGT = 0.06
SCALE_MIN = 0.70
N_EACH = 3
VOL_CAP = 2.5


def yuan(x: float | None) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return int(round(float(x)))


def instruction(
    inn: bool,
    lots: int,
    opens_long: list[str],
    opens_short: list[str],
    holds: list[tuple[str, int]],
    closes: list[tuple[str, int]],
    prev_lots: dict[str, int],
    sc: float,
) -> str:
    parts: list[str] = []
    fallback = 1
    if not inn:
        why = f"空仓（仓位倍数{sc:.2f}<0.70）"
        if closes:
            parts.append(why)
            parts.append("平仓" + "、".join(name_lots(s, prev_lots.get(s, fallback)) for s, _ in closes))
        else:
            parts.append(why)
        return " ".join(parts)
    if closes:
        parts.append("平仓" + "、".join(name_lots(s, prev_lots.get(s, fallback)) for s, _ in closes))
    for s in opens_long:
        parts.append(f"做多{name_lots(s, lots)}")
    for s in opens_short:
        parts.append(f"做空{name_lots(s, lots)}")
    for s, side in holds:
        tag = "多" if side > 0 else "空"
        parts.append(f"持仓{name_lots(s, lots)}({tag})")
    return " ".join(parts) if parts else "空仓"


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


def slot_cols(prefix: str, sym: str | None, meta: dict, lots: int, action: str, inn: bool) -> dict:
    empty = {
        f"{prefix}代码": None, f"{prefix}中文": None, f"{prefix}合约": None,
        f"{prefix}方向": None, f"{prefix}动作": None, f"{prefix}手数": None,
        f"{prefix}pos64": None, f"{prefix}收盘": None, f"{prefix}次日开盘": None,
        f"{prefix}合约价值": None, f"{prefix}保证金": None, f"{prefix}保证金率": None,
    }
    if not sym:
        return empty
    info = meta.get(sym, {})
    n = int(lots) if inn else 0
    px = info.get("close")
    m = money_of(sym, px, inn and n > 0)
    return {
        f"{prefix}代码": sym,
        f"{prefix}中文": cn(sym),
        f"{prefix}合约": info.get("code"),
        f"{prefix}方向": "多" if prefix.startswith("多") else "空",
        f"{prefix}动作": action,
        f"{prefix}手数": n,
        f"{prefix}pos64": fmt(info.get("pos64"), 4),
        f"{prefix}收盘": fmt(px, 2),
        f"{prefix}次日开盘": fmt(info.get("next_open"), 2),
        f"{prefix}合约价值": m["合约价值"],
        f"{prefix}保证金": m["保证金"],
        f"{prefix}保证金率": m["保证金率"],
    }


def main() -> None:
    print("panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01"))
    print("prices...", flush=True)
    px = load_px()
    panel = panel.merge(px, on=["date", "symbol", "code"], how="left")
    full = panel.dropna(subset=["pos_64"]).copy()
    hist = full.dropna(subset=["fwd_ret"])

    recs: list[dict] = []
    print("daily 3L3S pos64...", flush=True)
    for dt, day in hist.groupby("date"):
        if dt < BT_START:
            continue
        day = day.drop_duplicates("symbol").reset_index(drop=True)
        picks = pick_ls_n(day, day["pos_64"].to_numpy(), N_EACH)
        recs.append({
            "date": dt,
            "w": picks_to_w(picks),
            "picks": picks,
            "pnl": apply_hold(day, picks),
            "n_cs": int(day["symbol"].nunique()),
            "meta": meta_of(day),
        })

    last_full = pd.Timestamp(full["date"].max())
    last_hist = pd.Timestamp(recs[-1]["date"]) if recs else None
    if last_hist is None or last_full > last_hist:
        day = full[full["date"] == last_full].drop_duplicates("symbol").reset_index(drop=True)
        picks = pick_ls_close(day, day["pos_64"].to_numpy(), N_EACH)
        recs.append({
            "date": last_full,
            "w": picks_to_w(picks),
            "picks": picks,
            "pnl": np.nan,
            "n_cs": int(day["symbol"].nunique()),
            "meta": meta_of(day),
        })

    by_dt = {r["date"]: r for r in recs}
    raw = pd.Series({r["date"]: r["pnl"] for r in recs}).sort_index()
    raw_bt = raw.dropna()
    _vs, sc, vol20 = vol_stats(raw_bt, target=VOL_TGT, cap=VOL_CAP)
    sc = sc.reindex(raw.index)
    vol20 = vol20.reindex(raw.index)
    if sc.notna().any():
        sc = sc.ffill(limit=1).fillna(1.0)
        vol20 = vol20.ffill(limit=1)
    lots = sc.map(lambda x: 0 if float(x) < SCALE_MIN else 1).astype(int)
    lots = lots.reindex(raw.index).fillna(0).astype(int)
    live = (raw * lots).where(raw.notna())
    cash = lots <= 0
    ratio = book_ratio(raw_bt).reindex(raw.index)
    if ratio.notna().any():
        ratio = ratio.ffill(limit=1)

    full_bt = live[np.isfinite(pd.to_numeric(live, errors="coerce"))]
    y2026 = live[(live.index >= "2026-01-01") & (live.index <= "2026-12-31")]
    y2026_bt = y2026[np.isfinite(pd.to_numeric(y2026, errors="coerce"))]
    st_all = year_pack(full_bt)
    st_26 = year_pack(y2026_bt)
    cash_all = float(cash.reindex(full_bt.index).fillna(False).mean())
    cash_26 = float(cash.reindex(y2026.index).fillna(False).mean())
    print(
        f"all ret={st_all['return']:+.1%} sh={st_all['sharpe']:.2f} dd={st_all['max_dd']:.1%} "
        f"cash={cash_all:.0%} neg={st_all['n_neg_years']} | "
        f"2026 ret={st_26['return']:+.1%} sh={st_26['sharpe']:.2f} dd={st_26['max_dd']:.1%} cash={cash_26:.0%}",
        flush=True,
    )

    dates = list(raw.index)
    prev_held: dict[str, int] = {}
    prev_lots: dict[str, int] = {}
    daily_rows = []
    ledger_rows = []

    for dt in dates:
        r = by_dt[dt]
        sc_t = float(sc.loc[dt]) if dt in sc.index and np.isfinite(sc.loc[dt]) else 1.0
        lots_n = int(lots.loc[dt]) if dt in lots.index else 0
        inn = lots_n > 0
        b = float(ratio.loc[dt]) if dt in ratio.index and np.isfinite(ratio.loc[dt]) else None
        vol_t = float(vol20.loc[dt]) if dt in vol20.index and np.isfinite(vol20.loc[dt]) else None
        pnl_t = float(live.loc[dt]) if dt in live.index and np.isfinite(live.loc[dt]) else None
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
        hold_map = {s: side for s, side in holds}

        def act_of(sym: str, side: int) -> str:
            if not inn:
                return "空仓"
            if sym in opens_long or sym in opens_short:
                return "做多" if side > 0 else "做空"
            if sym in hold_map:
                return "持仓"
            return "做多" if side > 0 else "做空"

        note = instruction(inn, lots_n, opens_long, opens_short, holds, closes, prev_lots, sc_t)
        fill_note = "收盘后下单，次日开盘成交"
        if r["meta"] and all(not v.get("next_open") for v in meta.values()):
            fill_note = "收盘后下单；次日开盘尚未发生，成交价待定"

        legs = [("多1", longs[0] if len(longs) > 0 else None),
                ("多2", longs[1] if len(longs) > 1 else None),
                ("多3", longs[2] if len(longs) > 2 else None),
                ("空1", shorts[0] if len(shorts) > 0 else None),
                ("空2", shorts[1] if len(shorts) > 1 else None),
                ("空3", shorts[2] if len(shorts) > 2 else None)]

        tot_val = 0
        tot_mar = 0
        row = {
            "日期": str(pd.Timestamp(dt).date()),
            "是否空仓": (not inn),
            "BOOK": fmt(b, 3),
            "波动20日年化": fmt(vol_t, 4),
            "波动20日年化pct": fmt(None if vol_t is None else vol_t * 100, 2),
            "仓位倍数": fmt(sc_t, 4),
            "手数": lots_n,
            "截面品种数": r["n_cs"],
            "账户日收益": fmt(pnl_t, 5),
            "做多": "、".join(name_lots(s, lots_n) for s in opens_long) if inn else "",
            "做空": "、".join(name_lots(s, lots_n) for s in opens_short) if inn else "",
            "平仓": "、".join(name_lots(s, prev_lots.get(s, 1)) for s, _ in closes),
            "持仓": "、".join(f"{name_lots(s, lots_n)}({'多' if side > 0 else '空'})" for s, side in holds) if inn else "",
            "操作说明": note,
            "信号做多": "、".join(cn(s) for s in longs),
            "信号做空": "、".join(cn(s) for s in shorts),
            "成交说明": fill_note,
        }
        for prefix, sym in legs:
            side = 1 if prefix.startswith("多") else -1
            action = act_of(sym, side) if sym else ""
            cols = slot_cols(prefix, sym, meta, lots_n, action, inn)
            row.update(cols)
            if inn and cols.get(f"{prefix}合约价值"):
                tot_val += cols[f"{prefix}合约价值"] or 0
            if inn and cols.get(f"{prefix}保证金"):
                tot_mar += cols[f"{prefix}保证金"] or 0
        row["合计合约价值"] = tot_val if inn else 0
        row["合计保证金"] = tot_mar if inn else 0
        row["组合杠杆"] = fmt(tot_val / tot_mar, 2) if inn and tot_mar else None
        daily_rows.append(row)

        def add_ledger(action: str, sym: str, side: int, n: int):
            info = meta.get(sym, {})
            px = info.get("close")
            m = money_of(sym, px, action not in ("平仓", "空仓") and inn)
            ledger_rows.append({
                "日期": str(pd.Timestamp(dt).date()),
                "动作": action,
                "方向": "多" if side > 0 else "空",
                "品种代码": sym,
                "品种中文": cn(sym),
                "合约": info.get("code"),
                "手数": 0 if action in ("平仓", "空仓") else int(n),
                "pos64": fmt(info.get("pos64"), 4),
                "收盘": fmt(px, 2),
                "次日开盘": fmt(info.get("next_open"), 2),
                "次日日期": info.get("next_date"),
                "合约价值": m["合约价值"],
                "保证金": m["保证金"] if action not in ("平仓", "空仓") else 0,
                "保证金率": m["保证金率"],
                "BOOK": fmt(b, 3),
                "波动20日年化pct": fmt(None if vol_t is None else vol_t * 100, 2),
                "仓位倍数": fmt(sc_t, 4),
                "是否空仓": (not inn),
                "账户日收益": fmt(pnl_t, 5),
                "合计保证金": tot_mar if inn else 0,
                "操作说明": note,
                "成交说明": fill_note,
            })

        if not inn:
            if closes:
                for s, side in closes:
                    add_ledger("平仓", s, side, prev_lots.get(s, 1))
            else:
                ledger_rows.append({
                    "日期": str(pd.Timestamp(dt).date()),
                    "动作": "空仓",
                    "方向": "",
                    "品种代码": "",
                    "品种中文": "",
                    "合约": "",
                    "手数": 0,
                    "pos64": None,
                    "收盘": None,
                    "次日开盘": None,
                    "次日日期": None,
                    "合约价值": None,
                    "保证金": 0,
                    "保证金率": None,
                    "BOOK": fmt(b, 3),
                    "波动20日年化pct": fmt(None if vol_t is None else vol_t * 100, 2),
                    "仓位倍数": fmt(sc_t, 4),
                    "是否空仓": True,
                    "账户日收益": fmt(pnl_t, 5),
                    "合计保证金": 0,
                    "操作说明": note,
                    "成交说明": fill_note,
                })
        else:
            for s, side in closes:
                add_ledger("平仓", s, side, prev_lots.get(s, 1))
            for s in opens_long:
                add_ledger("做多", s, 1, lots_n)
            for s in opens_short:
                add_ledger("做空", s, -1, lots_n)
            for s, side in holds:
                add_ledger("持仓", s, side, lots_n)

        prev_held = today
        prev_lots = {s: lots_n for s in today}

    ddf = pd.DataFrame(daily_rows)
    ldf = pd.DataFrame(ledger_rows)
    ddf["日期"] = pd.to_datetime(ddf["日期"])
    ldf["日期"] = pd.to_datetime(ldf["日期"])
    d2026 = ddf[(ddf["日期"] >= "2026-01-01") & (ddf["日期"] <= "2026-12-31")].copy()
    l2026 = ldf[(ldf["日期"] >= "2026-01-01") & (ldf["日期"] <= "2026-12-31")].copy()
    d2026["日期"] = d2026["日期"].dt.strftime("%Y-%m-%d")
    l2026["日期"] = l2026["日期"].dt.strftime("%Y-%m-%d")

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
            "cash": fmt(float(cash.reindex(g.index).fillna(False).mean()), 3),
            "days": int(len(g)),
        })

    last = d2026.iloc[-1].to_dict() if not d2026.empty else {}
    last_dt = last.get("日期")

    d_path = OUT_DIR / "final_scheme_2026_daily.csv"
    p_path = OUT_DIR / "final_scheme_2026_ledger.csv"
    d2026.to_csv(d_path, index=False)
    l2026.to_csv(p_path, index=False)

    snap = {
        "scheme": "pos64 3L3S（六品种，每品种1手）→ 波动目标6%；仓位倍数<0.70则空仓，否则1手，不加2手",
        "rules": {
            "signal": "收盘后按 pos64 截面做多最高3、做空最低3",
            "fill": "次日开盘成交",
            "lots": "整数手。仓位倍数=0.06/昨20日年化波动；<0.70 → 0手，否则每品种1手",
            "margin": "合约价值=收盘价×交易单位；保证金按东方财富期货2026-09-09券商比例估算（含期货公司加收）",
        },
        "full_sample": {
            **{k: st_all[k] for k in ("return", "sharpe", "max_dd", "days", "n_neg_years", "neg_years", "worst_year_sharpe")},
            "cash": fmt(cash_all, 3),
        },
        "year_2026": {**{k: st_26[k] for k in ("return", "sharpe", "max_dd", "days")}, "cash": fmt(cash_26, 3)},
        "months": months,
        "last_day": {k: last.get(k) for k in (
            "日期", "是否空仓", "BOOK", "仓位倍数", "手数", "操作说明",
            "合计合约价值", "合计保证金", "组合杠杆",
        )},
        "files": {"daily": str(d_path), "ledger": str(p_path)},
    }
    jpath = OUT_DIR / "final_scheme_2026.json"
    jpath.write_text(json.dumps(snap, ensure_ascii=False, indent=2, default=str))
    print(f"last {last_dt} cash={last.get('是否空仓')} scale={last.get('仓位倍数')} "
          f"保证金={last.get('合计保证金')} 操作={last.get('操作说明')}")
    print(f"saved {d_path} {p_path} {jpath}")


if __name__ == "__main__":
    main()
