#!/usr/bin/env python3
"""2026 blotter: pos64 2L2S (4 names), vol-target 7%, flatten if BOOK>=1.14.

Signal after close; fill next open. Last day (no fwd_ret) still emits a live order.
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
from final_scheme_2026_blotter import CN, load_px, pick_ls_close
from linear_ridge_walkforward import OUT_DIR, build_panel
from six_name_select import pick_ls_n
from two_model_rotate import BT_START
from two_name_select import apply_hold

try:
    sys.path.insert(0, str(HERE.parents[1] / "news/code/train"))
    from symbol_map import SYMBOL_ALIASES  # type: ignore
    for alias, sym in SYMBOL_ALIASES.items():
        if sym not in CN or (len(alias) < len(str(CN.get(sym, ""))) and alias.isascii() is False):
            if sym not in CN:
                CN[sym] = alias
            elif len(alias) < len(CN[sym]) and not alias.isascii():
                CN[sym] = alias
except Exception:  # noqa: BLE001
    pass

VOL_TGT = 0.07
BOOK_THR = 1.14
N_EACH = 2
VOL_WIN = 20
VOL_CAP = 2.5

# Prefer short spoken names used in trading notes.
CN.update({
    "RU": "橡胶",
    "LU": "低硫燃油",
    "FU": "燃料油",
    "NR": "20号胶",
    "CF": "棉花",
    "AU": "黄金",
    "AG": "白银",
    "SN": "锡",
    "SS": "不锈钢",
    "LH": "生猪",
    "JD": "鸡蛋",
    "IC": "中证500",
    "SC": "原油",
    "CU": "铜",
    "AL": "铝",
    "ZN": "锌",
    "NI": "镍",
    "PB": "铅",
    "RB": "螺纹钢",
    "HC": "热卷",
    "I": "铁矿石",
    "J": "焦炭",
    "JM": "焦煤",
    "TA": "PTA",
    "MA": "甲醇",
    "PP": "聚丙烯",
    "L": "塑料",
    "V": "PVC",
    "EG": "乙二醇",
    "BU": "沥青",
    "M": "豆粕",
    "Y": "豆油",
    "P": "棕榈油",
    "OI": "菜油",
    "RM": "菜粕",
    "C": "玉米",
    "CS": "淀粉",
    "SR": "白糖",
    "AP": "苹果",
    "PK": "花生",
    "SA": "纯碱",
    "SH": "烧碱",
    "UR": "尿素",
    "PG": "LPG",
    "BR": "合成橡胶",
    "SI": "工业硅",
    "AO": "氧化铝",
    "LC": "碳酸锂",
    "IF": "沪深300",
    "IH": "上证50",
    "IM": "中证1000",
    "AD": "铸造铝合金",
    "PS": "多晶硅",
    "LC": "碳酸锂",
    "TL": "三十年期国债",
    "T": "十年期国债",
    "TF": "五年期国债",
    "TS": "二年期国债",
    "CJ": "红枣",
    "EC": "集运欧线",
    "SP": "纸浆",
    "LG": "原木",
    "PX": "PX",
    "PF": "短纤",
    "PR": "瓶片",
    "BZ": "纯苯",
    "EB": "苯乙烯",
    "SF": "硅铁",
    "SM": "锰硅",
    "FG": "玻璃",
    "A": "豆一",
    "B": "豆二",
    "CY": "棉纱",
    "PL": "丙烯",
    "PT": "铂",
    "PD": "钯",
})


def vol_stats(pnl: pd.Series, target: float = VOL_TGT, win: int = VOL_WIN, cap: float = VOL_CAP):
    vol = pnl.rolling(win).std(ddof=1) * np.sqrt(252)
    vol_lag = vol.shift(1)
    sc = (target / vol_lag).replace([np.inf, -np.inf], np.nan)
    sc = sc.clip(lower=0.0, upper=cap).fillna(1.0)
    return pnl * sc, sc, vol_lag


def picks_to_w(picks: list[tuple[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for s, wt in picks:
        out[s] = out.get(s, 0.0) + wt
    return out


def to_lots(scale: float, inn: bool = True) -> int:
    """Tradeable lots: round vol-scale to nearest integer; at least 1 when in market."""
    if not inn:
        return 0
    return max(1, int(round(abs(float(scale)))))


def fmt_lots(x: float) -> str:
    return str(int(round(abs(float(x)))))


def cn(sym: str) -> str:
    return CN.get(sym, sym)


def name_lots(sym: str, lots: float) -> str:
    return f"{cn(sym)} {fmt_lots(lots)}手"


def meta_of(day: pd.DataFrame) -> dict[str, dict]:
    m = day.drop_duplicates("symbol").set_index("symbol")
    out = {}
    for s, r in m.iterrows():
        out[s] = {
            "code": str(r.get("code", "")),
            "pos64": float(r["pos_64"]) if np.isfinite(r.get("pos_64", np.nan)) else None,
            "close": float(r["close"]) if "close" in r and np.isfinite(r["close"]) else None,
            "next_open": float(r["next_open"]) if "next_open" in r and np.isfinite(r.get("next_open", np.nan)) else None,
            "next_close": float(r["next_close"]) if "next_close" in r and np.isfinite(r.get("next_close", np.nan)) else None,
            "next_date": str(pd.Timestamp(r["next_date"]).date()) if "next_date" in r and pd.notna(r.get("next_date")) else None,
            "fwd_ret": float(r["fwd_ret"]) if "fwd_ret" in r and np.isfinite(r.get("fwd_ret", np.nan)) else None,
        }
    return out


def split_ls(w: dict[str, float]) -> tuple[list[str], list[str]]:
    longs = [s for s, wt in w.items() if wt > 0]
    shorts = [s for s, wt in w.items() if wt < 0]
    longs.sort(key=lambda s: -w[s])
    shorts.sort(key=lambda s: w[s])
    return longs, shorts


def action_lists(prev: dict[str, int], today: dict[str, int]):
    opens_long, opens_short, holds, closes = [], [], [], []
    for s, side in today.items():
        prev_side = prev.get(s)
        if prev_side == side:
            holds.append((s, side))
        else:
            if prev_side is not None:
                closes.append((s, prev_side))
            (opens_long if side > 0 else opens_short).append(s)
    for s, side in prev.items():
        if s not in today:
            closes.append((s, side))
    return opens_long, opens_short, holds, closes


def instruction(
    inn: bool,
    lots: int,
    opens_long: list[str],
    opens_short: list[str],
    holds: list[tuple[str, int]],
    closes: list[tuple[str, int]],
    prev_lots_n: int | None,
    prev_lots: dict[str, int],
    book: float | None,
) -> str:
    parts: list[str] = []
    book_s = f"BOOK={book:.3f}" if book is not None else "BOOK=NA"
    fallback = prev_lots_n or 0
    if not inn:
        if closes:
            parts.append(f"空仓（{book_s}≥1.14全平）")
            parts.append("平仓" + "、".join(name_lots(s, prev_lots.get(s, fallback)) for s, _ in closes))
        else:
            parts.append(f"空仓（{book_s}≥1.14）")
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
    if prev_lots_n is not None and holds and prev_lots_n != lots:
        parts.append(f"（手数{prev_lots_n}→{lots}）")
    return " ".join(parts) if parts else "空仓"


def slot_cols(prefix: str, sym: str | None, meta: dict, lots: float, action: str, inn: bool) -> dict:
    if not sym:
        return {
            f"{prefix}代码": None,
            f"{prefix}中文": None,
            f"{prefix}合约": None,
            f"{prefix}方向": None,
            f"{prefix}动作": None,
            f"{prefix}基准手数": None,
            f"{prefix}缩放手数": None,
            f"{prefix}pos64": None,
            f"{prefix}收盘": None,
            f"{prefix}次日开盘": None,
        }
    info = meta.get(sym, {})
    n = int(lots) if inn else 0
    return {
        f"{prefix}代码": sym,
        f"{prefix}中文": cn(sym),
        f"{prefix}合约": info.get("code"),
        f"{prefix}方向": "多" if prefix.startswith("多") else "空",
        f"{prefix}动作": action,
        f"{prefix}基准手数": 1 if inn else 0,
        f"{prefix}缩放手数": n,
        f"{prefix}pos64": fmt(info.get("pos64"), 4),
        f"{prefix}收盘": fmt(info.get("close"), 2),
        f"{prefix}次日开盘": fmt(info.get("next_open"), 2),
    }


def join_syms(items: list, with_lots: bool, sc: float) -> str:
    if not items:
        return ""
    names = []
    for it in items:
        s = it[0] if isinstance(it, tuple) else it
        if with_lots:
            names.append(name_lots(s, sc))
        else:
            names.append(cn(s))
    return "、".join(names)


def main() -> None:
    print("panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01"))
    print("prices...", flush=True)
    px = load_px()
    panel = panel.merge(px, on=["date", "symbol", "code"], how="left")
    full = panel.dropna(subset=["pos_64"]).copy()
    hist = full.dropna(subset=["fwd_ret"])

    recs: list[dict] = []
    print("daily 2L2S pos64...", flush=True)
    for dt, day in hist.groupby("date"):
        if dt < BT_START:
            continue
        day = day.drop_duplicates("symbol").reset_index(drop=True)
        picks = pick_ls_n(day, day["pos_64"].to_numpy(), N_EACH)
        w = picks_to_w(picks)
        recs.append({
            "date": dt,
            "w": w,
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
    vs, sc, vol20 = vol_stats(raw_bt)
    sc = sc.reindex(raw.index)
    vol20 = vol20.reindex(raw.index)
    if sc.notna().any():
        sc = sc.ffill(limit=1).fillna(1.0)
        vol20 = vol20.ffill(limit=1)
    vs_full = (raw * sc).where(raw.notna())

    ratio = book_ratio(vs.dropna()).reindex(raw.index)
    if ratio.notna().any():
        ratio = ratio.ffill(limit=1)
    cash = (ratio >= BOOK_THR).fillna(False)
    live = vs_full.copy()
    live[cash] = 0.0

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
    prev_lots_n: int | None = None
    prev_lots: dict[str, int] = {}
    daily_rows = []
    ledger_rows = []

    for dt in dates:
        r = by_dt[dt]
        inn = not bool(cash.loc[dt]) if dt in cash.index else True
        b = float(ratio.loc[dt]) if dt in ratio.index and np.isfinite(ratio.loc[dt]) else None
        sc_t = float(sc.loc[dt]) if dt in sc.index and np.isfinite(sc.loc[dt]) else 1.0
        lots_n = to_lots(sc_t, inn)
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

        note = instruction(inn, lots_n, opens_long, opens_short, holds, closes, prev_lots_n, prev_lots, b)
        fill_note = "收盘后下单，次日开盘成交"
        if r["meta"] and all(not v.get("next_open") for v in meta.values()):
            fill_note = "收盘后下单；次日开盘尚未发生，成交价待定"

        l1 = longs[0] if len(longs) > 0 else None
        l2 = longs[1] if len(longs) > 1 else None
        s1 = shorts[0] if len(shorts) > 0 else None
        s2 = shorts[1] if len(shorts) > 1 else None

        row = {
            "日期": str(pd.Timestamp(dt).date()),
            "是否空仓": (not inn),
            "BOOK": fmt(b, 3),
            "波动20日年化": fmt(vol_t, 4),
            "波动20日年化pct": fmt(None if vol_t is None else vol_t * 100, 2),
            "仓位倍数": fmt(sc_t, 4),
            "截面品种数": r["n_cs"],
            "账户日收益": fmt(pnl_t, 5),
            "做多": "、".join(name_lots(s, lots_n) for s in opens_long) if inn else "",
            "做空": "、".join(name_lots(s, lots_n) for s in opens_short) if inn else "",
            "平仓": "、".join(name_lots(s, prev_lots.get(s, prev_lots_n or 0)) for s, _ in closes),
            "持仓": "、".join(f"{name_lots(s, lots_n)}({'多' if side > 0 else '空'})" for s, side in holds) if inn else "",
            "操作说明": note,
            "信号做多": "、".join(cn(s) for s in longs),
            "信号做空": "、".join(cn(s) for s in shorts),
            "成交说明": fill_note,
        }
        for prefix, sym in (("多1", l1), ("多2", l2), ("空1", s1), ("空2", s2)):
            side = 1 if prefix.startswith("多") else -1
            action = act_of(sym, side) if sym else ""
            row.update(slot_cols(prefix, sym, meta, lots_n, action, inn))
        daily_rows.append(row)

        def add_ledger(action: str, sym: str, side: int, lots: int):
            info = meta.get(sym, {})
            if action == "平仓":
                signed_lots = int(prev_lots.get(sym, lots))
            elif action == "空仓":
                signed_lots = 0
            else:
                signed_lots = int(lots)
            ledger_rows.append({
                "日期": str(pd.Timestamp(dt).date()),
                "动作": action,
                "方向": "多" if side > 0 else "空",
                "品种代码": sym,
                "品种中文": cn(sym),
                "合约": info.get("code"),
                "基准手数": 0 if action in ("平仓", "空仓") or not inn else 1,
                "缩放手数": signed_lots,
                "pos64": fmt(info.get("pos64"), 4),
                "收盘": fmt(info.get("close"), 2),
                "次日开盘": fmt(info.get("next_open"), 2),
                "次日日期": info.get("next_date"),
                "BOOK": fmt(b, 3),
                "波动20日年化": fmt(vol_t, 4),
                "波动20日年化pct": fmt(None if vol_t is None else vol_t * 100, 2),
                "仓位倍数": fmt(sc_t, 4),
                "是否空仓": (not inn),
                "账户日收益": fmt(pnl_t, 5),
                "操作说明": note,
                "成交说明": fill_note,
            })

        if not inn:
            if closes:
                for s, side in closes:
                    add_ledger("平仓", s, side, prev_lots.get(s, prev_lots_n or 0))
            else:
                ledger_rows.append({
                    "日期": str(pd.Timestamp(dt).date()),
                    "动作": "空仓",
                    "方向": "",
                    "品种代码": "",
                    "品种中文": "",
                    "合约": "",
                    "基准手数": 0,
                    "缩放手数": 0,
                    "pos64": None,
                    "收盘": None,
                    "次日开盘": None,
                    "次日日期": None,
                    "BOOK": fmt(b, 3),
                    "波动20日年化": fmt(vol_t, 4),
                    "波动20日年化pct": fmt(None if vol_t is None else vol_t * 100, 2),
                    "仓位倍数": fmt(sc_t, 4),
                    "是否空仓": True,
                    "账户日收益": fmt(pnl_t, 5),
                    "操作说明": note,
                    "成交说明": fill_note,
                })
        else:
            for s, side in closes:
                add_ledger("平仓", s, side, prev_lots.get(s, prev_lots_n or 0))
            for s in opens_long:
                add_ledger("做多", s, 1, lots_n)
            for s in opens_short:
                add_ledger("做空", s, -1, lots_n)
            for s, side in holds:
                add_ledger("持仓", s, side, lots_n)

        prev_held = today
        prev_lots_n = lots_n if inn else 0
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
        "scheme": "pos64 2L2S（四品种，每品种基准1手）→ 波动目标7%（上限2.5）→ BOOK≥1.14全平",
        "rules": {
            "signal": "收盘后按 pos64 截面做多最高2、做空最低2",
            "fill": "次日开盘成交",
            "lots": "实盘手数=round(仓位倍数)，至少1手（空仓为0）；仓位倍数=0.07/昨20日年化波动，clip[0,2.5]",
            "book": "BOOK=策略20日波动/252日波动中位数（已滞后1日），≥1.14空仓",
        },
        "full_sample": {
            **{k: st_all[k] for k in ("return", "sharpe", "max_dd", "days", "n_neg_years", "neg_years", "worst_year_sharpe")},
            "cash": fmt(float(cash.reindex(full_bt.index).fillna(False).mean()), 3),
        },
        "year_2026": {**{k: st_26[k] for k in ("return", "sharpe", "max_dd", "days")}, "cash": fmt(cash_26, 3)},
        "months": months,
        "last_day": last,
        "files": {"daily": str(d_path), "ledger": str(p_path)},
    }
    jpath = OUT_DIR / "final_scheme_2026.json"
    jpath.write_text(json.dumps(snap, ensure_ascii=False, indent=2, default=str))
    print(f"last {last_dt} cash={last.get('是否空仓')} book={last.get('BOOK')} 操作={last.get('操作说明')}")
    print(f"saved {d_path} {p_path} {jpath}")


if __name__ == "__main__":
    main()
