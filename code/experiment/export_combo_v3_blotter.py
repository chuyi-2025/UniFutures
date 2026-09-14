#!/usr/bin/env python3
"""Export combo_v3 daily + ledger with per-sleeve open/close attribution.

Layout mirrors final_scheme_*_daily / *_ledger under:
  data/infer/results/daily/combo_v3/
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/workspace/lab/UniFutures")
OUT = ROOT / "data/infer/results/daily/combo_v3"
EQUITY = 3_000_000.0
W_WEATHER = 500_000.0
MACRO_SCALE = 0.12 / 0.35
DUAL_START = pd.Timestamp("2024-03-08")
W_NEWS = 1.25

CN_FALLBACK = {
    "AU": "黄金", "AG": "白银", "SC": "原油", "CU": "铜", "AL": "铝", "ZN": "锌",
    "NI": "镍", "PB": "铅", "SN": "锡", "CF": "棉花", "SR": "白糖", "AP": "苹果",
    "CJ": "红枣", "RB": "螺纹钢", "HC": "热卷", "I": "铁矿石", "J": "焦炭",
    "JM": "焦煤", "M": "豆粕", "Y": "豆油", "P": "棕榈油", "OI": "菜油",
    "RM": "菜粕", "TA": "PTA", "MA": "甲醇", "PP": "聚丙烯", "L": "塑料",
    "V": "PVC", "EG": "乙二醇", "EB": "苯乙烯", "PG": "LPG", "FU": "燃料油",
    "LU": "低硫燃油", "BU": "沥青", "NR": "20号胶", "RU": "橡胶", "SS": "不锈钢",
    "LH": "生猪", "PK": "花生", "SA": "纯碱", "FG": "玻璃", "SF": "硅铁",
    "SM": "锰硅", "ZC": "动力煤", "UR": "尿素", "SP": "纸浆", "CS": "玉米淀粉",
    "C": "玉米", "A": "豆一", "B": "豆二", "JD": "鸡蛋", "WR": "线材",
    "AO": "花生油", "PR": "瓶片", "PX": "对二甲苯", "SH": "烧碱", "BR": "丁二烯橡胶",
}


def cn(sym: str) -> str:
    try:
        from infer.config import SYMBOL_CN  # type: ignore

        return SYMBOL_CN.get(sym, CN_FALLBACK.get(sym, sym))
    except Exception:
        return CN_FALLBACK.get(sym, sym)


def _split_items(text) -> list[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return []
    parts = re.split(r"[、;；]", s)
    return [p.strip() for p in parts if p.strip()]


def _parse_final_piece(piece: str) -> dict | None:
    """Parse '棉花2701 1手@17120' / '橡胶2701 1手(多)@19315' / '...(收盘)'."""
    m = re.match(
        r"^(?P<body>.+?)\s+(?P<lots>\d+)手(?:\((?P<side>[多空])\))?@(?P<px>[0-9.]+)(?P<tag>\(收盘\))?$",
        piece.strip(),
    )
    if not m:
        return {"raw": piece}
    body = m.group("body")
    # body is like 棉花2701 or 不锈钢2701
    mm = re.match(r"^(.+?)(\d{3,4})$", body)
    name_cn, contract = (mm.group(1), body) if mm else (body, body)
    return {
        "品种中文": name_cn,
        "合约": contract,
        "手数": int(m.group("lots")),
        "方向标注": m.group("side") or "",
        "点位": float(m.group("px")),
        "点位说明": "收盘" if m.group("tag") else "",
        "raw": piece,
    }


def build_final_events(final: pd.DataFrame) -> list[dict]:
    rows = []
    sig = "pos64×RSI截面叠加；BOOK≥1.18全平；每品种≤1手；保证金帽5万"
    for _, r in final.iterrows():
        dt = pd.Timestamp(r["日期"])
        book = r.get("BOOK")
        cash = bool(r.get("是否空仓"))
        base = {
            "日期": dt,
            "策略": "量价",
            "策略代码": "final_scheme",
            "信号": sig,
            "BOOK": book,
            "是否空仓": cash,
            "账户日盈亏": float(r.get("账户日盈亏") or 0),
            "合计保证金": float(r.get("合计保证金") or 0) if pd.notna(r.get("合计保证金")) else 0.0,
        }
        if cash:
            rows.append({
                **base,
                "动作": "空仓",
                "方向": "",
                "品种代码": "",
                "品种中文": "",
                "合约": "",
                "手数": 0,
                "点位": "",
                "操作说明": str(r.get("操作说明") or f"空仓（BOOK={book}）"),
            })
            continue
        for act, col, default_dir in [
            ("开多", "做多", "多"),
            ("开空", "做空", "空"),
            ("平仓", "平仓", ""),
            ("持仓", "持仓", ""),
        ]:
            for piece in _split_items(r.get(col)):
                p = _parse_final_piece(piece) or {"raw": piece}
                side = p.get("方向标注") or default_dir
                if act == "平仓" and not side:
                    side = ""
                rows.append({
                    **base,
                    "动作": act,
                    "方向": side,
                    "品种代码": "",
                    "品种中文": p.get("品种中文", ""),
                    "合约": p.get("合约", p.get("raw", piece)),
                    "手数": p.get("手数", 0),
                    "点位": p.get("点位", ""),
                    "操作说明": f"[量价|{sig}] {act}{side} {p.get('raw', piece)}",
                })
    return rows


def build_macro_events() -> tuple[pd.DataFrame, list[dict]]:
    sys.path.insert(0, str(ROOT / "macro/code"))
    from search_macro_account import BookEngine, CAPITAL  # noqa: WPS433

    panel = pd.read_parquet(ROOT / "macro/data/results/search_account/expanded_aligned_panel.parquet")
    factors = [c for c in panel.columns if c.endswith("_z60") or c in (
        "usd_hawkish", "risk_off", "labor_soft", "dxy_chg5", "dxy_chg20",
        "hike_proxy_chg", "us_2y_chg5", "vix_chg", "unemp_chg",
    )]
    # BookEngine expects specific factor list from module — use its FACTORS intersect
    from search_macro_account import FACTORS  # noqa: WPS433

    facs = [f for f in FACTORS if f in panel.columns]
    eng = BookEngine(panel, facs)
    factor, mode, thr, direction, hold = "usd_broad_z60", "long_only", 0.25, 1.0, 5
    sym_list = ("AU",)
    max_names, max_lots, margin_cap, util = 2, 2, 200_000.0, 0.35
    js = [eng.sym_index[s] for s in sym_list]
    sig = eng._signal(factor, mode, thr, direction, hold)[:, js]
    fac_raw = eng.fac[factor][:, js]
    fwd = eng.fwd[:, js]
    m1 = eng.m1[:, js]
    notion = eng.notional[:, js]
    fee_o = eng.fee_o[:, js]
    fee_c = eng.fee_c[:, js]
    n_t, n_s = sig.shape
    prev = np.zeros(n_s, dtype=int)
    daily_rows = []
    events = []
    sig_txt = f"{factor} {mode} thr≥{thr} hold={hold}d dir={int(direction)} → AU"

    for t in range(n_t):
        dt = pd.Timestamp(eng.dates[t])
        scores = []
        for j in range(n_s):
            s = sig[t, j]
            if abs(s) < 1e-12:
                continue
            mj = m1[t, j]
            if not np.isfinite(mj) or mj <= 0:
                continue
            if margin_cap is not None and mj > margin_cap:
                continue
            scores.append((abs(s), s, j, mj))
        scores.sort(reverse=True)
        scores = scores[:max_names]
        target = np.zeros(n_s, dtype=int)
        budget = util * CAPITAL
        used = 0.0
        if scores:
            share = budget / len(scores)
            for _, s, j, mj in scores:
                lots = int(min(max_lots, max(0, np.floor(share / mj))))
                if lots <= 0 and mj <= budget - used:
                    lots = 1
                while lots > 0 and used + lots * mj > budget:
                    lots -= 1
                lots = min(lots, max_lots)
                if lots > 0:
                    target[j] = int(np.sign(s) * lots)
                    used += lots * mj
        fee = 0.0
        for j in range(n_s):
            old, new = int(prev[j]), int(target[j])
            fo = fee_o[t, j] if np.isfinite(fee_o[t, j]) else 0.0
            fc = fee_c[t, j] if np.isfinite(fee_c[t, j]) else fo
            if old != 0 and (new == 0 or np.sign(new) != np.sign(old)):
                fee += fc * abs(old)
            if new != 0 and old == 0:
                fee += fo * abs(new)
            elif new != 0 and old != 0 and np.sign(new) == np.sign(old):
                d = abs(new) - abs(old)
                if d > 0:
                    fee += fo * d
                elif d < 0:
                    fee += fc * (-d)
        gross = 0.0
        mar = 0.0
        hold_bits = []
        open_bits = []
        close_bits = []
        for j in range(n_s):
            sym = sym_list[j]
            old, new = int(prev[j]), int(target[j])
            z = float(fac_raw[t, j]) if np.isfinite(fac_raw[t, j]) else float("nan")
            z_s = f"{z:.3f}" if np.isfinite(z) else "nan"
            reason = f"{sig_txt}; 当日因子={z_s}"
            if old != 0 and (new == 0 or np.sign(new) != np.sign(old)):
                side = "多" if old > 0 else "空"
                close_bits.append(f"{cn(sym)} {abs(old)}手({side})")
                events.append({
                    "日期": dt, "策略": "宏观", "策略代码": "macro_A",
                    "信号": reason, "动作": "平仓", "方向": side,
                    "品种代码": sym, "品种中文": cn(sym), "合约": sym,
                    "手数": abs(old), "点位": "", "BOOK": "", "是否空仓": False,
                    "账户日盈亏": "", "合计保证金": "",
                    "操作说明": f"[宏观|{reason}] 平{side} {cn(sym)} {abs(old)}手",
                })
                old_eff = 0
            else:
                old_eff = old
            if new != 0 and old_eff == 0:
                side = "多" if new > 0 else "空"
                open_bits.append(f"{cn(sym)} {abs(new)}手({side})")
                events.append({
                    "日期": dt, "策略": "宏观", "策略代码": "macro_A",
                    "信号": reason, "动作": f"开{side}", "方向": side,
                    "品种代码": sym, "品种中文": cn(sym), "合约": sym,
                    "手数": abs(new), "点位": "", "BOOK": "", "是否空仓": False,
                    "账户日盈亏": "", "合计保证金": "",
                    "操作说明": f"[宏观|{reason}] 开{side} {cn(sym)} {abs(new)}手",
                })
            elif new != 0 and old_eff != 0 and np.sign(new) == np.sign(old_eff) and abs(new) != abs(old_eff):
                side = "多" if new > 0 else "空"
                events.append({
                    "日期": dt, "策略": "宏观", "策略代码": "macro_A",
                    "信号": reason, "动作": "调仓", "方向": side,
                    "品种代码": sym, "品种中文": cn(sym), "合约": sym,
                    "手数": abs(new), "点位": "", "BOOK": "", "是否空仓": False,
                    "账户日盈亏": "", "合计保证金": "",
                    "操作说明": f"[宏观|{reason}] 调仓 {cn(sym)} {abs(old_eff)}→{abs(new)}手({side})",
                })
            if new != 0:
                side = "多" if new > 0 else "空"
                hold_bits.append(f"{cn(sym)} {abs(new)}手({side})")
                if abs(new) == abs(old_eff) and np.sign(new) == np.sign(old_eff) and old_eff != 0:
                    events.append({
                        "日期": dt, "策略": "宏观", "策略代码": "macro_A",
                        "信号": reason, "动作": "持仓", "方向": side,
                        "品种代码": sym, "品种中文": cn(sym), "合约": sym,
                        "手数": abs(new), "点位": "", "BOOK": "", "是否空仓": False,
                        "账户日盈亏": "", "合计保证金": "",
                        "操作说明": f"[宏观|{reason}] 持仓 {cn(sym)} {abs(new)}手({side})",
                    })
            fr = fwd[t, j] if np.isfinite(fwd[t, j]) else 0.0
            val = notion[t, j]
            if not np.isfinite(val):
                val = (m1[t, j] / 0.15) if np.isfinite(m1[t, j]) else 0.0
            if new != 0:
                gross += new * fr * float(val)
                mar += abs(new) * (float(m1[t, j]) if np.isfinite(m1[t, j]) else 0.0)
        pnl = gross - fee
        daily_rows.append({
            "日期": dt,
            "账户日盈亏": pnl,
            "合计保证金": mar,
            "做多": "、".join([b for b in open_bits if "(多)" in b]),
            "做空": "、".join([b for b in open_bits if "(空)" in b]),
            "平仓": "、".join(close_bits),
            "持仓": "、".join(hold_bits),
            "操作说明": " ".join(
                ([f"开{x}" for x in open_bits] if open_bits else [])
                + ([f"平{x}" for x in close_bits] if close_bits else [])
                + ([f"持{x}" for x in hold_bits] if hold_bits and not open_bits and not close_bits else [])
                or (["空仓"] if not hold_bits and not open_bits and not close_bits else [])
            ),
            "信号": sig_txt,
            "因子值": float(fac_raw[t, 0]) if np.isfinite(fac_raw[t, 0]) else np.nan,
        })
        prev[:] = target

    daily = pd.DataFrame(daily_rows).set_index("日期").sort_index()
    return daily, events


def build_weather_events() -> tuple[pd.DataFrame, list[dict]]:
    schemes = json.loads((ROOT / "weather/data/results/search/picked_schemes_v3.json").read_text())
    pick_map = {p["symbol"]: p for p in schemes["picks"]}
    legs = pd.read_csv(ROOT / "weather/data/results/search/picked_legs_daily_v3.csv")
    legs["日期"] = pd.to_datetime(legs["日期"])
    book = pd.read_csv(ROOT / "weather/data/results/search/picked_book_daily_v3.csv")
    book["日期"] = pd.to_datetime(book["日期"])
    book = book.set_index("日期")
    book_yuan = book["pnl"].astype(float) * W_WEATHER

    events = []
    daily_bits: dict[pd.Timestamp, list[str]] = {}

    for sym, g in legs.groupby("symbol"):
        g = g.sort_values("日期")
        p = pick_map[sym]
        sig_txt = (
            f"{p['factor']} {p['mode']} thr={p['thr']} hold={p['hold']} "
            f"dir={int(p['direction'])} pheno={'Y' if p['pheno_only'] else 'N'}"
        )
        prev = 0.0
        for _, r in g.iterrows():
            dt = r["日期"]
            cur = float(r["signal"])
            log_pnl = float(r["pnl"])
            # leg share of book is research-only; attribution uses leg log, book pnl is EW
            if cur == prev:
                if cur != 0:
                    side = "多" if cur > 0 else "空"
                    events.append({
                        "日期": dt, "策略": "天气", "策略代码": "weather_v3",
                        "信号": sig_txt, "动作": "持仓", "方向": side,
                        "品种代码": sym, "品种中文": cn(sym), "合约": sym,
                        "手数": 1, "点位": "", "BOOK": "", "是否空仓": False,
                        "账户日盈亏": "", "合计保证金": "",
                        "操作说明": f"[天气|{sig_txt}] 持仓 {cn(sym)}({side})",
                    })
                    daily_bits.setdefault(dt, []).append(f"持{cn(sym)}({side})")
                prev = cur
                continue
            if prev != 0:
                side = "多" if prev > 0 else "空"
                events.append({
                    "日期": dt, "策略": "天气", "策略代码": "weather_v3",
                    "信号": sig_txt, "动作": "平仓", "方向": side,
                    "品种代码": sym, "品种中文": cn(sym), "合约": sym,
                    "手数": 1, "点位": "", "BOOK": "", "是否空仓": False,
                    "账户日盈亏": "", "合计保证金": "",
                    "操作说明": f"[天气|{sig_txt}] 平{side} {cn(sym)}（信号{prev:+.0f}→{cur:+.0f}）",
                })
                daily_bits.setdefault(dt, []).append(f"平{cn(sym)}({side})")
            if cur != 0:
                side = "多" if cur > 0 else "空"
                events.append({
                    "日期": dt, "策略": "天气", "策略代码": "weather_v3",
                    "信号": sig_txt, "动作": f"开{side}", "方向": side,
                    "品种代码": sym, "品种中文": cn(sym), "合约": sym,
                    "手数": 1, "点位": "", "BOOK": "", "是否空仓": False,
                    "账户日盈亏": "", "合计保证金": "",
                    "操作说明": f"[天气|{sig_txt}] 开{side} {cn(sym)}（信号{prev:+.0f}→{cur:+.0f}）",
                })
                daily_bits.setdefault(dt, []).append(f"开{cn(sym)}({side})")
            prev = cur

    daily = pd.DataFrame({
        "账户日盈亏": book_yuan,
        "操作说明": [ " ".join(daily_bits.get(d, ["空仓"])) for d in book_yuan.index ],
    })
    daily.index.name = "日期"
    return daily, events


def build_news_events() -> tuple[pd.DataFrame, list[dict]]:
    """Rebuild Dual-Gated news positions (from BT_START)."""
    # Prefer news/code/train over macro/code (both expose `common`).
    news_train = str(ROOT / "news/code/train")
    exp = str(ROOT / "code/experiment")
    for p in (str(ROOT / "macro/code"), news_train, exp):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, exp)
    sys.path.insert(0, news_train)
    # Drop shadowed macro `common` if already imported by build_macro_events
    mod = sys.modules.get("common")
    if mod is not None and "macro" in str(getattr(mod, "__file__", "") or ""):
        del sys.modules["common"]

    from common import DATA_DIR  # noqa: WPS433
    assert "news" in str(DATA_DIR), f"wrong common imported: {DATA_DIR}"
    from final_scheme_2026_blotter import load_px  # noqa: WPS433
    from linear_ridge_walkforward import build_panel  # noqa: WPS433
    from pos64_margin_integer_search import CAPITAL  # noqa: WPS433
    from rule_factor_mine import batch_aggregate, filter_docs, keep_single_symbol_docs  # noqa: WPS433
    from rule_news_alt_search import build_news_cache, hyst_update, lots_fixed1, lots_risk  # noqa: WPS433
    from rule_news_capacity_search import day_spread, dollar_multi, fees_multi  # noqa: WPS433
    from rule_news_quad_search import attach_vol  # noqa: WPS433
    from six_name_select import pick_ls_n  # noqa: WPS433

    cfg = dict(
        n_each=2, enter=0.40, exit_=0.15, min_hold=5, spread_ref=0.30, strength_cap=1.0,
        max_leg_margin=50_000.0, util=0.01, max_lots=2, max_leg_risk=3000.0,
    )
    print("news: docs/panel...", flush=True)
    docs = pd.read_parquet(DATA_DIR / "rule_docs_rich.parquet")
    docs["report_date"] = pd.to_datetime(docs["report_date"])
    docs = keep_single_symbol_docs(filter_docs(docs, "no_dianping"))
    panel = build_panel(DUAL_START).dropna(subset=["fwd_ret"])
    panel = panel.merge(load_px(), on=["date", "symbol", "code"], how="left")
    panel = panel[panel["date"] >= DUAL_START].copy()
    panel = attach_vol(panel)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    print("news factor L30...", flush=True)
    fac = batch_aggregate(docs, "rule_edge", 30, dates, "uniform")
    cache = build_news_cache(panel, fac)

    state = None
    days_in = 0
    prev: dict = {}
    daily_rows = []
    events = []
    sig_base = (
        f"研报情感截面 top{cfg['n_each']}多空; enter≥{cfg['enter']} exit≤{cfg['exit_']} "
        f"min_hold={cfg['min_hold']}; 仅当量价保证金>0时×{W_NEWS}计入组合"
    )

    for dt in cache["dates"]:
        info = cache["days"][dt]
        d = info["frame"]
        spread = day_spread(d["val"].to_numpy()) if len(d) >= 2 else 0.0
        cand = pick_ls_n(d, d["val"].to_numpy(), cfg["n_each"]) if len(d) >= 2 * cfg["n_each"] else []
        state, days_in = hyst_update(
            state, days_in, cand, spread, cfg["enter"], cfg["exit_"], cfg["min_hold"], set(d["symbol"])
        )
        if state and cfg.get("util", 0) > 0:
            strength = float(np.clip(spread / cfg["spread_ref"], 0.0, cfg["strength_cap"]))
            pos = lots_risk(
                state, info["m1"], info["risk1"], cfg["util"] * CAPITAL * strength,
                cfg["max_lots"], cfg["max_leg_risk"], cfg["max_leg_margin"],
            )
        else:
            pos = lots_fixed1(state, info["m1"], cfg.get("max_leg_margin"))
        cur = {s: n for s, n in pos}
        fee = fees_multi(info["meta"], prev, cur)
        if pos:
            gross, mar, _ = dollar_multi(pos, info["close"], info["ret"])
            pnl = gross - fee
        else:
            mar, pnl = 0.0, (-fee if fee else 0.0)

        # score map for explanation
        score = {}
        if len(d):
            for _, rr in d.iterrows():
                score[str(rr["symbol"])] = float(rr["val"]) if pd.notna(rr["val"]) else float("nan")

        ts = pd.Timestamp(dt)
        open_bits, close_bits, hold_bits = [], [], []
        all_syms = set(prev) | set(cur)
        for sym in sorted(all_syms):
            o = int(prev.get(sym, 0))
            n = int(cur.get(sym, 0))
            sc = score.get(sym, float("nan"))
            sc_s = f"{sc:.3f}" if np.isfinite(sc) else "nan"
            reason = f"{sig_base}; spread={spread:.3f}; {sym}分={sc_s}"
            if o != 0 and (n == 0 or np.sign(n) != np.sign(o)):
                side = "多" if o > 0 else "空"
                close_bits.append(f"{cn(sym)}{abs(o)}手({side})")
                events.append({
                    "日期": ts, "策略": "新闻", "策略代码": "news_dual_gated",
                    "信号": reason, "动作": "平仓", "方向": side,
                    "品种代码": sym, "品种中文": cn(sym), "合约": sym,
                    "手数": abs(o), "点位": "", "BOOK": "", "是否空仓": False,
                    "账户日盈亏": "", "合计保证金": "",
                    "操作说明": f"[新闻|{reason}] 平{side} {cn(sym)} {abs(o)}手",
                })
                o_eff = 0
            else:
                o_eff = o
            if n != 0 and o_eff == 0:
                side = "多" if n > 0 else "空"
                open_bits.append(f"{cn(sym)}{abs(n)}手({side})")
                events.append({
                    "日期": ts, "策略": "新闻", "策略代码": "news_dual_gated",
                    "信号": reason, "动作": f"开{side}", "方向": side,
                    "品种代码": sym, "品种中文": cn(sym), "合约": sym,
                    "手数": abs(n), "点位": "", "BOOK": "", "是否空仓": False,
                    "账户日盈亏": "", "合计保证金": "",
                    "操作说明": f"[新闻|{reason}] 开{side} {cn(sym)} {abs(n)}手",
                })
            if n != 0 and abs(n) == abs(o_eff) and np.sign(n) == np.sign(o_eff) and o_eff != 0:
                side = "多" if n > 0 else "空"
                hold_bits.append(f"{cn(sym)}{abs(n)}手({side})")
                events.append({
                    "日期": ts, "策略": "新闻", "策略代码": "news_dual_gated",
                    "信号": reason, "动作": "持仓", "方向": side,
                    "品种代码": sym, "品种中文": cn(sym), "合约": sym,
                    "手数": abs(n), "点位": "", "BOOK": "", "是否空仓": False,
                    "账户日盈亏": "", "合计保证金": "",
                    "操作说明": f"[新闻|{reason}] 持仓 {cn(sym)} {abs(n)}手({side})",
                })
            elif n != 0 and o_eff != 0 and np.sign(n) == np.sign(o_eff) and abs(n) != abs(o_eff):
                side = "多" if n > 0 else "空"
                events.append({
                    "日期": ts, "策略": "新闻", "策略代码": "news_dual_gated",
                    "信号": reason, "动作": "调仓", "方向": side,
                    "品种代码": sym, "品种中文": cn(sym), "合约": sym,
                    "手数": abs(n), "点位": "", "BOOK": "", "是否空仓": False,
                    "账户日盈亏": "", "合计保证金": "",
                    "操作说明": f"[新闻|{reason}] 调仓 {cn(sym)} {abs(o_eff)}→{abs(n)}手({side})",
                })
                hold_bits.append(f"{cn(sym)}{abs(n)}手({side})")

        note_parts = []
        if open_bits:
            note_parts.append("开" + "、".join(open_bits))
        if close_bits:
            note_parts.append("平" + "、".join(close_bits))
        if hold_bits and not open_bits and not close_bits:
            note_parts.append("持" + "、".join(hold_bits))
        if not note_parts:
            note_parts.append("空仓")

        daily_rows.append({
            "日期": ts,
            "账户日盈亏": pnl,
            "合计保证金": mar,
            "spread": spread,
            "操作说明": " ".join(note_parts),
            "信号": sig_base,
            "持仓明细": "、".join(hold_bits + open_bits) if (hold_bits or open_bits) else "",
        })
        prev = cur

    daily = pd.DataFrame(daily_rows).set_index("日期").sort_index()
    return daily, events


def join_ops(*parts: str) -> str:
    xs = [p.strip() for p in parts if p and str(p).strip() and str(p).strip() != "nan"]
    return " | ".join(xs) if xs else "空仓"


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    print("final...", flush=True)
    final = pd.read_csv(ROOT / "data/infer/results/daily/linear/final_scheme_all_daily.csv")
    final["日期"] = pd.to_datetime(final["日期"])
    final_events = build_final_events(final)
    final_i = final.set_index("日期").sort_index()

    print("macro...", flush=True)
    macro_d, macro_events = build_macro_events()

    print("weather...", flush=True)
    weather_d, weather_events = build_weather_events()

    print("news (may take a few minutes)...", flush=True)
    try:
        news_d, news_events = build_news_events()
        news_ok = True
    except Exception as e:
        print(f"news rebuild failed: {e!r}; falling back to pnl-only notes", flush=True)
        news_ok = False
        news_d = pd.DataFrame()
        news_events = []
        dual = pd.read_csv(
            ROOT / "news/data/results/rule_capacity_dual/audit/dual_gated_v1_proper_book_daily.csv"
        )
        dual["日期"] = pd.to_datetime(dual["日期"])
        dual = dual.set_index("日期")
        for dt, r in dual.iterrows():
            npnl = float(r["news门控加权"])
            if abs(npnl) < 1e-9 and float(r["合计保证金"]) <= 0:
                continue
            gated = float(r.get("final盈亏", 0))  # presence of dual day
            note = (
                f"[新闻|Dual门控×{W_NEWS}] "
                + ("开火" if abs(npnl) > 1e-9 else "候选未计入/Core空仓")
                + f" 加权盈亏={npnl:.1f}"
            )
            news_events.append({
                "日期": dt, "策略": "新闻", "策略代码": "news_dual_gated",
                "信号": f"Dual-Gated V1 ×{W_NEWS}（无持仓明细回退）",
                "动作": "盈亏" if abs(npnl) > 1e-9 else "空仓",
                "方向": "", "品种代码": "", "品种中文": "", "合约": "",
                "手数": 0, "点位": "", "BOOK": "", "是否空仓": abs(npnl) < 1e-9,
                "账户日盈亏": npnl, "合计保证金": "",
                "操作说明": note,
            })
        news_d = dual[["news门控加权"]].rename(columns={"news门控加权": "账户日盈亏"})

    # master calendar from final
    idx = final_i.index
    core_m = final_i["合计保证金"].astype(float).fillna(0.0)
    core_pnl = final_i["账户日盈亏"].astype(float).fillna(0.0)

    # dual overlay for news gate window
    dual_path = ROOT / "news/data/results/rule_capacity_dual/audit/dual_gated_v1_proper_book_daily.csv"
    dual = pd.read_csv(dual_path)
    dual["日期"] = pd.to_datetime(dual["日期"])
    dual = dual.set_index("日期")
    in_dual = pd.Series((idx >= DUAL_START) & idx.isin(dual.index), index=idx)

    news_raw = (
        news_d["账户日盈亏"].astype(float).reindex(idx).fillna(0.0)
        if len(news_d) else pd.Series(0.0, index=idx)
    )
    if isinstance(news_raw, pd.DataFrame):
        news_raw = news_raw.iloc[:, 0]

    dual_dates = idx[in_dual]
    news_official = dual.loc[dual_dates, "news门控加权"].astype(float)
    final_official = dual.loc[dual_dates, "final盈亏"].astype(float)
    mar_official = dual.loc[dual_dates, "合计保证金"].astype(float)

    # gate: only when core margin > 0; weight W_NEWS. Dual window uses official spine.
    if news_ok:
        news_gated = (news_raw * W_NEWS * (core_m > 0).astype(float)).copy()
    else:
        news_gated = news_raw.copy()
    news_gated.loc[dual_dates] = news_official.values

    core_pnl_use = core_pnl.copy()
    core_pnl_use.loc[dual_dates] = final_official.values
    core_m_use = core_m.copy()
    core_m_use.loc[dual_dates] = mar_official.values

    dual_pnl = core_pnl_use + news_gated

    macro_raw = macro_d["账户日盈亏"].astype(float).reindex(idx).fillna(0.0)
    macro_g = macro_raw * MACRO_SCALE * (0.5 + 0.5 * (core_m_use > 0).astype(float))
    weather_pnl = weather_d["账户日盈亏"].astype(float).reindex(idx).fillna(0.0)
    combo = dual_pnl + macro_g + weather_pnl

    # annotate news events with gate status
    gated_news_events = []
    for e in news_events:
        dt = pd.Timestamp(e["日期"])
        if dt not in core_m_use.index:
            continue
        core_on = float(core_m_use.loc[dt]) > 0
        e2 = dict(e)
        if not core_on:
            e2["操作说明"] = e2["操作说明"] + "（当日Core空仓→新闻不计入组合）"
            e2["信号"] = str(e2["信号"]) + " | 门控=关"
        else:
            e2["信号"] = str(e2["信号"]) + " | 门控=开"
            if e2.get("动作") in ("开多", "开空", "平仓", "持仓", "调仓"):
                e2["操作说明"] = e2["操作说明"] + f"（计入组合×{W_NEWS}）"
        gated_news_events.append(e2)

    # macro events: note derate
    macro_events_ann = []
    for e in macro_events:
        dt = pd.Timestamp(e["日期"])
        if dt not in core_m_use.index:
            # still keep if before final? skip
            if dt < idx.min() or dt > idx.max():
                continue
        core_on = float(core_m_use.reindex([dt]).fillna(0).iloc[0]) > 0 if dt in core_m_use.index else True
        scale = MACRO_SCALE * (1.0 if core_on else 0.5)
        e2 = dict(e)
        e2["信号"] = str(e2["信号"]) + f" | 组合盈亏缩放×{scale:.4f}"
        e2["操作说明"] = e2["操作说明"] + f"（组合内×{scale:.4f}）"
        macro_events_ann.append(e2)

    # daily table
    daily_rows = []
    for dt in idx:
        frow = final_i.loc[dt]
        mrow = macro_d.reindex([dt])
        wrow = weather_d.reindex([dt])
        nrow = news_d.reindex([dt]) if len(news_d) else None

        final_ops = str(frow.get("操作说明") or "")
        if final_ops and final_ops != "nan":
            final_ops = f"[量价] {final_ops}"
        else:
            final_ops = ""

        macro_ops = ""
        if dt in macro_d.index:
            mo = str(macro_d.loc[dt].get("操作说明") or "")
            if mo and mo != "nan":
                z = macro_d.loc[dt].get("因子值")
                z_s = f"{float(z):.3f}" if pd.notna(z) else "nan"
                core_on = float(core_m_use.loc[dt]) > 0
                sc = MACRO_SCALE * (1.0 if core_on else 0.5)
                macro_ops = f"[宏观|usd_broad_z60={z_s}×{sc:.4f}] {mo}"

        weather_ops = ""
        if dt in weather_d.index:
            wo = str(weather_d.loc[dt].get("操作说明") or "")
            if wo and wo != "nan" and wo != "空仓":
                weather_ops = f"[天气v3] {wo}"

        news_ops = ""
        if news_ok and dt in news_d.index:
            no = str(news_d.loc[dt].get("操作说明") or "")
            core_on = float(core_m_use.loc[dt]) > 0
            if no and no != "nan":
                gate = "门控开" if core_on else "门控关(不计入)"
                news_ops = f"[新闻|{gate}] {no}"
        elif (not news_ok) and dt in dual.index:
            npnl = float(dual.loc[dt, "news门控加权"])
            if abs(npnl) > 1e-9:
                news_ops = f"[新闻|门控开] 加权盈亏={npnl:.1f}"

        ops = join_ops(final_ops, news_ops, macro_ops, weather_ops)
        cash = bool(frow.get("是否空仓")) and abs(macro_g.loc[dt]) < 1e-9 and abs(weather_pnl.loc[dt]) < 1e-9 and abs(news_gated.loc[dt]) < 1e-9

        daily_rows.append({
            "日期": dt.strftime("%Y-%m-%d"),
            "是否空仓": cash,
            "BOOK": frow.get("BOOK"),
            "量价盈亏": round(float(core_pnl_use.loc[dt]), 4),
            "新闻门控盈亏": round(float(news_gated.loc[dt]), 4),
            "Dual盈亏": round(float(dual_pnl.loc[dt]), 4),
            "宏观降权盈亏": round(float(macro_g.loc[dt]), 4),
            "天气代理盈亏": round(float(weather_pnl.loc[dt]), 4),
            "账户日盈亏": round(float(combo.loc[dt]), 4),
            "账户日收益": round(float(combo.loc[dt] / EQUITY), 8),
            "量价保证金": round(float(core_m_use.loc[dt]), 2),
            "宏观保证金(账户口径)": 0.0,
            "新闻可用": int(bool(in_dual.loc[dt])),
            "做多_量价": frow.get("做多"),
            "做空_量价": frow.get("做空"),
            "平仓_量价": frow.get("平仓"),
            "持仓_量价": frow.get("持仓"),
            "信号做多_量价": frow.get("信号做多"),
            "信号做空_量价": frow.get("信号做空"),
            "宏观操作": macro_d.loc[dt, "操作说明"] if dt in macro_d.index else "",
            "宏观因子usd_broad_z60": macro_d.loc[dt, "因子值"] if dt in macro_d.index else "",
            "新闻操作": news_d.loc[dt, "操作说明"] if (len(news_d) and dt in news_d.index) else "",
            "天气操作": weather_d.loc[dt, "操作说明"] if dt in weather_d.index else "",
            "操作说明": ops,
        })

    # fix macro margin column cleanly
    macro_mar = macro_d["合计保证金"].astype(float).reindex(idx).fillna(0.0) if "合计保证金" in macro_d.columns else pd.Series(0.0, index=idx)
    for i, dt in enumerate(idx):
        daily_rows[i]["宏观保证金(账户口径)"] = round(float(macro_mar.loc[dt]), 2)

    daily = pd.DataFrame(daily_rows)
    daily.to_csv(OUT / "combo_v3_daily.csv", index=False)
    daily.tail(1).to_csv(OUT / "combo_v3_latest.csv", index=False)
    y2026 = daily[daily["日期"] >= "2026-01-01"]
    y2026.to_csv(OUT / "combo_v3_2026_daily.csv", index=False)

    # ledger: prefer open/close/adjust; keep holds but can be huge for weather — keep all for audit
    ledger = pd.DataFrame(final_events + gated_news_events + macro_events_ann + weather_events)
    if len(ledger):
        ledger["日期"] = pd.to_datetime(ledger["日期"]).dt.strftime("%Y-%m-%d")
        # order
        act_order = {"开多": 0, "开空": 1, "平仓": 2, "调仓": 3, "持仓": 4, "空仓": 5, "盈亏": 6}
        ledger["_o"] = ledger["动作"].map(lambda x: act_order.get(x, 9))
        ledger = ledger.sort_values(["日期", "_o", "策略"]).drop(columns=["_o"])
        cols = [
            "日期", "策略", "策略代码", "信号", "动作", "方向", "品种代码", "品种中文", "合约",
            "手数", "点位", "BOOK", "是否空仓", "账户日盈亏", "合计保证金", "操作说明",
        ]
        for c in cols:
            if c not in ledger.columns:
                ledger[c] = ""
        ledger = ledger[cols]
    ledger.to_csv(OUT / "combo_v3_ledger.csv", index=False)
    # action-only ledger (no 持仓/空仓) for readability
    if len(ledger):
        trades = ledger[~ledger["动作"].isin(["持仓", "空仓"])].copy()
        trades.to_csv(OUT / "combo_v3_trades.csv", index=False)
        trades[trades["日期"] >= "2026-01-01"].to_csv(OUT / "combo_v3_2026_trades.csv", index=False)

    meta = {
        "equity": EQUITY,
        "formula": "dual_or_final + macro*(0.12/0.35)*(0.5 if core flat) + weather_log*500k",
        "news_weight": W_NEWS,
        "news_positions_rebuilt": news_ok,
        "start": str(idx.min().date()),
        "end": str(idx.max().date()),
        "n_days": int(len(idx)),
        "n_ledger_rows": int(len(ledger)),
        "pnl_sum": round(float(combo.sum()), 2),
        "files": {
            "daily": "combo_v3_daily.csv",
            "daily_2026": "combo_v3_2026_daily.csv",
            "ledger": "combo_v3_ledger.csv",
            "trades": "combo_v3_trades.csv",
            "trades_2026": "combo_v3_2026_trades.csv",
            "latest": "combo_v3_latest.csv",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    readme = """# combo_v3 日表 / 流水

类似 `linear/final_scheme_*_daily.csv`，但覆盖 **Core–Satellite 组合 v3**，并在开平仓说明里标注策略与信号。

## 文件

| 文件 | 内容 |
|------|------|
| `combo_v3_daily.csv` | 全历史日表（盈亏分拆 + 汇总操作说明） |
| `combo_v3_2026_daily.csv` | 2026 年日表 |
| `combo_v3_ledger.csv` | 全流水（含持仓/空仓） |
| `combo_v3_trades.csv` | 仅开/平/调仓（可读性更好） |
| `combo_v3_2026_trades.csv` | 2026 开平调仓 |
| `combo_v3_latest.csv` | 最近一日 |

## 操作说明格式

`[策略|信号要点] 动作 品种…`

- **量价**：pos64×RSI 截面；BOOK≥1.18 全平
- **新闻**：研报情感截面多空 + 滞后门控；仅 Core 保证金>0 时 ×1.25 计入
- **宏观**：`usd_broad_z60` long_only thr0.25 hold5 → AU；组合内盈亏 ×0.12/0.35（Core 空仓再 ×0.5）
- **天气**：v3 四腿物候规则；研究 log×50 万代理

日收益分母：**300 万**。
"""
    (OUT / "README.md").write_text(readme)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
