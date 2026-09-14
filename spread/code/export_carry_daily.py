#!/usr/bin/env python3
"""Export speculator carry book daily CSV (same-product near/far only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

import yaml

from common import CONFIG_DIR, DATA_DIR, RESULT_DIR, ensure_dirs  # noqa: E402
from futures_lot_specs import lot_fee, lot_margin, lot_value  # noqa: E402
from search_spread_strategies import hold_signal, max_dd, sharpe  # noqa: E402

PANEL_DIR = DATA_DIR / "panels_speculator"
BEST_PATH = RESULT_DIR / "search_speculator" / "spread_search_best.csv"
BOOK_CFG = CONFIG_DIR / "carry_book.yaml"
OUT_DIR = RESULT_DIR / "daily"
MIRROR_DIR = Path("/home/workspace/lab/UniFutures/data/infer/results/daily/spread")
DEFAULT_BOOK = "carry4"
CARRY_LOTS = 1
BOOK_CAPITAL = 1_000_000.0

CN = {
    "AU": "黄金", "AG": "白银", "RB": "螺纹钢", "HC": "热卷",
    "M": "豆粕", "Y": "豆油", "P": "棕榈油",
}


def yuan(x: float | None) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return round(float(x), 2)


def fmt_px(px: float | None) -> str:
    if px is None or not np.isfinite(px):
        return ""
    s = f"{float(px):.2f}".rstrip("0").rstrip(".")
    return s


def contract_ym(code: str | None) -> str:
    if not code or not isinstance(code, str):
        return ""
    digits = "".join(ch for ch in code if ch.isdigit())
    return digits[-4:] if len(digits) >= 4 else digits


def cn_contract(sym: str, code: str | None) -> str:
    ym = contract_ym(code)
    name = CN.get(sym.upper(), sym.upper())
    return f"{name}{ym}" if ym else name


def leg_name(sym: str, code: str, lots: int, side: str, px: float) -> str:
    """e.g. 热卷2605 1手(多)@3850"""
    return f"{cn_contract(sym, code)} {lots}手({side})@{fmt_px(px)}"


def spread_fee_row(sym: str, near_px: float, far_px: float, *, opening: bool) -> float:
    kind = "open" if opening else "close_prev"
    try:
        return float(lot_fee(sym, near_px, kind, CARRY_LOTS)) + float(lot_fee(sym, far_px, kind, CARRY_LOTS))
    except Exception:
        return 0.0


def leg_margins(sym: str, near_px: float, far_px: float) -> tuple[float, float, float]:
    m1, _ = lot_margin(sym, near_px, broker=True)
    m2, _ = lot_margin(sym, far_px, broker=True)
    m1 = float(m1 or 0.0) * CARRY_LOTS
    m2 = float(m2 or 0.0) * CARRY_LOTS
    return m1, m2, m1 + m2


def position_text(row: pd.Series) -> str:
    sig = float(row["signal"])
    if sig == 0:
        return ""
    sym = str(row["symbol"])
    near_px = float(row["near_close"])
    far_px = float(row["far_close"])
    near_code = str(row["near_code"])
    far_code = str(row["far_code"])
    spread = float(row["spread_px"])
    if sig > 0:
        near = leg_name(sym, near_code, CARRY_LOTS, "多", near_px)
        far = leg_name(sym, far_code, CARRY_LOTS, "空", far_px)
        tag = "多carry"
    else:
        near = leg_name(sym, near_code, CARRY_LOTS, "空", near_px)
        far = leg_name(sym, far_code, CARRY_LOTS, "多", far_px)
        tag = "空carry"
    return f"{tag} {near} / {far} 价差={fmt_px(spread)}"


def run_scheme_daily(
    df: pd.DataFrame,
    *,
    factor: str,
    mode: str,
    thr: float,
    direction: float,
    hold: int,
    mean_revert: bool,
    symbol: str,
) -> pd.DataFrame | None:
    d = df.dropna(subset=[factor]).copy()
    if len(d) < 80:
        return None
    x = d[factor].to_numpy(dtype=float)
    raw = np.zeros(len(x))
    if mode == "sign":
        if mean_revert:
            raw = np.where(np.abs(x) >= thr, -np.sign(x), 0.0)
        else:
            raw = np.where(np.abs(x) >= thr, np.sign(x) * direction, 0.0)
    elif mode == "long_only":
        raw = np.where(x * direction >= thr, direction, 0.0)
    else:
        return None
    sig = hold_signal(raw, hold) if hold > 1 else raw
    d["signal"] = sig
    d["factor_z"] = d[factor]

    pnl_col = "pnl_long_spread" if "pnl_long_spread" in d.columns else "pnl_long_carry"
    prev = 0.0
    rows_meta = []
    for _, r in d.iterrows():
        cur = float(r["signal"])
        near_px = float(r["near_close"])
        far_px = float(r["far_close"])
        m_near, m_far, m_tot = leg_margins(symbol, near_px, far_px) if cur != 0 else (0.0, 0.0, 0.0)
        gross = cur * float(r[pnl_col]) if cur != 0 and np.isfinite(r[pnl_col]) else 0.0
        fee = 0.0
        if cur != prev:
            if prev != 0:
                fee += spread_fee_row(symbol, near_px, far_px, opening=False)
            if cur != 0:
                fee += spread_fee_row(symbol, near_px, far_px, opening=True)
        v_near = (lot_value(symbol, near_px) or 0.0) * CARRY_LOTS
        v_far = (lot_value(symbol, far_px) or 0.0) * CARRY_LOTS
        rows_meta.append({
            "gross_pnl": gross,
            "fee": fee,
            "pnl": gross - fee,
            "margin_near": m_near,
            "margin_far": m_far,
            "margin": m_tot,
            "lots_near": CARRY_LOTS if cur != 0 else 0,
            "lots_far": CARRY_LOTS if cur != 0 else 0,
            "lots_total": CARRY_LOTS * 2 if cur != 0 else 0,
            "value_near": v_near,
            "value_far": v_far,
            "value_total": v_near + v_far,
        })
        prev = cur

    meta = pd.DataFrame(rows_meta)
    d = pd.concat([d.reset_index(drop=True), meta], axis=1)
    d["ret"] = d["pnl"] / BOOK_CAPITAL
    d["symbol"] = symbol
    d["持仓"] = d.apply(position_text, axis=1)
    return d


def load_book_cfg(book_id: str) -> dict:
    cfg = yaml.safe_load(BOOK_CFG.read_text(encoding="utf-8"))
    book = cfg.get("books", {}).get(book_id)
    if not book:
        raise SystemExit(f"unknown book {book_id!r} in {BOOK_CFG}")
    return book


def build_book(
    start: pd.Timestamp,
    *,
    symbols: list[str],
    book_id: str,
    lots_per_leg: int,
    capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    global CARRY_LOTS, BOOK_CAPITAL  # noqa: PLW0603
    CARRY_LOTS = lots_per_leg
    BOOK_CAPITAL = capital

    best = pd.read_csv(BEST_PATH)
    best = best[best["kind"] == "carry"].copy()
    sym_set = {s.upper() for s in symbols}
    best["symbol"] = best["spread_id"].str.replace("_carry1", "", regex=False).str.upper()
    best = best[best["symbol"].isin(sym_set)].copy()
    if best.empty:
        raise SystemExit(f"no carry schemes for {sorted(sym_set)} in {BEST_PATH}")

    leg_parts: list[pd.DataFrame] = []
    meta_rows: list[dict] = []

    for _, spec in best.iterrows():
        sid = str(spec["spread_id"])
        sym = str(spec["symbol"])
        path = PANEL_DIR / f"carry_{sym.lower()}.parquet"
        if not path.exists():
            print(f"skip missing {path}", flush=True)
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        daily = run_scheme_daily(
            df,
            factor=str(spec["factor"]),
            mode=str(spec["mode"]),
            thr=float(spec["thr"]),
            direction=float(spec["direction"]),
            hold=int(spec["hold"]),
            mean_revert=bool(spec["mean_revert"]),
            symbol=sym,
        )
        if daily is None or daily.empty:
            continue
        daily = daily[daily["date"] >= start].copy()
        daily["spread_id"] = sid
        leg_parts.append(daily)
        meta_rows.append({
            "symbol": sym,
            "spread_id": sid,
            "factor": spec["factor"],
            "thr": float(spec["thr"]),
            "hold": int(spec["hold"]),
            "mean_revert": bool(spec["mean_revert"]),
            "sharpe": float(spec["sharpe"]),
            "return": float(spec["return"]),
            "lots_per_leg": CARRY_LOTS,
        })

    if not leg_parts:
        raise SystemExit("no carry daily legs built")

    legs = pd.concat(leg_parts, ignore_index=True).sort_values(["date", "symbol"])

    sym_cols = sorted(legs["symbol"].unique())
    dates = pd.DatetimeIndex(sorted(legs["date"].unique()))
    book = pd.DataFrame({"date": dates})

    for sym in sym_cols:
        sub = legs[legs["symbol"] == sym][["date", "pnl", "margin", "fee", "lots_total", "持仓"]].copy()
        book = book.merge(
            sub.rename(columns={
                "pnl": f"{sym}_pnl",
                "margin": f"保证金_{sym}",
                "fee": f"{sym}_fee",
                "lots_total": f"手数_{sym}",
                "持仓": f"持仓_{sym}",
            }),
            on="date", how="left",
        )

    pnl_cols = [f"{s}_pnl" for s in sym_cols]
    margin_cols = [f"保证金_{s}" for s in sym_cols]
    pos_cols = [f"持仓_{s}" for s in sym_cols]
    book[pnl_cols] = book[pnl_cols].fillna(0.0)
    book[margin_cols] = book[margin_cols].fillna(0.0)
    book["fee_total"] = book[[f"{s}_fee" for s in sym_cols]].fillna(0.0).sum(axis=1)
    book["pnl_eq"] = book[pnl_cols].mean(axis=1)
    book["margin_total"] = book[margin_cols].sum(axis=1)
    book["lots_total"] = book[[f"手数_{s}" for s in sym_cols]].fillna(0).sum(axis=1).astype(int)
    book["ret_eq"] = book["pnl_eq"] / BOOK_CAPITAL
    lot_cols = [f"手数_{s}" for s in sym_cols]
    book["active_symbols"] = (book[lot_cols].fillna(0) > 0).sum(axis=1)

    zh = book.copy()
    zh["日期"] = zh["date"].dt.strftime("%Y-%m-%d")
    zh["是否空仓"] = zh["lots_total"] == 0
    zh["账户日盈亏"] = zh["pnl_eq"].round(2)
    zh["账户日收益"] = zh["ret_eq"]
    zh["手续费"] = zh["fee_total"].round(2)
    zh["合计保证金"] = zh["margin_total"].round(2)
    zh["合计手数"] = zh["lots_total"]
    zh["活跃品种数"] = book["active_symbols"]

    def day_ops(g: pd.DataFrame) -> str:
        parts = []
        for r in g.itertuples():
            if float(r.signal) == 0:
                continue
            parts.append(f"[{r.symbol}] {r.持仓} z={float(r.factor_z):.2f}")
        return " | ".join(parts) if parts else "全品种空仓"

    op_map = legs.groupby("date").apply(day_ops, include_groups=False)
    zh["操作说明"] = zh["date"].map(op_map).fillna("全品种空仓")

    for sym in sym_cols:
        zh[f"{sym}盈亏"] = zh[f"{sym}_pnl"].round(2)

    zh_cols = [
        "日期", "是否空仓", "活跃品种数", "合计手数",
        *[f"{s}盈亏" for s in sym_cols],
        *margin_cols,
        "账户日盈亏", "手续费", "账户日收益", "合计保证金",
        *pos_cols,
        "操作说明",
    ]
    zh = zh[zh_cols]

    legs_export = legs[[
        "date", "symbol", "spread_id", "signal", "factor_z",
        "near_code", "far_code", "near_close", "far_close", "spread_px",
        "lots_near", "lots_far", "lots_total",
        "margin_near", "margin_far", "margin",
        "value_near", "value_far", "value_total",
        "gross_pnl", "fee", "pnl", "ret", "持仓",
    ]].copy()
    legs_export = legs_export.rename(columns={
        "near_close": "近月点位",
        "far_close": "远月点位",
        "spread_px": "价差",
        "lots_near": "手数_近",
        "lots_far": "手数_远",
        "lots_total": "手数_合计",
        "margin_near": "近月保证金",
        "margin_far": "远月保证金",
        "margin": "合计保证金",
        "value_near": "近月合约价值",
        "value_far": "远月合约价值",
        "value_total": "合约价值合计",
        "gross_pnl": "毛盈亏",
        "fee": "手续费",
        "pnl": "盈亏",
        "ret": "日收益",
        "factor_z": "z值",
        "near_code": "近月合约",
        "far_code": "远月合约",
        "signal": "信号",
    })
    legs_export["日期"] = legs_export["date"].dt.strftime("%Y-%m-%d")
    legs_export["信号说明"] = legs_export["信号"].map({1.0: "多carry", -1.0: "空carry", 0.0: "空仓"})

    # English wide book (compact)
    eng = book[["date", *pnl_cols, "pnl_eq", "margin_total", "lots_total", "fee_total", "ret_eq"]].copy()
    eng = eng.rename(columns={"margin_total": "margin_total", "fee_total": "fee_total", "lots_total": "lots_total"})

    summary = {
        "book_id": book_id,
        "mode": "speculator_carry",
        "start": str(start.date()),
        "capital": BOOK_CAPITAL,
        "lots_per_leg": CARRY_LOTS,
        "symbols": sym_cols,
        "schemes": meta_rows,
        "book_sharpe": sharpe(book["ret_eq"]),
        "book_return": float(book["ret_eq"].sum()),
        "book_max_dd": max_dd(book["ret_eq"]),
        "book_days": len(book),
        "avg_margin_when_active": float(book.loc[book["margin_total"] > 0, "margin_total"].mean()),
    }
    return eng, zh, legs_export, legs, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", default=DEFAULT_BOOK, help="carry_book.yaml book id (default carry4)")
    parser.add_argument("--start", default=None, help="override book start date")
    args = parser.parse_args()
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)

    bcfg = load_book_cfg(args.book)
    start = pd.Timestamp(args.start or bcfg.get("start", "2018-01-01"))
    symbols = [str(s).upper() for s in bcfg["symbols"]]
    lots = int(bcfg.get("lots_per_leg", 1))
    capital = float(bcfg.get("capital", 1_000_000.0))

    eng, zh, legs_out, _, summary = build_book(
        start, symbols=symbols, book_id=args.book, lots_per_leg=lots, capital=capital,
    )
    summary["label"] = bcfg.get("label", args.book)

    prefix = f"{args.book}_"
    book_path = OUT_DIR / f"{prefix}book_daily.csv"
    zh_path = OUT_DIR / f"{prefix}book_daily_zh.csv"
    legs_path = OUT_DIR / f"{prefix}legs_daily.csv"
    summary_path = OUT_DIR / f"{prefix}book_summary.json"

    eng.to_csv(book_path, index=False)
    zh.to_csv(zh_path, index=False)
    legs_out.to_csv(legs_path, index=False)

    for src, name in [
        (book_path, f"{prefix}book_daily.csv"),
        (zh_path, f"{prefix}book_daily_zh.csv"),
        (legs_path, f"{prefix}legs_daily.csv"),
    ]:
        (MIRROR_DIR / name).write_bytes(src.read_bytes())

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
    )
    (MIRROR_DIR / summary_path.name).write_bytes(summary_path.read_bytes())

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str), flush=True)
    print(f"\nWrote {zh_path}", flush=True)
    print(f"Mirror -> {MIRROR_DIR}", flush=True)


if __name__ == "__main__":
    main()
