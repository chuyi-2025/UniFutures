#!/usr/bin/env python3
"""Shared paths and main-contract return helpers for news sentiment experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NEWS_ROOT = Path("/home/workspace/lab/UniFutures/news")
CODE_DIR = NEWS_ROOT / "code"
TRAIN_DIR = CODE_DIR / "train"
BACKTEST_DIR = CODE_DIR / "backtest"
OCR_ROOT = NEWS_ROOT / "result" / "unlimited_ocr_by_deepseek"
DATA_DIR = NEWS_ROOT / "data" / "sentiment"
RESULTS_ROOT = NEWS_ROOT / "data" / "results"
CONTRACTS_DIR = Path("/home/workspace/lab/UniFutures/data/contracts")

WEIGHTS = {
    "finance_zh": Path("/home/workspace/weights/finance-sentiment-zh-base"),
    "modernbert": Path("/home/workspace/weights/modernbert-fingpt"),
    "finbert2": Path("/home/workspace/weights/FinBERT2-large"),
}

INIT_CAP = 1_000_000.0
RET_TH = 0.001
ROLL_OFFSET = 3
REMOVED_SYMBOLS = frozenset(
    {"WR", "ZC", "RR", "RI", "JR", "LR", "WH", "PM", "FB", "BB", "BC"}
)

ZEROSHOT_START = pd.Timestamp("2024-01-01")
ZEROSHOT_END = pd.Timestamp("2026-12-31")
FT_TRAIN_START = pd.Timestamp("2024-01-01")
FT_TRAIN_END = pd.Timestamp("2025-06-30")
FT_TEST_START = pd.Timestamp("2025-07-01")
FT_TEST_END = pd.Timestamp("2026-12-31")

LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def parse_code_ym(code: str) -> tuple[int, int]:
    yy = int(str(code)[-4:-2])
    mm = int(str(code)[-2:])
    return 2000 + yy, mm


def roll_threshold(dt: pd.Timestamp) -> tuple[int, int]:
    y, m = dt.year, dt.month + ROLL_OFFSET
    while m > 12:
        m -= 12
        y += 1
    return y, m


def pick_main_contract(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    codes = work["code"].astype(str).str.upper()
    # expect ...YYMM at end; drop non-digit tails
    tail = codes.str.extract(r"(\d{3,4})$")[0]
    # YYMM (4 digits) or YMM — normalize to last 4 if possible
    def _ym(s: str) -> int:
        if not isinstance(s, str) or not s.isdigit():
            return -1
        if len(s) == 3:
            s = "0" + s
        if len(s) < 4:
            return -1
        yy = int(s[-4:-2])
        mm = int(s[-2:])
        if mm < 1 or mm > 12:
            return -1
        return (2000 + yy) * 100 + mm

    work["_ym"] = tail.map(_ym)
    work = work[work["_ym"] > 0]
    if work.empty:
        return pd.DataFrame()
    picks: list[pd.DataFrame] = []
    for dt, grp in work.groupby("date", sort=True):
        ts = pd.Timestamp(dt)
        ty, tm = ts.year, ts.month + ROLL_OFFSET
        while tm > 12:
            tm -= 12
            ty += 1
        thr = ty * 100 + tm
        cand = grp[grp["_ym"] > thr]
        if cand.empty:
            continue
        picks.append(cand.loc[[cand["_ym"].idxmin()]])
    if not picks:
        return pd.DataFrame()
    out = pd.concat(picks, ignore_index=True).sort_values("date").reset_index(drop=True)
    return out.drop(columns=["_ym"], errors="ignore")


def load_symbol_ohlc(symbol: str) -> pd.DataFrame:
    folder = CONTRACTS_DIR / symbol.upper()
    if not folder.is_dir():
        return pd.DataFrame()
    frames = []
    for csv_path in sorted(folder.glob("*.csv")):
        df = pd.read_csv(csv_path, parse_dates=["date"])
        if "code" not in df.columns:
            df["code"] = csv_path.stem
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.dropna(subset=["date", "close"]).sort_values(["date", "code"]).reset_index(drop=True)
    main = pick_main_contract(raw)
    if main.empty:
        return pd.DataFrame()
    main = main.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    close = main["close"].astype(float)
    main["log_ret_1d"] = np.log(close / close.shift(1))
    return main[["date", "code", "close", "log_ret_1d"]].copy()


def ret_to_label(ret: float, horizon: int = 1) -> str:
    """Weak label from forward return; neutral band scales with sqrt(horizon)."""
    th = RET_TH * float(np.sqrt(max(int(horizon), 1)))
    if not np.isfinite(ret) or abs(ret) < th:
        return "neutral"
    return "positive" if ret > 0 else "negative"


def normalize_ts(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t.normalize()


def forward_horizon_ret(
    df: pd.DataFrame,
    report_date: pd.Timestamp,
    horizon: int,
) -> tuple[pd.Timestamp | None, float]:
    """First session after report_date as entry; sum of next `horizon` daily log_rets.

    Returns (entry_trade_date, cumulative_log_ret). Index of df must be date.
    """
    if df.empty or horizon < 1:
        return None, float("nan")
    rd = normalize_ts(report_date)
    future = df.index[df.index > rd]
    if len(future) < horizon:
        return None, float("nan")
    window = future[:horizon]
    rets = df.loc[window, "log_ret_1d"].astype(float)
    if rets.isna().any():
        return None, float("nan")
    return pd.Timestamp(window[0]), float(rets.sum())


def label_to_pos(label: str | int) -> int:
    if isinstance(label, (int, np.integer)):
        # 3-class ids: 0 pos, 1 neu, 2 neg  OR  modernbert 0..8
        i = int(label)
        if i in (0, 1, 2) and i <= 2:
            # ambiguous; caller should pass string for 3-class
            pass
        if 0 <= i <= 8 and i > 2 or i == 4:
            # treat as 9-class if clearly in that range when using modernbert helper
            pass
    s = str(label).lower()
    if s in {"positive", "pos", "0"} and s != "0":
        if s == "positive" or s == "pos":
            return 1
    if s == "positive":
        return 1
    if s == "negative":
        return -1
    if s == "neutral":
        return 0
    return 0


def class3_to_pos(class_id: int) -> int:
    """finance-zh / finetuned 3-class: 0=pos, 1=neu, 2=neg."""
    if class_id == 0:
        return 1
    if class_id == 2:
        return -1
    return 0


def modernbert9_to_pos(class_id: int) -> int:
    """0-3 short, 4 flat, 5-8 long."""
    i = int(class_id)
    if i <= 3:
        return -1
    if i >= 5:
        return 1
    return 0


def extract_body_text(md: str, max_chars: int = 2500) -> str:
    lines = md.splitlines()
    # skip header table (first ~6 lines often)
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            start = i
            break
    body = "\n".join(lines[start:]).strip()
    body = body.replace("\n", " ")
    return body[:max_chars]


def parse_ocr_header(md: str) -> tuple[str, pd.Timestamp | None, str] | None:
    """Return (institution, date, variety_field) from markdown header table."""
    lines = md.splitlines()
    for i, line in enumerate(lines[:30]):
        if "机构" in line and "日期" in line and "品种" in line:
            for j in range(i + 1, min(i + 6, len(lines))):
                row = lines[j].strip()
                if not row.startswith("|"):
                    continue
                if set(row.replace("|", "").replace(":", "").replace("-", "").strip()) == set():
                    continue
                if "---" in row:
                    continue
                parts = [p.strip() for p in row.strip("|").split("|")]
                if len(parts) < 3:
                    continue
                inst, date_s, variety = parts[0], parts[1], parts[2]
                dt = None
                for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"):
                    try:
                        dt = pd.to_datetime(date_s, format=fmt)
                        break
                    except Exception:
                        continue
                if dt is None or pd.isna(dt):
                    try:
                        dt = pd.to_datetime(date_s, errors="coerce")
                    except Exception:
                        dt = None
                if dt is not None and not pd.isna(dt):
                    dt = pd.Timestamp(dt)
                    if dt.tzinfo is not None:
                        dt = dt.tz_localize(None)
                    dt = dt.normalize()
                else:
                    dt = None
                return inst, dt, variety
    return None
