#!/usr/bin/env python3
"""Rule-based news factors from raw OCR markdown — no ML model.

Per document:
  rule_title — filename + heading keywords
  rule_lex   — bull/bear lexicon ratio on body
  rule_edge  — 0.5 * title + 0.5 * lex

Daily (symbol, date): L-day exp-weighted mean + log doc count.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_news_factors import _weights  # noqa: E402
from common import DATA_DIR, OCR_ROOT, extract_body_text, load_symbol_ohlc, parse_ocr_header  # noqa: E402
from symbol_map import map_varieties  # noqa: E402

BULL = (
    "上涨", "看多", "偏多", "偏强", "强势", "拉升", "走强", "反弹", "攀升", "利多",
    "供不应求", "去库", "缺口", "提涨", "涨停", "震荡偏强", "趋势上行", "震荡上行",
    "供需趋紧", "价格中枢抬升", "支撑有效",
)
BEAR = (
    "下跌", "看空", "偏空", "偏弱", "弱势", "承压", "走弱", "回调", "下挫", "利空",
    "过剩", "累库", "压制", "跌停", "震荡偏弱", "趋势下行", "震荡下行",
    "供需宽松", "价格中枢下移",
)


def first_heading(md: str) -> str:
    for line in md.splitlines():
        if line.startswith("#"):
            return re.sub(r"^#+\s*", "", line).strip()
    return ""


def title_score(stem: str, heading: str) -> float:
    s = f"{stem} {heading}"
    if any(x in s for x in ("上涨点评", "涨停", "强势拉涨", "强势延续")):
        return 1.0
    if any(x in s for x in ("下跌点评", "跌停", "大幅下挫")):
        return -1.0
    if any(x in s for x in ("上涨", "偏强", "走高", "反弹", "拉升")):
        return 0.6
    if any(x in s for x in ("下跌", "偏弱", "走低", "回调", "承压")):
        return -0.6
    if any(x in s for x in ("区间震荡", "震荡", "区间运行", "中性")):
        return 0.0
    return 0.0


def lex_score(text: str) -> float:
    nb = sum(text.count(w) for w in BULL)
    nk = sum(text.count(w) for w in BEAR)
    return float(nb - nk) / float(nb + nk + 1)


def scan_docs(ocr_root: Path, limit: int = 0) -> pd.DataFrame:
    files = sorted(ocr_root.rglob("*.md"))
    if limit > 0:
        files = files[:limit]
    rows: list[dict] = []
    for path in tqdm(files, desc="scan ocr"):
        try:
            md = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hdr = parse_ocr_header(md)
        if hdr is None:
            try:
                dt = pd.to_datetime(path.parent.name, format="%Y%m%d")
                variety, inst = "", ""
            except Exception:
                continue
        else:
            inst, dt, variety = hdr
            if dt is None:
                try:
                    dt = pd.to_datetime(path.parent.name, format="%Y%m%d")
                except Exception:
                    continue
        stem, head = path.stem, first_heading(md)
        syms = map_varieties(variety) if variety else map_varieties(stem + head)
        if not syms:
            continue
        body = extract_body_text(md, max_chars=3000)
        ts, lx = title_score(stem, head), lex_score(body)
        edge = 0.5 * ts + 0.5 * lx
        rd = pd.Timestamp(dt).normalize()
        for sym in syms:
            rows.append({
                "report_date": rd,
                "symbol": sym.upper(),
                "rule_title": ts,
                "rule_lex": lx,
                "rule_edge": edge,
            })
    return pd.DataFrame(rows)


def aggregate(docs: pd.DataFrame, col: str, lookback: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict] = []
    for sym in sorted(docs["symbol"].unique()):
        px = load_symbol_ohlc(sym)
        if px.empty:
            continue
        dates = px[(px["date"] >= start) & (px["date"] <= end)]["date"].map(
            lambda x: pd.Timestamp(x).normalize()
        )
        sub = docs[docs["symbol"] == sym].sort_values("report_date")
        if sub.empty:
            continue
        rdates = sub["report_date"].to_numpy()
        vals = sub[col].to_numpy(dtype=float)
        for T in dates:
            t0 = T - pd.Timedelta(days=lookback)
            mask = (rdates >= np.datetime64(t0)) & (rdates < np.datetime64(T))
            if not mask.any():
                continue
            ages = np.array([(T - pd.Timestamp(d)).days for d in rdates[mask]], dtype=float)
            ages = np.maximum(ages, 1.0)
            w = _weights(ages, lookback)
            rows.append({
                "date": T,
                "symbol": sym,
                col: float((vals[mask] * w).sum()),
                "rule_cnt": float(np.log1p(mask.sum())),
                "n_docs": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-root", type=Path, default=OCR_ROOT)
    ap.add_argument("--lookback", type=int, default=7)
    ap.add_argument("--start", default="2024-03-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    docs = scan_docs(args.ocr_root, limit=args.limit)
    doc_out = DATA_DIR / "rule_docs.parquet"
    docs.to_parquet(doc_out, index=False)
    print(f"docs {len(docs)} symbols={docs['symbol'].nunique()} -> {doc_out}", flush=True)

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    parts = []
    for col in ("rule_edge", "rule_title", "rule_lex"):
        f = aggregate(docs, col, args.lookback, start, end)
        parts.append(f[["date", "symbol", col, "rule_cnt", "n_docs"] if col == "rule_edge" else ["date", "symbol", col]])
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["date", "symbol"], how="outer")
    path = DATA_DIR / f"rule_factors_L{args.lookback}.parquet"
    out.to_parquet(path, index=False)
    print(f"saved {path} rows={len(out)}", flush=True)


if __name__ == "__main__":
    main()
