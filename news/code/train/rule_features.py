#!/usr/bin/env python3
"""Rich rule features per OCR document (no ML)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from build_rule_factors import BULL, BEAR, first_heading, lex_score, title_score  # noqa: F401
from common import OCR_ROOT, extract_body_text, parse_ocr_header
from symbol_map import map_varieties

# 前瞻/策略用语（偏预测）
FWD_BULL = (
    "有望", "预计", "或将", "大概率", "上行空间", "看涨", "做多", "供需趋紧", "去库",
    "库存下降", "需求改善", "供应偏紧", "偏强运行", "震荡偏强", "趋势上行", "价格中枢上移",
    "建议关注", "维持偏多", "谨慎偏多",
)
FWD_BEAR = (
    "或承压", "下行风险", "看跌", "做空", "累库", "库存上升", "供应过剩", "需求走弱",
    "供需宽松", "偏弱运行", "震荡偏弱", "趋势下行", "价格中枢下移", "谨慎偏空", "维持偏空",
)
# 事后描述（点评常用）
BACKWARD = (
    "今日", "收涨", "收跌", "收盘", "涨幅", "跌幅", "涨停", "跌停", "盘中", "截至收盘",
    "上涨点评", "下跌点评", "异动快评", "快评",
)
SUPPLY_BEAR = ("供应增加", "复产", "开工回升", "产量上升", "进口增加", "到港增加", "投放增加")
DEMAND_BULL = ("需求改善", "需求回升", "采购增加", "补库", "冬储", "旺季", "消费好转")


def report_kind(stem: str, heading: str) -> str:
    s = f"{stem} {heading}"
    if any(x in s for x in ("上涨点评", "下跌点评", "异动快评", "快评")):
        return "dianping"
    if any(x in s for x in ("策略报告", "策略", "月报", "季报", "展望", "专题", "二季度", "三季度", "四季度")):
        return "strategy"
    if any(x in s for x in ("早评", "日评", "晨报", "日报")):
        return "daily"
    if "周报" in s:
        return "weekly"
    return "other"


def extract_section(md: str, title: str) -> str:
    lines = md.splitlines()
    start = None
    for i, line in enumerate(lines):
        if title in line and line.strip().startswith("#"):
            start = i + 1
            break
    if start is None:
        return ""
    out = []
    for line in lines[start:]:
        if line.strip().startswith("#"):
            break
        out.append(line)
    return " ".join(out)[:2000]


def word_score(text: str, pos: tuple[str, ...], neg: tuple[str, ...]) -> float:
    nb = sum(text.count(w) for w in pos)
    nk = sum(text.count(w) for w in neg)
    return float(nb - nk) / float(nb + nk + 1)


def forward_lex(text: str) -> float:
    return word_score(text, FWD_BULL, FWD_BEAR)


def supply_demand_lex(text: str) -> float:
    """需求利好 − 供应利空 的粗略净方向。"""
    d = sum(text.count(w) for w in DEMAND_BULL)
    s = sum(text.count(w) for w in SUPPLY_BEAR)
    return float(d - s) / float(d + s + 1)


def backward_ratio(text: str) -> float:
    n = sum(text.count(w) for w in BACKWARD)
    return float(n) / float(len(text) / 80 + 1)


def scan_rich_docs(ocr_root: Path = OCR_ROOT, limit: int = 0) -> pd.DataFrame:
    files = sorted(ocr_root.rglob("*.md"))
    if limit > 0:
        files = files[:limit]
    rows: list[dict] = []
    for path in files:
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
        body = extract_body_text(md, max_chars=4000)
        core = extract_section(md, "核心观点") or extract_section(md, "结论与展望") or body[:800]
        kind = report_kind(stem, head)
        ts = title_score(stem, head)
        lx = lex_score(body)
        flx = forward_lex(core if len(core) > 50 else body)
        sdl = supply_demand_lex(body)
        br = backward_ratio(stem + head + body[:500])
        edge = 0.5 * ts + 0.5 * lx
        rd = pd.Timestamp(dt).normalize()
        for sym in syms:
            rows.append({
                "path": str(path),
                "report_date": rd,
                "symbol": sym.upper(),
                "kind": kind,
                "rule_title": ts,
                "rule_lex": lx,
                "rule_edge": edge,
                "fwd_lex": flx,
                "core_lex": lex_score(core),
                "sd_lex": sdl,
                "back_ratio": br,
            })
    return pd.DataFrame(rows)
