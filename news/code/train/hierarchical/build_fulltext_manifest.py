#!/usr/bin/env python3
"""Build a deduplicated, untruncated OCR document manifest."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from config import DATA_ROOT, OCR_ROOT  # noqa: E402
from common import CONTRACTS_DIR, REMOVED_SYMBOLS, parse_ocr_header  # noqa: E402
from symbol_map import map_varieties  # noqa: E402


def full_body(markdown: str) -> str:
    """Match the legacy body boundary, without the 2,500-character cap."""
    lines = markdown.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("#")), 0)
    return "\n".join(lines[start:]).strip()


def fallback_date(path: Path) -> pd.Timestamp | None:
    for parent in [path.parent, *path.parents]:
        try:
            return pd.to_datetime(parent.name, format="%Y%m%d").normalize()
        except (TypeError, ValueError):
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-root", type=Path, default=OCR_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    available = {
        p.name.upper()
        for p in CONTRACTS_DIR.iterdir()
        if p.is_dir() and p.name.upper() not in REMOVED_SYMBOLS
    }
    paths = sorted(args.ocr_root.rglob("*.md"))
    if args.limit:
        paths = paths[: args.limit]

    docs: list[dict] = []
    links: list[dict] = []
    skipped = {"header": 0, "date": 0, "symbol": 0, "short": 0, "duplicate": 0}
    seen_paths: set[str] = set()
    for path in tqdm(paths, desc="full OCR manifest"):
        canonical = str(path.resolve())
        if canonical in seen_paths:
            skipped["duplicate"] += 1
            continue
        seen_paths.add(canonical)
        markdown = path.read_text(encoding="utf-8", errors="ignore")
        header = parse_ocr_header(markdown)
        if header is None:
            skipped["header"] += 1
            continue
        institution, report_date, variety = header
        report_date = report_date or fallback_date(path)
        if report_date is None:
            skipped["date"] += 1
            continue
        symbols = sorted({s for s in map_varieties(variety) if s in available})
        if not symbols:
            skipped["symbol"] += 1
            continue
        text = full_body(markdown)
        if len(text) < 20:
            skipped["short"] += 1
            continue
        doc_id = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:20]
        docs.append(
            {
                "doc_idx": len(docs),
                "doc_id": doc_id,
                "path": canonical,
                "path_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "institution": institution,
                "report_date": pd.Timestamp(report_date).normalize(),
                "variety_raw": variety,
                "text": text,
                "char_count": len(text),
            }
        )
        links.extend(
            {
                "doc_idx": len(docs) - 1,
                "doc_id": doc_id,
                "symbol": symbol,
                "report_date": pd.Timestamp(report_date).normalize(),
            }
            for symbol in symbols
        )

    doc_df = pd.DataFrame(docs)
    link_df = pd.DataFrame(links)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    doc_df.to_parquet(args.out_dir / "documents.parquet", index=False)
    link_df.to_parquet(args.out_dir / "document_symbols.parquet", index=False)
    doc_df.drop(columns=["text"], errors="ignore").to_csv(
        args.out_dir / "documents.csv", index=False
    )
    link_df.to_csv(args.out_dir / "document_symbols.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "documents": len(doc_df),
                "symbol_links": len(link_df),
                "symbols": link_df["symbol"].nunique() if len(link_df) else 0,
                **{f"skipped_{k}": v for k, v in skipped.items()},
            }
        ]
    )
    summary.to_csv(args.out_dir / "manifest_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"[done] {args.out_dir / 'documents.parquet'}")


if __name__ == "__main__":
    main()
