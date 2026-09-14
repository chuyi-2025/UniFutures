#!/usr/bin/env python3
"""Batch OCR for 日报/月报/周报 datasets into flat YYYYMMDD/*.md layout."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from ocr_unlimited import (
    DOC_SUFFIXES,
    MODEL_PATH,
    build_output_path,
    extract_date_key,
    load_model,
    ocr_file_to_text,
)

DEFAULT_SRC = Path("/home/workspace/datasets/日报 月报 周报")
DEFAULT_OUT = Path("/home/workspace/lab/UniFutures/news/result/unlimited_ocr")
DEFAULT_LOG = DEFAULT_OUT / "batch_progress.jsonl"


def iter_source_files(src_root: Path) -> list[Path]:
    files = [
        p for p in src_root.rglob("*")
        if p.is_file() and p.suffix.lower() in DOC_SUFFIXES
    ]
    files.sort()
    return files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch Unlimited-OCR for research reports")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="source root directory")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output root directory")
    p.add_argument("--model", type=Path, default=MODEL_PATH, help="local model dir")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--mode", choices=("gundam", "base"), default="gundam")
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--limit", type=int, default=0, help="process at most N files (0=all)")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--log", type=Path, default=DEFAULT_LOG, help="jsonl progress log")
    p.add_argument("--files", type=Path, default=None, help="optional text file of input paths")
    return p.parse_args()


def append_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.files is not None:
        inputs = [Path(line.strip()) for line in args.files.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        inputs = iter_source_files(args.src)

    planned: list[tuple[Path, Path]] = []
    skipped_no_date: list[Path] = []
    skipped_exists: list[Path] = []

    for src in inputs:
        out_path = build_output_path(src, args.out)
        if out_path is None:
            skipped_no_date.append(src)
            continue
        if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            skipped_exists.append(src)
            continue
        planned.append((src, out_path))

    if args.limit > 0:
        planned = planned[: args.limit]

    print(f"[scan] total inputs={len(inputs)}")
    print(f"[scan] to_process={len(planned)} skip_existing={len(skipped_exists)} no_date={len(skipped_no_date)}")
    if skipped_no_date:
        print(f"[warn] first no-date file: {skipped_no_date[0]}")
    if not planned:
        print("[done] nothing to process")
        return 0

    print(f"[load] model={args.model}")
    tokenizer, model = load_model(args.model)

    ok = 0
    failed = 0
    for idx, (src, out_path) in enumerate(planned, start=1):
        date_key = extract_date_key(src)
        print(f"[{idx}/{len(planned)}] {date_key} {src.name}")
        try:
            text = ocr_file_to_text(
                src,
                tokenizer,
                model,
                dpi=args.dpi,
                max_length=args.max_length,
                mode=args.mode,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            ok += 1
            append_log(
                args.log,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "status": "ok",
                    "src": str(src),
                    "out": str(out_path),
                    "date_key": date_key,
                },
            )
            print(f"  -> {out_path}")
        except Exception as exc:
            failed += 1
            append_log(
                args.log,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "status": "error",
                    "src": str(src),
                    "out": str(out_path),
                    "date_key": date_key,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"  !! failed: {exc}", file=sys.stderr)

    print(f"[summary] ok={ok} failed={failed} skipped_existing={len(skipped_exists)} no_date={len(skipped_no_date)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
