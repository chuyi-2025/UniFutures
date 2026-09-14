#!/usr/bin/env python3
"""Unlimited-OCR: PDF / image document parsing for futures research reports."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = Path("/home/workspace/weights/Unlimited-OCR")
DEFAULT_PDF = Path(
    "/home/workspace/datasets/futures_text/期货研报/日报 月报 周报"
    "/2026年/7月/7.16/收评/有色/碳酸锂产业日报20260716(1).pdf"
)
DEFAULT_OUTPUT = Path("/home/workspace/lab/UniFutures/news/output/ocr")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DOC_SUFFIXES = {".pdf", *IMAGE_SUFFIXES}


def pdf_to_images(pdf_path: str | Path, dpi: int = 300) -> tuple[list[str], str]:
    """Rasterize each PDF page to PNG under a temp directory."""
    doc = fitz.open(str(pdf_path))
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths: list[str] = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths, tmp_dir


def load_model(model_path: str | Path):
    model_path = str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_safetensors=True,
        dtype=torch.bfloat16,
    )
    model = model.eval().cuda()
    return tokenizer, model


def _read_result_md(work_dir: Path) -> str:
    result_md = work_dir / "result.md"
    if not result_md.exists():
        return ""
    return result_md.read_text(encoding="utf-8")


def ocr_pdf_to_text(
    pdf_path: str | Path,
    tokenizer,
    model,
    dpi: int = 300,
    max_length: int = 32768,
    mode: str = "gundam",
    work_dir: str | Path | None = None,
) -> str:
    """Run OCR on a PDF and return markdown text."""
    pdf_path = Path(pdf_path)
    image_files, tmp_dir = pdf_to_images(pdf_path, dpi=dpi)
    cleanup_tmp = work_dir is None
    work_dir = Path(work_dir) if work_dir is not None else Path(tmp_dir) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if mode == "gundam":
            parts: list[str] = []
            for i, image_file in enumerate(image_files):
                page_out = work_dir / f"page_{i + 1:04d}"
                page_out.mkdir(parents=True, exist_ok=True)
                model.infer(
                    tokenizer,
                    prompt="<image>document parsing.",
                    image_file=image_file,
                    output_path=str(page_out),
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    max_length=max_length,
                    no_repeat_ngram_size=35,
                    ngram_window=128,
                    save_results=True,
                )
                parts.append(_read_result_md(page_out))
            if len(parts) == 1:
                return parts[0]
            return "<PAGE>\n" + "\n<PAGE>\n".join(parts)

        page_out = work_dir / "multi"
        page_out.mkdir(parents=True, exist_ok=True)
        model.infer_multi(
            tokenizer,
            prompt="<image>Multi page parsing.",
            image_files=image_files,
            output_path=str(page_out),
            image_size=1024,
            max_length=max_length,
            no_repeat_ngram_size=35,
            ngram_window=1024,
            save_results=True,
        )
        return _read_result_md(page_out)
    finally:
        if cleanup_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def ocr_image_to_text(
    image_path: str | Path,
    tokenizer,
    model,
    mode: str = "gundam",
    max_length: int = 32768,
    work_dir: str | Path | None = None,
) -> str:
    """Run OCR on a single image and return markdown text."""
    image_path = Path(image_path)
    if mode == "gundam":
        base_size, image_size, crop_mode = 1024, 640, True
        ngram_window = 128
    elif mode == "base":
        base_size, image_size, crop_mode = 1024, 1024, False
        ngram_window = 128
    else:
        raise ValueError(f"unknown mode={mode!r}, expect gundam|base")

    work_dir = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="img_ocr_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    model.infer(
        tokenizer,
        prompt="<image>document parsing.",
        image_file=str(image_path),
        output_path=str(work_dir),
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        max_length=max_length,
        no_repeat_ngram_size=35,
        ngram_window=ngram_window,
        save_results=True,
    )
    return _read_result_md(work_dir)


def ocr_file_to_text(
    input_path: str | Path,
    tokenizer,
    model,
    dpi: int = 300,
    max_length: int = 32768,
    mode: str = "gundam",
) -> str:
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        return ocr_pdf_to_text(
            input_path,
            tokenizer,
            model,
            dpi=dpi,
            max_length=max_length,
            mode=mode,
        )
    if suffix in IMAGE_SUFFIXES:
        return ocr_image_to_text(
            input_path,
            tokenizer,
            model,
            mode=mode,
            max_length=max_length,
        )
    raise ValueError(f"unsupported file type: {suffix}")


def extract_date_key(path: Path) -> str | None:
    """Map source path to YYYYMMDD output directory name."""
    s = str(path)
    m = re.search(r"/(\d{4})年/(\d{1,2})月/[^/]*?(\d{1,2})\.(\d{1,2})", s)
    if m:
        year, month, _, day = m.groups()
        return f"{int(year):04d}{int(month):02d}{int(day):02d}"

    m = re.search(r"(20\d{6})", path.stem)
    if m:
        return m.group(1)

    m = re.search(r"/(\d{4})年/(\d{1,2})月/", s)
    if m:
        year, month = m.groups()
        return f"{int(year):04d}{int(month):02d}01"

    m = re.search(r"/(\d{4})年/", s)
    if m:
        year = int(m.group(1))
        if "四季度" in s:
            return f"{year}1001"
        if "三季度" in s:
            return f"{year}0701"
        if "二季度" in s:
            return f"{year}0401"
        if "一季度" in s:
            return f"{year}0101"
        return f"{year}0101"

    return None


def build_output_path(input_path: Path, output_root: Path) -> Path | None:
    date_key = extract_date_key(input_path)
    if date_key is None:
        return None
    return output_root / date_key / f"{input_path.stem}.md"


def ocr_pdf(
    pdf_path: str | Path,
    output_path: str | Path,
    model_path: str | Path = MODEL_PATH,
    dpi: int = 300,
    max_length: int = 32768,
    mode: str = "gundam",
):
    """PDF OCR. gundam=per-page crop; base/multi=infer_multi image_size=1024."""
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[load] model={model_path}")
    tokenizer, model = load_model(model_path)

    print(f"[pdf] {pdf_path} -> mode={mode}")
    text = ocr_pdf_to_text(
        pdf_path,
        tokenizer,
        model,
        dpi=dpi,
        max_length=max_length,
        mode=mode,
        work_dir=output_path,
    )
    result_md = output_path / "result.md"
    result_md.write_text(text, encoding="utf-8")
    print(f"[done] saved: {result_md}")
    return text


def ocr_image(
    image_path: str | Path,
    output_path: str | Path,
    model_path: str | Path = MODEL_PATH,
    mode: str = "gundam",
    max_length: int = 32768,
):
    """Single-image OCR. mode: gundam (crop) or base."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[load] model={model_path}")
    tokenizer, model = load_model(model_path)

    print(f"[infer] single image mode={mode} -> {output_path}")
    text = ocr_image_to_text(
        image_path,
        tokenizer,
        model,
        mode=mode,
        max_length=max_length,
        work_dir=output_path,
    )
    result_md = output_path / "result.md"
    result_md.write_text(text, encoding="utf-8")
    print(f"[done] saved: {result_md}")
    return text


def parse_args():
    p = argparse.ArgumentParser(description="Unlimited-OCR for futures research PDFs")
    p.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="input PDF path")
    p.add_argument("--image", type=Path, default=None, help="optional single image")
    p.add_argument("--output", type=Path, default=None, help="output directory")
    p.add_argument("--model", type=Path, default=MODEL_PATH, help="local model dir")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--mode", choices=("gundam", "base"), default="gundam",
                   help="gundam=crop; base=full-page / multi-page")
    p.add_argument("--max-length", type=int, default=32768)
    return p.parse_args()


def main():
    args = parse_args()
    stem = (args.image or args.pdf).stem
    suffix = f"_{args.mode}" if args.mode != "base" else "_base"
    output = args.output or (DEFAULT_OUTPUT / f"{stem}{suffix}")

    if args.image is not None:
        ocr_image(
            args.image,
            output,
            model_path=args.model,
            mode=args.mode,
            max_length=args.max_length,
        )
    else:
        if not args.pdf.exists():
            raise FileNotFoundError(args.pdf)
        ocr_pdf(
            args.pdf,
            output,
            model_path=args.model,
            dpi=args.dpi,
            max_length=args.max_length,
            mode=args.mode,
        )


if __name__ == "__main__":
    main()
