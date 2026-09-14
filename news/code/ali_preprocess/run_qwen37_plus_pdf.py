#!/usr/bin/env python3
"""Call Qwen3.7-Plus multimodal on a PDF: rasterize each page, then analyze page-by-page."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import shutil
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

# From engineai_deployment .../llm/config.py (ali_omni)
API_KEY = (
    "sk-ws-H.RXPDRME.2SDp.MEYCIQDaeu81UzW85Dw8zzesS4hzZXjVNA_Cx6Ce8E-RwyxjOwIhALOUfOYTsWLBMRBrl-aKxkcJcMRl5t-mDQ6Gb_pbV-iK"
)
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.7-plus"

DEFAULT_PDF = Path(
    "/home/workspace/lab/UniFutures/news/data/2024年国庆期间部分品种现货情况.pdf"
)
DEFAULT_OUTPUT = Path(
    "/home/workspace/lab/UniFutures/news/result/ali_qwen37_plus"
)
DEFAULT_PROMPT = (
    "这是一份期货/现货研报的某一页扫描图。请用中文完整提取并整理本页文字、表格与关键结论；"
    "保持原有层级结构；表格用 Markdown 表格呈现；不要编造页面中不存在的信息。"
)


def encode_local_image(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None:
        mime = "image/png"
    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 200) -> list[Path]:
    """Rasterize each PDF page to PNG via pdftoppm (poppler)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    cmd = [
        "pdftoppm",
        "-png",
        "-r",
        str(dpi),
        str(pdf_path),
        str(prefix),
    ]
    subprocess.run(cmd, check=True)
    paths = sorted(out_dir.glob("page-*.png"))
    if not paths:
        raise RuntimeError(f"pdftoppm produced no images under {out_dir}")
    return paths


def ask_vl(
    prompt: str,
    image: str | Path,
    *,
    enable_thinking: bool = False,
    max_tokens: int = 4096,
) -> str:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    image_path = Path(image)
    image_url = encode_local_image(image_path) if image_path.is_file() else str(image)

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=max_tokens,
        extra_body={"enable_thinking": enable_thinking},
    )
    return completion.choices[0].message.content or ""


def analyze_pdf(
    pdf_path: Path,
    prompt: str,
    *,
    dpi: int = 200,
    enable_thinking: bool = False,
    max_tokens: int = 4096,
    max_pages: int | None = None,
    image_dir: Path | None = None,
) -> tuple[str, list[Path]]:
    cleanup_images = image_dir is None
    tmp_dir: str | None = None
    if image_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="qwen37_pdf_")
        image_dir = Path(tmp_dir)
    else:
        image_dir.mkdir(parents=True, exist_ok=True)

    try:
        page_images = pdf_to_images(pdf_path, image_dir, dpi=dpi)
        if max_pages is not None:
            page_images = page_images[: max(0, max_pages)]

        parts: list[str] = []
        for i, img in enumerate(page_images, start=1):
            print(f"[page {i}/{len(page_images)}] analyzing {img.name} ...", flush=True)
            page_prompt = f"当前为第 {i}/{len(page_images)} 页。\n{prompt}"
            text = ask_vl(
                page_prompt,
                img,
                enable_thinking=enable_thinking,
                max_tokens=max_tokens,
            )
            parts.append(f"<PAGE>\n{text.strip()}\n")

        return "\n".join(parts).rstrip() + "\n", page_images
    finally:
        if cleanup_images and tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3.7-Plus vision: PDF pages -> images -> text analysis"
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="input PDF path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="text instruction per page")
    parser.add_argument("--dpi", type=int, default=200, help="rasterize DPI")
    parser.add_argument("--thinking", action="store_true", help="enable thinking mode")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="only analyze the first N pages (debug)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="keep page PNGs under this directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write combined markdown result (default: result/ali_qwen37_plus/<stem>.md)",
    )
    args = parser.parse_args()

    pdf_path = args.pdf.expanduser().resolve()
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_path = args.output
    if out_path is None:
        out_path = DEFAULT_OUTPUT / f"{pdf_path.stem}.md"
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"model={MODEL}")
    print(f"base_url={BASE_URL}")
    print(f"pdf={pdf_path}")
    print(f"dpi={args.dpi}")
    print(f"prompt={args.prompt}")
    print(f"output={out_path}")
    print("-" * 60)

    text, page_images = analyze_pdf(
        pdf_path,
        args.prompt,
        dpi=args.dpi,
        enable_thinking=args.thinking,
        max_tokens=args.max_tokens,
        max_pages=args.max_pages,
        image_dir=args.image_dir,
    )

    out_path.write_text(text, encoding="utf-8")
    print("-" * 60)
    print(text)
    print("-" * 60)
    print(f"pages={len(page_images)} saved -> {out_path}")


if __name__ == "__main__":
    main()
