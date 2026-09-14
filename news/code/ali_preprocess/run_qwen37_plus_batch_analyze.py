#!/usr/bin/env python3
"""Analyze a full document (PDF / image / text) with Qwen3.7-Plus via Batch Chat.

- PDF: rasterize all pages -> send as multi-image in ONE request
- Image / text: send directly
- Uses Batch endpoint (≈50% off realtime)
- enable_thinking=True by default
- Output: Markdown table of 品种/预测/总结/分析 (not full OCR)
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from openai import OpenAI

# From engineai_deployment .../llm/config.py (ali_omni)
API_KEY = (
    "sk-ws-H.RXPDRME.2SDp.MEYCIQDaeu81UzW85Dw8zzesS4hzZXjVNA_Cx6Ce8E-RwyxjOwIhALOUfOYTsWLBMRBrl-aKxkcJcMRl5t-mDQ6Gb_pbV-iK"
)
# Batch Chat: same call style, ~50% price of realtime
BATCH_BASE_URL = "https://batch.dashscope.aliyuncs.com/compatible-mode/v1"
REALTIME_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.7-plus"

# Batch Chat context cap for qwen3.7-plus is 256K tokens
BATCH_CONTEXT_LIMIT = 256_000

DEFAULT_INPUT = Path(
    "/home/workspace/lab/UniFutures/news/data/2024年国庆期间部分品种现货情况.pdf"
)
DEFAULT_OUTPUT = Path(
    "/home/workspace/lab/UniFutures/news/result/ali_qwen37_plus_batch"
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json"}

DEFAULT_PROMPT = """请分析我上传的这份文件（PDF/图片/文字均可），提取其中涉及的所有商品/品种，并按以下要求输出一个 Markdown 表格：

表格列：
| 品种 | 预测 | 总结 | 分析 |

其中“分析”列需要从以下几个维度展开（没有相关信息的维度可以不写）：
- 宏观
- 供应
- 需求
- 库存
- 成本利润
- 资金情绪

要求：
1. 每个品种一行，简洁凝练，不要展开过多细节
2. “预测”列给出对未来一段时间的明确方向判断（只有五个，趋势上行、趋势下行、震荡上行、震荡下行、中性）
3. “总结”列用一两句话概括该品种当前的核心矛盾或主要特征
4. “分析”列按上述维度分别简述
5. 如果文件中某些品种信息不完整，只写有依据的内容，不要臆测
6. 直接输出为一个Markdown 表格，不要做其他说明"""


def encode_local_image(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None:
        mime = "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def pdf_to_images(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 120,
    max_pages: int | None = None,
) -> list[Path]:
    """Rasterize PDF pages to JPEG (smaller payload than PNG)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    cmd = [
        "pdftoppm",
        "-jpeg",
        "-jpegopt",
        "quality=85",
        "-r",
        str(dpi),
        str(pdf_path),
        str(prefix),
    ]
    if max_pages is not None:
        cmd[1:1] = ["-f", "1", "-l", str(max_pages)]
    subprocess.run(cmd, check=True)
    paths = sorted(out_dir.glob("page-*.jpg"))
    if not paths:
        raise RuntimeError(f"pdftoppm produced no images under {out_dir}")
    if max_pages is not None:
        paths = paths[: max(0, max_pages)]
    return paths


def estimate_image_tokens(width: int, height: int) -> int:
    """Rough Qwen vision token estimate: 1 token per 32x32 patch."""
    return max(1, (width // 32) * (height // 32))


def estimate_page_tokens(dpi: int, page_w_pt: float = 595.3, page_h_pt: float = 841.9) -> int:
    w = int(page_w_pt * dpi / 72)
    h = int(page_h_pt * dpi / 72)
    return estimate_image_tokens(w, h)


def build_user_content(
    prompt: str,
    *,
    images: list[Path] | None = None,
    text: str | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if images:
        for img in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_local_image(img)},
                }
            )
    if text:
        content.append({"type": "text", "text": text})
    content.append({"type": "text", "text": prompt})
    return content


def analyze_document(
    input_path: Path,
    prompt: str,
    *,
    use_batch: bool = True,
    enable_thinking: bool = True,
    thinking_budget: int = 8192,
    max_tokens: int = 4096,
    dpi: int = 120,
    max_pages: int | None = None,
    image_dir: Path | None = None,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Return dict with text, usage, pages, base_url, etc."""
    suffix = input_path.suffix.lower()
    images: list[Path] = []
    text_body: str | None = None
    tmp_dir: str | None = None
    cleanup = False

    try:
        if suffix == ".pdf":
            if image_dir is None:
                tmp_dir = tempfile.mkdtemp(prefix="qwen37_batch_")
                image_dir = Path(tmp_dir)
                cleanup = True
            else:
                image_dir.mkdir(parents=True, exist_ok=True)
            images = pdf_to_images(input_path, image_dir, dpi=dpi, max_pages=max_pages)
            # Soft check against Batch 256K context
            est = len(images) * estimate_page_tokens(dpi) + 800
            if use_batch and est > BATCH_CONTEXT_LIMIT:
                raise RuntimeError(
                    f"Estimated input tokens ~{est:,} exceed Batch Chat 256K limit "
                    f"({len(images)} pages @ {dpi} DPI). Lower --dpi / --max-pages, "
                    f"or use --realtime (1M context)."
                )
        elif suffix in IMAGE_SUFFIXES:
            images = [input_path]
        elif suffix in TEXT_SUFFIXES:
            text_body = input_path.read_text(encoding="utf-8", errors="ignore")
        else:
            raise SystemExit(
                f"Unsupported input type: {suffix}. Use PDF / image / text."
            )

        base_url = BATCH_BASE_URL if use_batch else REALTIME_BASE_URL
        client = OpenAI(api_key=API_KEY, base_url=base_url).with_options(
            timeout=timeout
        )

        content = build_user_content(prompt, images=images or None, text=text_body)
        print(
            f"calling model={MODEL} batch={use_batch} pages/images={len(images)} "
            f"thinking={enable_thinking} ...",
            flush=True,
        )

        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            extra_body={
                "enable_thinking": enable_thinking,
                "thinking_budget": thinking_budget,
            },
        )

        message = completion.choices[0].message
        text = message.content or ""
        # Some SDKs expose reasoning on the message object
        reasoning = getattr(message, "reasoning_content", None) or getattr(
            message, "reasoning", None
        )

        usage = completion.usage
        usage_dict: dict[str, Any] = {}
        if usage is not None:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                usage_dict["reasoning_tokens"] = getattr(
                    details, "reasoning_tokens", None
                )

        return {
            "text": text.strip() + "\n",
            "reasoning": reasoning,
            "usage": usage_dict,
            "n_images": len(images),
            "base_url": base_url,
            "model": MODEL,
        }
    finally:
        if cleanup and tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def estimate_cost_cny(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    batch: bool = True,
) -> dict[str, float]:
    """Batch Chat ≈ 50% of catalog; catalog ≤256K: in=2, out=8 元/MTok."""
    in_price = 2.0
    out_price = 8.0
    if batch:
        in_price *= 0.5
        out_price *= 0.5
    cost_in = prompt_tokens / 1e6 * in_price
    cost_out = completion_tokens / 1e6 * out_price
    return {
        "input_cny": cost_in,
        "output_cny": cost_out,
        "total_cny": cost_in + cost_out,
        "in_price_per_mtok": in_price,
        "out_price_per_mtok": out_price,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3.7-Plus Batch Chat: whole-document commodity analysis"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="PDF / image / text path",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="PDF rasterize DPI (lower = fewer tokens; Batch cap 256K)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="only first N PDF pages",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="disable enable_thinking (default: on)",
    )
    parser.add_argument("--thinking-budget", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="use realtime endpoint instead of Batch Chat",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="keep PDF page images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write markdown result",
    )
    parser.add_argument(
        "--usage-json",
        type=Path,
        default=None,
        help="optional path to dump token usage / cost estimate",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    out_path = args.output
    if out_path is None:
        out_path = DEFAULT_OUTPUT / f"{input_path.stem}_analyze.md"
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_batch = not args.realtime
    enable_thinking = not args.no_thinking

    print(f"model={MODEL}")
    print(f"endpoint={'batch' if use_batch else 'realtime'}")
    print(f"input={input_path}")
    print(f"dpi={args.dpi} max_pages={args.max_pages}")
    print(f"thinking={enable_thinking} budget={args.thinking_budget}")
    print(f"output={out_path}")
    print("-" * 60)

    result = analyze_document(
        input_path,
        args.prompt,
        use_batch=use_batch,
        enable_thinking=enable_thinking,
        thinking_budget=args.thinking_budget,
        max_tokens=args.max_tokens,
        dpi=args.dpi,
        max_pages=args.max_pages,
        image_dir=args.image_dir,
        timeout=args.timeout,
    )

    out_path.write_text(result["text"], encoding="utf-8")
    print(result["text"])
    print("-" * 60)

    usage = result["usage"]
    print(f"images={result['n_images']} usage={usage}")
    cost_info: dict[str, Any] = {"usage": usage, "batch": use_batch}
    if usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
        cost = estimate_cost_cny(
            int(usage["prompt_tokens"]),
            int(usage["completion_tokens"]),
            batch=use_batch,
        )
        cost_info["cost_cny"] = cost
        print(
            f"est_cost≈{cost['total_cny']:.4f} CNY "
            f"(in {cost['in_price_per_mtok']}/M + out {cost['out_price_per_mtok']}/M)"
        )

    if args.usage_json is not None:
        usage_path = args.usage_json.expanduser().resolve()
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        usage_path.write_text(
            json.dumps(cost_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"usage saved -> {usage_path}")

    if result.get("reasoning"):
        reasoning_path = out_path.with_suffix(".reasoning.md")
        reasoning_path.write_text(str(result["reasoning"]), encoding="utf-8")
        print(f"reasoning saved -> {reasoning_path}")

    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
