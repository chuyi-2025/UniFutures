#!/usr/bin/env python3
"""Clean Unlimited-OCR markdown reports with DeepSeek into readable research notes.

Input:  /home/workspace/lab/UniFutures/news/result/unlimited_ocr/YYYYMMDD/*.md
Output: /home/workspace/lab/UniFutures/news/result/unlimited_ocr_by_deepseek/YYYYMMDD/*.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from openai import OpenAI

DEFAULT_SRC = Path("/home/workspace/lab/UniFutures/news/result/unlimited_ocr")
DEFAULT_OUT = Path("/home/workspace/lab/UniFutures/news/result/unlimited_ocr_by_deepseek")
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_API_KEY = "sk-283fbfae9a5647b8a74d8afd587517a2"

# Rough char budget per API call (Chinese-heavy OCR text).
CHUNK_CHARS = 60_000
MAX_OUTPUT_TOKENS = 32768
# If concatenated segment results exceed this, skip merge API and keep page results as-is.
MERGE_API_MAX_CHARS = 32768

SYSTEM_PROMPT = """
你是期货/商品研报清洗与排版助手。输入是 OCR 得到的原始 markdown（可能含页面标记、广告、免责声明、页眉页脚、乱序文本）。

你的任务：输出一份**干净、简洁、人类可读**的研究笔记 Markdown。

## 必须删除（不要保留）
1. 开户/拉客广告（如“期货开户”“手续费+1分”“客户经理微信”“公众号：期之货”等）
2. 免责声明、版权声明、地址电话邮编网址、公司总部信息
3. 分析师资格号、电话、邮箱、二维码提示、关注公众号等联系信息（可保留作者姓名一次）
4. OCR 噪声：`[NO TEXT]`、`[Non-Text]`、`[No text]`、`![](images/...)`、页码（如 `16 / 16`）、重复页眉页脚、纯品牌行
5. 纯目录页、纯图表目录页（若目录外无正文信息）
6. 无实质观点/数据的空段
7. **空图注**：仅有“图X：xxx”“表X：xxx”或“数据来源：xxx”、没有实际数字/表格时整行删除；正文里“见图1”等可去掉图号

## 保留并整理
1. 核心观点、摘要、结论、风险提示
2. 供需/库存/宏观等实质分析与关键数据
3. 有数字的表格整理成 Markdown 表格；没有数据不要保留空图注
4. 分点用列表，可读数据优先表格

## 输出格式（严格）
只输出 Markdown，不要解释，不要包代码块。

全文开头必须是这一张表（仅一行数据）：

| 机构 | 日期 | 品种 |
|------|------|------|
| ... | ... | ... |

填写规则：
- 机构：报告发布/署名机构；多个用 `/` 连接，如 `中信期货/华泰期货`
- 日期：报告日期，优先 `YYYY-MM-DD`；不确定则写原文日期或 `未知`
- 品种：列出具体期货/商品中文名，多个用 `/` 连接，如 `集运/碳酸锂/豆粕/菜粕`；不要写“多品种”“大宗商品”等笼统词；实在抽不出具体品种才写 `未知`
- 不要拆成多行，也不要按品种开很多小节

表后接清洗后的正文，结构可按内容取舍：
# 标题
## 核心观点
## 正文分析
## 结论与展望
## 风险提示

语言简洁，合并重复，修正明显 OCR 错字（不改变原意）。
若几乎无有效内容，表后只写：`# 无有效内容`
""".strip()


AD_PATTERNS = [
    re.compile(r"期货开户[：:].{0,400}"),
    re.compile(r"客户经理微信[：:].{0,80}"),
    re.compile(r"公众号[：:]期之货"),
    re.compile(r"赠送头部机构研报群.{0,200}"),
]
NOISE_LINES = re.compile(
    r"^(\[NO TEXT\]|\[Non-Text\]|\[No text\]|CITIC Futures|HUATAI FUTURES|"
    r"MINMETALS FUTURES CO\.?,?LTD|"
    r"!\[\]\(images/[^)]+\)|"
    r"\d+\s*/\s*\d+)\s*$",
    re.IGNORECASE,
)
# Caption-only lines with no numeric/table content, e.g. "图 6：xxx｜数据来源：Kpler"
EMPTY_FIGURE_LINE = re.compile(
    r"^[\-\*\u2022\u00b7]?\s*(图|表)\s*\d+\s*[：:].+$"
)
SOURCE_ONLY_LINE = re.compile(
    r"^[\-\*\u2022\u00b7]?\s*(资料来源|数据来源|来源)\s*[：:].+$"
)
DISCLAIMER_START = re.compile(r"^(免责声明|重要提示|版权声明|除非另有说明)")


def _is_empty_figure_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if SOURCE_ONLY_LINE.match(s):
        return True
    if not EMPTY_FIGURE_LINE.match(s):
        return False
    # Keep if the caption line itself embeds multiple numbers (rare but possible).
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    # Ignore the figure index itself (图6 / 表12).
    meaningful = [n for n in nums if not re.fullmatch(r"\d{1,3}", n) or float(n) >= 100]
    return len(meaningful) < 2


def strip_empty_figures(text: str) -> str:
    """Remove caption-only figure/table lines from model output."""
    kept: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if _is_empty_figure_line(line):
            continue
        kept.append(line)
    out: list[str] = []
    blank = 0
    for line in kept:
        if not line.strip():
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(line)
    return "\n".join(out).strip() + "\n"


def pre_clean(text: str) -> str:
    """Cheap local cleanup before calling the API."""
    pages = re.split(r"\n?<PAGE>\n?", text)
    cleaned_pages: list[str] = []
    for page in pages:
        lines: list[str] = []
        skip_rest = False
        for raw in page.splitlines():
            line = raw.strip()
            if not line:
                continue
            if DISCLAIMER_START.match(line):
                skip_rest = True
                break
            if NOISE_LINES.match(line):
                continue
            if _is_empty_figure_line(line):
                continue
            if any(p.search(line) for p in AD_PATTERNS):
                continue
            if "期货开户" in line and ("手续费" in line or "客户经理" in line):
                continue
            lines.append(line)
        if skip_rest:
            # keep any content before disclaimer on this page
            pass
        body = "\n".join(lines).strip()
        if body:
            cleaned_pages.append(body)
    return "\n\n---\n\n".join(cleaned_pages)


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts = text.split("\n\n---\n\n")
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for part in parts:
        add = len(part) + 8
        if cur and cur_len + add > max_chars:
            chunks.append("\n\n---\n\n".join(cur))
            cur, cur_len = [part], len(part)
        else:
            cur.append(part)
            cur_len += add
    if cur:
        chunks.append("\n\n---\n\n".join(cur))
    return chunks


def call_deepseek(
    client: OpenAI,
    model: str,
    user_content: str,
    temperature: float,
    max_retries: int,
) -> str:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=MAX_OUTPUT_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "请清洗并重排下面这份 OCR 研报文本，直接输出整理后的 Markdown：\n\n"
                            + user_content
                        ),
                    },
                ],
            )
            content = resp.choices[0].message.content or ""
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:markdown|md)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            return content.strip()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"DeepSeek call failed after {max_retries} retries: {last_err}")


def process_one(
    src: Path,
    out: Path,
    client: OpenAI,
    model: str,
    temperature: float,
    max_retries: int,
) -> str:
    raw = src.read_text(encoding="utf-8", errors="ignore")
    cleaned = pre_clean(raw)
    if not cleaned.strip():
        text = "# 无有效内容\n\n原文经清洗后无可保留的研究信息。"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return text

    chunks = chunk_text(cleaned)
    if len(chunks) == 1:
        text = call_deepseek(client, model, chunks[0], temperature, max_retries)
    else:
        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            piece = call_deepseek(
                client,
                model,
                f"（这是长文档的第 {i}/{len(chunks)} 段，请整理本段；可保留小标题，勿写全文总结）\n\n{chunk}",
                temperature,
                max_retries,
            )
            parts.append(piece)
        # Merge via API only when total cleaned text is short enough.
        # Otherwise keep paginated segment results as the final document.
        joined = "\n\n".join(parts)
        if len(joined) > MERGE_API_MAX_CHARS:
            text = joined
        else:
            merged_input = "\n\n".join(
                f"## 分段整理 {i}\n\n{p}" for i, p in enumerate(parts, start=1)
            )
            text = call_deepseek(
                client,
                model,
                "下面是同一篇研报分段清洗后的结果，请合并为一篇完整、去重、结构统一的 Markdown：\n\n"
                + merged_input,
                temperature,
                max_retries,
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    text = strip_empty_figures(text)
    out.write_text(text.rstrip() + "\n", encoding="utf-8")
    return text


def iter_md_files(src_root: Path) -> list[Path]:
    files = [p for p in src_root.rglob("*.md") if p.is_file()]
    files.sort()
    return files


def relative_out_path(src: Path, src_root: Path, out_root: Path) -> Path:
    return out_root / src.relative_to(src_root)


def append_log(log_path: Path, record: dict, lock: threading.Lock | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    if lock is None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)
        return
    with lock:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)


def is_clean_done(out: Path) -> bool:
    """Skip only outputs that already have the required header table."""
    if not out.exists() or out.stat().st_size < 40:
        return False
    try:
        head = out.read_text(encoding="utf-8", errors="ignore")[:800]
    except OSError:
        return False
    return "| 机构 |" in head and "| 日期 |" in head and "| 品种 |" in head


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepSeek clean Unlimited-OCR reports")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--workers", type=int, default=64, help="thread workers")
    p.add_argument("--limit", type=int, default=0, help="process at most N files (0=all)")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--files", type=Path, default=None, help="optional list of input md paths")
    p.add_argument("--log", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.log or (args.out / "clean_progress.jsonl")
    workers = max(1, int(args.workers))

    if args.files is not None:
        inputs = [Path(x.strip()) for x in args.files.read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        inputs = iter_md_files(args.src)

    planned: list[tuple[Path, Path]] = []
    skipped = 0
    for src in inputs:
        out = relative_out_path(src, args.src, args.out) if src.is_relative_to(args.src) else (
            args.out / src.parent.name / src.name
        )
        if args.skip_existing and is_clean_done(out):
            skipped += 1
            continue
        planned.append((src, out))

    if args.limit > 0:
        planned = planned[: args.limit]

    print(
        f"[scan] inputs={len(inputs)} to_process={len(planned)} "
        f"skipped_existing={skipped} workers={workers}"
    )
    if not planned:
        print("[done] nothing to process")
        return 0

    api_key = os.getenv("DEEPSEEK_API_KEY", DEFAULT_API_KEY)
    log_lock = threading.Lock()
    print_lock = threading.Lock()
    ok = 0
    failed = 0
    done = 0
    counters_lock = threading.Lock()
    total = len(planned)

    def worker(item: tuple[Path, Path]) -> tuple[str, Path, Path, str | None, int]:
        src, out = item
        # One client per task avoids shared-session contention.
        client = OpenAI(api_key=api_key, base_url=args.base_url)
        try:
            text = process_one(
                src,
                out,
                client,
                model=args.model,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )
            return "ok", src, out, None, len(text)
        except Exception as exc:  # noqa: BLE001
            return "error", src, out, f"{exc}\n{traceback.format_exc()}", 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, item): item for item in planned}
        for fut in as_completed(futures):
            status, src, out, err, chars = fut.result()
            with counters_lock:
                done += 1
                idx = done
                if status == "ok":
                    ok += 1
                else:
                    failed += 1
            rel = src.relative_to(args.src) if src.is_relative_to(args.src) else src
            if status == "ok":
                append_log(
                    log_path,
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "status": "ok",
                        "src": str(src),
                        "out": str(out),
                        "chars": chars,
                    },
                    lock=log_lock,
                )
                with print_lock:
                    print(f"[{idx}/{total}] ok {rel} -> {chars} chars")
            else:
                append_log(
                    log_path,
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "status": "error",
                        "src": str(src),
                        "out": str(out),
                        "error": (err or "")[:2000],
                    },
                    lock=log_lock,
                )
                with print_lock:
                    print(f"[{idx}/{total}] !! failed {rel}: {(err or '').splitlines()[0]}", file=sys.stderr)

    print(f"[summary] ok={ok} failed={failed} skipped_existing={skipped} workers={workers}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
