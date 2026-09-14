#!/usr/bin/env python3
"""Compress each cleaned report into <=512-token, per-variety JSON records.

Long reports use map-reduce:
  chunk facts -> document-level merge/compression.
Outputs are resumable JSON files plus a consolidated parquet/csv manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE.parent / "train"
sys.path.insert(0, str(TRAIN_DIR))

from common import CONTRACTS_DIR, REMOVED_SYMBOLS  # noqa: E402
from symbol_map import map_variety  # noqa: E402

DEFAULT_DOCS = Path(
    "/home/workspace/lab/UniFutures/news/data/sentiment/hierarchical/documents.parquet"
)
DEFAULT_LINKS = Path(
    "/home/workspace/lab/UniFutures/news/data/sentiment/hierarchical/document_symbols.parquet"
)
DEFAULT_OUT = Path(
    "/home/workspace/lab/UniFutures/news/data/sentiment/deepseek_variety_compressed"
)
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
CHUNK_CHARS = 48_000
MAX_OUTPUT_TOKENS = 8192
MAX_ITEM_CHARS = 1_200  # conservative proxy for <=512 DeepSeek tokens in Chinese
FORECAST_DIRECTIONS = {
    "趋势上行",
    "震荡上行",
    "区间震荡",
    "震荡下行",
    "趋势下行",
    "中性",
}
FORECAST_HORIZONS = {
    "短期（1-4周）",
    "中期（1-3月）",
    "长期（3月以上）",
    "原文未说明",
}
DIMENSIONS = ("宏观", "供应", "需求", "库存", "成本利润", "资金情绪")

SYSTEM_PROMPT = """
你是一名商品期货研报结构化压缩助手。

任务：
仅依据输入研报中的内容，识别其中被实质分析的商品或期货品种，并为每个品种生成一份独立、可用于量化建模的精简摘要。

重要约束：
1. 严禁使用原文之外的知识、实时行情或常识补充。
2. 严禁推测原文未明确表达的供需、库存、价格或方向。
3. 只提到名称、用于横向比较、作为替代品或竞争作物，但没有实质分析的品种，不单独输出。
4. 同一品种的不同名称必须合并，例如“郑棉/棉花/CF”统一为“棉花”。
5. 国内外同一商品原则上合并；若原文明显给出不同市场判断，应在分析中分别说明。
6. 判断必须站在研报发布日期时点，不得引用发布日期之后的信息。
7. 每个品种的完整 JSON 对象必须控制在512个token以内。优先保留影响未来价格方向的信息，删除行情复盘、重复数据和无方向意义的细节。
8. 若原文没有明确方向，预测方向必须写“中性”，不能自行推断。
9. 只输出合法JSON，不要Markdown、代码块、解释或前后缀。

“分析”字段：
- 只能使用：宏观、供应、需求、库存、成本利润、资金情绪。
- 只输出原文有依据的维度；没有依据的维度不要输出。
- 每个维度不超过30个中文字符。
- 优先描述变化方向及其价格影响。
- 区分事实、预期和风险，不得把条件性判断写成确定事实。

“总结”字段：
- 不超过80个中文字符。
- 概括核心矛盾、主要驱动和关键风险，不机械重复分析分项。

“预测”字段：
- 方向只能是：趋势上行、震荡上行、区间震荡、震荡下行、趋势下行、中性。
- 期限只能是：短期（1-4周）、中期（1-3月）、长期（3月以上）、原文未说明。
- 依据不超过50个中文字符，且必须来自原文。
- 多情景判断选择基准情景，并注明关键条件。

严格输出结构：
{
  "品种分析": [
    {
      "品种": "规范化中文品种名",
      "分析": {"宏观": "...", "供应": "..."},
      "总结": "...",
      "预测": {
        "方向": "区间震荡",
        "期限": "中期（1-3月）",
        "依据": "原文中的核心依据"
      }
    }
  ]
}

若没有被实质分析的品种，输出：{"品种分析":[]}
""".strip()

CHUNK_INSTRUCTION = """
这是长研报的第 {index}/{total} 段。只提取本段中各品种有依据的事实、预期、风险和方向。
不要因为本段信息不完整而补充其他知识。输出仍使用规定 JSON 结构；
此阶段允许同一品种在不同段重复，后续会统一合并。

研报元数据：
- 机构：{institution}
- 发布日期：{report_date}
- 标题标注品种：{variety_raw}

正文：
{body}
""".strip()

FINAL_INSTRUCTION = """
请对下面这份完整研报执行按品种压缩。必须遵守系统规则和 JSON 格式。

研报元数据：
- 机构：{institution}
- 发布日期：{report_date}
- 标题标注品种：{variety_raw}

研报正文：
{body}
""".strip()

MERGE_INSTRUCTION = """
下面是同一篇研报各分段提取结果。请按品种合并、去重和解决重复表述。
只保留原分段中有依据的信息，不得补充外部知识。
每个品种最终对象必须控制在512个token以内，并严格输出规定 JSON。

研报元数据：
- 机构：{institution}
- 发布日期：{report_date}
- 标题标注品种：{variety_raw}

分段结果：
{partials}
""".strip()


def split_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    blocks = re.split(r"\n(?=#{1,4}\s)|\n\n+", text)
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for block in blocks:
        if current and size + len(block) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        if len(block) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, size = [], 0
            chunks.extend(
                block[i : i + max_chars] for i in range(0, len(block), max_chars)
            )
        else:
            current.append(block)
            size += len(block) + 2
    if current:
        chunks.append("\n\n".join(current))
    return [x for x in chunks if x.strip()]


def parse_json_object(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("response is not a JSON object")
    return obj


def clean_item(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    variety = str(raw.get("品种", "")).strip()
    if not variety:
        return None
    analysis_raw = raw.get("分析", {})
    analysis: dict[str, str] = {}
    if isinstance(analysis_raw, dict):
        for key in DIMENSIONS:
            value = str(analysis_raw.get(key, "")).strip()
            if value:
                analysis[key] = value[:30]
    summary = str(raw.get("总结", "")).strip()[:80]
    forecast_raw = raw.get("预测", {})
    if not isinstance(forecast_raw, dict):
        forecast_raw = {}
    direction = str(forecast_raw.get("方向", "中性")).strip()
    horizon = str(forecast_raw.get("期限", "原文未说明")).strip()
    basis = str(forecast_raw.get("依据", "")).strip()[:50]
    if direction not in FORECAST_DIRECTIONS:
        direction = "中性"
    if horizon not in FORECAST_HORIZONS:
        horizon = "原文未说明"
    item = {
        "品种": variety,
        "分析": analysis,
        "总结": summary,
        "预测": {"方向": direction, "期限": horizon, "依据": basis},
    }
    return item


def validate_result(obj: dict[str, Any]) -> dict[str, Any]:
    raw_items = obj.get("品种分析", [])
    if not isinstance(raw_items, list):
        raise ValueError("品种分析 must be a list")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_items:
        item = clean_item(raw)
        if item is None:
            continue
        key = re.sub(r"\s+", "", item["品种"])
        if key in seen:
            raise ValueError(f"duplicate variety after merge: {key}")
        seen.add(key)
        if len(json.dumps(item, ensure_ascii=False)) > MAX_ITEM_CHARS:
            raise ValueError(f"item too long: {item['品种']}")
        items.append(item)
    return {"品种分析": items}


def call_json(
    client: OpenAI,
    model: str,
    user_content: str,
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    prompt = user_content
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
                # V4 Flash defaults thinking ON; reasoning can exhaust max_tokens
                # and leave message.content empty (finish_reason=length).
                extra_body={"thinking": {"type": "disabled"}},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            if not content.strip():
                raise ValueError(
                    f"empty content finish_reason={finish_reason} "
                    f"usage={getattr(response, 'usage', None)}"
                )
            obj = parse_json_object(content)
            return validate_result(obj)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            prompt = (
                user_content
                + "\n\n上一次输出未通过 JSON/字段/长度校验。请重新输出更短的合法 JSON；"
                + f"校验错误：{type(exc).__name__}: {str(exc)[:200]}"
            )
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"DeepSeek failed after {max_retries} retries: {last_error}")


def process_document(
    client: OpenAI,
    model: str,
    row: Any,
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    text = str(row.text)
    metadata = {
        "institution": str(row.institution),
        "report_date": str(pd.Timestamp(row.report_date).date()),
        "variety_raw": str(row.variety_raw),
    }
    chunks = split_text(text)
    if len(chunks) == 1:
        result = call_json(
            client,
            model,
            FINAL_INSTRUCTION.format(body=chunks[0], **metadata),
            temperature,
            max_retries,
        )
    else:
        partials = []
        for index, chunk in enumerate(chunks, 1):
            partial = call_json(
                client,
                model,
                CHUNK_INSTRUCTION.format(
                    index=index, total=len(chunks), body=chunk, **metadata
                ),
                temperature,
                max_retries,
            )
            partials.append(partial)
        result = call_json(
            client,
            model,
            MERGE_INSTRUCTION.format(
                partials=json.dumps(partials, ensure_ascii=False), **metadata
            ),
            temperature,
            max_retries,
        )
    return {
        "schema_version": 1,
        "doc_idx": int(row.doc_idx),
        "doc_id": str(row.doc_id),
        "content_hash": str(row.content_hash),
        "institution": metadata["institution"],
        "report_date": metadata["report_date"],
        "variety_raw": metadata["variety_raw"],
        "model": model,
        "n_chunks": len(chunks),
        **result,
    }


def output_path(out_dir: Path, doc_id: str) -> Path:
    return out_dir / "json" / f"{doc_id}.json"


def is_complete(path: Path, content_hash: str, model: str) -> bool:
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        validate_result(obj)
        return obj.get("content_hash") == content_hash and obj.get("model") == model
    except Exception:  # noqa: BLE001
        return False


def render_training_text(symbol: str, item: dict[str, Any]) -> str:
    analysis = "；".join(f"{k}：{v}" for k, v in item["分析"].items())
    pred = item["预测"]
    return (
        f"[品种:{symbol}] "
        f"分析：{analysis}。"
        f"总结：{item['总结']}。"
        f"预测：{pred['方向']}；期限：{pred['期限']}；依据：{pred['依据']}。"
    )


def consolidate(
    docs: pd.DataFrame, links: pd.DataFrame, out_dir: Path, model: str
) -> pd.DataFrame:
    known_symbols = {
        path.name.upper()
        for path in CONTRACTS_DIR.iterdir()
        if path.is_dir() and path.name.upper() not in REMOVED_SYMBOLS
    }
    link_map = (
        links.groupby("doc_id")["symbol"]
        .apply(lambda x: sorted(set(str(v).upper() for v in x)))
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []
    doc_by_id = docs.set_index("doc_id")
    for path in sorted((out_dir / "json").glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if obj.get("model") != model or obj["doc_id"] not in doc_by_id.index:
                continue
            item_doc = doc_by_id.loc[obj["doc_id"]]
            allowed = link_map.get(obj["doc_id"], [])
            for item in validate_result(obj)["品种分析"]:
                symbol = map_variety(item["品种"])
                if symbol not in known_symbols:
                    if len(allowed) == 1:
                        symbol = allowed[0]
                    else:
                        unmapped.append(
                            {
                                "doc_id": obj["doc_id"],
                                "品种": item["品种"],
                                "mapped": symbol,
                                "allowed": allowed,
                            }
                        )
                        continue
                rows.append(
                    {
                        "doc_idx": int(obj["doc_idx"]),
                        "doc_id": obj["doc_id"],
                        "path": str(item_doc["path"]),
                        "content_hash": obj["content_hash"],
                        "institution": obj["institution"],
                        "report_date": pd.Timestamp(obj["report_date"]),
                        "variety_raw": obj["variety_raw"],
                        "variety": item["品种"],
                        "symbol": symbol,
                        "analysis_json": json.dumps(
                            item["分析"], ensure_ascii=False, sort_keys=True
                        ),
                        "summary": item["总结"],
                        "forecast_direction": item["预测"]["方向"],
                        "forecast_horizon": item["预测"]["期限"],
                        "forecast_basis": item["预测"]["依据"],
                        "text": render_training_text(symbol, item),
                        "compressed_chars": len(
                            json.dumps(item, ensure_ascii=False)
                        ),
                        "model": model,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            unmapped.append({"path": str(path), "error": str(exc)})
    out = pd.DataFrame(rows)
    if len(out):
        out = (
            out.sort_values(["doc_idx", "symbol"])
            .drop_duplicates(["doc_id", "symbol"], keep="first")
            .reset_index(drop=True)
        )
    out.to_parquet(out_dir / "compressed_items.parquet", index=False)
    out.drop(columns=["text"], errors="ignore").to_csv(
        out_dir / "compressed_items.csv", index=False
    )
    (out_dir / "unmapped.json").write_text(
        json.dumps(unmapped, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def append_jsonl(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--doc-id", action="append", default=[])
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "json").mkdir(parents=True, exist_ok=True)
    docs = pd.read_parquet(args.docs).sort_values("doc_idx").reset_index(drop=True)
    links = pd.read_parquet(args.links)
    if args.doc_id:
        docs = docs[docs["doc_id"].isin(set(args.doc_id))].reset_index(drop=True)
    if args.limit > 0:
        docs = docs.head(args.limit).reset_index(drop=True)

    if args.consolidate_only:
        out = consolidate(
            pd.read_parquet(args.docs), links, args.out_dir, args.model
        )
        print(f"[consolidated] rows={len(out)}")
        return 0

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        # Compatibility with the existing cleaner; rotate this legacy key and use env.
        from clean_ocr import DEFAULT_API_KEY  # noqa: PLC0415

        api_key = DEFAULT_API_KEY

    planned = []
    skipped = 0
    for row in docs.itertuples(index=False):
        path = output_path(args.out_dir, str(row.doc_id))
        if (
            not args.no_skip_existing
            and is_complete(path, str(row.content_hash), args.model)
        ):
            skipped += 1
        else:
            planned.append(row)
    print(
        f"[scan] docs={len(docs)} planned={len(planned)} skipped={skipped} "
        f"workers={args.workers} model={args.model}",
        flush=True,
    )

    log_path = args.out_dir / "progress.jsonl"
    log_lock = threading.Lock()
    print_lock = threading.Lock()

    def worker(row: Any) -> tuple[str, Any, dict[str, Any] | None, str | None]:
        client = OpenAI(api_key=api_key, base_url=args.base_url)
        try:
            result = process_document(
                client,
                args.model,
                row,
                args.temperature,
                args.max_retries,
            )
            path = output_path(args.out_dir, str(row.doc_id))
            path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return "ok", row, result, None
        except Exception as exc:  # noqa: BLE001
            return "error", row, None, f"{exc}\n{traceback.format_exc()}"

    ok = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(worker, row): row for row in planned}
        total = len(futures)
        for index, future in enumerate(as_completed(futures), 1):
            status, row, result, error = future.result()
            record = {
                "time": pd.Timestamp.now().isoformat(),
                "status": status,
                "doc_id": str(row.doc_id),
                "doc_idx": int(row.doc_idx),
                "content_hash": str(row.content_hash),
            }
            if status == "ok":
                ok += 1
                record["n_items"] = len(result["品种分析"]) if result else 0
            else:
                failed += 1
                record["error"] = (error or "")[:2000]
            append_jsonl(log_path, record, log_lock)
            with print_lock:
                suffix = (
                    f"items={record.get('n_items', 0)}"
                    if status == "ok"
                    else (error or "").splitlines()[0]
                )
                print(f"[{index}/{total}] {status} {row.doc_id} {suffix}", flush=True)

    all_docs = pd.read_parquet(args.docs)
    out = consolidate(all_docs, links, args.out_dir, args.model)
    summary = {
        "model": args.model,
        "docs_total": int(len(all_docs)),
        "planned": len(planned),
        "skipped": skipped,
        "ok": ok,
        "failed": failed,
        "compressed_rows": int(len(out)),
        "compressed_docs": int(out["doc_id"].nunique()) if len(out) else 0,
        "symbols": int(out["symbol"].nunique()) if len(out) else 0,
        "manifest_hash": hashlib.sha256(
            "\n".join(
                f"{r.doc_id}:{r.content_hash}"
                for r in all_docs.sort_values("doc_idx").itertuples()
            ).encode()
        ).hexdigest(),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
