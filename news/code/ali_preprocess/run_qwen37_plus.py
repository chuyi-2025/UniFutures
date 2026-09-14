#!/usr/bin/env python3
"""Call Qwen3.7-Plus multimodal (image + text -> text) via DashScope OpenAI-compatible API."""

from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path

from openai import OpenAI

# From engineai_deployment .../llm/config.py (ali_omni)
API_KEY = (
    "sk-ws-H.RXPDRME.2SDp.MEYCIQDaeu81UzW85Dw8zzesS4hzZXjVNA_Cx6Ce8E-RwyxjOwIhALOUfOYTsWLBMRBrl-aKxkcJcMRl5t-mDQ6Gb_pbV-iK"
)
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.7-plus"

DEFAULT_IMAGE_URL = (
    "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/"
    "zh-CN/20241022/emyrja/dog_and_girl.jpeg"
)
DEFAULT_PROMPT = "请用中文简要描述这张图片里有什么，并说明画面氛围。"


def encode_local_image(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None:
        mime = "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def build_image_url(image: str) -> str:
    path = Path(image)
    if path.is_file():
        return encode_local_image(path)
    return image


def ask_vl(
    prompt: str,
    image: str,
    *,
    enable_thinking: bool = False,
    max_tokens: int = 1024,
) -> str:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    image_url = build_image_url(image)

    kwargs: dict = {
        "enable_thinking": enable_thinking,
    }

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
        extra_body=kwargs,
    )
    return completion.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.7-Plus vision (image+text -> text)")
    parser.add_argument("--image", default=DEFAULT_IMAGE_URL, help="local path or public URL")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="text instruction")
    parser.add_argument("--thinking", action="store_true", help="enable thinking mode")
    parser.add_argument("--max-tokens", type=int, default=30720)
    args = parser.parse_args()

    print(f"model={MODEL}")
    print(f"base_url={BASE_URL}")
    print(f"image={args.image}")
    print(f"prompt={args.prompt}")
    print("-" * 60)

    text = ask_vl(
        args.prompt,
        args.image,
        enable_thinking=args.thinking,
        max_tokens=args.max_tokens,
    )
    print(text)


if __name__ == "__main__":
    main()
