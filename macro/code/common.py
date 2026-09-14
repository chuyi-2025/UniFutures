"""Shared paths for macro × super-commodity research."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FACTOR_DIR = DATA_DIR / "factors"
RESULT_DIR = DATA_DIR / "results"
DOCS_DIR = ROOT / "docs"
SOURCES_YAML = CONFIG_DIR / "macro_sources.yaml"

EXPERIMENT_DIR = Path("/home/workspace/lab/UniFutures/code/experiment")
TARGETS = ("AU", "AG", "SC")


def ensure_dirs() -> None:
    for p in (RAW_DIR, FACTOR_DIR, RESULT_DIR, CONFIG_DIR, DOCS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def load_sources(path: Path | None = None) -> dict[str, Any]:
    with (path or SOURCES_YAML).open(encoding="utf-8") as f:
        return yaml.safe_load(f)
