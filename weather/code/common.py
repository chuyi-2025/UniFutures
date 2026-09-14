"""Shared paths and helpers for weather × domestic ag strategies."""

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
REGIONS_YAML = CONFIG_DIR / "crop_regions.yaml"

EXPERIMENT_DIR = Path("/home/workspace/lab/UniFutures/code/experiment")
SYMBOLS = ("CF", "SR", "AP", "CJ")


def ensure_dirs() -> None:
    for p in (RAW_DIR, FACTOR_DIR, RESULT_DIR, CONFIG_DIR):
        p.mkdir(parents=True, exist_ok=True)


def load_regions(path: Path | None = None) -> dict[str, Any]:
    p = path or REGIONS_YAML
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_stations(cfg: dict[str, Any] | None = None):
    """Yield (symbol, station_dict) for all configured points."""
    cfg = cfg or load_regions()
    for sym, meta in cfg["symbols"].items():
        for st in meta["stations"]:
            yield sym, st


def all_station_ids(cfg: dict[str, Any] | None = None) -> list[str]:
    return [st["id"] for _, st in iter_stations(cfg)]
