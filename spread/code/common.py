"""Shared paths for spread / carry research."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
PANEL_DIR = DATA_DIR / "panels"
RESULT_DIR = DATA_DIR / "results"
DOCS_DIR = ROOT / "docs"

TREE = Path("/home/workspace/lab/Tree-Stock/futures/data/all_contracts")
EXPERIMENT_DIR = Path("/home/workspace/lab/UniFutures/code/experiment")


def ensure_dirs() -> None:
    for p in (PANEL_DIR, RESULT_DIR, CONFIG_DIR, DOCS_DIR):
        p.mkdir(parents=True, exist_ok=True)
