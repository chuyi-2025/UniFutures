#!/usr/bin/env python3
"""Build infer GAF + XGB features for configured symbols."""

from __future__ import annotations

from build_gaf_feature import main as build_gaf
from build_xgb_feature import main as build_xgb


def main() -> None:
    build_gaf()
    build_xgb()


if __name__ == "__main__":
    main()
