#!/usr/bin/env python3
"""Bag-of-words + Ridge on raw report text — simplest linear training.

Train: report_date < 2025-07-01
Predict: all docs -> bow_score; aggregate to daily rule_bow factor.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_rule_factors import aggregate  # noqa: E402
from common import DATA_DIR, FT_TEST_START  # noqa: E402

TOKEN = re.compile(r"[\u4e00-\u9fff]{2,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=DATA_DIR / "samples.parquet")
    ap.add_argument("--lookback", type=int, default=14)
    ap.add_argument("--max-features", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=10.0)
    args = ap.parse_args()

    df = pd.read_parquet(args.samples)
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.dropna(subset=["text", "log_ret_1d", "symbol"])
    df["symbol"] = df["symbol"].str.upper()

    train = df[df["report_date"] < FT_TEST_START].copy()
    test = df.copy()
    print(f"train={len(train)} test/all={len(test)}", flush=True)

    vec = CountVectorizer(
        max_features=args.max_features,
        token_pattern=r"(?u)\b[\u4e00-\u9fff]{2,}\b",
        min_df=5,
    )
    X_tr = vec.fit_transform(train["text"].astype(str))
    y_tr = train["log_ret_1d"].astype(float).to_numpy()
    model = Ridge(alpha=args.alpha)
    model.fit(X_tr, y_tr)

    X_all = vec.transform(test["text"].astype(str))
    test = test.copy()
    test["bow_score"] = model.predict(X_all)
    test[["report_date", "symbol", "bow_score"]].to_parquet(DATA_DIR / "bow_doc_scores.parquet", index=False)

    fac = aggregate(
        test.rename(columns={"bow_score": "rule_edge"}).assign(rule_edge=test["bow_score"]),
        "rule_edge",
        args.lookback,
        pd.Timestamp("2024-03-01"),
        pd.Timestamp("2026-12-31"),
    ).rename(columns={"rule_edge": "rule_bow"})
    out = DATA_DIR / f"bow_factors_L{args.lookback}.parquet"
    fac.to_parquet(out, index=False)

    # top positive/negative words
    coef = pd.Series(model.coef_, index=vec.get_feature_names_out())
    top = pd.concat([coef.nlargest(10), coef.nsmallest(10)])
    meta = {
        "alpha": args.alpha,
        "max_features": args.max_features,
        "train_n": len(train),
        "top_coef": top.to_dict(),
    }
    import json
    (DATA_DIR / "bow_ridge_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"saved {out} rows={len(fac)}", flush=True)
    print("top words:", top.head(5).to_dict(), "...", top.tail(5).to_dict(), flush=True)


if __name__ == "__main__":
    main()
