#!/usr/bin/env python3
"""Datasets / collators for encoder and aggregator rank training."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import HEAD_TAIL_HALF, HORIZONS, MAX_DOC_TOKENS, MAX_WINDOW_DOCS


def prefix_text(symbol: str, text: str) -> str:
    return f"[品种:{symbol}] {text}"


def tokenize_head_tail(tokenizer, text: str, max_length: int, half: int) -> dict[str, list[int]]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) > max_length:
        ids = ids[:half] + ids[-half:]
    eos = [tokenizer.eos_token_id] if getattr(tokenizer, "eos_token_id", None) else []
    budget = max_length - len(eos)
    if len(ids) > budget:
        ids = ids[:budget]
    full = ids + eos
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
    }


class DocRankDataset(Dataset):
    """Encoder stage: one (doc, symbol) per row; batches grouped by trade_date."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = MAX_DOC_TOKENS,
        half: int = HEAD_TAIL_HALF,
        split: str | None = None,
    ) -> None:
        work = df if split is None else df[df["split"] == split]
        # Need groups with >=2 valid symbols for pairwise
        work = work[work["group_size"] >= 2].reset_index(drop=True)
        self.df = work
        self.tok = tokenizer
        self.max_length = max_length
        self.half = half
        # Map trade_date → integer group id
        dates = sorted(work["trade_date"].unique())
        self.date_to_gid = {pd.Timestamp(d): i for i, d in enumerate(dates)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        text = prefix_text(str(row["symbol"]), str(row["text"]))
        enc = tokenize_head_tail(self.tok, text, self.max_length, self.half)
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "group_id": self.date_to_gid[pd.Timestamp(row["trade_date"])],
            "symbol": str(row["symbol"]),
            "trade_date": str(pd.Timestamp(row["trade_date"]).date()),
            "doc_idx": int(row["doc_idx"]),
        }
        for h in HORIZONS:
            item[f"label_h{h}"] = float(row[f"rank_h{h}"]) if pd.notna(row[f"rank_h{h}"]) else 0.0
            item[f"valid_h{h}"] = bool(row[f"valid_h{h}"]) and pd.notna(row[f"rank_h{h}"])
            item[f"ret_h{h}"] = float(row[f"ret_h{h}"]) if pd.notna(row[f"ret_h{h}"]) else 0.0
        return item


def collate_docs(batch: list[dict], pad_token_id: int = 151643) -> dict[str, Any]:
    max_len = max(len(b["input_ids"]) for b in batch)
    # Left pad (Qwen pad/eos id; overridden by make_collate_docs when available)
    input_ids, attn = [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append([pad_token_id] * pad + b["input_ids"])
        attn.append([0] * pad + b["attention_mask"])
    out: dict[str, Any] = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "group_id": torch.tensor([b["group_id"] for b in batch], dtype=torch.long),
        "symbol": [b["symbol"] for b in batch],
        "trade_date": [b["trade_date"] for b in batch],
        "doc_idx": torch.tensor([b["doc_idx"] for b in batch], dtype=torch.long),
    }
    for h in HORIZONS:
        out[f"label_h{h}"] = torch.tensor([b[f"label_h{h}"] for b in batch], dtype=torch.float)
        out[f"valid_h{h}"] = torch.tensor([b[f"valid_h{h}"] for b in batch], dtype=torch.bool)
        out[f"ret_h{h}"] = torch.tensor([b[f"ret_h{h}"] for b in batch], dtype=torch.float)
    return out


def make_collate_docs(tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _fn(batch: list[dict]) -> dict[str, Any]:
        return collate_docs(batch, pad_token_id=pad_id)

    return _fn


class DateGroupBatchSampler:
    """Yield indices of one or more complete trade_date groups per batch."""

    def __init__(
        self,
        df: pd.DataFrame,
        max_rows: int = 16,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.shuffle = shuffle
        self.max_rows = max_rows
        self.rng = np.random.default_rng(seed)
        # df index must match dataset positions 0..n-1
        groups: dict[Any, list[int]] = {}
        for i, d in enumerate(df["trade_date"].tolist()):
            groups.setdefault(pd.Timestamp(d), []).append(i)
        # Keep only groups with >=2
        self.groups = [idxs for idxs in groups.values() if len(idxs) >= 2]

    def __iter__(self):
        order = list(range(len(self.groups)))
        if self.shuffle:
            self.rng.shuffle(order)
        batch: list[int] = []
        for gi in order:
            g = self.groups[gi]
            if len(batch) + len(g) > self.max_rows and batch:
                yield batch
                batch = []
            batch.extend(g)
            if len(batch) >= self.max_rows:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        # Approximate
        total = sum(len(g) for g in self.groups)
        return max(1, (total + self.max_rows - 1) // self.max_rows)


class WindowRankDataset(Dataset):
    """Aggregator stage: window of cached doc embeddings + multi-horizon labels."""

    def __init__(
        self,
        windows: pd.DataFrame,
        embeds: np.ndarray,
        split: str | None = None,
        max_docs: int = MAX_WINDOW_DOCS,
    ) -> None:
        work = windows if split is None else windows[windows["split"] == split]
        work = work[work["group_size"] >= 2].reset_index(drop=True)
        self.df = work
        self.embeds = embeds
        self.max_docs = max_docs
        dates = sorted(work["trade_date"].unique())
        self.date_to_gid = {pd.Timestamp(d): i for i, d in enumerate(dates)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        doc_idxs = list(row["doc_idxs"])[: self.max_docs]
        ages = list(row["ages"])[: self.max_docs]
        n = len(doc_idxs)
        emb = np.zeros((self.max_docs, self.embeds.shape[1]), dtype=np.float32)
        age_arr = np.zeros(self.max_docs, dtype=np.int64)
        mask = np.zeros(self.max_docs, dtype=np.int64)
        if n:
            emb[:n] = self.embeds[doc_idxs]
            age_arr[:n] = np.asarray(ages, dtype=np.int64)
            mask[:n] = 1
        item = {
            "embeds": torch.from_numpy(emb),
            "ages": torch.from_numpy(age_arr),
            "mask": torch.from_numpy(mask),
            "group_id": self.date_to_gid[pd.Timestamp(row["trade_date"])],
            "symbol": str(row["symbol"]),
            "trade_date": str(pd.Timestamp(row["trade_date"]).date()),
        }
        for h in HORIZONS:
            item[f"label_h{h}"] = float(row[f"rank_h{h}"]) if pd.notna(row[f"rank_h{h}"]) else 0.0
            item[f"valid_h{h}"] = bool(row[f"valid_h{h}"]) and pd.notna(row[f"rank_h{h}"])
            item[f"ret_h{h}"] = float(row[f"ret_h{h}"]) if pd.notna(row[f"ret_h{h}"]) else 0.0
        return item


def collate_windows(batch: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "embeds": torch.stack([b["embeds"] for b in batch]),
        "ages": torch.stack([b["ages"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "group_id": torch.tensor([b["group_id"] for b in batch], dtype=torch.long),
        "symbol": [b["symbol"] for b in batch],
        "trade_date": [b["trade_date"] for b in batch],
    }
    for h in HORIZONS:
        out[f"label_h{h}"] = torch.tensor([b[f"label_h{h}"] for b in batch], dtype=torch.float)
        out[f"valid_h{h}"] = torch.tensor([b[f"valid_h{h}"] for b in batch], dtype=torch.bool)
        out[f"ret_h{h}"] = torch.tensor([b[f"ret_h{h}"] for b in batch], dtype=torch.float)
    return out
