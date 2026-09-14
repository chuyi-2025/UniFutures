#!/usr/bin/env python3
"""finance_zh / random BERT over frozen doc embedding sequences."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import BertConfig, BertForSequenceClassification


class HierarchicalDocClassifier(nn.Module):
    """
    Streams → L2 → concat|gate → LN+Linear(768) + age emb + CLS
    → BertForSequenceClassification(inputs_embeds).
    """

    def __init__(
        self,
        stream_dims: list[int],
        bert_path: Path | str,
        num_labels: int = 3,
        fusion: str = "concat",
        bert_init: str = "finance_zh",
        use_age: bool = True,
        max_age: int = 60,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stream_dims = list(stream_dims)
        self.fusion = fusion
        self.use_age = use_age
        self.n_streams = len(stream_dims)

        if fusion == "gate":
            self.stream_projs = nn.ModuleList([nn.Linear(d, 768) for d in stream_dims])
            self.gate = nn.Linear(768 * self.n_streams, self.n_streams)
            self.proj = nn.Identity()
        else:
            self.stream_projs = None
            self.gate = None
            in_dim = int(sum(stream_dims))
            self.proj = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 768),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(768),
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        nn.init.normal_(self.cls_token, std=0.02)
        self.age_emb = nn.Embedding(max_age + 1, 768) if use_age else None

        if bert_init == "random":
            cfg = BertConfig.from_pretrained(str(bert_path))
            cfg.num_labels = num_labels
            self.bert = BertForSequenceClassification(cfg)
        else:
            self.bert = BertForSequenceClassification.from_pretrained(
                str(bert_path),
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )

    def _fuse(self, stream_tensors: list[torch.Tensor]) -> torch.Tensor:
        if self.fusion == "gate":
            assert self.stream_projs is not None and self.gate is not None
            projs = [proj(x) for proj, x in zip(self.stream_projs, stream_tensors)]
            pooled = [p.mean(dim=1) for p in projs]
            w = torch.softmax(self.gate(torch.cat(pooled, dim=-1)), dim=-1)
            stacked = torch.stack(projs, dim=-1)  # [B,T,768,n]
            return (stacked * w.unsqueeze(1).unsqueeze(2)).sum(dim=-1)
        return torch.cat(stream_tensors, dim=-1)

    def forward(
        self,
        stream_embeds: list[torch.Tensor],
        attention_mask: torch.Tensor,
        ages: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        norms = [
            torch.nn.functional.normalize(x.float(), p=2, dim=-1).to(dtype=x.dtype)
            for x in stream_embeds
        ]
        fused = self._fuse(norms)
        hidden = self.proj(fused) if self.fusion != "gate" else fused

        if self.use_age and ages is not None and self.age_emb is not None:
            hidden = hidden + self.age_emb(ages.clamp(0, self.age_emb.num_embeddings - 1))

        bsz = hidden.size(0)
        cls = self.cls_token.expand(bsz, -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        cls_mask = torch.ones(
            bsz, 1, device=attention_mask.device, dtype=attention_mask.dtype
        )
        mask = torch.cat([cls_mask, attention_mask], dim=1)
        return self.bert(inputs_embeds=hidden, attention_mask=mask, labels=labels)
