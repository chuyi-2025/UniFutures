#!/usr/bin/env python3
"""Qwen3 FinSent encoder with LoRA + multi-horizon rank heads; L60 aggregators."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModel, AutoTokenizer

from config import (
    EXP_HALFLIFE,
    HIDDEN_SIZE,
    HORIZONS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGETS,
    LOOKBACK_DAYS,
)


def last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padded = bool(attention_mask[:, 0].sum().item() == attention_mask.shape[0])
    if left_padded:
        return last_hidden[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    batch = torch.arange(last_hidden.size(0), device=last_hidden.device)
    return last_hidden[batch, seq_lens]


def load_tokenizer(path: Path | str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_base_encoder(path: Path | str, use_lora: bool = True) -> nn.Module:
    """Load Qwen3Model body in BF16; optionally wrap with LoRA (adapters stay trainable)."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(
        str(path),
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",  # eager materializes 16k² masks and OOMs
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if use_lora:
        cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=list(LORA_TARGETS),
            bias="none",
        )
        model = get_peft_model(model, cfg)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        # Keep LoRA params in FP32 for stable Adam updates
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
    else:
        for p in model.parameters():
            p.requires_grad = False
    return model


class MultiHorizonRankEncoder(nn.Module):
    """Doc text → last-token L2 embed → H∈{1,7,14} scalar scores."""

    def __init__(self, base_path: Path | str, use_lora: bool = True) -> None:
        super().__init__()
        self.encoder = load_base_encoder(base_path, use_lora=use_lora)
        self.heads = nn.ModuleDict(
            {f"h{h}": nn.Linear(HIDDEN_SIZE, 1) for h in HORIZONS}
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        pooled = last_token_pool(out.last_hidden_state, attention_mask)
        return nn.functional.normalize(pooled.float(), p=2, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        emb = self.encode(input_ids, attention_mask)
        scores = {h: self.heads[f"h{h}"](emb).squeeze(-1) for h in HORIZONS}
        return emb, scores


def exp_weights(ages: torch.Tensor, half_life: float = EXP_HALFLIFE) -> torch.Tensor:
    # ages: [B, T]
    w = torch.exp(-torch.log(torch.tensor(2.0, device=ages.device)) * ages.float() / half_life)
    return w


def linear_weights(ages: torch.Tensor, lookback: int = LOOKBACK_DAYS) -> torch.Tensor:
    return (lookback + 1.0 - ages.float()).clamp(min=0.0)


class WeightedMeanAggregator(nn.Module):
    def __init__(self, mode: str = "exp") -> None:
        super().__init__()
        assert mode in {"exp", "linear"}
        self.mode = mode
        self.heads = nn.ModuleDict(
            {f"h{h}": nn.Linear(HIDDEN_SIZE, 1) for h in HORIZONS}
        )

    def aggregate(
        self,
        embeds: torch.Tensor,
        ages: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # embeds [B,T,D], ages [B,T], mask [B,T]
        if self.mode == "exp":
            w = exp_weights(ages)
        else:
            w = linear_weights(ages)
        w = w * mask.float()
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return (embeds * w.unsqueeze(-1)).sum(dim=1)

    def forward(
        self,
        embeds: torch.Tensor,
        ages: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        pooled = self.aggregate(embeds, ages, mask)
        pooled = nn.functional.normalize(pooled, p=2, dim=-1)
        return {h: self.heads[f"h{h}"](pooled).squeeze(-1) for h in HORIZONS}


class BiGRUPosAggregator(nn.Module):
    """BiGRU over doc sequence with calendar-age + relative-position embeddings."""

    def __init__(
        self,
        hidden: int = HIDDEN_SIZE,
        gru_hidden: int = 256,
        max_age: int = LOOKBACK_DAYS,
        max_pos: int = 256,
    ) -> None:
        super().__init__()
        self.age_emb = nn.Embedding(max_age + 1, 64)
        self.pos_emb = nn.Embedding(max_pos, 64)
        self.in_proj = nn.Linear(hidden + 128, gru_hidden)
        self.gru = nn.GRU(
            gru_hidden,
            gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Linear(gru_hidden * 2, 1)
        self.heads = nn.ModuleDict(
            {f"h{h}": nn.Linear(gru_hidden * 2, 1) for h in HORIZONS}
        )
        self.max_age = max_age
        self.max_pos = max_pos

    def forward(
        self,
        embeds: torch.Tensor,
        ages: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        b, t, _ = embeds.shape
        age_idx = ages.clamp(0, self.max_age)
        pos_idx = torch.arange(t, device=embeds.device).unsqueeze(0).expand(b, -1)
        pos_idx = pos_idx.clamp(0, self.max_pos - 1)
        x = torch.cat(
            [embeds, self.age_emb(age_idx), self.pos_emb(pos_idx)], dim=-1
        )
        x = torch.tanh(self.in_proj(x))
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=t)
        attn_logits = self.attn(out).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask.bool(), -1e4)
        attn = torch.softmax(attn_logits, dim=1)
        pooled = (out * attn.unsqueeze(-1)).sum(dim=1)
        return {h: self.heads[f"h{h}"](pooled).squeeze(-1) for h in HORIZONS}


def build_aggregator(name: str) -> nn.Module:
    if name in {"exp", "linear"}:
        return WeightedMeanAggregator(mode=name)
    if name == "bigru_pos":
        return BiGRUPosAggregator()
    raise ValueError(name)
