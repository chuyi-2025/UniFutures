#!/usr/bin/env python3
"""Pairwise logistic ranking loss (XGB rank:pairwise style) for date groups."""

from __future__ import annotations

import torch
from torch import nn


def pairwise_logistic_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor | None = None,
    max_pairs: int = 2048,
) -> torch.Tensor:
    """scores/labels: [N]. Prefer higher label → higher score.

    Samples up to max_pairs preference pairs within the group.
    """
    if valid is not None:
        m = valid.bool()
        scores = scores[m]
        labels = labels[m]
    n = int(scores.numel())
    if n < 2:
        return scores.sum() * 0.0

    # All pairs i prefers j when label_i > label_j
    # For efficiency sample when n is large
    if n * (n - 1) // 2 > max_pairs:
        # Random sample of index pairs
        i = torch.randint(0, n, (max_pairs,), device=scores.device)
        j = torch.randint(0, n, (max_pairs,), device=scores.device)
        keep = i != j
        i, j = i[keep], j[keep]
        # Orient so label_i >= label_j; drop ties
        swap = labels[i] < labels[j]
        i, j = torch.where(swap, j, i), torch.where(swap, i, j)
        keep2 = labels[i] > labels[j]
        i, j = i[keep2], j[keep2]
        if i.numel() == 0:
            return scores.sum() * 0.0
        diff = scores[i] - scores[j]
        return nn.functional.softplus(-diff).mean()

    # Exact all pairs
    li = labels.unsqueeze(1)  # [N,1]
    lj = labels.unsqueeze(0)  # [1,N]
    prefer = li > lj  # [N,N]
    if not prefer.any():
        return scores.sum() * 0.0
    si = scores.unsqueeze(1)
    sj = scores.unsqueeze(0)
    diff = si - sj
    return nn.functional.softplus(-diff[prefer]).mean()


def multi_horizon_pairwise_loss(
    score_dict: dict[int, torch.Tensor],
    label_dict: dict[int, torch.Tensor],
    valid_dict: dict[int, torch.Tensor],
    group_ids: torch.Tensor,
    horizons: tuple[int, ...] = (1, 7, 14),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average pairwise loss over horizons and date groups in the batch."""
    losses = []
    metrics: dict[str, float] = {}
    unique_groups = torch.unique(group_ids)
    for h in horizons:
        h_losses = []
        scores = score_dict[h]
        labels = label_dict[h]
        valid = valid_dict[h]
        for g in unique_groups:
            mask = group_ids == g
            if int(mask.sum()) < 2:
                continue
            loss = pairwise_logistic_loss(scores[mask], labels[mask], valid[mask])
            if torch.isfinite(loss) and loss.numel() == 1:
                h_losses.append(loss)
        if h_losses:
            hl = torch.stack(h_losses).mean()
            losses.append(hl)
            metrics[f"loss_h{h}"] = float(hl.detach().item())
        else:
            metrics[f"loss_h{h}"] = 0.0
    if not losses:
        # Dummy zero that still connects to graph
        any_score = next(iter(score_dict.values()))
        return any_score.sum() * 0.0, metrics
    total = torch.stack(losses).mean()
    metrics["loss"] = float(total.detach().item())
    return total, metrics


@torch.no_grad()
def daily_rank_ic(
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    group_ids: torch.Tensor,
) -> float:
    """Mean Spearman-like IC: corr(score, label) within each date group."""
    ics = []
    for g in torch.unique(group_ids):
        mask = (group_ids == g) & valid.bool()
        if int(mask.sum()) < 3:
            continue
        s = scores[mask].float()
        y = labels[mask].float()
        s = s - s.mean()
        y = y - y.mean()
        denom = s.std(unbiased=False) * y.std(unbiased=False)
        if float(denom) < 1e-8:
            continue
        ics.append(float((s * y).mean() / denom))
    return float(sum(ics) / len(ics)) if ics else 0.0
