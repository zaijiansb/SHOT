from __future__ import annotations

import torch
import torch.nn.functional as F


def information_maximization_loss(
    logits: torch.Tensor,
    epsilon: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return SHOT IM loss and its entropy/diversity parts.

    The minimized objective is conditional entropy plus diversity:
    E[sum -p log p] + sum mean_p log mean_p.
    """

    probs = F.softmax(logits, dim=1)
    entropy = torch.sum(-probs * torch.log(probs + epsilon), dim=1).mean()
    mean_probs = probs.mean(dim=0)
    diversity = torch.sum(mean_probs * torch.log(mean_probs + epsilon))
    return entropy + diversity, entropy, diversity


def pseudo_label_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)
