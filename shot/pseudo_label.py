from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_target_outputs(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    all_features: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []

    for images, _labels, indices in loader:
        images = images.to(device)
        features, logits = model(images)
        all_features.append(features.cpu())
        all_probs.append(F.softmax(logits, dim=1).cpu())
        all_indices.append(indices.cpu())

    return (
        torch.cat(all_features, dim=0),
        torch.cat(all_probs, dim=0),
        torch.cat(all_indices, dim=0),
    )


def _safe_centroids(
    features: torch.Tensor,
    weights: torch.Tensor,
    fallback_features: torch.Tensor,
) -> torch.Tensor:
    numerators = weights.t().matmul(features)
    denominators = weights.sum(dim=0).clamp_min(1e-8).unsqueeze(1)
    centroids = numerators / denominators

    empty = weights.sum(dim=0) <= 1e-8
    if empty.any():
        centroids[empty] = fallback_features[empty]
    return F.normalize(centroids, dim=1)


def generate_pseudo_labels(
    features: torch.Tensor,
    probs: torch.Tensor,
    dataset_size: int,
    indices: torch.Tensor,
    refine_rounds: int = 1,
) -> torch.Tensor:
    """Generate SHOT target pseudo labels by weighted and hard centroids."""

    normalized_features = F.normalize(features, dim=1)
    num_classes = probs.size(1)

    confidence = probs.max(dim=0).indices
    fallback = normalized_features[confidence[:num_classes]]
    centroids = _safe_centroids(normalized_features, probs, fallback)

    pseudo = torch.matmul(normalized_features, centroids.t()).argmax(dim=1)
    for _ in range(refine_rounds):
        hard_weights = F.one_hot(pseudo, num_classes=num_classes).float()
        centroids = _safe_centroids(normalized_features, hard_weights, centroids)
        pseudo = torch.matmul(normalized_features, centroids.t()).argmax(dim=1)

    labels = torch.full((dataset_size,), -1, dtype=torch.long)
    labels[indices.long()] = pseudo.long()
    return labels


@torch.no_grad()
def update_pseudo_labels(
    model,
    loader: DataLoader,
    device: torch.device,
    dataset_size: int,
    refine_rounds: int = 1,
) -> torch.Tensor:
    features, probs, indices = collect_target_outputs(model, loader, device)
    return generate_pseudo_labels(
        features=features,
        probs=probs,
        dataset_size=dataset_size,
        indices=indices,
        refine_rounds=refine_rounds,
    )
