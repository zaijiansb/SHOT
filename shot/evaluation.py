from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def _snrs_from_loader(loader: DataLoader, indices: torch.Tensor | None) -> torch.Tensor | None:
    dataset = loader.dataset
    base = getattr(dataset, "dataset", dataset)
    sample_snrs = getattr(base, "sample_snrs", None)
    if sample_snrs is None:
        return None
    if indices is None:
        return sample_snrs
    return sample_snrs[indices.cpu()]


@torch.no_grad()
def collect_outputs(model, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    features_parts: list[torch.Tensor] = []
    logits_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    snr_parts: list[torch.Tensor] = []

    for batch in loader:
        if len(batch) == 3:
            signals, labels, indices = batch
        else:
            signals, labels = batch
            indices = None

        signals = signals.to(device)
        features, logits = model(signals)
        features_parts.append(features.cpu())
        logits_parts.append(logits.cpu())
        labels_parts.append(labels.cpu())

        snrs = _snrs_from_loader(loader, indices)
        if snrs is not None:
            snr_parts.append(snrs.cpu())

    logits_all = torch.cat(logits_parts, dim=0)
    output = {
        "features": torch.cat(features_parts, dim=0).numpy(),
        "logits": logits_all.numpy(),
        "preds": logits_all.argmax(dim=1).numpy(),
        "labels": torch.cat(labels_parts, dim=0).numpy(),
    }
    if snr_parts:
        output["snrs"] = torch.cat(snr_parts, dim=0).numpy()
    return output


def accuracy_by_snr(labels: np.ndarray, preds: np.ndarray, snrs: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for snr in sorted(np.unique(snrs).tolist()):
        mask = snrs == snr
        rows.append(
            {
                "snr": float(snr),
                "accuracy": float((preds[mask] == labels[mask]).mean()),
                "count": float(mask.sum()),
            }
        )
    return rows


def save_per_snr_accuracy(rows: list[dict[str, float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["snr", "accuracy", "count"])
        writer.writeheader()
        writer.writerows(rows)


def save_tsne_plot(
    features: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
    title: str,
    class_names: list[str] | None = None,
    max_samples: int = 2000,
    seed: int = 0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if features.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        selected = rng.choice(features.shape[0], size=max_samples, replace=False)
        features = features[selected]
        labels = labels[selected]

    from matplotlib import pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from sklearn.manifold import TSNE

    num_classes = len(class_names) if class_names is not None else int(labels.max()) + 1
    base_cmap = plt.get_cmap("tab20", num_classes)
    colors = [base_cmap(class_id) for class_id in range(num_classes)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, num_classes + 0.5, 1), num_classes)

    embedded = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=30,
        random_state=seed,
    ).fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=labels,
        s=6,
        cmap=cmap,
        norm=norm,
        alpha=0.8,
    )
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    if class_names is not None:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=colors[class_id],
                markersize=6,
                label=class_name,
            )
            for class_id, class_name in enumerate(class_names)
        ]
        plt.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
        )
    else:
        plt.colorbar(scatter, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
