from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import information_maximization_loss, pseudo_label_loss
from .models import SHOTNet
from .pseudo_label import update_pseudo_labels


def train_source_epoch(
    model: SHOTNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float = 0.1,
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch in tqdm(loader, desc="source", leave=False):
        if len(batch) == 3:
            images, labels, _indices = batch
        else:
            images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        _features, logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size

    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": total_correct / max(total_seen, 1),
    }


def adapt_target_epoch(
    model: SHOTNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pseudo_labels: torch.Tensor,
    device: torch.device,
    beta: float = 0.3,
) -> dict[str, float]:
    model.train()
    model.freeze_classifier()

    total_loss = 0.0
    total_im = 0.0
    total_pl = 0.0
    total_seen = 0

    for images, _labels, indices in tqdm(loader, desc="target", leave=False):
        images = images.to(device)
        labels = pseudo_labels[indices].to(device)
        valid = labels >= 0

        _features, logits = model(images)
        im_loss, _entropy, _diversity = information_maximization_loss(logits)
        if valid.any():
            pl_loss = pseudo_label_loss(logits[valid], labels[valid])
        else:
            pl_loss = logits.new_tensor(0.0)
        loss = im_loss + beta * pl_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_im += im_loss.item() * batch_size
        total_pl += pl_loss.item() * batch_size
        total_seen += batch_size

    return {
        "loss": total_loss / max(total_seen, 1),
        "im": total_im / max(total_seen, 1),
        "pseudo_ce": total_pl / max(total_seen, 1),
    }


@torch.no_grad()
def evaluate(model: SHOTNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_correct = 0
    total_seen = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc="eval", leave=False):
        if len(batch) == 3:
            images, labels, _indices = batch
        else:
            images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        _features, logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size

    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": total_correct / max(total_seen, 1),
    }


def adapt_target(
    model: SHOTNet,
    train_loader: DataLoader,
    pseudo_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    beta: float,
    pseudo_interval: int = 1,
    refine_rounds: int = 1,
    eval_loader: DataLoader | None = None,
) -> list[dict[str, float]]:
    model.freeze_classifier()
    history: list[dict[str, float]] = []

    pseudo_labels = update_pseudo_labels(
        model=model,
        loader=pseudo_loader,
        device=device,
        dataset_size=len(train_loader.dataset),
        refine_rounds=refine_rounds,
    )

    for epoch in range(1, epochs + 1):
        if epoch == 1 or epoch % pseudo_interval == 0:
            pseudo_labels = update_pseudo_labels(
                model=model,
                loader=pseudo_loader,
                device=device,
                dataset_size=len(train_loader.dataset),
                refine_rounds=refine_rounds,
            )

        metrics = adapt_target_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            pseudo_labels=pseudo_labels,
            device=device,
            beta=beta,
        )
        metrics["epoch"] = float(epoch)
        if eval_loader is not None:
            eval_metrics = evaluate(model, eval_loader, device)
            metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
        history.append(metrics)

    return history


def save_checkpoint(model: SHOTNet, path: str | Path, **metadata) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(model.config),
            "metadata": metadata,
        },
        path,
    )
