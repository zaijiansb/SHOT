from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shot.data import build_dataset
from shot.models import build_model
from shot.results import save_history_csv, save_json
from shot.train import evaluate, save_checkpoint, train_source_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SHOT source model.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--sample-len", type=int)
    parser.add_argument("--mods")
    parser.add_argument("--snrs")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output", default="checkpoints/source.pt")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--backbone", default="cnn", choices=["cnn"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = build_dataset(
        args.data_root,
        mods=args.mods,
        snrs=args.snrs,
        split="train",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    eval_set = build_dataset(
        args.data_root,
        mods=args.mods,
        snrs=args.snrs,
        class_to_idx=train_set.class_to_idx,
        split="val",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    num_classes = args.num_classes or len(train_set.classes)
    sample_len = args.sample_len or train_set.sample_len
    if num_classes != len(train_set.classes):
        raise ValueError(
            f"--num-classes={num_classes} does not match selected classes "
            f"({len(train_set.classes)})"
        )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        num_classes=num_classes,
        sample_len=sample_len,
        backbone=args.backbone,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_source_epoch(
            model,
            train_loader,
            optimizer,
            device,
            label_smoothing=args.label_smoothing,
        )
        eval_metrics = evaluate(model, eval_loader, device)
        best_acc = max(best_acc, eval_metrics["acc"])
        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": eval_metrics["loss"],
            "val_acc": eval_metrics["acc"],
        }
        history.append(row)
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"eval_acc={eval_metrics['acc']:.4f}"
        )

    save_checkpoint(
        model,
        args.output,
        best_acc=best_acc,
        classes=train_set.classes,
        class_to_idx=train_set.class_to_idx,
        source_data=args.data_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        history=history,
    )
    save_history_csv(history, Path(args.results_root) / "source" / "history.csv")
    save_json(
        {
            "source_data": args.data_root,
            "classes": train_set.classes,
            "class_to_idx": train_set.class_to_idx,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
            "best_acc": best_acc,
            "checkpoint": args.output,
        },
        Path(args.results_root) / "source" / "metrics.json",
    )


if __name__ == "__main__":
    main()
