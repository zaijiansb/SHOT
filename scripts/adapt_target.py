from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shot.data import IndexedDataset, build_dataset
from shot.evaluation import (
    accuracy_by_snr,
    collect_outputs,
    save_per_snr_accuracy,
    save_tsne_plot,
)
from shot.models import build_model
from shot.results import make_target_result_dir, save_history_csv, save_json
from shot.train import adapt_target, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt a SHOT source model to a target domain.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--mods")
    parser.add_argument("--snrs")
    parser.add_argument("--target-split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--eval-split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--pseudo-interval", type=int, default=1)
    parser.add_argument("--refine-rounds", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-target-labels", action="store_true", default=True)
    parser.add_argument("--no-eval-target-labels", dest="eval_target_labels", action="store_false")
    parser.add_argument("--no-tsne", action="store_true")
    parser.add_argument("--tsne-samples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.source_checkpoint, map_location="cpu")
    config = checkpoint["config"]
    metadata = checkpoint.get("metadata", {})
    class_to_idx = metadata.get("class_to_idx")
    source_data = metadata.get("source_data")
    result_dir = make_target_result_dir(args.results_root, source_data, args.data_root)
    output = args.output or str(result_dir / "target_checkpoint.pt")

    target_base = build_dataset(
        args.data_root,
        mods=args.mods,
        snrs=args.snrs,
        class_to_idx=class_to_idx,
        split=args.target_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    eval_base = build_dataset(
        args.data_root,
        mods=args.mods,
        snrs=args.snrs,
        class_to_idx=class_to_idx,
        split=args.eval_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    train_set = IndexedDataset(target_base)
    eval_set = IndexedDataset(eval_base)
    pseudo_set = IndexedDataset(target_base)
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
    pseudo_loader = DataLoader(
        pseudo_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.freeze_classifier()

    before_outputs = None
    if args.eval_target_labels:
        before_outputs = collect_outputs(model, eval_loader, device)

    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(
        trainable,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    history = adapt_target(
        model=model,
        train_loader=train_loader,
        pseudo_loader=pseudo_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        beta=args.beta,
        pseudo_interval=args.pseudo_interval,
        refine_rounds=args.refine_rounds,
        eval_loader=eval_loader if args.eval_target_labels else None,
    )

    for metrics in history:
        print(
            f"epoch={int(metrics['epoch'])} "
            f"loss={metrics['loss']:.4f} "
            f"im={metrics['im']:.4f} "
            f"pseudo_ce={metrics['pseudo_ce']:.4f} "
            f"eval_acc={metrics.get('eval_acc', 0.0):.4f}"
        )

    final_outputs = None
    per_snr_rows = []
    if args.eval_target_labels:
        final_outputs = collect_outputs(model, eval_loader, device)
        if "snrs" in final_outputs:
            per_snr_rows = accuracy_by_snr(
                final_outputs["labels"],
                final_outputs["preds"],
                final_outputs["snrs"],
            )
            save_per_snr_accuracy(per_snr_rows, result_dir / "accuracy_by_snr.csv")

        if not args.no_tsne and before_outputs is not None:
            save_tsne_plot(
                before_outputs["features"],
                before_outputs["labels"],
                result_dir / "tsne_before.png",
                title="Before Adaptation",
                max_samples=args.tsne_samples,
            )
            save_tsne_plot(
                final_outputs["features"],
                final_outputs["labels"],
                result_dir / "tsne_after.png",
                title="After Adaptation",
                max_samples=args.tsne_samples,
            )

    save_checkpoint(
        model,
        output,
        source_checkpoint=args.source_checkpoint,
        target_data=args.data_root,
        target_split=args.target_split,
        eval_split=args.eval_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        history=history,
    )
    save_history_csv(history, result_dir / "history.csv")
    save_json(
        {
            "source_checkpoint": args.source_checkpoint,
            "source_data": source_data,
            "target_data": args.data_root,
            "target_split": args.target_split,
            "eval_split": args.eval_split,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
            "classes": target_base.classes,
            "checkpoint": output,
            "accuracy_by_snr": per_snr_rows,
            "history": history,
        },
        result_dir / "metrics.json",
    )


if __name__ == "__main__":
    main()
