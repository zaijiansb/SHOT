from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _parse_csv_values(value: str | None, cast=str):
    if value is None or value == "":
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _split_samples(
    samples: np.ndarray,
    split: str,
    train_ratio: float,
    val_ratio: float,
) -> np.ndarray:
    if train_ratio <= 0.0 or val_ratio < 0.0 or train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio and val_ratio must satisfy 0 < train, 0 <= val, train + val <= 1")
    if split == "all":
        return samples

    train_end = int(samples.shape[0] * train_ratio)
    val_end = train_end + int(samples.shape[0] * val_ratio)
    if split == "train":
        return samples[:train_end]
    if split == "val":
        return samples[train_end:val_end]
    if split == "test":
        return samples[val_end:]

    raise ValueError("split must be one of: train, val, test, all")


class IndexedDataset(Dataset):
    """Wrap a dataset so each item also returns its stable integer index."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label, index


class RadioDatDataset(Dataset):
    """Dataset for modulation-recognition pickle .dat files.

    Expected file format:
        dict[(modulation_name, snr)] -> ndarray[num_samples, 2, sample_len]
    """

    def __init__(
        self,
        path: str | Path,
        mods: list[str] | None = None,
        snrs: list[int] | None = None,
        class_to_idx: dict[str, int] | None = None,
        split: str = "all",
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ):
        self.path = Path(path)
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        with self.path.open("rb") as file:
            raw = pickle.load(file, encoding="latin1")

        available_mods = sorted({mod for mod, _snr in raw.keys()})
        selected_mods = mods or available_mods
        self.class_to_idx = class_to_idx or {
            mod: idx for idx, mod in enumerate(selected_mods)
        }
        self.classes = [
            mod for mod, _idx in sorted(self.class_to_idx.items(), key=lambda item: item[1])
        ]
        self.snrs = sorted(snrs if snrs is not None else {snr for _mod, snr in raw.keys()})

        data_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        snr_parts: list[np.ndarray] = []
        for mod in selected_mods:
            if mod not in self.class_to_idx:
                continue
            for snr in self.snrs:
                samples = raw.get((mod, snr))
                if samples is None:
                    continue
                samples = np.asarray(samples, dtype=np.float32)
                if samples.ndim != 3 or samples.shape[1] != 2:
                    raise ValueError(
                        f"{self.path} key {(mod, snr)} has shape {samples.shape}, "
                        "expected [N, 2, sample_len]"
                    )
                samples = _split_samples(
                    samples,
                    split=split,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                )
                data_parts.append(samples)
                label_parts.append(
                    np.full(samples.shape[0], self.class_to_idx[mod], dtype=np.int64)
                )
                snr_parts.append(np.full(samples.shape[0], snr, dtype=np.int64))

        if not data_parts:
            raise ValueError(f"No samples selected from {self.path}")

        self.data = torch.from_numpy(np.concatenate(data_parts, axis=0))
        self.labels = torch.from_numpy(np.concatenate(label_parts, axis=0))
        self.sample_snrs = torch.from_numpy(np.concatenate(snr_parts, axis=0))
        self.sample_len = int(self.data.shape[-1])

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]


def build_radio_dat(
    path: str | Path,
    mods: str | list[str] | None = None,
    snrs: str | list[int] | None = None,
    class_to_idx: dict[str, int] | None = None,
    split: str = "all",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> RadioDatDataset:
    if isinstance(mods, str):
        mods = _parse_csv_values(mods, str)
    if isinstance(snrs, str):
        snrs = _parse_csv_values(snrs, int)
    return RadioDatDataset(
        path=path,
        mods=mods,
        snrs=snrs,
        class_to_idx=class_to_idx,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )


def build_dataset(
    path: str | Path,
    mods: str | list[str] | None = None,
    snrs: str | list[int] | None = None,
    class_to_idx: dict[str, int] | None = None,
    split: str = "all",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dataset:
    path = Path(path)
    if path.suffix.lower() == ".dat":
        return build_radio_dat(
            path,
            mods=mods,
            snrs=snrs,
            class_to_idx=class_to_idx,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

    raise ValueError(f"Unsupported dataset path: {path}")


"""
ImageFolder helpers kept for future image experiments. They import torchvision
only when called, so radio-signal runs do not require torchvision.
"""


def image_transforms(train: bool = True, image_size: int = 224):
    from torchvision import transforms

    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_imagefolder(root: str | Path, train: bool = True, image_size: int = 224):
    from torchvision.datasets import ImageFolder

    return ImageFolder(str(root), transform=image_transforms(train, image_size))
