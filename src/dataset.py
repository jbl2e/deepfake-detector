# src/dataset.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import cv2

PathLabelList = List[Tuple[str, int]]  # [(img_path, label), ...]


class DeepfakeDataset(Dataset):
    """
    Moved from concatenated (DeepfakeDataset).
    Keeps:
    - cv2.imread + BGR->RGB
    - albumentations transform(image=...)
    - retry logic (up to max_attempts)
    """
    def __init__(self, data_list: PathLabelList, transform=None, max_attempts: int = 10):
        self.data_list = data_list
        self.transform = transform
        self.max_attempts = int(max_attempts)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        attempts = 0
        current_idx = idx

        while attempts < self.max_attempts:
            img_path, label = self.data_list[current_idx]
            try:
                image = cv2.imread(img_path)
                if image is None:
                    raise FileNotFoundError(f"Failed to read: {img_path}")

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if self.transform is not None:
                    out = self.transform(image=image)
                    image = out["image"]

                # BCEWithLogitsLoss expects float labels (0.0/1.0)
                return image, torch.tensor(label, dtype=torch.float32)

            except Exception:
                current_idx = random.randint(0, len(self.data_list) - 1)
                attempts += 1

        raise RuntimeError(f"Could not fetch a valid image after {self.max_attempts} attempts (last={img_path}).")


def get_weighted_sampler(data_list: PathLabelList) -> WeightedRandomSampler:
    """
    Moved from concatenated get_sampler().
    """
    labels = np.array([y for _, y in data_list], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)  # assume binary
    class_counts = np.clip(class_counts, 1, None)   # avoid div-by-zero
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    sample_weights = class_weights[torch.from_numpy(labels)]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


@dataclass
class LoaderConfig:
    batch_size: int = 48
    num_workers: int = 8
    pin_memory: bool = True
    drop_last: bool = True


def create_loaders(
    train_data: PathLabelList,
    valid_data: PathLabelList,
    train_transform,
    valid_transform,
    cfg: LoaderConfig,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Wraps your original DataLoader creation.
    """
    train_ds = DeepfakeDataset(train_data, transform=train_transform)
    valid_ds = DeepfakeDataset(valid_data, transform=valid_transform)

    if use_weighted_sampler:
        sampler = get_weighted_sampler(train_data)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
        )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    return train_loader, valid_loader
