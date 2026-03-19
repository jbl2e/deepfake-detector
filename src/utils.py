# src/utils.py
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ----------------------------
# (A) Augmentations (그대로 유지)
# ----------------------------
def get_transforms(
    input_size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> Tuple[A.Compose, A.Compose]:
    train_transform = A.Compose([
        A.Resize(input_size, input_size, p=1.0),

        A.HorizontalFlip(p=0.7),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=(-30, 30),
            p=0.7,
        ),

        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.8),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),

        A.OneOf([
            A.ImageCompression(quality_range=(40, 70), p=0.8),
            A.MotionBlur(blur_limit=7, p=0.5),
        ], p=0.5),

        A.GaussianBlur(blur_limit=(3, 3), p=0.5),

        A.Normalize(mean=mean, std=std, p=1.0),
        ToTensorV2(p=1.0),
    ])

    valid_transform = A.Compose([
        A.Resize(input_size, input_size, p=1.0),
        A.Normalize(mean=mean, std=std, p=1.0),
        ToTensorV2(p=1.0),
    ])

    return train_transform, valid_transform


# ----------------------------
# (B) Safe ZIP extract only
# ----------------------------
def _is_within_directory(base: Path, target: Path) -> bool:
    try:
        base = base.resolve()
        target = target.resolve()
        return str(target).startswith(str(base))
    except Exception:
        return False


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = dest_dir / member.filename
            if not _is_within_directory(dest_dir, member_path):
                raise RuntimeError(f"[ZIP] Unsafe path detected: {member.filename}")
        zf.extractall(dest_dir)


def extract_zip(archive_path: str | Path, dest_dir: str | Path) -> Path:
    """
    Extracts .zip to dest_dir.
    Returns dest_dir as Path.
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    if archive_path.suffix.lower() != ".zip":
        raise ValueError(f"Only .zip is supported now: {archive_path.name}")
    safe_extract_zip(archive_path, dest_dir)
    return dest_dir
