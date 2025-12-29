from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataConfig:
    data_dir: str = "data"
    image_size: int = 64
    batch_size: int = 32
    num_workers: int = 2


def build_transforms(image_size: int) -> transforms.Compose:
    # Keep it simple and consistent for train/val/predict.
    # Add Normalize later if you want; just keep it the same everywhere.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def build_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Expects:
      data/train/<class>/*
      data/val/<class>/*
    Returns:
      train_loader, val_loader, class_to_idx
    """
    root = Path(cfg.data_dir)
    train_dir = root / "train"
    val_dir = root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected '{train_dir}' and '{val_dir}' to exist.\n"
            f"Create:\n"
            f"  {train_dir}/cat/*\n"
            f"  {train_dir}/fish/*\n"
            f"  {val_dir}/cat/*\n"
            f"  {val_dir}/fish/*\n"
        )

    tfm = build_transforms(cfg.image_size)

    train_ds = datasets.ImageFolder(train_dir, transform=tfm)
    val_ds = datasets.ImageFolder(val_dir, transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, train_ds.class_to_idx
