from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataConfig:
    data_dir: str = "data"
    image_size: int = 64
    batch_size: int = 32
    num_workers: int = 2

    # Demo mode: helps when you have very few training images.
    # - Adds light augmentation on TRAIN only
    # - Optionally samples with replacement to create more batches/epoch
    demo_mode: bool = False
    demo_min_batches: int = 20  # ensures you see >1 batch per epoch on tiny datasets


def build_train_transforms(image_size: int, demo_mode: bool) -> transforms.Compose:
    if not demo_mode:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    # Light, safe augmentations for a tiny demo dataset.
    # Keep val/predict deterministic.
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
        ]
    )


def build_eval_transforms(image_size: int) -> transforms.Compose:
    # Deterministic transforms for validation/prediction.
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

    train_tfm = build_train_transforms(cfg.image_size, cfg.demo_mode)
    eval_tfm = build_eval_transforms(cfg.image_size)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfm)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfm)

    # If demo_mode is on and dataset is tiny, sample with replacement so you get
    # more batches per epoch (and new augmented views each time).
    train_sampler = None
    shuffle = True

    if cfg.demo_mode:
        # At least demo_min_batches per epoch.
        desired_samples = max(len(train_ds), cfg.batch_size * cfg.demo_min_batches)
        train_sampler = RandomSampler(train_ds, replacement=True, num_samples=desired_samples)
        shuffle = False  # cannot use shuffle with a sampler

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
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
