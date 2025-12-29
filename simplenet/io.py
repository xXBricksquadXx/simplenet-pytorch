file:simplenet/io.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model_state: dict,
    optimizer_state: dict | None,
    epoch: int,
    class_to_idx: dict,
    config: dict,
) -> None:
    """
    Save a training checkpoint that can be used to resume training or run inference.

    path: output file path (e.g. runs/simplenet_checkpoint.pt)
    model_state: model.state_dict()
    optimizer_state: optimizer.state_dict() or None
    epoch: current epoch number (1-based or 0-based; be consistent)
    class_to_idx: mapping from class name -> integer index (from ImageFolder)
    config: any metadata you want to persist (model/data/optimizer hyperparams)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model": model_state,
        "optimizer": optimizer_state,
        "epoch": epoch,
        "class_to_idx": class_to_idx,
        "config": config,
    }
    torch.save(payload, str(path))


def load_checkpoint(
    path: str | Path,
    *,
    device: str,
) -> Dict[str, Any]:
    """
    Load a dict-style checkpoint saved by save_checkpoint().
    """
    path = Path(path)
    payload = torch.load(str(path), map_location=device)

    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Checkpoint at {path} is not a dict-style checkpoint.")

    return payload
