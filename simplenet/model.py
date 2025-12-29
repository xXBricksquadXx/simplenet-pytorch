from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    name = name.lower().strip()
    if name == "relu":
        return nn.ReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU(0.1)
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unknown activation: {name!r} (try relu|leakyrelu|gelu|elu)")


@dataclass(frozen=True)
class SimpleNetConfig:
    image_size: int = 64
    in_channels: int = 3
    num_classes: int = 2
    hidden_dims: List[int] = None  # e.g. [256, 128]
    activation: str = "relu"
    extra_layer: bool = False

    def resolved_hidden_dims(self) -> List[int]:
        if self.hidden_dims is None:
            return [256, 128]
        return list(self.hidden_dims)


class SimpleNet(nn.Module):
    """
    A simple MLP classifier:
      image -> flatten -> Linear -> Act -> Linear -> Act -> (optional extra layer) -> Linear(num_classes)

    This mirrors the "artisanal construction" idea:
    - image_size changes parameter count
    - hidden_dims changes capacity
    - activation swaps behavior
    - extra_layer adds depth
    """

    def __init__(self, cfg: SimpleNetConfig):
        super().__init__()
        hidden_dims = cfg.resolved_hidden_dims()
        act = _activation(cfg.activation)

        in_features = cfg.in_channels * cfg.image_size * cfg.image_size

        layers: List[nn.Module] = []
        prev = in_features

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(_activation(cfg.activation))
            prev = h

        if cfg.extra_layer:
            # A small extra layer; tweak to taste
            layers.append(nn.Linear(prev, prev))
            layers.append(_activation(cfg.activation))

        layers.append(nn.Linear(prev, cfg.num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.view(x.size(0), -1)
        return self.net(x)
