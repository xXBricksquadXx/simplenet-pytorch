from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from simplenet.data import DataConfig, build_dataloaders
from simplenet.io import save_checkpoint
from simplenet.model import SimpleNet, SimpleNetConfig


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_one_epoch(model, loader, optimizer, loss_fn, device: str) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in tqdm(loader, desc="train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += bs

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(model, loader, loss_fn, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in tqdm(loader, desc="val", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = loss_fn(logits, targets)

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += bs

    return total_loss / total_examples, total_correct / total_examples


def build_optimizer(name: str, params, lr: float):
    name = name.lower().strip()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError("Unknown optimizer. Use: adam|sgd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--activation", default="relu", choices=["relu", "leakyrelu", "gelu", "elu"])
    ap.add_argument("--hidden-dims", nargs="*", type=int, default=[256, 128])
    ap.add_argument("--extra-layer", action="store_true")
    ap.add_argument("--device", default="cpu")

    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu")

    data_cfg = DataConfig(
    data_dir=args.data_dir,
    image_size=args.image_size,
    batch_size=args.batch_size,
    num_workers=2,
    demo_mode=True,
    demo_min_batches=20,
)

    train_loader, val_loader, class_to_idx = build_dataloaders(data_cfg)

    model_cfg = SimpleNetConfig(
        image_size=args.image_size,
        in_channels=3,
        num_classes=len(class_to_idx),
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        extra_layer=args.extra_layer,
    )
    model = SimpleNet(model_cfg).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), lr=args.lr)

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)

    # 1) Full model object (brittle but shown for learning)
    # 2) state_dict (recommended weights-only)
    # 3) checkpoint dict (best for resuming training)
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        # Save weights every epoch
        torch.save(model.state_dict(), out_dir / "simplenet_state_dict.pt")

        # Save a checkpoint every epoch (resume training)
        save_checkpoint(
            out_dir / "simplenet_checkpoint.pt",
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            class_to_idx=class_to_idx,
            config={
                "model_cfg": asdict(model_cfg),
                "data_cfg": asdict(data_cfg),
                "optimizer": args.optimizer,
                "lr": args.lr,
            },
        )

        # Save "best" full-model object (demonstration only)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, out_dir / "simplenet_full_model.pt")

    print(f"Done. Best val acc: {best_val_acc:.4f}")
    print("Saved:")
    print(" - runs/simplenet_state_dict.pt")
    print(" - runs/simplenet_checkpoint.pt")
    print(" - runs/simplenet_full_model.pt (brittle demo)")


if __name__ == "__main__":
    main()
