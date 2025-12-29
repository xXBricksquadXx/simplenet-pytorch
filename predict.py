from __future__ import annotations

import argparse

import torch
from PIL import Image
from torchvision import transforms

from simplenet.io import load_checkpoint
from simplenet.model import SimpleNet, SimpleNetConfig


def build_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to an image file (jpg/png/etc).")
    ap.add_argument("--checkpoint", default="runs/simplenet_checkpoint.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu")

    ckpt = load_checkpoint(args.checkpoint, device=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model_cfg_dict = ckpt["config"]["model_cfg"]
    model_cfg = SimpleNetConfig(
        image_size=int(model_cfg_dict["image_size"]),
        in_channels=int(model_cfg_dict["in_channels"]),
        num_classes=int(model_cfg_dict["num_classes"]),
        hidden_dims=list(model_cfg_dict["hidden_dims"]),
        activation=str(model_cfg_dict["activation"]),
        extra_layer=bool(model_cfg_dict["extra_layer"]),
    )

    model = SimpleNet(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = Image.open(args.image).convert("RGB")
    tfm = build_transforms(model_cfg.image_size)
    x = tfm(img).unsqueeze(0).to(device)  # [1, C, H, W]

    logits = model(x)
    pred_idx = int(logits.argmax(dim=1).item())
    pred_label = idx_to_class[pred_idx]

    print(f"prediction: {pred_label} (class index {pred_idx})")


if __name__ == "__main__":
    main()
