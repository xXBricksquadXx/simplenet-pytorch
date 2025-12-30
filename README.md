## ![Header](assets/header-banner.png)

# SimpleNet (PyTorch) — training, prediction, saving/loading

A compact, practical reference for the basics covered in the chapter:

- train a simple feedforward network (MLP) on images ("SimpleNet")
- validate and compute accuracy
- run single-image predictions
- save and restore models (state_dict + checkpoint + full-model demo)

This repo is intentionally small so you can iterate on:

- layer sizes / depth
- activation functions
- optimizer + learning rate
- batch size
- input image size (affects parameter count)

---

## Repo layout

```
assets/                 # README cosmetics (NOT training data)
  header-banner.png
  icon.png

data/                   # training/validation images (gitignored)
  train/                # "clean" images
    cat/
    fish/
  val/                  # "challenge" images
    cat/
    fish/

runs/                   # outputs/checkpoints (gitignored)
  simplenet_checkpoint.pt
  simplenet_state_dict.pt
  simplenet_full_model.pt

simplenet/              # package
  __init__.py
  data.py
  io.py
  model.py

train.py
predict.py
requirements.txt
```

Notes:

- `data/` is for model training only.
- `assets/` is for visuals (banners/icons/screenshots) so they never leak into training.

---

## 1) Setup

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Sanity:

```bash
python -c "import torch, torchvision; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

---

## 2) Dataset (ImageFolder)

This uses `torchvision.datasets.ImageFolder`, so **PNG is fine**.

Expected folder structure:

```
data/
  train/
    cat/   *.png
    fish/  *.png
  val/
    cat/   *.png
    fish/  *.png
```

chosen convention:

- `data/train/*` = **clean** examples
- `data/val/*` = **challenge** examples

---

## 3) Train

Minimal smoke test:

```bash
python train.py --data-dir data --device cpu --epochs 2 --batch-size 16 --image-size 64
```

Outputs:

- `runs/simplenet_state_dict.pt` (recommended weights-only)
- `runs/simplenet_checkpoint.pt` (recommended for resume + inference)
- `runs/simplenet_full_model.pt` (brittle demo; breaks if code structure changes)

---

## 4) Predict

Single file:

```bash
python predict.py --image "data\val\cat\cat-challenge.png" --checkpoint "runs\simplenet_checkpoint.pt" --device cpu
```

Directory input (picks the first image in the folder):

```bash
python predict.py --image "data\val\cat" --checkpoint "runs\simplenet_checkpoint.pt" --device cpu
```

---

## 5) Saving & loading (what to remember)

### A) Full model object (works, but brittle)

```python
torch.save(model, "runs/simplenet_full_model.pt")
model = torch.load("runs/simplenet_full_model.pt", map_location=device)
```

### B) Weights only (recommended)

```python
torch.save(model.state_dict(), "runs/simplenet_state_dict.pt")
model = SimpleNet(cfg)
model.load_state_dict(torch.load("runs/simplenet_state_dict.pt", map_location=device))
```

### C) Checkpoint dict (recommended for real work)

Includes:

- model weights
- optimizer state
- epoch
- `class_to_idx`
- config metadata

This is what `predict.py` uses.

---

## 6) Demo mode augmentation (for tiny datasets)

If you only have a few images per class, training will look "stuck" (e.g. accuracy ~0.50) because the model can’t learn much.

This repo supports a **demo mode** in `simplenet/data.py` that:

- applies light random augmentations to the **train** split
- (optionally) uses sampling-with-replacement so you get more than 1 batch per epoch

Recommended when you have < ~50 images per class.

---

## 7) Experiments to try

- Wider layers: `--hidden-dims 512 256`
- Narrower layers: `--hidden-dims 128 64`
- Add depth: `--extra-layer`
- Activation: `--activation gelu` (or `leakyrelu`, `elu`)
- Optimizer: `--optimizer sgd --lr 0.01`
- Batch size: `--batch-size 16` vs `64`
- Image size: `--image-size 32` vs `128`

---
