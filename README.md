file:README.md

# SimpleNet (PyTorch) - training, prediction, saving/loading

This repo is a compact reference implementation for:

- training a simple feedforward network ("SimpleNet") on images
- validating and computing accuracy
- predicting on a single image
- saving/loading models (full model, state_dict, checkpoint)

## 1) Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```
