# BactoRCANet

BactoRCANet is a standalone Python package for image-based multi-task bacterial concentration analysis. It replaces the single-script workflow with configurable modules for data loading, transforms, model definition, losses, training, evaluation, reporting, and inference.

Package mark: `jiangxuyue`.

## Install

```bash
pip install -e .
```

## Train

```bash
bactorcanet train --config configs/default.yaml --data-dir data --output-dir runs/default
```

## Evaluate

```bash
bactorcanet evaluate --checkpoint runs/default/best.pt --data-dir data --split test --output-dir reports/test
```

## Predict

```bash
bactorcanet predict --checkpoint runs/default/best.pt --image path/to/image.png --output prediction.json
```

## Data

Each split contains a CSV file named after the split and columns `image_path`, `x`, `y`, and `z`. The values in `x`, `y`, and `z` are used both as regression targets and as source values for binary labels.


## Legacy note

The previous script `multitask_cnn.py` remains available for reference; new work should use the `bactorcanet` package.
