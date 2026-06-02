# BactoRCANet: Functionally Equivalent Reimplementation Package Specification

> Target paper to reproduce: W. Xue, R. Li, K. Wang, K. Song, Z. Zhang, et al., **"Deep Learning-Enhanced DNAzyme-Driven Rolling-Circle Amplification Encoding for Multibacterial Detection,"** *Angewandte Chemie International Edition* 65, no. 22 (2026): e4446117. DOI: <https://doi.org/10.1002/anie.4446117>.
>
> Proposed English package name: **BactoRCANet** (`bactorcanet`).
>
> Scope: This document defines a clean-room, functionally equivalent Python package that reproduces the machine-learning workflow represented by the current repository: image-based multi-task bacterial detection with three binary classification heads and three continuous concentration regression heads.

---

## 1. Reproduction Goal

`BactoRCANet` should provide a maintainable, testable, and reusable implementation of the deep-learning portion of the paper's workflow. The rewritten package should preserve the functional behavior of the existing single-file script while replacing it with a modular package architecture.

The package should support:

1. Loading image datasets from `train/`, `val/`, and `test/` folders.
2. Reading per-split CSV files with `image_path`, `x`, `y`, and `z` columns.
3. Converting continuous concentration values into binary presence/absence labels.
4. Training a multi-task CNN for:
   - three binary classification tasks: `x`, `y`, and `z` bacterial target presence;
   - three continuous regression tasks: `x`, `y`, and `z` concentration prediction.
5. Evaluating classification, combination-label, and regression performance.
6. Saving best/final checkpoints and metrics artifacts.
7. Loading a checkpoint for single-image or batch inference.

Because the full experimental dataset is not publicly included in this repository, this package should be designed to reproduce the pipeline when users provide either the original dataset under permission from the authors or a compatible replacement dataset.

---

## 2. Package Name and Rationale

### 2.1 Name

**BactoRCANet**

### 2.2 Import Name

```python
import bactorcanet
```

### 2.3 Meaning

- **Bacto**: bacterial detection.
- **RCA**: rolling-circle amplification.
- **Net**: neural-network model.

The name is English, concise, searchable, and aligned with the paper's theme: DNAzyme-driven RCA encoding plus deep-learning-based multibacterial detection.

---

## 3. Functional Equivalence to the Current Repository

The current repository contains a monolithic `multitask_cnn.py` script and a pretrained checkpoint. `BactoRCANet` should preserve the following behavior while improving structure.

| Existing capability | Rewritten package component |
| --- | --- |
| Hard-coded `base_dir = './data'` | Configurable `DataConfig.data_dir` |
| CSV loading from `train/train.csv`, `val/val.csv`, `test/test.csv` | `bactorcanet.data.load_split()` and `load_dataset()` |
| Custom PyTorch dataset | `bactorcanet.data.ConcentrationDataset` |
| Training transforms and validation/test transforms | `bactorcanet.transforms.build_transforms()` |
| Multi-task CNN model | `bactorcanet.models.MultiTaskConcentrationCNN` |
| Classification and regression losses | `bactorcanet.training.MultiTaskLoss` |
| Training loop with checkpointing | `bactorcanet.training.Trainer` |
| Evaluation, mistakes CSV, all-results CSV | `bactorcanet.evaluation.evaluate()` |
| Regression plots and CSV outputs | `bactorcanet.reporting` |
| `predict_unknown_image()` | `bactorcanet.inference.Predictor.predict_image()` |

---

## 4. Recommended Repository Layout

```text
bactorcanet/
  pyproject.toml
  README.md
  LICENSE
  configs/
    default.yaml
    paper_reproduction.yaml
  src/
    bactorcanet/
      __init__.py
      cli.py
      config.py
      data.py
      transforms.py
      models.py
      losses.py
      metrics.py
      training.py
      evaluation.py
      inference.py
      reporting.py
      utils.py
  tests/
    test_data.py
    test_model_shapes.py
    test_metrics.py
    test_inference.py
  examples/
    train_from_csv.py
    predict_single_image.py
    export_metrics.py
  docs/
    data_format.md
    reproduction_protocol.md
    model_card.md
```

---

## 5. Dataset Contract

### 5.1 Folder Structure

The package should expect this default structure:

```text
data/
  train/
    train.csv
    image_001.png
    image_002.png
    ...
  val/
    val.csv
    ...
  test/
    test.csv
    ...
```

Alternative paths should be configurable from CLI arguments or YAML config.

### 5.2 CSV Schema

Each split CSV must contain:

| Column | Type | Required | Description |
| --- | --- | --- | --- |
| `image_path` | string | yes | Absolute path or path relative to the CSV file/dataset root. |
| `x` | float | yes | Continuous concentration for target X. |
| `y` | float | yes | Continuous concentration for target Y. |
| `z` | float | yes | Continuous concentration for target Z. |

### 5.3 Classification Label Rule

A functionally equivalent implementation should convert concentrations to binary labels with a clear, configurable threshold:

```python
label = 1 if concentration > threshold else 0
```

Default threshold: `0.0`.

This preserves the current logic implied by concentration-based presence/absence classification while making the threshold explicit and auditable.

### 5.4 Missing or Invalid Images

The loader should support two modes:

- `strict=True`: fail immediately if any image path is missing.
- `strict=False`: drop missing images and log a warning.

For reproducibility, the default should be `strict=True` in published experiments and `strict=False` only for exploratory runs.

---

## 6. Core Model Design

### 6.1 Model Interface

```python
class MultiTaskConcentrationCNN(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "class_x": logits_x,
            "class_y": logits_y,
            "class_z": logits_z,
            "reg_x": concentration_x,
            "reg_y": concentration_y,
            "reg_z": concentration_z,
        }
```

### 6.2 Output Shapes

For batch size `B`:

| Output | Shape | Meaning |
| --- | --- | --- |
| `class_x` | `[B, 2]` | binary logits for target X |
| `class_y` | `[B, 2]` | binary logits for target Y |
| `class_z` | `[B, 2]` | binary logits for target Z |
| `reg_x` | `[B, 1]` | predicted continuous concentration X |
| `reg_y` | `[B, 1]` | predicted continuous concentration Y |
| `reg_z` | `[B, 1]` | predicted continuous concentration Z |

### 6.3 Architecture Requirements

The rewritten package should expose a default CNN architecture equivalent in purpose to the original script:

1. Shared image feature extractor.
2. Classification heads for X/Y/Z.
3. Regression heads for X/Y/Z.
4. Optional dropout and batch normalization.
5. Configurable input image size.
6. Checkpoint compatibility layer, if needed, for loading the existing `best_concentration_model.pth`.

---

## 7. Training Objective

### 7.1 Classification Loss

Use cross-entropy loss independently for each classification target:

```text
L_class = CE(class_x, label_x) + CE(class_y, label_y) + CE(class_z, label_z)
```

### 7.2 Regression Loss

Use mean squared error or smooth L1 loss independently for each concentration target:

```text
L_reg = MSE(reg_x, x) + MSE(reg_y, y) + MSE(reg_z, z)
```

### 7.3 Combined Loss

```text
L_total = class_weight * L_class + reg_weight * L_reg
```

The default weights should match the original training behavior as closely as possible after auditing the original script. Package configs should make both weights explicit.

---

## 8. Evaluation Metrics

### 8.1 Per-Target Classification

For each target `x`, `y`, and `z`:

- accuracy;
- precision;
- recall;
- F1 score;
- confusion matrix;
- predicted probability distribution.

### 8.2 Combination Classification

Combine the three binary labels into one multi-label pattern:

```text
000, 001, 010, 011, 100, 101, 110, 111
```

Report:

- combination accuracy;
- raw confusion matrix;
- normalized confusion matrix;
- per-combination support.

### 8.3 Regression

For each target `x`, `y`, and `z`:

- MAE;
- MSE;
- RMSE;
- R²;
- predicted-vs-true scatter plots;
- CSV files containing true and predicted concentrations.

---

## 9. Command-Line Interface

The package should expose a CLI named `bactorcanet`.

### 9.1 Train

```bash
bactorcanet train \
  --config configs/paper_reproduction.yaml \
  --data-dir data \
  --output-dir runs/paper_reproduction
```

### 9.2 Evaluate

```bash
bactorcanet evaluate \
  --checkpoint runs/paper_reproduction/best.pt \
  --data-dir data \
  --split test \
  --output-dir reports/test
```

### 9.3 Predict a Single Image

```bash
bactorcanet predict \
  --checkpoint best_concentration_model.pth \
  --image path/to/image.jpg \
  --output prediction.json
```

### 9.4 Predict a CSV Batch

```bash
bactorcanet predict-batch \
  --checkpoint best_concentration_model.pth \
  --csv data/test/test.csv \
  --output predictions.csv
```

---

## 10. Configuration Template

```yaml
seed: 42

data:
  data_dir: data
  train_split: train
  val_split: val
  test_split: test
  csv_encoding: gbk
  image_size: 224
  threshold: 0.0
  strict_paths: true

model:
  name: multitask_concentration_cnn
  dropout: 0.5
  num_classes: 2

training:
  epochs: 100
  batch_size: 32
  num_workers: 0
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 0.0
  class_weight: 1.0
  reg_weight: 1.0
  scheduler: reduce_on_plateau
  early_stopping_patience: 20

output:
  run_dir: runs/default
  save_best: true
  save_final: true
  save_mistakes: true
  save_all_results: true
```

---

## 11. Python API

### 11.1 Training API

```python
from bactorcanet.config import load_config
from bactorcanet.training import Trainer

config = load_config("configs/paper_reproduction.yaml")
trainer = Trainer(config)
trainer.fit()
trainer.test()
```

### 11.2 Inference API

```python
from bactorcanet.inference import Predictor

predictor = Predictor.from_checkpoint("best_concentration_model.pth")
result = predictor.predict_image("path/to/your/image.jpg")

print(result.classification)
print(result.regression)
```

---

## 12. Reproduction Protocol

### 12.1 Environment

1. Create a clean Python environment.
2. Install the package with development dependencies.
3. Record Python, PyTorch, CUDA, and torchvision versions.
4. Set all random seeds.
5. Save the final YAML configuration with each run.

### 12.2 Data Preparation

1. Obtain the paper dataset from the corresponding authors, if permission is granted.
2. Place the dataset under the configured `data_dir`.
3. Validate all CSV schemas.
4. Validate all image paths.
5. Confirm that concentration units match the paper.
6. Confirm that split membership matches the paper.

### 12.3 Training

1. Train using `configs/paper_reproduction.yaml`.
2. Monitor validation total loss and per-target metrics.
3. Save the best checkpoint according to validation performance.
4. Save the final checkpoint after the last epoch.

### 12.4 Evaluation

1. Evaluate the best checkpoint on the test split.
2. Export classification mistakes.
3. Export all test predictions.
4. Export regression CSV files.
5. Export confusion matrices and regression plots.
6. Compare final metrics with paper-reported values, when available.

---

## 13. Minimum Test Suite

The rewritten package should include automated tests for:

1. CSV schema validation.
2. Dataset length and item structure.
3. Transform output shape.
4. Model forward-pass output keys and tensor shapes.
5. Loss calculation with dummy tensors.
6. Metric calculation on toy predictions.
7. Checkpoint save/load round trip.
8. Single-image inference with a synthetic image.

Example:

```bash
pytest -q
```

---

## 14. Packaging Requirements

Use `pyproject.toml` with modern Python packaging.

Recommended dependencies:

```toml
[project]
name = "bactorcanet"
version = "0.1.0"
description = "Multi-task CNN package for RCA-encoded multibacterial detection"
requires-python = ">=3.10"
dependencies = [
  "torch",
  "torchvision",
  "numpy",
  "pandas",
  "pillow",
  "scikit-learn",
  "matplotlib",
  "pyyaml",
  "tqdm"
]

[project.scripts]
bactorcanet = "bactorcanet.cli:main"
```

---

## 15. Clean-Room Rewrite Rules

To make the new package a true rewrite rather than a direct copy:

1. Use new module names, class names, and function boundaries.
2. Preserve behavior, not line-by-line implementation.
3. Replace hard-coded global variables with dataclass/YAML configuration.
4. Add explicit type hints and docstrings.
5. Separate data loading, model definition, training, evaluation, and inference.
6. Add tests for each major component.
7. Prefer small pure functions where possible.
8. Make all output paths configurable.
9. Add deterministic seed control.
10. Document dataset limitations and citation requirements.

---

## 16. Expected Deliverables

A complete `BactoRCANet` rewrite should deliver:

1. A pip-installable Python package.
2. A CLI for training, evaluation, and inference.
3. A default YAML configuration.
4. Dataset validation utilities.
5. Multi-task CNN implementation.
6. Training and checkpointing pipeline.
7. Evaluation reports and plots.
8. Inference API for new images.
9. Unit tests.
10. Documentation for reproducing the paper workflow.

---

## 17. Citation and Data Availability Note

When using this package to reproduce or extend the published work, cite:

```text
W. Xue, R. Li, K. Wang, K. Song, Z. Zhang, et al.,
Deep Learning-Enhanced DNAzyme-Driven Rolling-Circle Amplification Encoding for Multibacterial Detection,
Angewandte Chemie International Edition 65, no. 22 (2026): e4446117.
https://doi.org/10.1002/anie.4446117
```

The current repository states that the full dataset used in the paper is not publicly available and may be obtained from the corresponding authors upon reasonable academic request. Therefore, any exact reproduction depends on access to the original image dataset and concentration annotations.
