from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score

TARGETS = ("x", "y", "z")
COMBINATION_TYPES = ("0", "X", "Y", "Z", "X+Y", "X+Z", "Y+Z", "X+Y+Z")


def combination_label(labels: np.ndarray | list[int]) -> str:
    x, y, z = [int(v) for v in labels]
    if x == 0 and y == 0 and z == 0:
        return "0"
    return "+".join(name for name, value in zip(("X", "Y", "Z"), (x, y, z)) if value > 0)


def classification_metrics(true: np.ndarray, pred: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, object]:
    true = np.asarray(true).astype(int)
    pred = np.asarray(pred).astype(int)
    result: dict[str, object] = {}
    for idx, name in enumerate(TARGETS):
        result[name] = {
            "accuracy": float(accuracy_score(true[:, idx], pred[:, idx])),
            "precision": float(precision_score(true[:, idx], pred[:, idx], zero_division=0)),
            "recall": float(recall_score(true[:, idx], pred[:, idx], zero_division=0)),
            "f1": float(f1_score(true[:, idx], pred[:, idx], zero_division=0)),
            "confusion_matrix": confusion_matrix(true[:, idx], pred[:, idx], labels=[0, 1]).tolist(),
        }
    true_combo = [combination_label(row) for row in true]
    pred_combo = [combination_label(row) for row in pred]
    matrix = confusion_matrix(true_combo, pred_combo, labels=list(COMBINATION_TYPES))
    support = {label: int(true_combo.count(label)) for label in COMBINATION_TYPES}
    result["combination"] = {"accuracy": float(accuracy_score(true_combo, pred_combo)), "labels": list(COMBINATION_TYPES), "confusion_matrix": matrix.tolist(), "support": support}
    if probabilities is not None:
        result["probability_mean"] = {name: float(np.asarray(probabilities)[:, idx].mean()) for idx, name in enumerate(TARGETS)}
    return result


def regression_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, dict[str, float]]:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    metrics: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(TARGETS):
        mse = float(mean_squared_error(true[:, idx], pred[:, idx]))
        metrics[name] = {
            "mae": float(mean_absolute_error(true[:, idx], pred[:, idx])),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(true[:, idx], pred[:, idx])) if len(true) > 1 else float("nan"),
        }
    return metrics
