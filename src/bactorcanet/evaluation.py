from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .metrics import classification_metrics, regression_metrics
from .utils import write_json

TARGETS = ("x", "y", "z")


def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device | str) -> dict[str, object]:
    device = torch.device(device)
    model.eval()
    true_class: list[np.ndarray] = []
    pred_class: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    true_reg: list[np.ndarray] = []
    pred_reg: list[np.ndarray] = []
    paths: list[str] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            probs = [torch.softmax(outputs[f"class_{name}"], dim=1)[:, 1] for name in TARGETS]
            preds = [outputs[f"class_{name}"].argmax(dim=1) for name in TARGETS]
            regs = [outputs[f"reg_{name}"].squeeze(1) for name in TARGETS]
            true_class.append(batch["class"].cpu().numpy())
            pred_class.append(torch.stack(preds, dim=1).cpu().numpy())
            probabilities.append(torch.stack(probs, dim=1).cpu().numpy())
            true_reg.append(batch["regression"].cpu().numpy())
            pred_reg.append(torch.stack(regs, dim=1).cpu().numpy())
            paths.extend(batch["image_path"])
    return {
        "paths": paths,
        "true_class": np.concatenate(true_class, axis=0),
        "pred_class": np.concatenate(pred_class, axis=0),
        "probabilities": np.concatenate(probabilities, axis=0),
        "true_regression": np.concatenate(true_reg, axis=0),
        "pred_regression": np.concatenate(pred_reg, axis=0),
    }


def build_results_frame(collected: dict[str, object]) -> pd.DataFrame:
    data = {"image_path": collected["paths"]}
    for idx, name in enumerate(TARGETS):
        data[f"true_class_{name}"] = collected["true_class"][:, idx]
        data[f"pred_class_{name}"] = collected["pred_class"][:, idx]
        data[f"prob_{name}"] = collected["probabilities"][:, idx]
        data[f"true_reg_{name}"] = collected["true_regression"][:, idx]
        data[f"pred_reg_{name}"] = collected["pred_regression"][:, idx]
    return pd.DataFrame(data)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device | str = "cpu", output_dir: str | Path | None = None, save_mistakes: bool = False, save_all_results: bool = False) -> dict[str, object]:
    collected = collect_predictions(model, loader, device)
    metrics = {
        "classification": classification_metrics(collected["true_class"], collected["pred_class"], collected["probabilities"]),
        "regression": regression_metrics(collected["true_regression"], collected["pred_regression"]),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(metrics, output_dir / "metrics.json")
        frame = build_results_frame(collected)
        if save_all_results:
            frame.to_csv(output_dir / "all_results.csv", index=False)
        if save_mistakes:
            mask = (collected["true_class"] != collected["pred_class"]).any(axis=1)
            frame.loc[mask].to_csv(output_dir / "mistakes.csv", index=False)
    return metrics
