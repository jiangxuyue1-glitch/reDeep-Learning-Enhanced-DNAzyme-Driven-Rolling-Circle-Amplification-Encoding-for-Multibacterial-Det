from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TARGETS = ("x", "y", "z")


def save_regression_tables(frame: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in TARGETS:
        frame[["image_path", f"true_reg_{name}", f"pred_reg_{name}"]].to_csv(output_dir / f"regression_{name}.csv", index=False)


def save_regression_plots(frame: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in TARGETS:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(frame[f"true_reg_{name}"], frame[f"pred_reg_{name}"], s=18, alpha=0.75)
        ax.set_xlabel("True concentration")
        ax.set_ylabel("Predicted concentration")
        ax.set_title(name.upper())
        fig.tight_layout()
        fig.savefig(output_dir / f"regression_{name}.png", dpi=160)
        plt.close(fig)
