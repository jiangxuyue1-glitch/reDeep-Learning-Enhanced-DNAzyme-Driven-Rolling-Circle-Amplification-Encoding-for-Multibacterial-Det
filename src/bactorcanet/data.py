from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

TARGETS = ("x", "y", "z")
REQUIRED_COLUMNS = ("image_path", "x", "y", "z")


@dataclass(frozen=True)
class SplitData:
    image_paths: list[Path]
    concentrations: np.ndarray
    frame: pd.DataFrame


def resolve_image_path(value: str, csv_path: Path, data_dir: Path) -> Path:
    raw = Path(str(value).strip())
    candidates = [raw]
    if not raw.is_absolute():
        candidates = [csv_path.parent / raw, data_dir / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def validate_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")


def load_split(data_dir: str | Path, split: str, encoding: str = "gbk", strict: bool = True) -> SplitData:
    data_dir = Path(data_dir)
    csv_path = data_dir / split / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    frame = pd.read_csv(csv_path, encoding=encoding).dropna(subset=["image_path"])
    validate_frame(frame)
    frame = frame.copy()
    frame["image_path"] = frame["image_path"].astype(str)
    paths = [resolve_image_path(path, csv_path, data_dir) for path in frame["image_path"].tolist()]
    exists = [path.exists() for path in paths]
    if not all(exists):
        missing = [str(path) for path, ok in zip(paths, exists) if not ok]
        if strict:
            raise FileNotFoundError("Missing image files: " + "; ".join(missing[:10]))
        warnings.warn(f"Dropped {len(missing)} missing images", RuntimeWarning, stacklevel=2)
        frame = frame.loc[exists].reset_index(drop=True)
        paths = [path for path, ok in zip(paths, exists) if ok]
    concentrations = frame.loc[:, TARGETS].astype("float32").to_numpy()
    return SplitData(paths, concentrations, frame.reset_index(drop=True))


def load_dataset(data_dir: str | Path, train_split: str = "train", val_split: str = "val", test_split: str = "test", encoding: str = "gbk", strict: bool = True) -> dict[str, SplitData]:
    return {
        "train": load_split(data_dir, train_split, encoding, strict),
        "val": load_split(data_dir, val_split, encoding, strict),
        "test": load_split(data_dir, test_split, encoding, strict),
    }


class ConcentrationDataset(Dataset):
    def __init__(self, image_paths: list[str | Path], concentrations: np.ndarray, transform: Callable | None = None, threshold: float = 0.1):
        self.image_paths = [Path(path) for path in image_paths]
        self.concentrations = np.asarray(concentrations, dtype=np.float32)
        self.transform = transform
        self.threshold = float(threshold)
        if self.concentrations.ndim != 2 or self.concentrations.shape[1] != 3:
            raise ValueError("concentrations must have shape [N, 3]")
        if len(self.image_paths) != len(self.concentrations):
            raise ValueError("image path and concentration counts differ")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        values = torch.as_tensor(self.concentrations[index], dtype=torch.float32)
        labels = (values > self.threshold).to(torch.long)
        return {"image": image, "class": labels, "regression": values, "image_path": str(path)}
