from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bactorcanet.data import ConcentrationDataset, load_split, validate_frame
from bactorcanet.transforms import build_transforms


def test_validate_frame_requires_schema():
    with pytest.raises(ValueError):
        validate_frame(pd.DataFrame({"image_path": ["a.png"]}))


def test_dataset_item_structure(tmp_path: Path):
    image = tmp_path / "a.png"
    Image.new("RGB", (20, 20), "white").save(image)
    dataset = ConcentrationDataset([image], np.array([[0.0, 0.2, 1.0]], dtype=np.float32), build_transforms(32), threshold=0.1)
    item = dataset[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["class"].tolist() == [0, 1, 1]
    assert item["regression"].shape == (3,)


def test_load_split_relative_paths(tmp_path: Path):
    folder = tmp_path / "train"
    folder.mkdir()
    Image.new("RGB", (16, 16), "black").save(folder / "a.png")
    pd.DataFrame({"image_path": ["a.png"], "x": [0.0], "y": [1.0], "z": [2.0]}).to_csv(folder / "train.csv", index=False)
    split = load_split(tmp_path, "train", encoding="utf-8")
    assert len(split.image_paths) == 1
    assert split.concentrations.shape == (1, 3)
