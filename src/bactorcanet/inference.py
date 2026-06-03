from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from .models import MultiTaskConcentrationCNN
from .transforms import build_transforms
from .utils import pick_device

TARGETS = ("x", "y", "z")


@dataclass(frozen=True)
class PredictionResult:
    classification: dict[str, int]
    probabilities: dict[str, float]
    regression: dict[str, float]


class Predictor:
    def __init__(self, model: torch.nn.Module, image_size: int = 128, device: str | torch.device | None = None):
        self.device = pick_device(str(device) if device else None)
        self.model = model.to(self.device).eval()
        self.transform = build_transforms(image_size, train=False)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, image_size: int = 128, device: str | torch.device | None = None, num_classes: int = 2):
        payload = torch.load(checkpoint, map_location="cpu")
        model_kwargs = payload.get("model_kwargs", {"num_classes": num_classes}) if isinstance(payload, dict) else {"num_classes": num_classes}
        model = MultiTaskConcentrationCNN(**model_kwargs)
        state = payload.get("model_state", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
        if isinstance(state, dict):
            clean_state = {key.removeprefix("module."): value for key, value in state.items()}
            model.load_state_dict(clean_state, strict=False)
        return cls(model, image_size=image_size, device=device)

    def predict_image(self, image_path: str | Path) -> PredictionResult:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
        classification: dict[str, int] = {}
        probabilities: dict[str, float] = {}
        regression: dict[str, float] = {}
        for name in TARGETS:
            probs = torch.softmax(outputs[f"class_{name}"], dim=1)[0]
            classification[name] = int(probs.argmax().item())
            probabilities[name] = float(probs[1].item())
            regression[name] = float(outputs[f"reg_{name}"].squeeze().item())
        return PredictionResult(classification, probabilities, regression)

    def predict_csv(self, csv_path: str | Path, output: str | Path | None = None) -> pd.DataFrame:
        frame = pd.read_csv(csv_path)
        rows = []
        for path in frame["image_path"].astype(str):
            result = self.predict_image(path)
            row = {"image_path": path}
            for name in TARGETS:
                row[f"class_{name}"] = result.classification[name]
                row[f"prob_{name}"] = result.probabilities[name]
                row[f"reg_{name}"] = result.regression[name]
            rows.append(row)
        output_frame = pd.DataFrame(rows)
        if output is not None:
            output_frame.to_csv(output, index=False)
        return output_frame
