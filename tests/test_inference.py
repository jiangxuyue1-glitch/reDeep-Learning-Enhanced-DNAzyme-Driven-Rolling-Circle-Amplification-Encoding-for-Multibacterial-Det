from pathlib import Path

import torch
from PIL import Image

from bactorcanet.inference import Predictor
from bactorcanet.models import MultiTaskConcentrationCNN


def test_checkpoint_inference_roundtrip(tmp_path: Path):
    image = tmp_path / "image.png"
    checkpoint = tmp_path / "model.pt"
    Image.new("RGB", (64, 64), "white").save(image)
    model = MultiTaskConcentrationCNN().eval()
    torch.save({"model_state": model.state_dict(), "model_kwargs": {"num_classes": 2, "dropout": 0.5, "width": 1}, "stamp": "jiangxuyue"}, checkpoint)
    predictor = Predictor.from_checkpoint(checkpoint, image_size=64, device="cpu")
    result = predictor.predict_image(image)
    assert set(result.classification) == {"x", "y", "z"}
