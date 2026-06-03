import torch

from bactorcanet.losses import MultiTaskLoss
from bactorcanet.models import MultiTaskConcentrationCNN


def test_model_forward_shapes_eval_mode():
    model = MultiTaskConcentrationCNN(num_classes=2).eval()
    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 64, 64))
    assert set(outputs) == {"class_x", "class_y", "class_z", "reg_x", "reg_y", "reg_z"}
    assert outputs["class_x"].shape == (2, 2)
    assert outputs["reg_z"].shape == (2, 1)


def test_loss_calculation():
    outputs = {
        "class_x": torch.randn(2, 2),
        "class_y": torch.randn(2, 2),
        "class_z": torch.randn(2, 2),
        "reg_x": torch.randn(2, 1),
        "reg_y": torch.randn(2, 1),
        "reg_z": torch.randn(2, 1),
    }
    labels = torch.tensor([[0, 1, 1], [1, 0, 0]])
    values = torch.randn(2, 3)
    losses = MultiTaskLoss()(outputs, labels, values)
    assert losses["loss"].item() > 0
