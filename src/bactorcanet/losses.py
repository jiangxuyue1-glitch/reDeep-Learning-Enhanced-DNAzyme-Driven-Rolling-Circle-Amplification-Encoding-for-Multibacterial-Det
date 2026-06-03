from __future__ import annotations

import torch
from torch import nn

TARGETS = ("x", "y", "z")


class MultiTaskLoss(nn.Module):
    def __init__(self, class_weight: float = 1.0, reg_weight: float = 1.0, regression_loss: str = "smooth_l1"):
        super().__init__()
        self.class_weight = class_weight
        self.reg_weight = reg_weight
        self.class_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss() if regression_loss == "mse" else nn.SmoothL1Loss()

    def forward(self, outputs: dict[str, torch.Tensor], labels: torch.Tensor, values: torch.Tensor) -> dict[str, torch.Tensor]:
        class_total = sum(self.class_loss(outputs[f"class_{name}"], labels[:, idx].long()) for idx, name in enumerate(TARGETS))
        reg_total = sum(self.reg_loss(outputs[f"reg_{name}"].squeeze(1), values[:, idx].float()) for idx, name in enumerate(TARGETS))
        total = self.class_weight * class_total + self.reg_weight * reg_total
        return {"loss": total, "class_loss": class_total, "reg_loss": reg_total}
