from __future__ import annotations

import torch
from torch import nn


class FeatureBranch(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(128 * channels, 256 * channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(256 * channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256 * channels, 512 * channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(512 * channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout * 0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)


class MultiTaskConcentrationCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.5, width: int = 1):
        super().__init__()
        width = max(1, int(width))
        c64 = 64 * width
        c128 = 128 * width
        self.stem = nn.Sequential(
            nn.Conv2d(3, c64, 3, padding=1, bias=False),
            nn.BatchNorm2d(c64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c64, c128, 3, padding=1, bias=False),
            nn.BatchNorm2d(c128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.branches = nn.ModuleDict({name: FeatureBranch(width, dropout) for name in ("x", "y", "z")})
        features = 512 * width
        self.class_heads = nn.ModuleDict({name: self._classifier(features, num_classes, dropout) for name in ("x", "y", "z")})
        self.reg_heads = nn.ModuleDict({name: self._regressor(features, dropout) for name in ("x", "y", "z")})
        self.apply(self._init_weights)

    @staticmethod
    def _classifier(features: int, num_classes: int, dropout: float) -> nn.Sequential:
        hidden = max(128, features // 2)
        return nn.Sequential(nn.Linear(features, hidden), nn.ReLU(inplace=True), nn.BatchNorm1d(hidden), nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    @staticmethod
    def _regressor(features: int, dropout: float) -> nn.Sequential:
        hidden = max(128, features // 2)
        return nn.Sequential(nn.Linear(features, hidden), nn.ReLU(inplace=True), nn.BatchNorm1d(hidden), nn.Dropout(dropout * 0.6), nn.Linear(hidden, 1))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.stem(images)
        outputs: dict[str, torch.Tensor] = {}
        for name in ("x", "y", "z"):
            features = self.branches[name](shared)
            outputs[f"class_{name}"] = self.class_heads[name](features)
            outputs[f"reg_{name}"] = self.reg_heads[name](features)
        return outputs
