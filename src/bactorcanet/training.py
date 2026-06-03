from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import AppConfig, save_config
from .data import ConcentrationDataset, load_split
from .evaluation import evaluate
from .losses import MultiTaskLoss
from .models import MultiTaskConcentrationCNN
from .transforms import build_transforms
from .utils import pick_device, set_seed, write_json


class Trainer:
    def __init__(self, config: AppConfig, device: str | None = None):
        self.config = config
        set_seed(config.seed)
        self.device = pick_device(device)
        self.run_dir = Path(config.output.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model = MultiTaskConcentrationCNN(config.model.num_classes, config.model.dropout, config.model.width).to(self.device)
        self.criterion = MultiTaskLoss(config.training.class_weight, config.training.reg_weight, config.training.regression_loss)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min") if config.training.scheduler == "reduce_on_plateau" else None
        self.best_loss = float("inf")
        self.history: list[dict[str, float]] = []
        save_config(config, self.run_dir / "config.yaml")

    def _loader(self, split: str, train: bool) -> DataLoader:
        split_data = load_split(self.config.data.data_dir, split, self.config.data.csv_encoding, self.config.data.strict_paths)
        dataset = ConcentrationDataset(split_data.image_paths, split_data.concentrations, build_transforms(self.config.data.image_size, train), self.config.data.threshold)
        return DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=train, num_workers=self.config.training.num_workers)

    def _step(self, batch: dict[str, torch.Tensor], train: bool) -> dict[str, float]:
        images = batch["image"].to(self.device)
        labels = batch["class"].to(self.device)
        values = batch["regression"].to(self.device)
        with torch.set_grad_enabled(train):
            outputs = self.model(images)
            losses = self.criterion(outputs, labels, values)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                self.optimizer.step()
        return {key: float(value.detach().cpu().item()) for key, value in losses.items()}

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.model.train(train)
        totals = {"loss": 0.0, "class_loss": 0.0, "reg_loss": 0.0}
        count = 0
        for batch in tqdm(loader, leave=False):
            losses = self._step(batch, train)
            batch_size = int(batch["image"].shape[0])
            count += batch_size
            for key in totals:
                totals[key] += losses[key] * batch_size
        return {key: value / max(count, 1) for key, value in totals.items()}

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: dict[str, float]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "model_kwargs": {"num_classes": self.config.model.num_classes, "dropout": self.config.model.dropout, "width": self.config.model.width},
            "epoch": epoch,
            "metrics": metrics,
            "stamp": self.config.stamp,
        }, path)

    def fit(self) -> list[dict[str, float]]:
        train_loader = self._loader(self.config.data.train_split, True)
        val_loader = self._loader(self.config.data.val_split, False)
        patience = 0
        for epoch in range(1, self.config.training.epochs + 1):
            train_metrics = self._run_epoch(train_loader, True)
            val_metrics = self._run_epoch(val_loader, False)
            row = {f"train_{key}": value for key, value in train_metrics.items()}
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            row["epoch"] = float(epoch)
            self.history.append(row)
            if self.scheduler is not None:
                self.scheduler.step(val_metrics["loss"])
            if val_metrics["loss"] < self.best_loss:
                self.best_loss = val_metrics["loss"]
                patience = 0
                if self.config.output.save_best:
                    self.save_checkpoint(self.run_dir / "best.pt", epoch, val_metrics)
            else:
                patience += 1
            if patience >= self.config.training.early_stopping_patience:
                break
        if self.config.output.save_final:
            self.save_checkpoint(self.run_dir / "final.pt", int(self.history[-1]["epoch"]), self.history[-1])
        write_json(self.history, self.run_dir / "history.json")
        return self.history

    def test(self) -> dict[str, object]:
        loader = self._loader(self.config.data.test_split, False)
        return evaluate(self.model, loader, self.device, self.run_dir / "test", self.config.output.save_mistakes, self.config.output.save_all_results)
