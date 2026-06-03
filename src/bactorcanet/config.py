from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class DataConfig:
    data_dir: str = "data"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    csv_encoding: str = "gbk"
    image_size: int = 128
    threshold: float = 0.1
    strict_paths: bool = True


@dataclass
class ModelConfig:
    name: str = "multitask_concentration_cnn"
    dropout: float = 0.5
    num_classes: int = 2
    width: int = 1


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 0
    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    class_weight: float = 1.0
    reg_weight: float = 1.0
    scheduler: str = "reduce_on_plateau"
    early_stopping_patience: int = 20
    regression_loss: str = "smooth_l1"


@dataclass
class OutputConfig:
    run_dir: str = "runs/default"
    save_best: bool = True
    save_final: bool = True
    save_mistakes: bool = True
    save_all_results: bool = True


@dataclass
class AppConfig:
    seed: int = 42
    stamp: str = "jiangxuyue"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    names = set(instance.__dataclass_fields__)
    for key, value in values.items():
        if key in names:
            setattr(instance, key, value)
    return instance


def from_dict(values: dict[str, Any]) -> AppConfig:
    cfg = AppConfig()
    for key, value in values.items():
        if key == "data" and isinstance(value, dict):
            _merge_dataclass(cfg.data, value)
        elif key == "model" and isinstance(value, dict):
            _merge_dataclass(cfg.model, value)
        elif key == "training" and isinstance(value, dict):
            _merge_dataclass(cfg.training, value)
        elif key == "output" and isinstance(value, dict):
            _merge_dataclass(cfg.output, value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    values: dict[str, Any] = {}
    if path:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            values = json.loads(text)
        else:
            import yaml
            values = yaml.safe_load(text) or {}
    if overrides:
        values = {**values, **overrides}
    return from_dict(values)


def save_config(config: AppConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    else:
        import yaml
        path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
