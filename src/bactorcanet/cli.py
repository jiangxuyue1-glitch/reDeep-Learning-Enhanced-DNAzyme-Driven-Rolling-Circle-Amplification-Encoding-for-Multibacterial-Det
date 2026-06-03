from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import ConcentrationDataset, load_split
from .evaluation import evaluate
from .inference import Predictor
from .models import MultiTaskConcentrationCNN
from .training import Trainer
from .transforms import build_transforms
from .utils import pick_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bactorcanet")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config")
    train.add_argument("--data-dir")
    train.add_argument("--output-dir")
    eva = sub.add_parser("evaluate")
    eva.add_argument("--checkpoint", required=True)
    eva.add_argument("--data-dir", required=True)
    eva.add_argument("--split", default="test")
    eva.add_argument("--output-dir", required=True)
    eva.add_argument("--config")
    predict = sub.add_parser("predict")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--image", required=True)
    predict.add_argument("--output")
    batch = sub.add_parser("predict-batch")
    batch.add_argument("--checkpoint", required=True)
    batch.add_argument("--csv", required=True)
    batch.add_argument("--output", required=True)
    return parser


def cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.output_dir:
        config.output.run_dir = args.output_dir
    trainer = Trainer(config)
    trainer.fit()


def load_model_for_eval(checkpoint: str | Path, config) -> MultiTaskConcentrationCNN:
    payload = torch.load(checkpoint, map_location="cpu")
    kwargs = payload.get("model_kwargs", {"num_classes": config.model.num_classes, "dropout": config.model.dropout, "width": config.model.width})
    model = MultiTaskConcentrationCNN(**kwargs)
    model.load_state_dict(payload.get("model_state", payload), strict=False)
    return model


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    config.data.data_dir = args.data_dir
    device = pick_device(None)
    model = load_model_for_eval(args.checkpoint, config).to(device)
    split = load_split(config.data.data_dir, args.split, config.data.csv_encoding, config.data.strict_paths)
    dataset = ConcentrationDataset(split.image_paths, split.concentrations, build_transforms(config.data.image_size, False), config.data.threshold)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)
    metrics = evaluate(model, loader, device, args.output_dir, True, True)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def cmd_predict(args: argparse.Namespace) -> None:
    predictor = Predictor.from_checkpoint(args.checkpoint)
    result = predictor.predict_image(args.image)
    payload = {"classification": result.classification, "probabilities": result.probabilities, "regression": result.regression}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


def cmd_predict_batch(args: argparse.Namespace) -> None:
    Predictor.from_checkpoint(args.checkpoint).predict_csv(args.csv, args.output)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {"train": cmd_train, "evaluate": cmd_evaluate, "predict": cmd_predict, "predict-batch": cmd_predict_batch}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
