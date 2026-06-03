# Reproduction protocol

Install the package in a clean environment, place an authorized compatible dataset under `data`, verify the CSV schema, and run `bactorcanet train --config configs/paper_reproduction.yaml --data-dir data --output-dir runs/paper_reproduction`.

Evaluate the best checkpoint with `bactorcanet evaluate --checkpoint runs/paper_reproduction/best.pt --data-dir data --split test --output-dir reports/test`.
