import numpy as np

from bactorcanet.metrics import classification_metrics, combination_label, regression_metrics


def test_combination_label():
    assert combination_label([1, 0, 1]) == "X+Z"
    assert combination_label([0, 0, 0]) == "0"


def test_metrics_on_toy_values():
    true = np.array([[0, 1, 1], [1, 0, 0]])
    pred = np.array([[0, 1, 0], [1, 0, 0]])
    cls = classification_metrics(true, pred)
    reg = regression_metrics(true.astype(float), pred.astype(float))
    assert cls["x"]["accuracy"] == 1.0
    assert reg["x"]["mae"] == 0.0
