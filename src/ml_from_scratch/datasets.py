"""Small licensed data used by the offline curriculum projects; no model answers."""

import csv
from importlib.resources import files

import numpy as np

IRIS_FEATURES = ("sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm")
IRIS_CLASSES = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")


def load_iris():
    """Return fresh (150, 4) float features and (150,) integer class indices.

    Read the unchanged UCI iris.data artifact packaged with the project. No
    network, preprocessing, shuffling, or discrepancy corrections occur here.
    See data/NOTICE.md and iris-provenance.json for attribution and exact version.
    """
    text = files("ml_from_scratch").joinpath("data", "iris.data").read_text(encoding="ascii")
    rows = [row for row in csv.reader(text.splitlines()) if row]
    if len(rows) != 150 or any(len(row) != 5 for row in rows):
        raise ValueError("bundled Iris data have an unexpected shape")
    X = np.array([[float(value) for value in row[:4]] for row in rows], dtype=float)
    labels = {name: index for index, name in enumerate(IRIS_CLASSES)}
    try:
        y = np.array([labels[row[4]] for row in rows], dtype=np.int64)
    except KeyError as error:
        raise ValueError("bundled Iris data contain an unknown class") from error
    if not np.isfinite(X).all() or not np.array_equal(np.bincount(y), [50, 50, 50]):
        raise ValueError("bundled Iris data failed their finite-value/class-count check")
    return X, y
