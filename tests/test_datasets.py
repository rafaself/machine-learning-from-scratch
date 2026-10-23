"""Reproducible offline data, distinct from algorithm/reference-answer tests."""

import hashlib
import json
from importlib.resources import files

import numpy as np

from ml_from_scratch.datasets import IRIS_CLASSES, IRIS_FEATURES, load_iris


def test_iris_bytes_match_recorded_upstream_member():
    resources = files("ml_from_scratch").joinpath("data")
    provenance = json.loads(resources.joinpath("iris-provenance.json").read_text())
    raw = resources.joinpath("iris.data").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == provenance["member_sha256"]
    assert provenance["member_sha256"] == "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0"
    assert provenance["license"] == "CC-BY-4.0"
    assert provenance["doi"] == "10.24432/C56C76"
    assert "Creative Commons Attribution 4.0" in resources.joinpath("NOTICE.md").read_text()


def test_iris_shape_units_classes_and_unchanged_discrepancy_records():
    X, y = load_iris()
    assert X.shape == (150, 4) and y.shape == (150,)
    assert X.dtype == np.float64 and np.issubdtype(y.dtype, np.integer)
    assert len(IRIS_FEATURES) == 4 and all(name.endswith("_cm") for name in IRIS_FEATURES)
    assert IRIS_CLASSES == ("Iris-setosa", "Iris-versicolor", "Iris-virginica")
    np.testing.assert_array_equal(np.bincount(y), [50, 50, 50])
    np.testing.assert_allclose(X[0], [5.1, 3.5, 1.4, .2])
    np.testing.assert_allclose(X[34], [4.9, 3.1, 1.5, .1])
    np.testing.assert_allclose(X[37], [4.9, 3.1, 1.5, .1])
    assert np.isfinite(X).all() and len(np.unique(X, axis=0)) == 147


def test_iris_calls_return_independent_arrays():
    X, y = load_iris()
    X[:] = 0
    y[:] = 99
    fresh_X, fresh_y = load_iris()
    assert fresh_X[0, 0] == 5.1 and fresh_y[0] == 0
    assert np.all(fresh_y < 3)
