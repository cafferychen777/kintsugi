"""Tests for the toy dataset and demo pipeline."""

from __future__ import annotations

import hashlib

import numpy as np
import scipy.sparse as sp

import kintsugi
from kintsugi._demo import make_toy_dataset


def test_toy_dataset_deterministic() -> None:
    g1 = make_toy_dataset()
    g2 = make_toy_dataset()
    assert np.array_equal(g1.counts.toarray(), g2.counts.toarray())
    assert np.array_equal(g1.mask, g2.mask)


def test_toy_dataset_properties() -> None:
    grid = make_toy_dataset()
    assert grid.rows == 60
    assert grid.cols == 60
    assert grid.n_genes == 100
    assert grid.mask.sum() > 0
    assert sp.issparse(grid.counts)
    assert grid.gene_names is not None
    assert grid.gene_names.shape[0] == 100


def test_toy_tessellation_deterministic() -> None:
    grid = make_toy_dataset()
    r1 = grid.tessellate()
    r2 = grid.tessellate()

    h1 = hashlib.sha256(r1.labels.tobytes()).hexdigest()[:16]
    h2 = hashlib.sha256(r2.labels.tobytes()).hexdigest()[:16]
    assert h1 == h2
    assert h1 == "f6e40d11efb7527d"


def test_toy_tessellation_produces_multiple_regions() -> None:
    grid = make_toy_dataset()
    result = grid.tessellate()
    assert result.n_regions > 1


def test_toy_report_runs_without_error() -> None:
    grid = make_toy_dataset()
    result = grid.tessellate()
    report = kintsugi.tessellation_report(result, grid)
    assert report.n_regions == result.n_regions
    assert report.stationarity_pass_rate > 0


def test_make_toy_dataset_via_public_api() -> None:
    """Verify make_toy_dataset is accessible via kintsugi.make_toy_dataset."""
    grid = kintsugi.make_toy_dataset()
    assert grid.rows == 60
    assert grid.n_genes == 100


def test_examples_toy_data_delegates() -> None:
    """Verify examples/toy_data.py delegates to the same function."""
    from examples.toy_data import make_toy_dataset as example_make

    assert example_make is make_toy_dataset
