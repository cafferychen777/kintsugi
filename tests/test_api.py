from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import scipy.sparse as sp

import kintsugi


def test_tessellate_smoke_runs_on_minimal_grid() -> None:
    counts = sp.csr_matrix(
        np.array(
            [
                [3, 0, 1],
                [2, 0, 0],
                [0, 4, 1],
                [0, 3, 0],
            ],
            dtype=float,
        )
    )
    mask = np.ones((2, 2), dtype=bool)

    result = kintsugi.tessellate(
        counts,
        rows=2,
        cols=2,
        mask=mask,
        min_seed_distance=1,
        smooth_sigma=1.0,
    )

    assert isinstance(result, kintsugi.TessellationResult)
    assert isinstance(result, Mapping)
    assert set(result) == {
        "adjacency",
        "areas",
        "centroids",
        "depths",
        "labels",
        "residuals",
        "trace",
    }
    assert result["labels"].shape == (2, 2)
    assert result.labels.shape == (2, 2)
    assert result["trace"].shape == (2, 2)
    assert result["residuals"].shape[1] == counts.shape[1]


def test_tessellate_matches_manual_pipeline_components() -> None:
    counts = sp.csr_matrix(
        np.array(
            [
                [5, 0, 1],
                [4, 0, 1],
                [0, 6, 1],
                [0, 5, 1],
            ],
            dtype=float,
        )
    )
    mask = np.ones((2, 2), dtype=bool)

    result = kintsugi.tessellate(
        counts,
        rows=2,
        cols=2,
        mask=mask,
        min_seed_distance=1,
        smooth_sigma=1.0,
    )

    umi = np.asarray(counts.sum(axis=1)).ravel().reshape(2, 2)
    excess = kintsugi.directional_semivariance(umi, lag=2, mask=mask)
    trace, _lambda1, _lambda2, evec1 = kintsugi.boundary_tensor(excess)
    labels = kintsugi.adaptive_tessellation(
        umi,
        trace,
        evec1,
        kappa=2.0,
        min_seed_distance=1,
        smooth_sigma=1.0,
        mask=mask,
    )
    residuals, areas, depths, centroids = kintsugi.aggregate_counts(counts, labels, mask=mask)
    adjacency = kintsugi.build_spatial_graph(labels)

    assert np.array_equal(result["labels"], labels)
    assert np.allclose(result["trace"], trace)
    assert np.allclose(result["residuals"], residuals)
    assert np.allclose(result["areas"], areas)
    assert np.allclose(result["depths"], depths)
    assert np.allclose(result["centroids"], centroids)
    assert np.array_equal(result["adjacency"].toarray(), adjacency.toarray())


def test_tessellate_accepts_griddata_instance() -> None:
    counts = sp.csr_matrix(np.array([[1, 0], [0, 1], [2, 0], [0, 2]], dtype=float))
    grid = kintsugi.GridData(counts, rows=2, cols=2)

    result = kintsugi.tessellate(grid, min_seed_distance=1, smooth_sigma=1.0)

    assert isinstance(result, kintsugi.TessellationResult)
    assert result.labels.shape == (2, 2)


def test_tessellate_rejects_ambiguous_griddata_and_explicit_shape() -> None:
    counts = sp.csr_matrix(np.ones((4, 2), dtype=float))
    grid = kintsugi.GridData(counts, rows=2, cols=2)

    with pytest.raises(ValueError, match="must not be passed separately"):
        kintsugi.tessellate(grid, rows=2, cols=2)


def test_tessellate_accepts_non_csr_sparse_inputs() -> None:
    counts = sp.csc_matrix(np.array([[1, 0], [0, 1], [1, 1], [0, 2]], dtype=float))

    result = kintsugi.tessellate(
        counts,
        rows=2,
        cols=2,
        mask=np.ones((2, 2), dtype=bool),
        min_seed_distance=1,
        smooth_sigma=1.0,
    )

    assert result["labels"].shape == (2, 2)


def test_tessellate_rejects_bad_mask_shape() -> None:
    counts = sp.csr_matrix(np.ones((4, 2), dtype=float))

    with pytest.raises(ValueError, match="mask has shape"):
        kintsugi.tessellate(counts, rows=2, cols=2, mask=np.ones((3, 3), dtype=bool))


def test_tessellate_rejects_dense_inputs() -> None:
    counts = np.ones((4, 2), dtype=float)

    with pytest.raises(TypeError, match="SciPy sparse matrix"):
        kintsugi.tessellate(counts, rows=2, cols=2, mask=np.ones((2, 2), dtype=bool))


def test_tessellate_rejects_negative_raw_sparse_counts() -> None:
    counts = sp.csr_matrix(np.array([[1.0], [-1.0], [2.0], [3.0]]))

    with pytest.raises(ValueError, match="finite non-negative"):
        kintsugi.tessellate(counts, rows=2, cols=2)


def test_tessellate_requires_shape_for_raw_sparse_input() -> None:
    counts = sp.csr_matrix(np.ones((4, 2), dtype=float))

    with pytest.raises(TypeError, match="rows and cols are required"):
        kintsugi.tessellate(counts)


def test_tessellate_all_false_mask_returns_empty_region_outputs() -> None:
    counts = sp.csr_matrix(np.ones((4, 2), dtype=float))
    mask = np.zeros((2, 2), dtype=bool)

    result = kintsugi.tessellate(
        counts,
        rows=2,
        cols=2,
        mask=mask,
        min_seed_distance=1,
        smooth_sigma=1.0,
    )

    assert np.array_equal(result["labels"], np.full((2, 2), -1, dtype=np.int32))
    assert result["residuals"].shape == (0, 2)
    assert result["areas"].shape == (0,)
    assert result["depths"].shape == (0,)
    assert result["centroids"].shape == (0, 2)
    assert result["adjacency"].shape == (0, 0)
