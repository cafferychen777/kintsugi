from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import polygamma

from kintsugi.aggregate import aggregate_counts
from kintsugi.graph import build_spatial_graph
from kintsugi.models import GridData, validate_grid_data
from kintsugi.partition import adaptive_tessellation
from kintsugi.tensor import boundary_tensor
from kintsugi.variogram import directional_semivariance, poisson_baseline


def test_aggregate_counts_matches_closed_form_residuals_and_centroids() -> None:
    counts = sp.csr_matrix(
        np.array(
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ],
            dtype=float,
        )
    )
    labels = np.array([[0, 0], [1, 1]], dtype=np.int32)

    residuals, areas, depths, centroids = aggregate_counts(counts, labels)

    assert residuals.shape == (2, 2)
    assert np.allclose(areas, [2.0, 2.0])
    assert np.allclose(depths, [2.0, 2.0])
    assert np.allclose(centroids, [[0.0, 0.5], [1.0, 0.5]])
    assert np.allclose(residuals, [[1.0, -1.0], [-1.0, 1.0]])


def test_aggregate_counts_returns_empty_arrays_when_mask_excludes_everything() -> None:
    counts = sp.csr_matrix(np.ones((4, 3), dtype=float))
    labels = np.array([[0, 0], [1, 1]], dtype=np.int32)
    mask = np.zeros((2, 2), dtype=bool)

    residuals, areas, depths, centroids = aggregate_counts(counts, labels, mask=mask)

    assert residuals.shape == (0, 3)
    assert areas.shape == (0,)
    assert depths.shape == (0,)
    assert centroids.shape == (0, 2)


def test_aggregate_counts_rejects_bad_mask_shape() -> None:
    counts = sp.csr_matrix(np.ones((4, 2), dtype=float))
    labels = np.zeros((2, 2), dtype=np.int32)

    with pytest.raises(ValueError, match="mask has shape"):
        aggregate_counts(counts, labels, mask=np.ones((3, 3), dtype=bool))


def test_build_spatial_graph_is_binary_and_compact() -> None:
    labels = np.array(
        [
            [0, 0, 1],
            [0, 2, 1],
        ],
        dtype=np.int32,
    )

    adjacency = build_spatial_graph(labels)

    assert adjacency.dtype == np.uint8
    assert adjacency.shape == (3, 3)
    assert adjacency[0, 1] == 1
    assert adjacency[0, 2] == 1
    assert adjacency[1, 2] == 1
    assert adjacency[0, 0] == 0


def test_build_spatial_graph_returns_empty_matrix_without_valid_regions() -> None:
    labels = np.full((2, 3), -1, dtype=np.int32)

    adjacency = build_spatial_graph(labels)

    assert adjacency.shape == (0, 0)
    assert adjacency.dtype == np.uint8


def test_directional_semivariance_is_zero_on_constant_field() -> None:
    umi = np.full((3, 3), 7.0)
    excess = directional_semivariance(umi, lag=1, mask=np.ones_like(umi, dtype=bool))

    assert excess.shape == (3, 3, 4)
    assert np.allclose(excess, 0.0)


def test_directional_semivariance_rejects_bad_mask_shape() -> None:
    umi = np.ones((3, 3), dtype=float)

    with pytest.raises(ValueError, match="mask has shape"):
        directional_semivariance(umi, lag=1, mask=np.ones((2, 2), dtype=bool))


def test_poisson_baseline_matches_exact_variance_on_uniform_field() -> None:
    umi = np.full((4, 4), 3.0)
    baseline = poisson_baseline(umi, smooth_halfwidth=1)

    # Exact Var[log(N+1)] for N ~ Poisson(3)
    from kintsugi._poisson_log_var import _exact_scalar
    exact = _exact_scalar(3.0)
    assert np.allclose(baseline, exact, rtol=1e-4)


def test_exact_variance_less_than_trigamma_at_small_lambda() -> None:
    from kintsugi._poisson_log_var import poisson_log_variance
    for lam in [0.5, 1.0, 2.0, 5.0]:
        exact = float(poisson_log_variance(lam))
        trig = float(polygamma(1, lam + 1.0))
        assert exact < trig
        assert exact > 0


def test_exact_variance_converges_to_trigamma_at_large_lambda() -> None:
    from kintsugi._poisson_log_var import poisson_log_variance
    for lam in [100.0, 500.0, 600.0]:
        exact = float(poisson_log_variance(lam))
        trig = float(polygamma(1, lam + 1.0))
        assert abs(exact - trig) / trig < 1e-4


def test_exact_variance_rejects_invalid_lambda() -> None:
    from kintsugi._poisson_log_var import poisson_log_variance

    with pytest.raises(ValueError, match="finite non-negative"):
        poisson_log_variance(-1.0)
    with pytest.raises(ValueError, match="finite non-negative"):
        poisson_log_variance(np.array([1.0, np.nan]))


def test_poisson_baseline_rejects_bad_mask_shape() -> None:
    umi = np.ones((3, 3), dtype=float)

    with pytest.raises(ValueError, match="mask has shape"):
        poisson_baseline(umi, mask=np.ones((2, 2), dtype=bool))


def test_boundary_tensor_recovers_axis_aligned_anisotropy() -> None:
    excess = np.zeros((2, 3, 4), dtype=float)
    excess[:, :, 0] = 2.5
    trace, lambda1, lambda2, evec1 = boundary_tensor(excess)

    assert trace.shape == (2, 3)
    assert np.allclose(trace, 2.5)
    assert np.allclose(lambda1, 2.5)
    assert np.allclose(lambda2, 0.0)
    assert np.allclose(evec1[:, :, 0], 1.0)
    assert np.allclose(evec1[:, :, 1], 0.0)


def test_adaptive_tessellation_rejects_misaligned_inputs() -> None:
    umi = np.ones((3, 3), dtype=float)
    trace = np.ones((2, 2), dtype=float)
    evec1 = np.ones((3, 3, 2), dtype=float)

    with pytest.raises(ValueError, match="must align with umi"):
        adaptive_tessellation(umi, trace, evec1)


def test_validate_grid_data_rejects_negative_counts() -> None:
    counts = sp.csr_matrix(np.array([[1.0], [-1.0]]))

    with pytest.raises(ValueError, match="finite non-negative"):
        grid = GridData(counts, rows=2, cols=1)
        validate_grid_data(grid)
