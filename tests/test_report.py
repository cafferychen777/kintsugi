"""Tests for the kintsugi.report module."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import kintsugi
from kintsugi.report import TessellationReport, tessellation_report


def _make_grid_and_result(
    rows: int = 20,
    cols: int = 20,
    n_genes: int = 10,
    seed: int = 42,
):
    """Helper: create a small grid and tessellation result."""
    N = rows * cols
    counts = sp.random(N, n_genes, density=0.3, format="csr", random_state=seed)
    counts.data = np.round(counts.data * 10).astype(float)
    counts.data = np.maximum(counts.data, 0)
    grid = kintsugi.GridData(counts, rows=rows, cols=cols)
    result = grid.tessellate(min_seed_distance=2, smooth_sigma=2.0)
    return grid, result


def test_report_returns_correct_type() -> None:
    grid, result = _make_grid_and_result()
    report = tessellation_report(result, grid)
    assert isinstance(report, TessellationReport)


def test_report_region_count_matches_result() -> None:
    grid, result = _make_grid_and_result()
    report = tessellation_report(result, grid)
    assert report.n_regions == result.n_regions


def test_report_median_umi_matches_depths() -> None:
    grid, result = _make_grid_and_result()
    report = tessellation_report(result, grid)
    expected = float(np.median(result.depths))
    assert abs(report.median_umi_per_region - expected) < 1e-10


def test_report_density_cv_is_nonnegative() -> None:
    grid, result = _make_grid_and_result()
    report = tessellation_report(result, grid)
    assert report.density_cv >= 0.0


def test_report_stationarity_rate_in_unit_interval() -> None:
    grid, result = _make_grid_and_result()
    report = tessellation_report(result, grid)
    assert 0.0 <= report.stationarity_pass_rate <= 1.0


def test_report_holdout_likelihoods_are_finite() -> None:
    grid, result = _make_grid_and_result()
    report = tessellation_report(result, grid)
    assert np.isfinite(report.composition_holdout_ll)
    assert np.isfinite(report.density_holdout_ll)
    assert np.isfinite(report.composition_ll_ratio)
    assert np.isfinite(report.density_ll_ratio)


def test_report_accepts_raw_sparse_matrix_input() -> None:
    rows, cols, n_genes = 10, 10, 5
    counts = sp.random(rows * cols, n_genes, density=0.3, format="csr", random_state=0)
    counts.data = np.round(counts.data * 10).astype(float)
    counts.data = np.maximum(counts.data, 0)
    mask = np.ones((rows, cols), dtype=bool)

    result = kintsugi.tessellate(
        counts, rows=rows, cols=cols, mask=mask,
        min_seed_distance=1, smooth_sigma=1.0,
    )
    report = tessellation_report(
        result, counts, rows=rows, cols=cols, mask=mask,
    )
    assert report.n_regions == result.n_regions


def test_report_rejects_ambiguous_griddata_and_explicit_shape() -> None:
    grid, result = _make_grid_and_result(rows=8, cols=8, n_genes=4)

    with pytest.raises(ValueError, match="must not be passed separately"):
        tessellation_report(result, grid, rows=8, cols=8)


def test_report_rejects_bad_sparse_input_shape() -> None:
    counts = sp.csr_matrix(np.ones((3, 2), dtype=float))
    result = kintsugi.tessellate(
        sp.csr_matrix(np.ones((4, 2), dtype=float)),
        rows=2,
        cols=2,
        min_seed_distance=1,
        smooth_sigma=1.0,
    )

    with pytest.raises(ValueError, match="counts has 3 rows"):
        tessellation_report(result, counts, rows=2, cols=2)


def test_report_rejects_bad_mask_shape() -> None:
    counts = sp.csr_matrix(np.ones((4, 2), dtype=float))
    result = kintsugi.tessellate(
        counts,
        rows=2,
        cols=2,
        min_seed_distance=1,
        smooth_sigma=1.0,
    )

    with pytest.raises(ValueError, match="mask has shape"):
        tessellation_report(
            result,
            counts,
            rows=2,
            cols=2,
            mask=np.ones((3, 3), dtype=bool),
        )


def test_report_rejects_invalid_holdout_fraction() -> None:
    grid, result = _make_grid_and_result(rows=8, cols=8, n_genes=4)

    with pytest.raises(ValueError, match="holdout_fraction"):
        tessellation_report(result, grid, holdout_fraction=1.0)
    with pytest.raises(ValueError, match="holdout_fraction"):
        tessellation_report(result, grid, holdout_fraction=0.0)


def test_report_is_reproducible_with_same_seed() -> None:
    grid, result = _make_grid_and_result()
    r1 = tessellation_report(result, grid, seed=123)
    r2 = tessellation_report(result, grid, seed=123)
    assert r1.composition_holdout_ll == r2.composition_holdout_ll
    assert r1.density_holdout_ll == r2.density_holdout_ll


def test_report_str_contains_key_fields() -> None:
    grid, result = _make_grid_and_result()
    report = tessellation_report(result, grid)
    text = str(report)
    assert "Regions" in text
    assert "Median UMI" in text
    assert "Density CV" in text
    assert "Stationarity pass" in text
    assert "Composition holdout" in text
    assert "Density holdout" in text


def test_report_composition_dominates_warning_is_bool() -> None:
    grid, result = _make_grid_and_result()
    report = tessellation_report(result, grid)
    assert isinstance(report.composition_dominates_warning, bool)


def test_report_on_empty_tessellation() -> None:
    counts = sp.csr_matrix(np.ones((4, 2), dtype=float))
    mask = np.zeros((2, 2), dtype=bool)

    result = kintsugi.tessellate(
        counts, rows=2, cols=2, mask=mask,
        min_seed_distance=1, smooth_sigma=1.0,
    )
    report = tessellation_report(result, counts, rows=2, cols=2, mask=mask)
    assert report.n_regions == 0
    assert report.density_cv == 0.0
    assert report.stationarity_pass_rate == 0.0
