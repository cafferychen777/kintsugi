"""Tests for the kintsugi.io.anndata module."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import kintsugi

anndata = pytest.importorskip("anndata")


def _make_grid_and_result():
    counts = sp.random(400, 20, density=0.3, format="csr", random_state=42)
    counts.data = np.round(counts.data * 10).astype(float)
    counts.data = np.maximum(counts.data, 0)
    gene_names = [f"Gene_{g}" for g in range(20)]
    grid = kintsugi.GridData(
        counts, rows=20, cols=20,
        gene_names=np.array(gene_names),
    )
    result = grid.tessellate(min_seed_distance=2, smooth_sigma=2.0)
    return grid, result


def test_to_anndata_returns_anndata() -> None:
    grid, result = _make_grid_and_result()
    adata = kintsugi.to_anndata(result, grid=grid)
    assert isinstance(adata, anndata.AnnData)


def test_to_anndata_shape_matches_result() -> None:
    grid, result = _make_grid_and_result()
    adata = kintsugi.to_anndata(result, grid=grid)
    assert adata.shape == (result.n_regions, grid.n_genes)


def test_to_anndata_obs_columns() -> None:
    grid, result = _make_grid_and_result()
    adata = kintsugi.to_anndata(result, grid=grid)
    assert "area" in adata.obs.columns
    assert "depth" in adata.obs.columns
    assert np.allclose(adata.obs["area"].values, result.areas)
    assert np.allclose(adata.obs["depth"].values, result.depths)


def test_to_anndata_spatial_coordinates() -> None:
    grid, result = _make_grid_and_result()
    adata = kintsugi.to_anndata(result, grid=grid)
    assert "spatial" in adata.obsm
    assert adata.obsm["spatial"].shape == (result.n_regions, 2)
    assert np.allclose(adata.obsm["spatial"], result.centroids)


def test_to_anndata_adjacency_graph() -> None:
    grid, result = _make_grid_and_result()
    adata = kintsugi.to_anndata(result, grid=grid)
    assert "adjacency" in adata.obsp
    assert adata.obsp["adjacency"].shape == (result.n_regions, result.n_regions)


def test_to_anndata_gene_names_from_grid() -> None:
    grid, result = _make_grid_and_result()
    adata = kintsugi.to_anndata(result, grid=grid)
    assert list(adata.var_names) == [f"Gene_{g}" for g in range(20)]


def test_to_anndata_gene_names_override() -> None:
    grid, result = _make_grid_and_result()
    custom_names = [f"custom_{g}" for g in range(20)]
    adata = kintsugi.to_anndata(result, gene_names=custom_names)
    assert list(adata.var_names) == custom_names


def test_to_anndata_raw_counts_layer() -> None:
    grid, result = _make_grid_and_result()
    adata = kintsugi.to_anndata(result, grid=grid, use_raw_counts=True)
    assert "counts" in adata.layers
    assert adata.layers["counts"].shape == adata.shape
    # Raw counts should be non-negative
    assert np.all(adata.layers["counts"] >= 0)


def test_to_anndata_raw_counts_requires_grid() -> None:
    _grid, result = _make_grid_and_result()
    with pytest.raises(ValueError, match="grid is required"):
        kintsugi.to_anndata(result, use_raw_counts=True)


def test_to_anndata_x_contains_residuals() -> None:
    grid, result = _make_grid_and_result()
    adata = kintsugi.to_anndata(result, grid=grid)
    assert np.allclose(adata.X, result.residuals)


def test_to_anndata_without_grid_uses_default_gene_names() -> None:
    grid, result = _make_grid_and_result()
    adata = kintsugi.to_anndata(result)
    assert adata.var_names[0] == "gene_0"


def test_to_anndata_rejects_wrong_gene_names_length() -> None:
    grid, result = _make_grid_and_result()
    with pytest.raises(ValueError, match="gene_names has length"):
        kintsugi.to_anndata(result, gene_names=["a", "b"])


def test_to_anndata_empty_result_preserves_gene_dimension() -> None:
    counts = sp.csr_matrix(np.ones((4, 3), dtype=float))
    mask = np.zeros((2, 2), dtype=bool)
    grid = kintsugi.GridData(
        counts,
        rows=2,
        cols=2,
        mask=mask,
        gene_names=np.array(["GeneA", "GeneB", "GeneC"]),
    )
    result = grid.tessellate(min_seed_distance=1, smooth_sigma=1.0)

    adata = kintsugi.to_anndata(result, grid=grid, use_raw_counts=True)

    assert adata.shape == (0, 3)
    assert list(adata.var_names) == ["GeneA", "GeneB", "GeneC"]
    assert adata.layers["counts"].shape == (0, 3)
