"""Kintsugi: adaptive spatial tessellation for sub-cellular transcriptomics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import TessellationResult

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "GridData",
    "TessellationReport",
    "TessellationResult",
    "build_regular_grid",
    "load_10x_feature_matrix",
    "load_visium_hd",
    "load_visium_hd_from_dir",
    "parse_visium_barcode_coordinates",
    "read_tissue_positions",
    "validate_grid_data",
    "adaptive_tessellation",
    "aggregate_counts",
    "boundary_tensor",
    "build_spatial_graph",
    "directional_semivariance",
    "poisson_baseline",
    "poisson_log_variance",
    "make_toy_dataset",
    "tessellate",
    "tessellation_report",
    "to_anndata",
]

_LAZY_EXPORTS = {
    "GridData": (".models", "GridData"),
    "TessellationReport": (".report", "TessellationReport"),
    "TessellationResult": (".models", "TessellationResult"),
    "validate_grid_data": (".models", "validate_grid_data"),
    "build_regular_grid": (".io", "build_regular_grid"),
    "load_10x_feature_matrix": (".io", "load_10x_feature_matrix"),
    "load_visium_hd": (".io", "load_visium_hd"),
    "load_visium_hd_from_dir": (".io", "load_visium_hd_from_dir"),
    "parse_visium_barcode_coordinates": (".io", "parse_visium_barcode_coordinates"),
    "read_tissue_positions": (".io", "read_tissue_positions"),
    "to_anndata": (".io", "to_anndata"),
    "directional_semivariance": (".variogram", "directional_semivariance"),
    "poisson_baseline": (".variogram", "poisson_baseline"),
    "poisson_log_variance": ("._poisson_log_var", "poisson_log_variance"),
    "boundary_tensor": (".tensor", "boundary_tensor"),
    "adaptive_tessellation": (".partition", "adaptive_tessellation"),
    "aggregate_counts": (".aggregate", "aggregate_counts"),
    "build_spatial_graph": (".graph", "build_spatial_graph"),
    "tessellation_report": (".report", "tessellation_report"),
    "make_toy_dataset": ("._demo", "make_toy_dataset"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def tessellate(
    counts,
    rows: int | None = None,
    cols: int | None = None,
    mask=None,
    lag: int = 2,
    kappa: float = 2.0,
    min_seed_distance: int = 4,
    smooth_sigma: float = 4.0,
) -> "TessellationResult":
    """Run the full Kintsugi pipeline: variogram -> tensor -> partition -> aggregate -> graph.

    Parameters
    ----------
    counts : GridData or sparse matrix
        Either a normalized ``GridData`` container or a sparse UMI count
        matrix with shape ``(rows * cols, genes)`` in row-major grid order.
    rows, cols : int
        Grid dimensions. Required when ``counts`` is not a ``GridData`` instance.
    mask : ndarray, shape (rows, cols), dtype bool, optional
        True for in-tissue bins. Ignored when ``counts`` is a ``GridData`` instance.
    lag : int
        Variogram lag in bin units (2 bins = 4 um on a 2 um grid).
    kappa : float
        Stationarity tolerance (in SE units) for region growing.
    min_seed_distance : int
        Minimum seed separation in bins.
    smooth_sigma : float
        Gaussian sigma (bins) for trace smoothing before seed detection.

    Returns
    -------
    result : TessellationResult with fields
        ``"labels"``      : ndarray (rows, cols) int32, region labels (0-indexed, -1 outside mask)
        ``"residuals"``   : ndarray (K, G) float64, Pearson residuals
        ``"areas"``       : ndarray (K,) float64, bins per region
        ``"depths"``      : ndarray (K,) float64, total UMI per region
        ``"centroids"``   : ndarray (K, 2) float64, (row, col) centroids
        ``"adjacency"``   : csr_matrix (K, K), spatial adjacency graph
        ``"trace"``       : ndarray (rows, cols) float64, boundary-tensor trace
    """
    from .models import GridData, TessellationResult

    if isinstance(counts, GridData):
        if rows is not None or cols is not None or mask is not None:
            raise ValueError(
                "rows, cols, and mask must not be passed separately when counts is a GridData instance."
            )
        grid = counts
        counts = grid.counts
        rows = grid.rows
        cols = grid.cols
        mask = grid.mask
    else:
        if rows is None or cols is None:
            raise TypeError(
                "rows and cols are required when counts is not a GridData instance."
            )
        grid = GridData(counts, rows=rows, cols=cols, mask=mask)
        counts = grid.counts
        rows = grid.rows
        cols = grid.cols
        mask = grid.mask

    import numpy as np

    from .aggregate import aggregate_counts
    from .graph import build_spatial_graph
    from .partition import adaptive_tessellation
    from .tensor import boundary_tensor
    from .variogram import directional_semivariance

    umi_flat = np.asarray(counts.sum(axis=1)).ravel()
    umi = umi_flat.reshape(rows, cols)

    excess = directional_semivariance(umi, lag=lag, mask=mask)
    trace, _lambda1, _lambda2, evec1 = boundary_tensor(excess)
    labels = adaptive_tessellation(
        umi, trace, evec1,
        kappa=kappa,
        min_seed_distance=min_seed_distance,
        smooth_sigma=smooth_sigma,
        mask=mask,
    )

    residuals, areas, depths, centroids = aggregate_counts(counts, labels, mask=mask)
    adjacency = build_spatial_graph(labels)

    return TessellationResult(
        labels=labels,
        residuals=residuals,
        areas=areas,
        depths=depths,
        centroids=centroids,
        adjacency=adjacency,
        trace=trace,
    )
