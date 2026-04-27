"""Kintsugi: adaptive spatial tessellation for sub-cellular transcriptomics."""

from __future__ import annotations

__version__ = "0.1.0"

import numpy as np
import scipy.sparse as sp

from .models import GridData, TessellationResult, validate_grid_data
from .variogram import directional_semivariance, poisson_baseline
from ._poisson_log_var import poisson_log_variance
from .tensor import boundary_tensor
from .partition import adaptive_tessellation
from .aggregate import aggregate_counts
from .graph import build_spatial_graph
from .io import (
    build_regular_grid,
    load_10x_feature_matrix,
    load_visium_hd,
    load_visium_hd_from_dir,
    parse_visium_barcode_coordinates,
    read_tissue_positions,
)

__all__ = [
    "__version__",
    "GridData",
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
    "tessellate",
]


def tessellate(
    counts: GridData | sp.spmatrix,
    rows: int | None = None,
    cols: int | None = None,
    mask: np.ndarray | None = None,
    lag: int = 2,
    kappa: float = 2.0,
    min_seed_distance: int = 4,
    smooth_sigma: float = 4.0,
) -> TessellationResult:
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

    # Step 0: build UMI field on the 2D grid.
    umi_flat = np.asarray(counts.sum(axis=1)).ravel()  # (N,)
    umi = umi_flat.reshape(rows, cols)

    # Step 1: directional semivariance.
    excess = directional_semivariance(umi, lag=lag, mask=mask)

    # Step 2: boundary tensor.
    trace, _lambda1, _lambda2, evec1 = boundary_tensor(excess)

    # Step 3: adaptive tessellation.
    labels = adaptive_tessellation(
        umi, trace, evec1,
        kappa=kappa,
        min_seed_distance=min_seed_distance,
        smooth_sigma=smooth_sigma,
        mask=mask,
    )

    # Step 4: aggregate counts.
    residuals, areas, depths, centroids = aggregate_counts(counts, labels, mask=mask)

    # Step 5: spatial graph.
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
