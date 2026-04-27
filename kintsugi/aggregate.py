"""Null-aware count aggregation and Pearson residual normalization.

Aggregates a sparse (N x G) count matrix over an irregular partition
into a dense (K x G) residual matrix, where K << N.  All operations
keep the input sparse until the final K x G product.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def aggregate_counts(
    counts: sp.csr_matrix,
    labels: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate bin-level counts into region-level Pearson residuals.

    Parameters
    ----------
    counts : csr_matrix, shape (N, G)
        Sparse UMI count matrix.  Row order must match the row-major
        (C-order) flattening of the 2D lattice: bin (r, c) maps to
        row ``r * C + c``.
    labels : ndarray, shape (R, C), dtype int32
        Region labels from ``adaptive_tessellation``.  Values in
        ``[0, K)``.  Bins with label -1 are excluded.
    mask : ndarray, shape (R, C), dtype bool, optional
        If provided, only bins where mask is True are aggregated.
        Bins outside the mask are ignored regardless of their label.

    Returns
    -------
    residuals : ndarray, shape (K, G)
        Pearson residuals ``(observed - expected) / sqrt(expected)``.
    areas : ndarray, shape (K,)
        Number of native bins per region.
    depths : ndarray, shape (K,)
        Total UMI per region.
    centroids : ndarray, shape (K, 2)
        Region centroids in (row, col) grid coordinates.
    """
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array.")
    if not sp.issparse(counts):
        raise TypeError("counts must be a SciPy sparse matrix.")
    if not sp.isspmatrix_csr(counts):
        counts = counts.tocsr()

    R, C = labels.shape
    N = R * C

    if counts.shape[0] != N:
        raise ValueError(
            f"Count matrix has {counts.shape[0]} rows but label grid has "
            f"{N} bins ({R} x {C})."
        )
    if mask is not None and mask.shape != labels.shape:
        raise ValueError(
            f"mask has shape {mask.shape}, expected {labels.shape}."
        )

    flat_labels = labels.ravel()  # (N,)

    # Build inclusion mask.
    include = flat_labels >= 0
    if mask is not None:
        include = include & mask.ravel()

    G = counts.shape[1]

    # --- Sparse indicator matrix (K x N) ----------------------------------
    valid_idx = np.where(include)[0]
    if valid_idx.size == 0:
        empty = np.zeros((0, G), dtype=np.float64)
        return (
            empty,
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
        )

    valid_labels = flat_labels[valid_idx].astype(np.int64, copy=False)
    K = int(valid_labels.max()) + 1

    indicator = sp.csr_matrix(
        (np.ones(valid_idx.size, dtype=np.uint8), (valid_labels, valid_idx)),
        shape=(K, N),
    )

    # --- Aggregated counts Y' = indicator @ counts  (K x G) ---------------
    # counts is (N, G) sparse; indicator is (K, N) sparse.
    # Result is (K, G) — dense since K ~ 10^3-10^5, fits in memory.
    agg_counts = (indicator @ counts).toarray().astype(np.float64, copy=False)  # (K, G)

    # --- Region statistics ------------------------------------------------
    areas = np.bincount(valid_labels, minlength=K).astype(np.float64, copy=False)
    depths = agg_counts.sum(axis=1)                     # (K,)
    N_total = depths.sum()

    # Gene totals across all included bins.
    gene_totals = np.asarray(agg_counts.sum(axis=0)).ravel()  # (G,)

    # --- Centroids --------------------------------------------------------
    row_coords, col_coords = np.divmod(valid_idx, C)
    centroid_rows = np.bincount(
        valid_labels,
        weights=row_coords.astype(np.float64),
        minlength=K,
    )
    centroid_cols = np.bincount(
        valid_labels,
        weights=col_coords.astype(np.float64),
        minlength=K,
    )
    safe_areas = np.maximum(areas, 1.0)
    centroids = np.column_stack([centroid_rows / safe_areas, centroid_cols / safe_areas])

    # --- Pearson residuals ------------------------------------------------
    # mu_kj = depth_k * gene_j / N_total
    if N_total == 0:
        return np.zeros((K, G), dtype=np.float64), areas, depths, centroids

    expected = (depths[:, None] * gene_totals[None, :]) / N_total
    np.subtract(agg_counts, expected, out=agg_counts)
    np.maximum(expected, 1e-12, out=expected)
    np.sqrt(expected, out=expected)
    np.divide(agg_counts, expected, out=agg_counts)

    return agg_counts, areas, depths, centroids
