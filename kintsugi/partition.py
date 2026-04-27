"""Variogram-guided adaptive tessellation via watershed.

Partitions a regular lattice into Poisson-stationary regions using a
three-phase strategy:

1. **Seed detection**: Local minima of the smoothed boundary-tensor trace
   identify the centers of homogeneous patches.
2. **Watershed flooding**: The trace field serves as an elevation map;
   regions grow from seeds by flooding uphill, meeting naturally at trace
   ridges (biological boundaries).  No threshold parameter needed.
3. **Stationarity refinement**: Regions that fail a Poisson stationarity
   test are recursively bisected along the primary eigenvector direction.

The watershed formulation is resolution-invariant: it partitions based on
the *relative* structure of the trace field (minima vs ridges) rather than
requiring an absolute threshold.  This handles both the 2 µm regime (where
biological variation dominates Poisson noise) and coarser scales equally.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    find_objects,
    watershed_ift,
)
from ._poisson_log_var import poisson_log_variance


def adaptive_tessellation(
    umi: np.ndarray,
    trace: np.ndarray,
    evec1: np.ndarray,
    kappa: float = 2.0,
    min_seed_distance: int = 4,
    smooth_sigma: float = 4.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Partition a regular lattice into Poisson-stationary regions.

    Parameters
    ----------
    umi : ndarray, shape (R, C)
        Total UMI counts per bin.
    trace : ndarray, shape (R, C)
        Boundary-tensor trace (total boundary strength).
    evec1 : ndarray, shape (R, C, 2)
        Primary eigenvector of the boundary tensor.
    kappa : float
        Stationarity tolerance in units of SE for the Poisson
        stationarity test during subdivision.
    min_seed_distance : int
        Minimum distance (in bins) between seeds.  Controls the spatial
        scale of the tessellation: larger values produce fewer, larger
        regions.
    smooth_sigma : float
        Gaussian sigma (in bins) for smoothing the trace field before
        seed detection.
    mask : ndarray, shape (R, C), dtype bool, optional
        True for in-tissue bins. If None, all bins are used.

    Returns
    -------
    labels : ndarray, shape (R, C), dtype int32
        Region labels, 0-indexed.  Bins outside ``mask`` get label -1.
    """
    if trace.shape != umi.shape or evec1.shape[:2] != umi.shape:
        raise ValueError("trace and evec1 must align with umi.")

    R, C = umi.shape
    z = np.log1p(umi.astype(np.float64))

    if mask is None:
        mask = np.ones((R, C), dtype=bool)
    elif mask.shape != (R, C):
        raise ValueError(f"mask has shape {mask.shape}, expected {(R, C)}.")

    # --- Phase 1: seed detection via smoothed trace minima ----------------
    trace_smooth = gaussian_filter(trace.astype(np.float64), sigma=smooth_sigma)

    # Suppress out-of-tissue bins with a large finite sentinel (np.inf
    # causes NaN propagation in scipy's running-sum filter internals).
    trace_ceil = trace_smooth[mask].max() if mask.any() else 1.0
    trace_smooth[~mask] = trace_ceil * 10.0

    # Seeds = one per non-overlapping block of size win × win.  Within each
    # block, the in-tissue bin with the lowest smoothed trace becomes the
    # seed.  This guarantees uniform seed density regardless of trace
    # flatness (minimum_filter fails in constant-trace regions where every
    # pixel ties for the local minimum).
    win = 2 * min_seed_distance + 1
    seeds = np.zeros((R, C), dtype=bool)
    sentinel = trace_ceil * 10.0
    for r0 in range(0, R, win):
        for c0 in range(0, C, win):
            r1 = min(r0 + win, R)
            c1 = min(c0 + win, C)
            block_mask = mask[r0:r1, c0:c1]
            if not block_mask.any():
                continue
            block_trace = np.where(block_mask, trace_smooth[r0:r1, c0:c1], sentinel)
            min_idx = np.argmin(block_trace)
            mr, mc = np.unravel_index(min_idx, block_trace.shape)
            seeds[r0 + mr, c0 + mc] = True

    n_seeds = seeds.sum()
    if n_seeds == 0:
        labels = np.full((R, C), -1, dtype=np.int32)
        labels[mask] = 0
        return labels

    # --- Phase 2: watershed from seeds on the trace elevation map ---------
    # Each seed grows outward, claiming pixels in order of increasing trace
    # value, stopping where basins meet (trace ridges = biological
    # boundaries).  Uses scipy's Image Foresting Transform watershed.

    # Create integer markers: 0 = unlabeled, 1..n_seeds = seed labels.
    markers = np.zeros((R, C), dtype=np.int32)
    seed_rows, seed_cols = np.where(seeds)
    markers[seed_rows, seed_cols] = np.arange(1, n_seeds + 1, dtype=np.int32)

    # Quantize trace to uint16 for watershed_ift (preserves relative order).
    trace_min = trace_smooth[mask].min()
    trace_range = trace_ceil - trace_min
    if trace_range > 0:
        trace_uint = ((trace_smooth - trace_min) / trace_range * 65534).astype(np.uint16)
    else:
        trace_uint = np.zeros((R, C), dtype=np.uint16)
    # Set out-of-tissue to maximum elevation (barrier).
    trace_uint[~mask] = 65535

    # 4-connectivity structure (rook-adjacent).
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    # watershed_ift floods from markers on the elevation surface.
    ws_labels = watershed_ift(trace_uint, markers, structure=structure)

    # Convert to 0-indexed labels; mask out-of-tissue.
    labels = (ws_labels - 1).astype(np.int32)
    labels[~mask] = -1

    # --- Phase 3: Poisson stationarity refinement -------------------------
    # Watershed basins respect trace topology but may still be non-
    # stationary in UMI density (e.g., a basin spanning a gradual gradient).
    # Recursively bisect violators along the primary eigenvector direction.
    # Minimum sub-region size scales with seed spacing to prevent
    # over-fragmentation.
    min_region_bins = max(min_seed_distance ** 2, _MIN_REGION_BINS)
    n_regions = int(labels.max()) + 1
    labels = _stationarity_refinement(
        labels, n_regions, z, umi, evec1, kappa, min_region_bins,
    )

    return labels


# --- Stationarity refinement ---------------------------------------------

_MAX_SUBDIVISION_DEPTH = 5
_MIN_REGION_BINS = 16


def _poisson_stationary(z_vals: np.ndarray, umi_vals: np.ndarray, kappa: float) -> bool:
    """Test whether a set of bins satisfies Poisson stationarity.

    The test compares the sample variance of z = log(1+UMI) against the
    exact Poisson-log variance under a homogeneous Poisson model:
        H0: Var(z) <= gamma_0(lambda_bar) + kappa * SE

    Returns True if stationary (pass), False if not (needs subdivision).
    """
    n = z_vals.size
    if n < 4:
        return True

    sample_var = np.var(z_vals, ddof=1)
    lambda_bar = np.mean(umi_vals)
    gamma0 = poisson_log_variance(lambda_bar)

    # Standard error of sample variance under chi-squared model.
    se = np.sqrt(2.0 * gamma0 * gamma0 / (n - 1.0))

    return sample_var <= gamma0 + kappa * se


def _stationarity_refinement(
    labels: np.ndarray,
    n_regions: int,
    z: np.ndarray,
    umi: np.ndarray,
    evec1: np.ndarray,
    kappa: float,
    min_region_bins: int = 16,
) -> np.ndarray:
    """Refine watershed basins: recursively subdivide non-stationary regions."""
    R, C = z.shape
    refined = np.full((R, C), -1, dtype=np.int32)
    next_label = [0]
    slices = find_objects(labels + 1)  # find_objects uses 1-indexed labels

    for region_id in range(n_regions):
        sl = slices[region_id]
        if sl is None:
            continue
        region_mask = labels[sl] == region_id
        _subdivide_recursive(
            sl, region_mask, z, umi, evec1, kappa, refined, next_label,
            depth=0, min_bins=min_region_bins,
        )

    return refined


def _subdivide_recursive(
    sl: tuple,
    region_mask: np.ndarray,
    z: np.ndarray,
    umi: np.ndarray,
    evec1: np.ndarray,
    kappa: float,
    labels: np.ndarray,
    next_label: list,
    depth: int,
    min_bins: int = 16,
) -> None:
    """Recursively bisect a region until Poisson stationarity holds."""
    z_sub = z[sl]
    umi_sub = umi[sl]

    z_vals = z_sub[region_mask]
    umi_vals = umi_sub[region_mask]

    if z_vals.size == 0:
        return

    # Accept if stationary, too small to split, or max depth reached.
    if (
        _poisson_stationary(z_vals, umi_vals, kappa)
        or z_vals.size < 2 * min_bins
        or depth >= _MAX_SUBDIVISION_DEPTH
    ):
        labels[sl][region_mask] = next_label[0]
        next_label[0] += 1
        return

    # Bisect along the mean primary eigenvector within this region.
    ev_sub = evec1[sl]
    rows_idx, cols_idx = np.where(region_mask)

    cy, cx = np.mean(rows_idx), np.mean(cols_idx)
    dy, dx = rows_idx - cy, cols_idx - cx

    mean_ev = np.mean(ev_sub[region_mask], axis=0)
    norm = np.linalg.norm(mean_ev)
    if norm < 1e-12:
        mean_ev = np.array([1.0, 0.0])
    else:
        mean_ev = mean_ev / norm

    proj = dy * mean_ev[0] + dx * mean_ev[1]
    median_proj = np.median(proj)

    half_a = proj <= median_proj
    half_b = ~half_a

    for half in (half_a, half_b):
        if not np.any(half):
            continue
        sub_mask = np.zeros_like(region_mask)
        sub_mask[rows_idx[half], cols_idx[half]] = True
        _subdivide_recursive(
            sl, sub_mask, z, umi, evec1, kappa, labels, next_label,
            depth + 1, min_bins=min_bins,
        )
