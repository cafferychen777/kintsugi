"""Directional variogram estimation on a regular 2-micron lattice.

Exploits the grid structure: directional pairs at a given lag are obtained
by shifted array operations, making the computation O(N) and fully vectorized.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from ._poisson_log_var import poisson_log_variance


def directional_semivariance(
    umi: np.ndarray,
    lag: int = 2,
    directions: int = 4,
    smooth_halfwidth: int = 3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute excess directional semivariance on a 2D UMI grid.

    Parameters
    ----------
    umi : ndarray, shape (R, C)
        Total UMI counts per bin on the regular lattice.
    lag : int
        Lag in bin units (default 2 = 4 um on a 2 um grid).
    directions : int
        Number of equispaced directions in [0, pi). Must be 4.
    smooth_halfwidth : int
        Half-width of the uniform smoothing window applied to raw
        pointwise semivariance (actual window is 2*smooth_halfwidth + 1).
    mask : ndarray, shape (R, C), dtype bool, optional
        True for in-tissue bins.  Out-of-tissue bins are filled with
        the tissue median to prevent artificial boundary detection at
        tissue edges.

    Returns
    -------
    excess : ndarray, shape (R, C, D)
        Excess semivariance per direction, clipped to non-negative.
    """
    if directions != 4:
        raise ValueError("Only 4 directions (0, 45, 90, 135 deg) supported.")

    z = np.log1p(umi.astype(np.float64))
    R, C = z.shape

    # Fill out-of-tissue bins with tissue median to avoid edge artefacts.
    if mask is not None:
        if mask.shape != (R, C):
            raise ValueError(f"mask has shape {mask.shape}, expected {(R, C)}.")
        tissue_median = np.median(z[mask]) if np.any(mask) else 0.0
        z[~mask] = tissue_median

    h = lag

    # Direction shifts: (row_offset, col_offset) for 0, 45, 90, 135 degrees.
    # 0 deg  = east      -> (0, +h)
    # 45 deg = northeast -> (-h, +h)
    # 90 deg = north     -> (-h, 0)
    # 135 deg= northwest -> (-h, -h)
    shifts = [(0, h), (-h, h), (-h, 0), (-h, -h)]

    window = 2 * smooth_halfwidth + 1
    excess = np.zeros((R, C, directions), dtype=np.float64)

    gamma0 = poisson_baseline(umi, smooth_halfwidth=smooth_halfwidth, mask=mask)

    for d, (dr, dc) in enumerate(shifts):
        # Compute pointwise semivariance via shifted arrays.
        # Source and target slices that align after the shift.
        src_r = slice(max(0, -dr), R - max(0, dr))
        src_c = slice(max(0, -dc), C - max(0, dc))
        tgt_r = slice(max(0, dr), R - max(0, -dr))
        tgt_c = slice(max(0, dc), C - max(0, -dc))

        dz = z[tgt_r, tgt_c] - z[src_r, src_c]
        gamma_raw = 0.5 * dz * dz

        # Pad back to full grid (edges get zero, will be overwritten by smoothing).
        gamma_full = np.zeros((R, C), dtype=np.float64)
        # Count how many pairs contribute to each pixel (for proper averaging).
        count_full = np.zeros((R, C), dtype=np.float64)

        gamma_full[src_r, src_c] += gamma_raw
        gamma_full[tgt_r, tgt_c] += gamma_raw
        count_full[src_r, src_c] += 1.0
        count_full[tgt_r, tgt_c] += 1.0

        # Smooth both numerator and denominator, then divide -> local average.
        gamma_smooth = uniform_filter(gamma_full, size=window, mode="nearest")
        count_smooth = uniform_filter(count_full, size=window, mode="nearest")
        count_smooth = np.maximum(count_smooth, 1e-12)  # avoid division by zero

        gamma_local = gamma_smooth / count_smooth

        excess[:, :, d] = np.maximum(gamma_local - gamma0, 0.0)

    return excess


def poisson_baseline(
    umi: np.ndarray,
    smooth_halfwidth: int = 3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the Poisson null semivariance field.

    Parameters
    ----------
    umi : ndarray, shape (R, C)
        Total UMI counts per bin.
    smooth_halfwidth : int
        Half-width of the uniform smoothing window.
    mask : ndarray, shape (R, C), dtype bool, optional
        True for in-tissue bins. Out-of-tissue bins are filled with the
        in-tissue mean before local smoothing.

    Returns
    -------
    gamma0 : ndarray, shape (R, C)
        Expected semivariance under Poisson null.
    """
    umi_f = umi.astype(np.float64)
    if mask is not None:
        if mask.shape != umi_f.shape:
            raise ValueError(f"mask has shape {mask.shape}, expected {umi_f.shape}.")
        tissue_mean = np.mean(umi_f[mask]) if np.any(mask) else 1.0
        umi_f[~mask] = tissue_mean

    window = 2 * smooth_halfwidth + 1
    lambda_local = uniform_filter(umi_f, size=window, mode="nearest")
    return poisson_log_variance(lambda_local)
