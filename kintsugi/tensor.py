"""Local boundary tensor from directional excess semivariance.

Constructs a 2x2 symmetric PSD tensor at each bin by summing outer
products of direction vectors weighted by the excess semivariance.
Returns the eigendecomposition (trace, eigenvalues, primary eigenvector).
"""

from __future__ import annotations

import numpy as np


def boundary_tensor(
    excess_semivar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the boundary tensor and its eigendecomposition.

    Parameters
    ----------
    excess_semivar : ndarray, shape (R, C, D)
        Excess semivariance per direction from ``directional_semivariance``.
    Returns
    -------
    trace : ndarray, shape (R, C)
        Tr(T) = total boundary strength.
    lambda1 : ndarray, shape (R, C)
        Largest eigenvalue.
    lambda2 : ndarray, shape (R, C)
        Smallest eigenvalue.
    evec1 : ndarray, shape (R, C, 2)
        Unit eigenvector corresponding to lambda1.
    """
    if excess_semivar.ndim != 3 or excess_semivar.shape[2] != 4:
        raise ValueError("excess_semivar must have shape (R, C, 4).")

    R, C, D = excess_semivar.shape
    angles = np.arange(D) * (np.pi / D)
    cos_a = np.cos(angles)  # (D,)
    sin_a = np.sin(angles)  # (D,)

    cc = cos_a * cos_a
    ss = sin_a * sin_a
    cs = cos_a * sin_a

    T_xx = np.dot(excess_semivar, cc)
    T_yy = np.dot(excess_semivar, ss)
    T_xy = np.dot(excess_semivar, cs)

    trace = T_xx + T_yy

    half_diff = 0.5 * (T_xx - T_yy)
    discriminant = np.sqrt(half_diff * half_diff + T_xy * T_xy)

    half_sum = 0.5 * trace
    lambda1 = half_sum + discriminant
    lambda2 = half_sum - discriminant

    ev_x = T_xy
    ev_y = lambda1 - T_xx

    norm = np.sqrt(ev_x * ev_x + ev_y * ev_y)
    degen = norm < 1e-15
    ev_x = np.where(degen, 1.0, ev_x / np.maximum(norm, 1e-15))
    ev_y = np.where(degen, 0.0, ev_y / np.maximum(norm, 1e-15))

    evec1 = np.stack([ev_x, ev_y], axis=-1)

    return trace, lambda1, lambda2, evec1
