"""Local boundary tensor from directional excess semivariance.

Constructs a 2x2 symmetric PSD tensor at each bin by summing outer
products of direction vectors weighted by the excess semivariance.
Returns the eigendecomposition (trace, eigenvalues, primary eigenvector).
"""

from __future__ import annotations

import numpy as np


def boundary_tensor(
    excess_semivar: np.ndarray,
    directions: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the boundary tensor and its eigendecomposition.

    Parameters
    ----------
    excess_semivar : ndarray, shape (R, C, D)
        Excess semivariance per direction from ``directional_semivariance``.
    directions : int
        Number of equispaced directions in [0, pi). Must match axis 2.

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
    if excess_semivar.shape[2] != directions:
        raise ValueError(
            f"Expected {directions} directions, got {excess_semivar.shape[2]}."
        )
    if directions != 4:
        raise ValueError("Only 4 directions supported.")

    R, C, D = excess_semivar.shape

    # Direction angles: 0, pi/4, pi/2, 3pi/4
    angles = np.arange(D) * (np.pi / D)
    cos_a = np.cos(angles)  # (D,)
    sin_a = np.sin(angles)  # (D,)

    # Outer product components for each direction:
    # e_theta (x) e_theta = [[cos^2, cos*sin], [cos*sin, sin^2]]
    cc = cos_a * cos_a  # (D,)
    ss = sin_a * sin_a  # (D,)
    cs = cos_a * sin_a  # (D,)

    # Sum over directions: T_ij(r, c) = sum_d excess(r, c, d) * component_d
    # excess_semivar is (R, C, D), components are (D,), broadcast via dot.
    T_xx = np.dot(excess_semivar, cc)  # (R, C)
    T_yy = np.dot(excess_semivar, ss)  # (R, C)
    T_xy = np.dot(excess_semivar, cs)  # (R, C)

    trace = T_xx + T_yy

    # Eigenvalues of 2x2 symmetric matrix [[a, b], [b, c]]:
    # lambda = (a+c)/2 +/- sqrt(((a-c)/2)^2 + b^2)
    half_diff = 0.5 * (T_xx - T_yy)
    discriminant = np.sqrt(half_diff * half_diff + T_xy * T_xy)

    half_sum = 0.5 * trace
    lambda1 = half_sum + discriminant
    lambda2 = half_sum - discriminant

    # Eigenvector for lambda1: (T_xy, lambda1 - T_xx) normalized.
    # When T_xy ~ 0 and T_xx >= T_yy, the eigenvector is (1, 0).
    ev_x = T_xy
    ev_y = lambda1 - T_xx

    norm = np.sqrt(ev_x * ev_x + ev_y * ev_y)
    # Handle degenerate case (isotropic or zero tensor).
    degen = norm < 1e-15
    ev_x = np.where(degen, 1.0, ev_x / np.maximum(norm, 1e-15))
    ev_y = np.where(degen, 0.0, ev_y / np.maximum(norm, 1e-15))

    evec1 = np.stack([ev_x, ev_y], axis=-1)  # (R, C, 2)

    return trace, lambda1, lambda2, evec1
