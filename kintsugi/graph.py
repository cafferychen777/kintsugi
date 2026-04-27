"""Spatial graph construction from an irregular tessellation.

Builds a region adjacency graph: two regions are connected if they share
at least one pair of rook-adjacent (4-connected) native bins.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_spatial_graph(
    labels: np.ndarray,
) -> sp.csr_matrix:
    """Build a symmetric binary adjacency matrix from region labels.

    Parameters
    ----------
    labels : ndarray, shape (R, C), dtype int32
        Region labels from ``adaptive_tessellation``.  Values in
        ``[0, K)``.  Bins with label -1 are ignored.

    Returns
    -------
    adjacency : csr_matrix, shape (K, K)
        Symmetric binary adjacency (no self-loops).
    """
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array.")

    R, C = labels.shape
    valid_labels = labels[labels >= 0]
    if valid_labels.size == 0:
        return sp.csr_matrix((0, 0), dtype=np.uint8)
    K = int(valid_labels.max()) + 1

    edges_r: list[np.ndarray] = []
    edges_c: list[np.ndarray] = []

    # Horizontal neighbours: compare labels[:, :-1] with labels[:, 1:]
    left = labels[:, :-1].ravel()
    right = labels[:, 1:].ravel()
    valid = (left >= 0) & (right >= 0) & (left != right)
    edges_r.append(left[valid])
    edges_c.append(right[valid])

    # Vertical neighbours: compare labels[:-1, :] with labels[1:, :]
    top = labels[:-1, :].ravel()
    bot = labels[1:, :].ravel()
    valid = (top >= 0) & (bot >= 0) & (top != bot)
    edges_r.append(top[valid])
    edges_c.append(bot[valid])

    if not any(edge.size for edge in edges_r):
        return sp.csr_matrix((K, K), dtype=np.uint8)

    row = np.concatenate(edges_r)
    col = np.concatenate(edges_c)

    # Symmetrize.
    row_sym = np.concatenate([row, col])
    col_sym = np.concatenate([col, row])
    data = np.ones(row_sym.size, dtype=np.uint8)

    adj = sp.coo_matrix((data, (row_sym, col_sym)), shape=(K, K))
    # Remove duplicates and ensure binary.
    adj = adj.tocsr()
    adj.data[:] = 1
    # Remove any self-loops (should not exist but be safe).
    adj.setdiag(0)
    adj.eliminate_zeros()

    return adj
