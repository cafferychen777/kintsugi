"""AnnData export for Kintsugi tessellation results.

Converts a ``TessellationResult`` into an :class:`anndata.AnnData` object
compatible with the scverse ecosystem (Scanpy, Squidpy, etc.).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from ..models import GridData


def to_anndata(
    result,
    gene_names: np.ndarray | list[str] | None = None,
    *,
    use_raw_counts: bool = False,
    grid: GridData | None = None,
):
    """Convert a Kintsugi tessellation result to AnnData.

    Parameters
    ----------
    result : TessellationResult
        Output of ``kintsugi.tessellate``.
    gene_names : array-like of str, optional
        Gene names for ``var_names``.  If *grid* is provided and has
        ``gene_names``, those are used by default.
    use_raw_counts : bool
        If True and *grid* is provided, store aggregated raw counts in
        ``adata.layers["counts"]`` in addition to residuals in ``X``.
    grid : GridData, optional
        The original input grid.  When provided, gene names are read
        from it (unless overridden) and raw counts can be aggregated.

    Returns
    -------
    adata : anndata.AnnData
        Region-by-gene AnnData with:

        - ``X``: Pearson residuals (n_regions × n_genes).
        - ``obs["area"]``, ``obs["depth"]``: region statistics.
        - ``obsm["spatial"]``: region centroids as (row, col).
        - ``obsp["adjacency"]``: spatial adjacency graph.
        - ``layers["counts"]`` (optional): aggregated raw UMI counts.

    Raises
    ------
    ImportError
        If ``anndata`` is not installed.
    """
    ad = _require_anndata()

    K = result.n_regions
    G = result.residuals.shape[1]

    if use_raw_counts and grid is None:
        raise ValueError("grid is required when use_raw_counts=True.")

    # Resolve gene names
    if gene_names is None and grid is not None and grid.gene_names is not None:
        gene_names = grid.gene_names
    if gene_names is not None:
        gene_names = np.asarray(gene_names, dtype=str)
        if gene_names.shape[0] != G:
            raise ValueError(
                f"gene_names has length {gene_names.shape[0]}, expected {G}."
            )

    # Build obs DataFrame
    import pandas as pd

    obs = pd.DataFrame(
        {
            "area": result.areas,
            "depth": result.depths,
        },
        index=[f"region_{k}" for k in range(K)],
    )

    # Build var DataFrame
    if gene_names is not None and G > 0:
        var = pd.DataFrame(index=gene_names)
    else:
        var = pd.DataFrame(index=[f"gene_{g}" for g in range(G)])

    adata = ad.AnnData(
        X=result.residuals,
        obs=obs,
        var=var,
    )

    # Spatial coordinates
    adata.obsm["spatial"] = result.centroids.copy()

    # Adjacency graph
    adj = result.adjacency
    if sp.issparse(adj):
        adata.obsp["adjacency"] = adj.astype(np.float64).tocsr()
    else:
        adata.obsp["adjacency"] = sp.csr_matrix(adj, dtype=np.float64)

    # Optional raw counts layer
    if use_raw_counts and grid is not None:
        flat_labels = result.labels.ravel()
        flat_mask = grid.mask.ravel()
        include = (flat_labels >= 0) & flat_mask
        valid_idx = np.where(include)[0]
        valid_labels = flat_labels[valid_idx].astype(np.int64, copy=False)

        indicator = sp.csr_matrix(
            (np.ones(valid_idx.size, dtype=np.float64), (valid_labels, valid_idx)),
            shape=(K, grid.counts.shape[0]),
        )
        raw_counts = (indicator @ grid.counts).toarray().astype(np.float64)
        adata.layers["counts"] = raw_counts

    return adata


def _require_anndata():
    try:
        return importlib.import_module("anndata")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "anndata is required for AnnData export. Install it with: "
            "pip install 'kintsugi-st[anndata]'"
        ) from exc
