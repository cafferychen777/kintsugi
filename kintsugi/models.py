"""Typed containers for Kintsugi inputs and outputs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

_RESULT_KEYS = (
    "labels",
    "residuals",
    "areas",
    "depths",
    "centroids",
    "adjacency",
    "trace",
)


def validate_grid_data(data: "GridData") -> None:
    """Validate the normalized grid contract expected by Kintsugi."""
    if not sp.issparse(data.counts):
        raise TypeError("counts must be a SciPy sparse matrix.")
    if data.rows <= 0 or data.cols <= 0:
        raise ValueError("rows and cols must both be positive integers.")
    if data.counts.shape[0] != data.rows * data.cols:
        raise ValueError(
            f"counts has {data.counts.shape[0]} rows, expected "
            f"{data.rows * data.cols} = {data.rows} x {data.cols}."
        )
    if data.mask.shape != (data.rows, data.cols):
        raise ValueError(
            f"mask has shape {data.mask.shape}, expected {(data.rows, data.cols)}."
        )
    if data.counts.data.size and (
        np.any(~np.isfinite(data.counts.data)) or np.any(data.counts.data < 0)
    ):
        raise ValueError("counts must contain only finite non-negative values.")
    if data.gene_names is not None:
        if data.gene_names.ndim != 1:
            raise ValueError("gene_names must be a 1D array when provided.")
        if data.gene_names.shape[0] != data.counts.shape[1]:
            raise ValueError(
                f"gene_names has length {data.gene_names.shape[0]}, expected "
                f"{data.counts.shape[1]}."
            )


@dataclass(frozen=True, slots=True)
class GridData:
    """Normalized regular-grid input for Kintsugi."""

    counts: sp.csr_matrix
    rows: int
    cols: int
    mask: np.ndarray | None = None
    gene_names: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not sp.issparse(self.counts):
            raise TypeError("counts must be a SciPy sparse matrix.")

        counts = self.counts.tocsr() if not sp.isspmatrix_csr(self.counts) else self.counts
        counts.sum_duplicates()
        object.__setattr__(self, "counts", counts)

        mask = (
            np.ones((self.rows, self.cols), dtype=bool)
            if self.mask is None
            else np.asarray(self.mask, dtype=bool)
        )
        object.__setattr__(self, "mask", mask)

        gene_names = None
        if self.gene_names is not None:
            gene_names = np.asarray(self.gene_names, dtype=str)
        object.__setattr__(self, "gene_names", gene_names)

        validate_grid_data(self)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def n_bins(self) -> int:
        return self.rows * self.cols

    @property
    def n_genes(self) -> int:
        return self.counts.shape[1]

    def tessellate(
        self,
        *,
        lag: int = 2,
        kappa: float = 2.0,
        min_seed_distance: int = 4,
        smooth_sigma: float = 4.0,
    ) -> "TessellationResult":
        """Run Kintsugi directly from a normalized grid container."""
        from . import tessellate

        return tessellate(
            self,
            lag=lag,
            kappa=kappa,
            min_seed_distance=min_seed_distance,
            smooth_sigma=smooth_sigma,
        )


@dataclass(frozen=True, slots=True)
class TessellationResult(Mapping[str, object]):
    """Structured output of the Kintsugi pipeline."""

    labels: np.ndarray
    residuals: np.ndarray
    areas: np.ndarray
    depths: np.ndarray
    centroids: np.ndarray
    adjacency: sp.csr_matrix
    trace: np.ndarray

    def __post_init__(self) -> None:
        adjacency = (
            self.adjacency.tocsr()
            if sp.issparse(self.adjacency) and not sp.isspmatrix_csr(self.adjacency)
            else self.adjacency
        )
        object.__setattr__(self, "adjacency", adjacency)

        if self.labels.ndim != 2 or self.trace.shape != self.labels.shape:
            raise ValueError("labels and trace must be 2D arrays with the same shape.")

        n_regions = self.areas.shape[0]
        if self.residuals.ndim != 2 or self.residuals.shape[0] != n_regions:
            raise ValueError("residuals must have shape (K, G) with K = len(areas).")
        if self.depths.shape != (n_regions,):
            raise ValueError("depths must have shape (K,).")
        if self.centroids.shape != (n_regions, 2):
            raise ValueError("centroids must have shape (K, 2).")
        if self.adjacency.shape != (n_regions, n_regions):
            raise ValueError("adjacency must have shape (K, K).")

    def __getitem__(self, key: str) -> object:
        if key not in _RESULT_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(_RESULT_KEYS)

    def __len__(self) -> int:
        return len(_RESULT_KEYS)

    @property
    def n_regions(self) -> int:
        return self.areas.shape[0]

    def as_dict(self) -> dict[str, object]:
        return {key: getattr(self, key) for key in _RESULT_KEYS}
