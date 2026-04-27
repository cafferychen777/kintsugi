"""Regular-grid input builders for Kintsugi."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..models import GridData


def parse_visium_barcode_coordinates(
    barcodes: np.ndarray | list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Parse row and column coordinates from Visium-style barcodes."""
    barcode_array = np.asarray(barcodes, dtype=str)
    if barcode_array.ndim != 1:
        raise ValueError("barcodes must be a 1D array-like object.")

    rows = np.empty(barcode_array.shape[0], dtype=np.int64)
    cols = np.empty(barcode_array.shape[0], dtype=np.int64)

    for i, barcode in enumerate(barcode_array):
        parts = barcode.split("_")
        if len(parts) < 2:
            raise ValueError(
                f"barcode {barcode!r} does not contain trailing row/col coordinates."
            )
        try:
            rows[i] = int(parts[-2])
            cols[i] = int(parts[-1].split("-")[0])
        except ValueError as exc:
            raise ValueError(
                f"barcode {barcode!r} does not encode integer row/col coordinates."
            ) from exc

    return rows, cols


def build_regular_grid(
    counts: sp.spmatrix,
    row_coords: np.ndarray | list[int],
    col_coords: np.ndarray | list[int],
    *,
    rows: int | None = None,
    cols: int | None = None,
    mask: np.ndarray | None = None,
    gene_names: np.ndarray | list[str] | None = None,
) -> GridData:
    """Map occupied-bin counts onto a full row-major regular lattice."""
    if not sp.issparse(counts):
        raise TypeError("counts must be a SciPy sparse matrix.")

    counts = counts.tocsr() if not sp.isspmatrix_csr(counts) else counts.copy()
    counts.sum_duplicates()

    row_coords = _normalize_coords("row_coords", row_coords)
    col_coords = _normalize_coords("col_coords", col_coords)

    if counts.shape[0] != row_coords.shape[0] or counts.shape[0] != col_coords.shape[0]:
        raise ValueError(
            "counts, row_coords, and col_coords must describe the same number of bins."
        )

    mask_arr = None
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.ndim != 2:
            raise ValueError("mask must be a 2D array when provided.")

    rows = _infer_extent("rows", rows, row_coords, mask_arr, axis=0)
    cols = _infer_extent("cols", cols, col_coords, mask_arr, axis=1)

    if row_coords.size:
        if np.any(row_coords < 0) or np.any(col_coords < 0):
            raise ValueError("row_coords and col_coords must be non-negative.")
        if np.any(row_coords >= rows) or np.any(col_coords >= cols):
            raise ValueError("row_coords and col_coords must fall within the grid bounds.")

    flat_idx = row_coords * cols + col_coords
    if np.unique(flat_idx).shape[0] != flat_idx.shape[0]:
        raise ValueError("each occupied grid coordinate must be unique.")

    if mask_arr is None:
        mask_arr = np.zeros((rows, cols), dtype=bool)
        if row_coords.size:
            mask_arr[row_coords, col_coords] = True
    elif row_coords.size and not np.all(mask_arr[row_coords, col_coords]):
        raise ValueError("all occupied coordinates must fall inside the provided mask.")

    row_nnz = np.diff(counts.indptr)
    new_rows = np.repeat(flat_idx, row_nnz)
    full_counts = sp.coo_matrix(
        (counts.data, (new_rows, counts.indices)),
        shape=(rows * cols, counts.shape[1]),
    ).tocsr()

    return GridData(
        full_counts,
        rows=rows,
        cols=cols,
        mask=mask_arr,
        gene_names=None if gene_names is None else np.asarray(gene_names, dtype=str),
    )


def _normalize_coords(name: str, coords: np.ndarray | list[int]) -> np.ndarray:
    arr = np.asarray(coords)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array-like object.")
    if arr.size == 0:
        return arr.astype(np.int64)
    if not np.issubdtype(arr.dtype, np.integer):
        rounded = np.rint(arr)
        if not np.allclose(arr, rounded):
            raise ValueError(f"{name} must contain integer coordinates.")
        arr = rounded
    return arr.astype(np.int64, copy=False)


def _infer_extent(
    name: str,
    value: int | None,
    coords: np.ndarray,
    mask: np.ndarray | None,
    *,
    axis: int,
) -> int:
    if value is not None:
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    if mask is not None:
        return int(mask.shape[axis])

    if coords.size == 0:
        raise ValueError(f"{name} must be provided when coordinates are empty and no mask is given.")

    return int(coords.max()) + 1
