"""Visium HD input adapters for Kintsugi."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..models import GridData
from .grid import build_regular_grid, parse_visium_barcode_coordinates


def load_visium_hd(
    matrix_h5_path: str | Path,
    tissue_positions_path: str | Path,
) -> GridData:
    """Load a Visium HD sample into the normalized Kintsugi grid contract."""
    counts, barcodes, gene_names = load_10x_feature_matrix(matrix_h5_path)
    row_coords, col_coords = parse_visium_barcode_coordinates(barcodes)

    tissue_positions = read_tissue_positions(tissue_positions_path)
    all_rows = tissue_positions["array_row"].to_numpy(dtype=np.int64)
    all_cols = tissue_positions["array_col"].to_numpy(dtype=np.int64)
    in_tissue = tissue_positions["in_tissue"].to_numpy(dtype=bool)

    rows = int(all_rows.max()) + 1
    cols = int(all_cols.max()) + 1
    mask = np.zeros((rows, cols), dtype=bool)
    mask[all_rows[in_tissue], all_cols[in_tissue]] = True

    return build_regular_grid(
        counts,
        row_coords,
        col_coords,
        rows=rows,
        cols=cols,
        mask=mask,
        gene_names=gene_names,
    )


def load_visium_hd_from_dir(
    directory: str | Path,
    *,
    matrix_filename: str = "filtered_feature_bc_matrix.h5",
    tissue_positions_filename: str | None = None,
) -> GridData:
    """Load a Visium HD sample from a directory with common 10x-style paths."""
    directory = Path(directory)
    matrix_path = directory / matrix_filename
    if not matrix_path.exists():
        raise FileNotFoundError(f"matrix file not found: {matrix_path}")

    if tissue_positions_filename is not None:
        tissue_positions_path = directory / tissue_positions_filename
        if not tissue_positions_path.exists():
            raise FileNotFoundError(f"tissue_positions file not found: {tissue_positions_path}")
    else:
        candidates = [
            directory / "spatial" / "tissue_positions.parquet",
            directory / "spatial" / "tissue_positions.csv",
            directory / "spatial" / "tissue_positions_list.csv",
            directory / "tissue_positions.parquet",
            directory / "tissue_positions.csv",
            directory / "tissue_positions_list.csv",
        ]
        matches = [path for path in candidates if path.exists()]
        if not matches:
            raise FileNotFoundError(
                "could not find a tissue_positions file; checked: "
                + ", ".join(str(path) for path in candidates)
            )
        tissue_positions_path = matches[0]

    return load_visium_hd(matrix_path, tissue_positions_path)


def load_10x_feature_matrix(
    path: str | Path,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Load a 10x feature matrix as `(barcodes x genes)` CSR plus metadata."""
    h5py = _require_optional_dependency("h5py")

    with h5py.File(path, "r") as handle:
        shape = handle["matrix/shape"][()]
        data = handle["matrix/data"][()]
        indices = handle["matrix/indices"][()]
        indptr = handle["matrix/indptr"][()]
        barcodes = np.array([value.decode() for value in handle["matrix/barcodes"][()]])
        gene_names = np.array([value.decode() for value in handle["matrix/features/name"][()]])

    csc = sp.csc_matrix((data, indices, indptr), shape=(int(shape[0]), int(shape[1])))
    return csc.T.tocsr(), barcodes, gene_names


def read_tissue_positions(path: str | Path):
    """Read a Visium-style tissue positions table from parquet or CSV."""
    pandas = _require_optional_dependency("pandas")

    path = Path(path)
    if path.suffix == ".parquet":
        frame = pandas.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pandas.read_csv(path)
        if not {"array_row", "array_col", "in_tissue"}.issubset(frame.columns):
            frame = pandas.read_csv(path, header=None)
            if frame.shape[1] < 4:
                raise ValueError(
                    "CSV tissue_positions files must either include array_row/array_col/"
                    "in_tissue columns or use the 10x legacy 6-column layout."
                )
            frame.columns = [
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_in_fullres",
                "pxl_col_in_fullres",
            ][: frame.shape[1]]
    else:
        raise ValueError("tissue_positions_path must end in .parquet or .csv.")

    required = {"array_row", "array_col", "in_tissue"}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"tissue positions table is missing required columns: {missing}")

    return frame


def _require_optional_dependency(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{module_name} is required for Visium HD loading. Reinstall with "
            "`pip install kintsugi`."
        ) from exc
