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

    tissue_positions_path = _resolve_tissue_positions_path(
        directory,
        tissue_positions_filename,
    )
    return load_visium_hd(matrix_path, tissue_positions_path)


def _resolve_tissue_positions_path(
    directory: Path,
    filename: str | None,
) -> Path:
    if filename is not None:
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(f"tissue_positions file not found: {path}")
        return path

    candidates = [
        directory / "spatial" / "tissue_positions.parquet",
        directory / "spatial" / "tissue_positions.csv",
        directory / "spatial" / "tissue_positions_list.csv",
        directory / "tissue_positions.parquet",
        directory / "tissue_positions.csv",
        directory / "tissue_positions_list.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "could not find a tissue_positions file; checked: "
        + ", ".join(str(path) for path in candidates)
    )


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
        barcodes = _decode_h5_strings(handle["matrix/barcodes"][()])
        gene_names = _decode_h5_strings(handle["matrix/features/name"][()])

    csc = sp.csc_matrix((data, indices, indptr), shape=(int(shape[0]), int(shape[1])))
    return csc.T.tocsr(), barcodes, gene_names


def _decode_h5_strings(values) -> np.ndarray:
    return np.array([
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values
    ])


def read_tissue_positions(path: str | Path):
    """Read a Visium-style tissue positions table from parquet or CSV."""
    pandas = _require_optional_dependency("pandas")

    path = Path(path)
    if path.suffix == ".parquet":
        frame = pandas.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pandas.read_csv(path, header=0 if _csv_has_header(path) else None)
        if not {"array_row", "array_col", "in_tissue"}.issubset(frame.columns):
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


def _csv_has_header(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    columns = {column.strip() for column in first_line.split(",")}
    return {"array_row", "array_col", "in_tissue"}.issubset(columns)


def _require_optional_dependency(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{module_name} is required for Visium HD loading. Reinstall with "
            "`pip install kintsugi`."
        ) from exc
