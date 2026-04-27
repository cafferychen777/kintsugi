from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import kintsugi

h5py = pytest.importorskip("h5py")
pandas = pytest.importorskip("pandas")


def test_build_regular_grid_places_counts_in_row_major_order() -> None:
    counts = sp.csr_matrix(np.array([[5, 0], [0, 7]], dtype=float))
    row_coords = np.array([0, 1])
    col_coords = np.array([1, 0])
    mask = np.array([[False, True], [True, True]], dtype=bool)

    grid = kintsugi.build_regular_grid(
        counts,
        row_coords,
        col_coords,
        mask=mask,
    )

    assert isinstance(grid, kintsugi.GridData)
    assert grid.shape == (2, 2)
    assert np.array_equal(grid.mask, mask)
    assert np.array_equal(
        grid.counts.toarray(),
        np.array(
            [
                [0.0, 0.0],
                [5.0, 0.0],
                [0.0, 7.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_build_regular_grid_rejects_duplicate_coordinates() -> None:
    counts = sp.csr_matrix(np.array([[1.0], [2.0]]))
    row_coords = np.array([0, 0])
    col_coords = np.array([1, 1])

    with pytest.raises(ValueError, match="must be unique"):
        kintsugi.build_regular_grid(counts, row_coords, col_coords, rows=1, cols=2)


def test_build_regular_grid_rejects_occupied_coordinates_outside_mask() -> None:
    counts = sp.csr_matrix(np.array([[1.0]], dtype=float))
    mask = np.array([[True, False]], dtype=bool)

    with pytest.raises(ValueError, match="inside the provided mask"):
        kintsugi.build_regular_grid(
            counts,
            row_coords=[0],
            col_coords=[1],
            mask=mask,
        )


def test_parse_visium_barcode_coordinates_reads_trailing_indices() -> None:
    rows, cols = kintsugi.parse_visium_barcode_coordinates(
        ["s_008um_00012_00034-1", "prefix_anything_7_9-1"]
    )

    assert np.array_equal(rows, [12, 7])
    assert np.array_equal(cols, [34, 9])


def test_load_visium_hd_reads_h5_and_csv(tmp_path: Path) -> None:
    counts = sp.csr_matrix(
        np.array(
            [
                [2, 0],
                [0, 3],
                [4, 1],
            ],
            dtype=np.int32,
        )
    )
    barcodes = [
        "s_008um_00000_00001-1",
        "s_008um_00001_00000-1",
        "s_008um_00001_00002-1",
    ]
    gene_names = ["GeneA", "GeneB"]

    matrix_path = tmp_path / "filtered_feature_bc_matrix.h5"
    tissue_positions_path = tmp_path / "tissue_positions.csv"

    _write_10x_h5(matrix_path, counts, barcodes, gene_names)
    pandas.DataFrame(
        {
            "array_row": [0, 0, 1, 1],
            "array_col": [0, 1, 0, 2],
            "in_tissue": [False, True, True, True],
        }
    ).to_csv(tissue_positions_path, index=False)

    grid = kintsugi.load_visium_hd(matrix_path, tissue_positions_path)

    assert isinstance(grid, kintsugi.GridData)
    assert grid.shape == (2, 3)
    assert np.array_equal(grid.gene_names, np.array(gene_names))
    assert np.array_equal(
        grid.mask,
        np.array(
            [
                [False, True, False],
                [True, False, True],
            ],
            dtype=bool,
        ),
    )
    assert np.array_equal(
        grid.counts.toarray(),
        np.array(
            [
                [0, 0],
                [2, 0],
                [0, 0],
                [0, 3],
                [0, 0],
                [4, 1],
            ]
        ),
    )


def test_load_visium_hd_from_dir_finds_common_paths(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    spatial_dir = sample_dir / "spatial"
    spatial_dir.mkdir(parents=True)

    counts = sp.csr_matrix(np.array([[1, 0]], dtype=np.int32))
    barcodes = ["s_008um_00000_00000-1"]
    gene_names = ["GeneA", "GeneB"]

    _write_10x_h5(sample_dir / "filtered_feature_bc_matrix.h5", counts, barcodes, gene_names)
    pandas.DataFrame(
        {
            "array_row": [0],
            "array_col": [0],
            "in_tissue": [True],
        }
    ).to_csv(spatial_dir / "tissue_positions.csv", index=False)

    grid = kintsugi.load_visium_hd_from_dir(sample_dir)

    assert grid.shape == (1, 1)
    assert np.array_equal(grid.counts.toarray(), np.array([[1, 0]]))


def test_load_visium_hd_from_dir_falls_back_to_root_level_tissue_positions(
    tmp_path: Path,
) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()

    counts = sp.csr_matrix(np.array([[1, 2]], dtype=np.int32))
    barcodes = ["s_008um_00000_00001-1"]
    gene_names = ["GeneA", "GeneB"]

    _write_10x_h5(sample_dir / "filtered_feature_bc_matrix.h5", counts, barcodes, gene_names)
    pandas.DataFrame(
        {
            "array_row": [0, 0],
            "array_col": [0, 1],
            "in_tissue": [False, True],
        }
    ).to_csv(sample_dir / "tissue_positions.csv", index=False)

    grid = kintsugi.load_visium_hd_from_dir(sample_dir)

    assert grid.shape == (1, 2)
    assert np.array_equal(grid.mask, np.array([[False, True]], dtype=bool))
    assert np.array_equal(grid.counts.toarray(), np.array([[0, 0], [1, 2]]))


def test_read_tissue_positions_accepts_legacy_10x_csv_layout(tmp_path: Path) -> None:
    tissue_positions_path = tmp_path / "tissue_positions_list.csv"
    tissue_positions_path.write_text(
        "\n".join(
            [
                "barcode0,1,2,3,100,200",
                "barcode1,0,4,5,300,400",
            ]
        )
    )

    frame = kintsugi.read_tissue_positions(tissue_positions_path)

    assert list(frame.columns) == [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]
    assert frame["array_row"].tolist() == [2, 4]
    assert frame["array_col"].tolist() == [3, 5]
    assert frame["in_tissue"].tolist() == [1, 0]


def _write_10x_h5(
    path: Path,
    counts: sp.csr_matrix,
    barcodes: list[str],
    gene_names: list[str],
) -> None:
    csc = counts.T.tocsc()

    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("shape", data=np.array(csc.shape, dtype=np.int64))
        matrix.create_dataset("data", data=csc.data)
        matrix.create_dataset("indices", data=csc.indices)
        matrix.create_dataset("indptr", data=csc.indptr)
        matrix.create_dataset("barcodes", data=np.array(barcodes, dtype="S"))
        features = matrix.create_group("features")
        features.create_dataset("name", data=np.array(gene_names, dtype="S"))
