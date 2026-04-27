# Kintsugi

`kintsugi` builds adaptive spatial regions from 10x Genomics Visium HD and other
regular-grid spatial transcriptomics data.

It starts from a 2D bin lattice and returns larger spatial regions whose
boundaries follow local changes in molecule density. The package is meant for the
step between raw binned counts and downstream biological analysis: it produces
region labels, region-level residuals, region sizes, depths, centroids, and a
spatial adjacency graph.

Kintsugi does not do clustering, marker testing, plotting, or manuscript-specific
analysis. Those choices stay downstream of the tessellation.

## Installation

Install Kintsugi with pip:

```bash
pip install kintsugi
```

This includes the dependencies needed to load standard Visium HD feature matrices
and tissue-position files.

From a local checkout:

```bash
git clone https://github.com/cafferychen777/kintsugi.git
cd kintsugi
pip install -e .
```

Conda and mamba users can create an environment from source:

```bash
mamba env create -f environment.yml
mamba activate kintsugi
```

Use `conda env create -f environment.yml` instead if you prefer conda.

## Quick start: Visium HD

If your sample directory follows the standard 10x layout, use the directory
loader:

```python
import kintsugi

grid = kintsugi.load_visium_hd_from_dir("sample_dir")
result = grid.tessellate()

labels = result.labels
residuals = result.residuals
adjacency = result.adjacency
```

By default, `load_visium_hd_from_dir(...)` looks for
`filtered_feature_bc_matrix.h5`, then auto-detects
`spatial/tissue_positions.parquet` or `spatial/tissue_positions.csv`.

If your files are in custom locations, pass them explicitly:

```python
import kintsugi

grid = kintsugi.load_visium_hd(
    "filtered_feature_bc_matrix.h5",
    "spatial/tissue_positions.parquet",
)

result = kintsugi.tessellate(grid)
```

## If you already have a grid

Kintsugi can also work with any regular-grid spatial transcriptomics data. The
count matrix must have one row per grid bin in row-major order: row
`r * cols + c` is grid location `(r, c)`.

```python
import numpy as np
import scipy.sparse as sp

import kintsugi

counts = sp.csr_matrix(
    [
        [3, 0, 1],
        [2, 0, 0],
        [0, 4, 1],
        [0, 3, 0],
    ],
    dtype=float,
)

result = kintsugi.tessellate(
    counts,
    rows=2,
    cols=2,
    mask=np.ones((2, 2), dtype=bool),
    min_seed_distance=1,
    smooth_sigma=1.0,
)
```

Use `build_regular_grid(...)` when your matrix only contains occupied bins and
you have their row/column coordinates:

```python
import numpy as np
import scipy.sparse as sp

import kintsugi

occupied_counts = sp.csr_matrix(
    [
        [3, 0, 1],
        [2, 0, 0],
        [0, 4, 1],
        [0, 3, 0],
    ],
    dtype=float,
)

row_coords = np.array([0, 0, 1, 1])
col_coords = np.array([0, 1, 0, 1])

grid = kintsugi.build_regular_grid(
    occupied_counts,
    row_coords,
    col_coords,
    rows=2,
    cols=2,
)

result = kintsugi.tessellate(grid, min_seed_distance=1, smooth_sigma=1.0)
```

## Input contract

Kintsugi operates on a normalized grid:

- `counts`: SciPy sparse matrix with shape `(rows * cols, genes)`.
- `rows`, `cols`: dimensions of the 2D grid.
- `mask`: optional boolean array with shape `(rows, cols)`; `True` marks bins to
  tessellate.
- Matrix rows must be in row-major order: row `r * cols + c` corresponds to grid
  bin `(r, c)`.
- Count values must be finite and non-negative.

`GridData` is the package container for this contract. You can pass a `GridData`
object to `kintsugi.tessellate(grid)` or call `grid.tessellate()` directly.

## Output

`kintsugi.tessellate(...)` returns a `TessellationResult` with:

- `labels`: integer region labels on the original grid; `-1` outside `mask`.
- `residuals`: dense region-by-gene Pearson residual matrix.
- `areas`: number of native bins in each region.
- `depths`: total UMI count in each region.
- `centroids`: region centroids in `(row, col)` coordinates.
- `adjacency`: sparse binary graph connecting neighboring regions.
- `trace`: boundary-tensor trace on the original grid.

The result supports both attribute access and dictionary-style access:

```python
result.labels
result["labels"]
```

## Main parameters

The defaults are intended as conservative starting points for Visium HD-style
regular grids. They are deterministic, but not universal; tune them for grid
resolution, expected domain size, and the spatial scale of the structure you want
to preserve.

| Parameter | Default | Used by | How to think about it |
| --- | --- | --- | --- |
| `lag` | `2` | `tessellate` | Grid offset for local directional semivariance. On a 2 µm Visium HD grid, `lag=2` is a 4 µm offset. Larger values look at broader spatial variation. |
| `kappa` | `2.0` | `tessellate` | Stationarity tolerance during region refinement. Larger values usually allow broader regions. |
| `min_seed_distance` | `4` | `tessellate` | Minimum distance between seed points in grid bins. Larger values reduce over-fragmentation. |
| `smooth_sigma` | `4.0` | `tessellate` | Gaussian smoothing scale, in grid bins, applied before seed detection. Larger values favor smoother, larger-scale boundaries. |
| `rows`, `cols` | `None` | `build_regular_grid` | If omitted, grid dimensions are inferred from occupied coordinates. |
| `mask` | `None` | `build_regular_grid`, `tessellate` | If omitted in `build_regular_grid`, occupied bins define the mask. If omitted in `tessellate`, all bins are considered in tissue. |
| `matrix_filename` | `"filtered_feature_bc_matrix.h5"` | `load_visium_hd_from_dir` | Default 10x Genomics feature matrix filename. |
| `tissue_positions_filename` | `None` | `load_visium_hd_from_dir` | Auto-detect tissue-position parquet or CSV file when not specified. |

For very small toy examples, you may need smaller `min_seed_distance` and
`smooth_sigma` values than the defaults because the default values are designed
for real spatial grids rather than 2-by-2 examples.

## API overview

Most users only need:

- `kintsugi.load_visium_hd_from_dir(...)`
- `kintsugi.load_visium_hd(...)`
- `kintsugi.tessellate(...)`
- `kintsugi.GridData`
- `kintsugi.TessellationResult`

For non-Visium inputs, use `kintsugi.build_regular_grid(...)` to construct the
same grid contract from occupied-bin coordinates.

Lower-level functions are exposed for advanced users who want to inspect or
customize the pipeline:

- `directional_semivariance`
- `boundary_tensor`
- `adaptive_tessellation`
- `aggregate_counts`
- `build_spatial_graph`

## Performance notes

- Keep `counts` sparse before calling Kintsugi.
- Non-CSR sparse inputs are converted to CSR once.
- `residuals` is dense with shape `(regions, genes)`, so upstream gene filtering
  is the main memory lever for large datasets.
- `adjacency` is stored as a sparse graph.

## License

Kintsugi is released under the MIT License. See [LICENSE](LICENSE).
