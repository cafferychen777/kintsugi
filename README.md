# Kintsugi

> Named after the Japanese art of repairing broken ceramics with gold — honouring
> boundaries rather than erasing them.

Kintsugi builds adaptive spatial regions from Visium HD and other regular-grid
spatial transcriptomics data.  It starts from a 2D bin lattice and returns
larger spatial regions whose boundaries follow local changes in molecule
density.  The package sits between raw binned counts and downstream biological
analysis: it produces region labels, region-level Pearson residuals, region
sizes, depths, centroids, and a spatial adjacency graph.

Kintsugi does not do clustering, marker testing, plotting, or
manuscript-specific analysis.  Those choices stay downstream.

## Installation

### From PyPI (available upon publication)

```bash
pip install kintsugi-st
```

### From GitHub

```bash
pip install git+https://github.com/cafferychen777/kintsugi.git
```

### With AnnData support

```bash
pip install "kintsugi-st[anndata]"
```

### Conda / Mamba (from source)

```bash
git clone https://github.com/cafferychen777/kintsugi.git
cd kintsugi
mamba env create -f environment.yml
mamba activate kintsugi
```

### Docker

```bash
docker build -t kintsugi .
docker run --rm kintsugi                               # run demo
docker run --rm -it kintsugi python               # interactive
docker run --rm -v $(pwd)/data:/data kintsugi \
    python -c "import kintsugi; ..."              # mount data
```

### Singularity / Apptainer (HPC)

```bash
singularity build kintsugi.sif Singularity.def
singularity exec kintsugi.sif python -c "import kintsugi"

# or, on systems that provide Apptainer:
apptainer build kintsugi.sif Singularity.def
apptainer exec kintsugi.sif python -c "import kintsugi"
```

### System requirements

- **OS**: Linux, macOS, Windows (any platform with Python support).
- **Python**: 3.10, 3.11, or 3.12.
- **Dependencies**: NumPy (>=1.24), SciPy (>=1.11), h5py (>=3.10),
  pandas (>=2.0), PyArrow (>=14.0).  Kintsugi itself is pure Python and
  ships no compiled extension modules.
- **Install time**: < 30 seconds on a standard machine with pip.
- **Hardware**: No GPU required.  16 GB RAM is sufficient for most Visium HD
  samples; 64 GB recommended for very large grids (> 500k bins).

## Quick start (30 seconds)

```python
import kintsugi

grid = kintsugi.load_visium_hd_from_dir("sample_dir")
result = grid.tessellate()

print(kintsugi.tessellation_report(result, grid))
```

That's it.  Three lines: load, tessellate, report.

## One-command demo

Run the bundled demo on a synthetic dataset (no data download needed):

```bash
kintsugi-demo           # CLI entry point (after pip install)
python -m kintsugi.demo # module invocation
```

Expected output:

```
Kintsugi v0.1.0
=======================================================

1. Generating toy dataset...
   Grid: 60×60, 100 genes, 2,472 tissue bins
   Time: 0.02s

2. Running tessellation...
   Regions: 75
   Time: 0.12s

3. Diagnostic report:
── Kintsugi Tessellation Report ──────────────────────
  Regions              : 75
  Median UMI / region  : 167.0
  Density CV           : 0.3488
  Stationarity pass    : 84.0%
  Composition holdout  : -30.3192 nats/bin
  Density holdout      : -2.4633 nats/bin
  Composition ΔLL      : +3.3670 nats/bin
  Density ΔLL          : +0.4569 nats/bin
  Composition dominates: False
─────────────────────────────────────────────────────

4. Determinism check:
   Label hash (sha256[:16]): 3653373eec97b137

Total runtime: 0.16s
Done.
```

The label hash `3653373eec97b137` verifies deterministic output across
platforms.

## 10x Space Ranger to Kintsugi tutorial

If your data comes from 10x Genomics Space Ranger, the directory typically
looks like:

```
sample/
├── filtered_feature_bc_matrix.h5
└── spatial/
    └── tissue_positions.parquet   (or .csv)
```

### Step 1: Load

```python
import kintsugi

grid = kintsugi.load_visium_hd_from_dir("sample/")
print(f"Grid: {grid.rows}×{grid.cols}, {grid.n_genes} genes, {grid.mask.sum()} tissue bins")
```

If your files are in non-standard locations:

```python
grid = kintsugi.load_visium_hd(
    "path/to/filtered_feature_bc_matrix.h5",
    "path/to/tissue_positions.parquet",
)
```

### Step 2: Tessellate

```python
result = grid.tessellate()
```

With custom parameters:

```python
result = grid.tessellate(
    lag=2,                # variogram lag in bins (2 bins = 4 µm at 2 µm resolution)
    kappa=2.0,            # stationarity tolerance
    min_seed_distance=4,  # minimum seed separation in bins
    smooth_sigma=4.0,     # Gaussian smoothing for seed detection
)
```

### Step 3: Inspect the result

```python
result.labels      # (rows, cols) int32 — region labels, -1 outside tissue
result.residuals   # (K, G)  float64 — Pearson residuals
result.areas       # (K,)    float64 — bins per region
result.depths      # (K,)    float64 — total UMI per region
result.centroids   # (K, 2)  float64 — (row, col) centroids
result.adjacency   # (K, K)  sparse   — spatial adjacency graph
result.trace       # (rows, cols) float64 — boundary-tensor trace
result.n_regions   # int — number of regions
```

### Step 4: Diagnostic report

```python
report = kintsugi.tessellation_report(result, grid)
print(report)
```

The report automatically computes:

| Metric | What it measures |
| --- | --- |
| Region count | Number of tessellated regions |
| Median UMI/region | Depth distribution |
| Density CV | Coefficient of variation of per-bin UMI density |
| Stationarity pass rate | Fraction passing Poisson stationarity test |
| Composition holdout LL | Held-out log-likelihood for multinomial composition |
| Density holdout LL | Held-out log-likelihood for Poisson density |
| Composition/Density ΔLL | Improvement over single-global-model null |
| Composition dominates | Warning if boundaries are driven by composition, not density |

### Step 5: Export to AnnData (scverse integration)

```python
adata = kintsugi.to_anndata(result, grid=grid, use_raw_counts=True)

# adata.X         = Pearson residuals
# adata.obs       = area, depth
# adata.obsm      = spatial centroids
# adata.obsp      = adjacency graph
# adata.layers    = raw aggregated counts

adata.write("tessellation.h5ad")
```

The AnnData object is directly usable with Scanpy, Squidpy, and other scverse
tools for clustering, visualisation, and spatial analysis.

## Parameters

All parameters have fixed defaults and are documented.  There is no hidden
tuning.

| Parameter | Default | How to think about it |
| --- | --- | --- |
| `lag` | `2` | Grid offset for directional semivariance. On a 2 µm Visium HD grid, `lag=2` is a 4 µm offset. Larger values look at broader spatial variation. |
| `kappa` | `2.0` | Stationarity tolerance during region refinement (in SE units). Larger values allow broader regions. |
| `min_seed_distance` | `4` | Minimum distance between seed points in grid bins. Controls the spatial scale: larger = fewer, larger regions. |
| `smooth_sigma` | `4.0` | Gaussian sigma for smoothing the trace field before seed detection. Larger values favor smoother boundaries. |

For very small toy examples, use smaller `min_seed_distance` and
`smooth_sigma` values (the defaults target real Visium HD grids).

## Performance

Measured on an Apple M1 Max (single thread, pure Python) with
`tracemalloc` peak-memory tracking.

### Synthetic benchmarks

| Dataset | Grid | Genes | Tissue bins | Regions | Time | Peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| Toy | 60 × 60 | 100 | 2,644 | 46 | 0.01 s | 0.5 MB |
| Small | 100 × 100 | 200 | 7,556 | 129 | 0.02 s | 1.7 MB |
| Medium | 200 × 200 | 500 | 30,792 | 508 | 0.08 s | 8 MB |
| Large | 500 × 500 | 1,000 | 194,824 | 2,936 | 0.43 s | 72 MB |
| XL | 800 × 800 | 2,000 | 500,200 | 7,462 | 1.15 s | 298 MB |

### Real Visium HD (8 µm resolution)

| Tissue | Tissue bins | Regions | Time |
| --- | --- | --- | --- |
| Human CRC (P2_CRC) | 545,913 | 26,107 | 10–21 s |
| Mouse brain | 393,543 | 18,673 | 6–11 s |
| Mouse intestine | 351,817 | 17,854 | 7 s |

Memory scales primarily with `n_regions × n_genes` (the dense residual
matrix).  Upstream gene filtering is the main memory lever for large datasets.
Benchmarks are reproducible via `scripts/benchmark_runtime_memory.py`.

## Input contract

Kintsugi operates on a normalized grid:

- `counts`: SciPy sparse matrix with shape `(rows * cols, genes)`.
- `rows`, `cols`: dimensions of the 2D grid.
- `mask`: optional boolean array with shape `(rows, cols)`; `True` marks
  in-tissue bins.
- Matrix rows are in row-major order: row `r * cols + c` corresponds to grid
  bin `(r, c)`.
- Count values must be finite and non-negative.

`GridData` is the package container for this contract.

## Non-Visium inputs

For any regular-grid data where you have occupied-bin counts and coordinates:

```python
grid = kintsugi.build_regular_grid(
    counts,         # sparse (n_occupied, genes)
    row_coords,     # 1D array of row indices
    col_coords,     # 1D array of column indices
    rows=R, cols=C, # grid extent
)
result = grid.tessellate()
```

## API overview

Most users need only:

- `kintsugi.load_visium_hd_from_dir(...)` — load from Space Ranger output
- `kintsugi.tessellate(...)` — run the full pipeline
- `kintsugi.tessellation_report(...)` — diagnostic report
- `kintsugi.to_anndata(...)` — export to AnnData

Lower-level functions for advanced users:

- `directional_semivariance` — variogram estimation
- `boundary_tensor` — tensor eigendecomposition
- `adaptive_tessellation` — watershed + stationarity refinement
- `aggregate_counts` — region-level Pearson residuals
- `build_spatial_graph` — adjacency from labels

## Reproducibility

- **Deterministic**: no random seeds, no stochastic algorithms.  The same
  input always produces the same output (verified by SHA-256 hash in tests).
- **No hidden tuning**: all parameters are explicit and documented.
- **Tested package surface**: unit tests cover core algorithms, I/O, report,
  AnnData export, the demo dataset, and edge-case validation.
- **Coverage reporting**: use the pytest-cov command below to reproduce the
  module-level coverage table.
- **CI on Python 3.10, 3.11, 3.12** via GitHub Actions.
- **Docker and Singularity** images for containerised reproduction.

## Testing

```bash
python -m pip install -e ".[dev]"
python -m ruff check kintsugi tests examples
python -m pytest --cov=kintsugi --cov-report=term-missing
```

## License

Kintsugi is released under the [MIT License](LICENSE).

## Citation

If you use Kintsugi in your research, please cite:

> [Citation will be added upon publication]
