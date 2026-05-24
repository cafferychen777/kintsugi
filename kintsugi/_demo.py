"""Built-in demo: toy data generation and one-command pipeline.

This module is shipped inside the wheel so that ``pip install`` users
can run the demo without cloning the repository::

    kintsugi-demo            # CLI entry point
    python -m kintsugi.demo  # module invocation
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Toy dataset (self-contained, no external dependency)
# ---------------------------------------------------------------------------

def make_toy_dataset(
    rows: int = 60,
    cols: int = 60,
    n_genes: int = 100,
    seed: int = 2024,
):
    """Create a synthetic spatial transcriptomics grid.

    Returns a ``kintsugi.GridData`` with four spatially distinct domains:

    - **Top-left** (domain 0): high density, marker genes in first quarter.
    - **Top-right** (domain 1): medium density, marker genes in second quarter.
    - **Bottom-left** (domain 2): medium density, marker genes in third quarter.
    - **Bottom-right** (domain 3): low density, marker genes in last quarter.

    A circular tissue mask excludes corner bins.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions (default 60×60 = 3,600 bins).
    n_genes : int
        Number of genes (default 100).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    grid : kintsugi.GridData
        Normalized grid ready for ``grid.tessellate()``.
    """
    from . import GridData

    rng = np.random.default_rng(seed)

    rr, cc = np.mgrid[:rows, :cols]
    centre_r, centre_c = rows / 2 - 0.5, cols / 2 - 0.5
    dist = np.sqrt((rr - centre_r) ** 2 + (cc - centre_c) ** 2)
    mask = dist <= 28.0

    domain = np.full((rows, cols), -1, dtype=np.int32)
    domain[(rr < rows // 2) & (cc < cols // 2)] = 0
    domain[(rr < rows // 2) & (cc >= cols // 2)] = 1
    domain[(rr >= rows // 2) & (cc < cols // 2)] = 2
    domain[(rr >= rows // 2) & (cc >= cols // 2)] = 3

    base_rates = {0: 12.0, 1: 7.0, 2: 7.0, 3: 4.0}

    block = n_genes // 4
    gene_profiles = np.ones((4, n_genes)) * 0.1
    gene_profiles[0, 0:block] = 2.0
    gene_profiles[1, block:2 * block] = 2.0
    gene_profiles[2, 2 * block:3 * block] = 2.0
    gene_profiles[3, 3 * block:n_genes] = 2.0
    gene_profiles = gene_profiles / gene_profiles.sum(axis=1, keepdims=True)

    N = rows * cols
    data_rows = []
    data_cols = []
    data_vals = []

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            d = domain[r, c]
            if d < 0:
                continue
            total_umi = rng.poisson(base_rates[d])
            if total_umi == 0:
                continue
            gene_counts = rng.multinomial(total_umi, gene_profiles[d])
            nz_idx = np.where(gene_counts > 0)[0]
            flat_idx = r * cols + c
            for g in nz_idx:
                data_rows.append(flat_idx)
                data_cols.append(g)
                data_vals.append(float(gene_counts[g]))

    counts = sp.coo_matrix(
        (data_vals, (data_rows, data_cols)),
        shape=(N, n_genes),
    ).tocsr()

    gene_names = np.array([f"Gene_{g:03d}" for g in range(n_genes)])

    return GridData(
        counts, rows=rows, cols=cols, mask=mask, gene_names=gene_names,
    )


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full Kintsugi demo pipeline."""
    from . import __version__, tessellation_report, to_anndata

    print(f"Kintsugi v{__version__}")
    print("=" * 55)

    print("\n1. Generating toy dataset...")
    t0 = time.perf_counter()
    grid = make_toy_dataset()
    t_data = time.perf_counter() - t0
    print(f"   Grid: {grid.rows}\u00d7{grid.cols}, {grid.n_genes} genes, "
          f"{int(grid.mask.sum()):,} tissue bins")
    print(f"   Time: {t_data:.2f}s")

    print("\n2. Running tessellation...")
    t0 = time.perf_counter()
    result = grid.tessellate()
    t_tess = time.perf_counter() - t0
    print(f"   Regions: {result.n_regions}")
    print(f"   Time: {t_tess:.2f}s")

    print("\n3. Diagnostic report:")
    t0 = time.perf_counter()
    report = tessellation_report(result, grid)
    t_report = time.perf_counter() - t0
    print(report)
    print(f"   Report time: {t_report:.2f}s")

    label_hash = hashlib.sha256(result.labels.tobytes()).hexdigest()[:16]
    print("\n4. Determinism check:")
    print(f"   Label hash (sha256[:16]): {label_hash}")

    print("\n5. AnnData export:")
    try:
        adata = to_anndata(result, grid=grid, use_raw_counts=True)
        print(f"   Shape: {adata.shape}")
        print(f"   obs:   {list(adata.obs.columns)}")
        print(f"   obsm:  {list(adata.obsm.keys())}")
        print(f"   obsp:  {list(adata.obsp.keys())}")
        print(f"   layers: {list(adata.layers.keys())}")
    except ImportError:
        print("   Skipped (anndata not installed).")
        print("   Install with: pip install 'kintsugi-st[anndata]'")

    total = t_data + t_tess + t_report
    print(f"\nTotal runtime: {total:.2f}s")
    print("Done.")


if __name__ == "__main__":
    main()
