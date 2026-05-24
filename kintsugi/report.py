"""Automated tessellation diagnostics for quality assessment.

Produces a structured report summarising the statistical properties of a
Kintsugi tessellation: region count, depth distribution, density
homogeneity, Poisson stationarity pass rate, and held-out log-likelihood
for both density and composition models.  A warning is issued when the
composition signal appears to dominate over density, suggesting that the
tessellation boundaries may be driven primarily by gene-expression
differences rather than molecule-density structure.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

from .models import GridData, _validate_positive_int
from .partition import _poisson_stationary


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TessellationReport:
    """Diagnostic summary of a Kintsugi tessellation.

    Attributes
    ----------
    n_regions : int
        Number of tessellated regions.
    median_umi_per_region : float
        Median total UMI across regions.
    density_cv : float
        Coefficient of variation of mean UMI density across regions
        (std / mean of ``depths / areas``, where each region contributes
        one value).
    stationarity_pass_rate : float
        Fraction of regions that pass the Poisson stationarity test at
        the given *kappa* threshold.
    composition_holdout_ll : float
        Mean per-bin held-out log-likelihood under the within-region
        multinomial composition model (nats).
    density_holdout_ll : float
        Mean per-bin held-out log-likelihood under the within-region
        Poisson density model (nats).
    composition_ll_ratio : float
        Improvement in per-bin composition log-likelihood over a single
        global composition (nats).  Higher = stronger composition signal.
    density_ll_ratio : float
        Improvement in per-bin density log-likelihood over a single
        global rate (nats).  Higher = stronger density signal.
    composition_dominates_warning : bool
        True when the composition signal appears to dominate density,
        suggesting that boundaries may not reflect density structure.
    """

    n_regions: int
    median_umi_per_region: float
    density_cv: float
    stationarity_pass_rate: float
    composition_holdout_ll: float
    density_holdout_ll: float
    composition_ll_ratio: float
    density_ll_ratio: float
    composition_dominates_warning: bool

    def __str__(self) -> str:
        warn_tag = " [WARNING]" if self.composition_dominates_warning else ""
        return textwrap.dedent(f"""\
            ── Kintsugi Tessellation Report ──────────────────────
              Regions              : {self.n_regions}
              Median UMI / region  : {self.median_umi_per_region:,.1f}
              Density CV           : {self.density_cv:.4f}
              Stationarity pass    : {self.stationarity_pass_rate:.1%}
              Composition holdout  : {self.composition_holdout_ll:+.4f} nats/bin
              Density holdout      : {self.density_holdout_ll:+.4f} nats/bin
              Composition ΔLL      : {self.composition_ll_ratio:+.4f} nats/bin
              Density ΔLL          : {self.density_ll_ratio:+.4f} nats/bin
              Composition dominates: {self.composition_dominates_warning}{warn_tag}
            ─────────────────────────────────────────────────────""")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def tessellation_report(
    result,
    counts: sp.csr_matrix,
    rows: int | None = None,
    cols: int | None = None,
    mask: np.ndarray | None = None,
    *,
    kappa: float = 2.0,
    holdout_fraction: float = 0.5,
    seed: int = 0,
) -> TessellationReport:
    """Generate a diagnostic report for a tessellation result.

    Parameters
    ----------
    result : TessellationResult
        Output of ``kintsugi.tessellate``.
    counts : GridData or csr_matrix, shape (N, G)
        The original count matrix used for tessellation.
    rows, cols : int, optional
        Grid dimensions.  Required when *counts* is not a ``GridData``.
    mask : ndarray, shape (rows, cols), dtype bool, optional
        Tissue mask.  Ignored when *counts* is a ``GridData``.
    kappa : float
        Stationarity tolerance (same units as ``tessellate``).
    holdout_fraction : float
        Fraction of bins per region reserved for held-out evaluation.
    seed : int
        Random seed for reproducible holdout splits.

    Returns
    -------
    TessellationReport
    """
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and non-negative.")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be greater than 0 and less than 1.")

    if isinstance(counts, GridData):
        if rows is not None or cols is not None or mask is not None:
            raise ValueError(
                "rows, cols, and mask must not be passed separately when counts is a GridData instance."
            )
        mask = counts.mask
        rows = counts.rows
        cols = counts.cols
        counts = counts.counts
    else:
        if rows is None or cols is None:
            raise TypeError("rows and cols are required when counts is not a GridData.")
        rows = _validate_positive_int("rows", rows)
        cols = _validate_positive_int("cols", cols)
        if not sp.issparse(counts):
            raise TypeError("counts must be a SciPy sparse matrix or GridData.")
        if not sp.isspmatrix_csr(counts):
            counts = counts.tocsr()
        if counts.shape[0] != rows * cols:
            raise ValueError(
                f"counts has {counts.shape[0]} rows, expected "
                f"{rows * cols} = {rows} x {cols}."
            )
        if mask is None:
            mask = np.ones((rows, cols), dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (rows, cols):
                raise ValueError(f"mask has shape {mask.shape}, expected {(rows, cols)}.")

    labels = result.labels
    if labels.shape != (rows, cols):
        raise ValueError(f"result labels have shape {labels.shape}, expected {(rows, cols)}.")

    areas = result.areas
    depths = result.depths
    n_regions = result.n_regions

    # --- Basic statistics ---------------------------------------------------
    median_umi = float(np.median(depths)) if n_regions > 0 else 0.0

    density = depths / np.maximum(areas, 1.0)
    mean_density = density.mean() if n_regions > 0 else 0.0
    density_cv = float(density.std() / mean_density) if mean_density > 0 else 0.0

    # --- Stationarity pass rate ---------------------------------------------
    umi_flat = np.asarray(counts.sum(axis=1)).ravel()
    umi_2d = umi_flat.reshape(rows, cols)
    z_2d = np.log1p(umi_2d.astype(np.float64))

    pass_count = 0
    for k in range(n_regions):
        region_mask = (labels == k)
        z_vals = z_2d[region_mask]
        umi_vals = umi_2d[region_mask]
        if _poisson_stationary(z_vals, umi_vals, kappa):
            pass_count += 1
    stationarity_rate = pass_count / n_regions if n_regions > 0 else 0.0

    # --- Holdout evaluation -------------------------------------------------
    rng = np.random.default_rng(seed)
    flat_labels = labels.ravel()
    flat_mask = mask.ravel()

    # Global statistics for null model
    include = (flat_labels >= 0) & flat_mask
    global_counts = counts[include]
    global_depth = np.asarray(global_counts.sum(axis=1)).ravel()
    global_gene_totals = np.asarray(global_counts.sum(axis=0)).ravel()
    global_total_umi = global_gene_totals.sum()
    global_rate = global_depth.mean() if global_depth.size > 0 else 1.0
    n_genes = counts.shape[1]
    # Laplace (add-one) smoothing prevents zero probabilities that cause
    # catastrophic log-likelihood penalties on held-out data.
    global_pi = (global_gene_totals + 1) / (global_total_umi + n_genes)

    # Precompute global null log-probabilities (constant across regions).
    log_pi_global = np.log(global_pi)
    safe_global_rate = max(global_rate, 1e-12)
    log_global_rate = np.log(safe_global_rate)

    comp_ll_sum = 0.0
    comp_null_ll_sum = 0.0
    dens_ll_sum = 0.0
    dens_null_ll_sum = 0.0
    n_test_bins_total = 0

    for k in range(n_regions):
        bin_indices = np.where((flat_labels == k) & flat_mask)[0]
        n_bins = bin_indices.size
        if n_bins < 4:
            continue

        n_test = max(1, int(n_bins * holdout_fraction))
        perm = rng.permutation(n_bins)
        train_idx = bin_indices[perm[n_test:]]
        test_idx = bin_indices[perm[:n_test]]

        if train_idx.size == 0 or test_idx.size == 0:
            continue

        # --- Composition holdout ---
        train_counts = counts[train_idx]
        test_counts = counts[test_idx]

        train_gene_totals = np.asarray(train_counts.sum(axis=0)).ravel()
        train_total = train_gene_totals.sum()
        # Laplace smoothing: ensures no zero probabilities.
        pi_k = (train_gene_totals + 1) / (train_total + n_genes)

        log_pi_k = np.log(pi_k)

        test_dense = test_counts.toarray() if sp.issparse(test_counts) else test_counts
        comp_ll_sum += np.sum(test_dense * log_pi_k[np.newaxis, :])
        comp_null_ll_sum += np.sum(test_dense * log_pi_global[np.newaxis, :])

        # --- Density holdout ---
        train_umi = np.asarray(train_counts.sum(axis=1)).ravel()
        test_umi = np.asarray(test_counts.sum(axis=1)).ravel()

        lambda_k = train_umi.mean() if train_umi.size > 0 else global_rate
        safe_lambda_k = max(lambda_k, 1e-12)

        dens_ll_sum += np.sum(
            test_umi * np.log(safe_lambda_k) - safe_lambda_k - gammaln(test_umi + 1)
        )
        dens_null_ll_sum += np.sum(
            test_umi * log_global_rate - safe_global_rate - gammaln(test_umi + 1)
        )

        n_test_bins_total += test_idx.size

    # Normalize to per-bin
    if n_test_bins_total > 0:
        comp_holdout_ll = comp_ll_sum / n_test_bins_total
        dens_holdout_ll = dens_ll_sum / n_test_bins_total
        comp_ll_ratio = (comp_ll_sum - comp_null_ll_sum) / n_test_bins_total
        dens_ll_ratio = (dens_ll_sum - dens_null_ll_sum) / n_test_bins_total
    else:
        comp_holdout_ll = 0.0
        dens_holdout_ll = 0.0
        comp_ll_ratio = 0.0
        dens_ll_ratio = 0.0

    # --- Composition-dominates warning --------------------------------------
    # Fire when composition signal is present but density signal is absent
    # or negligible relative to composition.
    composition_dominates = bool(
        comp_ll_ratio > 0.01
        and (dens_ll_ratio < 0.01 or comp_ll_ratio > 10 * max(dens_ll_ratio, 1e-12))
    )

    return TessellationReport(
        n_regions=n_regions,
        median_umi_per_region=median_umi,
        density_cv=density_cv,
        stationarity_pass_rate=stationarity_rate,
        composition_holdout_ll=comp_holdout_ll,
        density_holdout_ll=dens_holdout_ll,
        composition_ll_ratio=comp_ll_ratio,
        density_ll_ratio=dens_ll_ratio,
        composition_dominates_warning=composition_dominates,
    )
