# SPDX-License-Identifier: MIT
"""
analysis.py
-----------
Distance-clustered spectral analysis for MREG data with BIDS-derivative I/O.

This module preserves prior algorithmic behavior (voxelwise demean, |rFFT|/N
amplitude spectra, band values as **sum of amplitudes** within predefined
frequency ranges). It standardizes filenames/outputs in a BIDS-derivatives
layout and adds class tokens (``arteries``, ``veins``, ``pvs``) to avoid
collisions.

Pipeline steps
--------------
1. Load 4D MREG, read TR from JSON, compute voxelwise rFFT amplitudes.
2. Write per-band amplitude-sum maps and a global mean-amplitude map.
3. (Optional) Build distance-based clusters from a distance map.
4. Analyze binned means across clusters (ANOVA/Kruskal) and plot.
5. Summarize mean spectra per cluster and plot up to `max_hz`.
6. Fit continuous regressions of log1p(band power) vs distance and plot.
7. (Optional) Regress log1p(band power) vs radii at centerline voxels.

Inputs / Outputs
----------------
Inputs  : SubjectPaths (BIDS roots/IDs and derivative dirs); 4D MREG
          (`sp.func_mreg_bold` + JSON sidecar); distance map(s); class masks.
Outputs : NIfTI band maps, integer cluster masks, NPZ spectra, CSV stats,
          and PNG figures, all under `derivatives/neurofluid-mreg/sub-<ID>/`.

Files written
-------------
- bandmaps/
  `sub-<ID>_space-MREG_band-<BAND>_desc-power_map.nii.gz`
  `sub-<ID>_space-MREG_desc-meanamp_map.nii.gz`
- clusters/
  `sub-<ID>_space-MREG_class-<CLASS>_desc-clusters_mask.nii.gz`
- spectra/
  `sub-<ID>_space-MREG_class-<CLASS>_desc-cluster_spectra.npz`
- stats/
  `sub-<ID>_space-MREG_class-<CLASS>_desc-binned_stats.csv` (semicolon CSV)
  `sub-<ID>_space_MREG_class-<CLASS>-desc-continuous_stats.csv` (semicolon CSV)
  `sub-<ID>_space-MREG_class-<CLASS>_desc-radius_vs_power.csv` (comma CSV)
- figures/
  `sub-<ID>_space-MREG_class-<CLASS>_desc-binned_bandpower.png`
  `sub-<ID>_space-MREG_class-<CLASS>_desc-cluster_spectra.png`
  `sub-<ID>_space-MREG_class-<CLASS>_band-<BAND>-desc-continuous.png`
  `sub-<ID>_space-MREG_class-<CLASS>_desc-radius_vs_power.png`

Assumptions / Preconditions
---------------------------
- Space: Operates in **native MREG space**. If affines differ, continue in
  image space after snapping masks/labels with **nearest-neighbor**.
- TR: Read exactly from the JSON sidecar; header 4th zoom is a last resort.
- Shapes/dtypes: Band maps/mean maps are float32; cluster labels are int16;
  spectra NPZ store float arrays; CSVs use semicolon except radii CSV (comma).
- BIDS naming: Uses `sub-<ID>_space-MREG_*` stems for NIfTI derivatives.

Warnings
--------
- Nearest-neighbor snapping preserves labels but may alias boundaries.
- NaNs may be introduced (brain masking) and must be handled downstream.
- No multiple-comparison control is applied in the statistical outputs.

Public API
----------
- compute_bandpower_maps
- make_distance_clusters
- analyze_binned
- cluster_spectra
- analyze_continuous
- frequency_map
- analyze_radius_vs_power
"""

import csv
import json
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import statsmodels.api as sm

from nibabel.processing import resample_from_to
from patsy import dmatrix
from scipy.stats import f_oneway, kruskal, linregress, spearmanr
from sklearn.linear_model import HuberRegressor

from .io import SubjectPaths, deriv_name


# -------------------------------------------------------------
# Defaults (bands, distance bins)
# -------------------------------------------------------------
# Default frequency bands (Hz)
BANDS_DEFAULT = {
    "cardiac": (0.80, 1.20),
    "respiratory": (0.20, 0.30),
    "LF": (0.027, 0.073),
    "VLF": (0.010, 0.027),
    # "vasomotor_broad": (0.001, 0.027)  # optional band
}

# Default distance bins (mm), use "max" for the last edge if needed
DIST_BINS_DEFAULT = [0, 2, 5, 10, "max"]

# -------------------------------------------------------------
# Config surface (tunable tolerances / thresholds)
# -------------------------------------------------------------
ATOL_DEFAULT: float = 1e-3
MASK_THR_DEFAULT: float = 0.5

# -------------------------------------------------------------
# I/O helpers (BIDS naming, reference resolution)
# -------------------------------------------------------------
def _resolve_mreg_ref(sp) -> nib.Nifti1Image:
    """
    Resolve a 3D MREG reference image for saving/aligning outputs.

    Preference order
    ----------------
    1) `sp.mreg_meanamp_path`
    2) `sp.mreg_mean_path`
    3) `sp.mreg_ref_path`
    4) `<bandmaps>/<sub>_space-MREG_desc-meanamp_map.nii.gz`
    5) `<mreg>/<sub>_space-MREG_class-brain_desc-mean_map.nii.gz`
    6) `sp.func_mreg_bold` (4D; only geometry used)

    Parameters
    ----------
    sp : SubjectPaths
        Subject context with paths. Expected attributes (if present):
        `mreg_meanamp_path`, `mreg_mean_path`, `mreg_ref_path`,
        `bandmaps_dir`, `mreg_dir`, `func_mreg_bold`, and `sub`.

    Returns
    -------
    nib.Nifti1Image
        The chosen reference image. If a 4D BOLD is used, only the geometry
        (affine and first three dimensions) is intended.

    Raises
    ------
    FileNotFoundError
        If none of the locations above exist.
    """
    # SubjectPaths attributes (if present)
    for attr in ("mreg_meanamp_path", "mreg_mean_path", "mreg_ref_path"):
        p = getattr(sp, attr, None)
        if p and Path(p).exists():
            return nib.load(str(p))

    # On-disk fallbacks
    meanamp = Path(sp.bandmaps_dir) / f"{sp.sub}_space-MREG_desc-meanamp_map.nii.gz"
    if meanamp.exists():
        return nib.load(str(meanamp))

    # brain mean in mreg dir (uses your deriv_name convention)
    mean = Path(sp.mreg_dir) / deriv_name(sp.sub, "MREG", "brain", "mean", "map")
    if mean.exists():
        return nib.load(str(mean))

    # Last resort: 4D BOLD (use its 3D grid)
    if getattr(sp, "func_mreg_bold", None) and Path(sp.func_mreg_bold).exists():
        return nib.load(str(sp.func_mreg_bold))

    raise FileNotFoundError("No MREG reference found (meanamp/mean/4D BOLD).")


# -------------------------------------------------------------
# Preprocessing (demean, rFFT amplitude)
# -------------------------------------------------------------
def _load_mreg_data(sp):
    """
    Load 4D MREG and compute voxelwise rFFT amplitude spectra (|rFFT|/N).

    Parameters
    ----------
    sp : SubjectPaths
        Must provide `func_mreg_bold` pointing to `<...>_task-mreg_bold.nii.gz`
        with an adjacent JSON sidecar containing `"RepetitionTime"`.

    Returns
    -------
    amp : ndarray, shape (n_voxels, n_freqs), dtype=float32
        Voxelwise amplitude spectra after per-voxel demean.
    freqs : ndarray, shape (n_freqs,), dtype=float64
        rFFT frequency axis in Hz.
    affine : ndarray, shape (4, 4)
        Affine copied from the input NIfTI.
    vol_shape : tuple of int
        3D volume shape `(nx, ny, nz)` for reshaping flat arrays.

    Assumptions / Preconditions
    ---------------------------
    - TR is read from the JSON sidecar if present; header 4th zoom is fallback.
    - Data are loaded as float32; rFFT is along time.

    Warnings
    --------
    - If multiple candidate MREG files exist, the first existing path in the
      preference list is used.

    Notes
    -----
    - Each voxel time series is demeaned before rFFT.
    """
    mreg_dir = Path(sp.mreg_dir)
    sub = sp.sub
    candidates = [
        mreg_dir / deriv_name(sub, "MREG", "brain", "detrended", "bold"),
        mreg_dir / deriv_name(sub, "MREG", "brain", "motionrealigned", "bold"),
        Path(sp.func_mreg_bold),
    ]
    mreg_path = next((p for p in candidates if p and p.exists()), None)
    if mreg_path is None:
        raise FileNotFoundError(f"No MREG 4D found (checked: {candidates})")
    print(f"[mreg] Source 4D: {mreg_path}")

    img = nib.load(str(mreg_path))
    data = img.get_fdata().astype(np.float32)  # (nx, ny, nz, nt)
    affine = img.affine
    nx, ny, nz, nt = data.shape
    ts = data.reshape(-1, nt)
    ts = ts - ts.mean(axis=1, keepdims=True)  # demean each voxel time series

    # TR resolution: prefer JSON next to chosen file; fallback to raw JSON; else header
    json_path = mreg_path.with_suffix("").with_suffix(".json")
    raw_json = Path(sp.func_mreg_bold).with_suffix("").with_suffix(".json")

    TR = None
    if json_path.exists():
        with open(json_path, "r") as f:
            info = json.load(f)
        TR = info.get("RepetitionTime")
    if TR is None and raw_json.exists():
        with open(raw_json, "r") as f:
            info = json.load(f)
        TR = info.get("RepetitionTime")
    if TR is None:
        TR = float(img.header.get_zooms()[3])  # safe fallback

    freqs = np.fft.rfftfreq(nt, d=TR)
    amp = np.abs(np.fft.rfft(ts, axis=1)) / nt
    return amp, freqs, affine, (nx, ny, nz)


# -------------------------------------------------------------
# Bandpower maps / single-frequency maps
# -------------------------------------------------------------
def compute_bandpower_maps(
    sp,
    tr,  # ignored (TR read from JSON in your loader)
    bands: dict[str, tuple[float, float]] | None = None,
    mask_path: Path | None = None,
    overwrite: bool = False,
    *,
    atol: float = ATOL_DEFAULT,
    mask_threshold: float = MASK_THR_DEFAULT,
) -> dict[str, Path]:
    """
    Compute per-band amplitude-sum maps and a global mean-amplitude map in
    MREG space. Optionally set NaN outside a brain mask.

    Parameters
    ----------
    sp : SubjectPaths
        Provides derivative dirs (`bandmaps_dir`) and MREG paths.
    tr : float
        Ignored. TR is read from the JSON sidecar in `_load_mreg_data`.
    bands : dict[str, tuple[float, float]] | None
        Frequency bands in Hz. If None, use `BANDS_DEFAULT`.
    mask_path : Path | None
        Optional brain mask in **MREG** space; snapped with nearest if needed.
    overwrite : bool, default False
        If False, skip writing existing outputs.
    atol : float, default 1e-3
        Absolute tolerance for affine equality when checking mask alignment
        to the MREG grid. If exceeded, the mask is snapped with nearest-neighbor.
    mask_threshold : float, default 0.5
        Threshold applied to the (possibly resampled) brain mask to create a
        boolean mask; values > mask_threshold are treated as inside brain.
    
    Returns
    -------
    dict[str, Path]
        Mapping `{band_name: band_map_path}`.

    Files written
    -------------
    - bandmaps/
      `sub-<ID>_space-MREG_band-<BAND>_desc-power_map.nii.gz` (float32)
      `sub-<ID>_space-MREG_desc-meanamp_map.nii.gz` (float32)

    Assumptions / Preconditions
    ---------------------------
    - Space: **MREG** grid. If mask is off-grid, snap with nearest.
    - Values are amplitude sums (a.u.) across FFT bins within band.
    - Mask alignment uses `atol` for affine comparisons; masks are snapped
      with nearest-neighbor when tolerance is exceeded.

    Warnings
    --------
    - NaNs may be introduced outside brain if `mask_path` is provided.
    """
    sp.bandmaps_dir.mkdir(parents=True, exist_ok=True)

    # Load FFT amplitudes already aligned to MREG grid
    amp, freqs, affine, vol_shape = _load_mreg_data(sp)
    if bands is None:
        bands = BANDS_DEFAULT

    # Build an in-memory reference NIfTI for resampling/writing headers
    ref_img = nib.Nifti1Image(np.zeros(vol_shape, dtype=np.float32), affine)
    ref_shape, ref_aff = ref_img.shape, ref_img.affine

    # Optional brain mask (snap to MREG grid if needed)
    brain_bool = None
    if mask_path is not None and Path(mask_path).exists():
        brain_img = nib.load(str(mask_path))
        if (brain_img.shape != ref_shape) or (
            not np.allclose(brain_img.affine, ref_aff, atol=atol)
        ):
            brain_img = resample_from_to(brain_img, (ref_shape, ref_aff), order=0)
            print("[bandmaps] [WARN] Brain mask snapped to MREG grid (nearest)")
        brain_bool = brain_img.get_fdata() > float(mask_threshold)

    band_paths: dict[str, Path] = {}
    nx, ny, nz = vol_shape

    # Per-band maps
    for band_name, (fmin, fmax) in bands.items():
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        if idx.size == 0:
            # Fallback to nearest FFT bin to band center
            center = 0.5 * (fmin + fmax)
            idx = [int(np.argmin(np.abs(freqs - center)))]

        power = amp[:, idx].sum(axis=1)  # sum amplitudes in band
        band_map = power.reshape(nx, ny, nz).astype(np.float32, copy=False)

        # Apply brain mask → NaN outside brain
        if brain_bool is not None:
            band_map = band_map.copy()
            band_map[~brain_bool] = np.nan

        out_path = (
            sp.bandmaps_dir
            / f"{sp.sub}_space-MREG_band-{band_name}_desc-power_map.nii.gz"
        )
        band_paths[band_name] = out_path

        if overwrite or (not out_path.exists()):
            hdr = ref_img.header.copy()
            hdr.set_data_dtype(np.float32)
            nib.save(nib.Nifti1Image(band_map, ref_aff, hdr), str(out_path))
            print(f"[bandmaps] Saved → {out_path}")
        else:
            print(f"[bandmaps] [SKIP] Exists: {out_path.name}")

    # Global mean-amplitude map (masked the same way)
    mean_power = amp.mean(axis=1).reshape(nx, ny, nz).astype(np.float32, copy=False)
    if brain_bool is not None:
        mean_power = mean_power.copy()
        mean_power[~brain_bool] = np.nan

    mean_path = sp.bandmaps_dir / f"{sp.sub}_space-MREG_desc-meanamp_map.nii.gz"
    if overwrite or (not mean_path.exists()):
        hdr = ref_img.header.copy()
        hdr.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(mean_power, ref_aff, hdr), str(mean_path))
        print(f"[bandmaps] Saved → {mean_path}")
    else:
        print(f"[bandmaps] [SKIP] Exists: {mean_path.name}")

    return band_paths


def frequency_map(sp, freq_hz, overwrite=False):
    """
    Create a single-frequency amplitude map (nearest rFFT bin) and save NIfTI.

    Parameters
    ----------
    sp : SubjectPaths
        Must provide `func_mreg_bold` and `bandmaps_dir`.
    freq_hz : float
        Frequency of interest (Hz); nearest rFFT bin is used.
    overwrite : bool, default False
        If False, skip when output already exists.

    Returns
    -------
    Path
        Path to `sub-<ID>_space-MREG_freq-{freq_hz:.3f}_desc-amp_map.nii.gz`.

    Files written
    -------------
    - bandmaps/
      `sub-<ID>_space-MREG_freq-{freq_hz:.3f}_desc-amp_map.nii.gz` (float32)

    Assumptions / Preconditions
    ---------------------------
    - Amplitude is |rFFT|/N after voxelwise demean.
    """
    amp, freqs, affine, vol_shape = _load_mreg_data(sp)
    idx = int(np.argmin(np.abs(freqs - freq_hz)))
    amp_flat = amp[:, idx]
    amp_map = amp_flat.reshape(vol_shape).astype(np.float32)
    fname = f"{sp.sub}_space-MREG_freq-{freq_hz:.3f}_desc-amp_map.nii.gz"
    sp.bandmaps_dir.mkdir(parents=True, exist_ok=True)
    amp_path = sp.bandmaps_dir / fname
    if not amp_path.exists() or overwrite:
        nib.save(nib.Nifti1Image(amp_map, affine), str(amp_path))
        print(f"[bandmaps] Saved → {amp_path}")
    else:
        print(f"[bandmaps] [SKIP] Exists: {amp_path.name}")
        
    return amp_path


# -------------------------------------------------------------
# Thresholding / post-processing / clustering
# -------------------------------------------------------------
def make_distance_clusters(sp, dist_map_path, klass: str, bins=None, overwrite=False, *, atol: float = ATOL_DEFAULT):
    """
    Create distance-based cluster labels per class and write a class-tagged
    integer mask in MREG space.

    Parameters
    ----------
    sp : SubjectPaths
        Provides `sub` and `clusters_dir`. Reference space is resolved
        internally for saving geometry.
    dist_map_path : str or os.PathLike
        Path to a 3D distance NIfTI (float), aligned to the MREG grid.
    klass : {'arteries', 'veins', 'pvs'}
        Vascular class token used in output filenames.
    bins : sequence[float] or None
        Monotonic bin edges (mm). If last edge is `'max'`, it is replaced by
        the image maximum. If None, uses `DIST_BINS_DEFAULT`.
    overwrite : bool, default False
        If False, skip when output exists.
    atol : float, default 1e-3
        Absolute tolerance for affine equality when verifying the distance-map
    grid against the chosen MREG reference. Labels are snapped with nearest-
    neighbor when tolerance is exceeded.

    Returns
    -------
    pathlib.Path
        `sub-<ID>_space-MREG_class-<klass>_desc-clusters_mask.nii.gz`.

    Files written
    -------------
    - clusters/
      `sub-<ID>_space-MREG_class-<klass>_desc-clusters_mask.nii.gz` (int16)

    Assumptions / Preconditions
    ---------------------------
    - Distance map in mm; NaNs are allowed and digitized to `-1` by `nan_to_num`.

    Warnings
    --------
    - Label resampling occurs when shape/affine mismatch exceeds `atol`
      (nearest-neighbor; integer labels preserved).
    """
    if bins is None:
        bins = DIST_BINS_DEFAULT

    dist_path = Path(dist_map_path)
    if not dist_path.exists():
        raise FileNotFoundError(f"Distance map not found: {dist_path}")

    # Load distance map
    dist_img = nib.load(str(dist_path))
    dist = np.asarray(dist_img.get_fdata(), dtype=np.float32)

    # Resolve MREG reference (shape & affine anchor)
    ref_img = _resolve_mreg_ref(sp)
    ref_shape = ref_img.shape[:3]
    ref_aff = ref_img.affine

    # Handle 'max' in bins
    finite_max = float(np.nanmax(dist)) if np.isfinite(dist).any() else 0.0
    def _normalize_bins(bins_in, vmax):
        raw = []
        has_max = False
        for b in bins_in:
            if isinstance(b, str) and str(b).lower() == "max":
                has_max = True
            else:
                raw.append(float(b))

        arr = np.asarray(raw, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        arr.sort()
        arr = np.unique(arr)

        if has_max:
            # keep only edges ≤ vmax, then append vmax as the last edge
            arr = arr[arr <= (vmax + 1e-6)]
            if arr.size == 0 or arr[-1] < vmax:
                arr = np.concatenate([arr, [np.float32(vmax)]])

        if arr.size < 2:
            raise ValueError(f"distance bin edges invalid after normalization: {arr}")
        if np.any(np.diff(arr) <= 0):
            raise ValueError(f"distance bin edges not strictly increasing: {arr}")

        return arr
    numeric_bins = _normalize_bins(bins, finite_max)        

    # Digitize -> labels (bin index), map NaNs to -1 explicitly
    valid = np.isfinite(dist)
    labels = np.full(dist.shape, -1, dtype=np.int16)
    if np.any(valid):
        labels[valid] = np.digitize(dist[valid], numeric_bins, right=True).astype(np.int16) - 1


    # Ensure labels are on MREG grid (nearest)
    if (labels.shape != tuple(ref_shape)) or (
        not np.allclose(dist_img.affine, ref_aff, atol=atol)
    ):
        lbl_src = nib.Nifti1Image(labels, dist_img.affine, dist_img.header)
        lbl_res = resample_from_to(lbl_src, (ref_shape, ref_aff), order=0)
        print("[clusters] [WARN] Labels snapped to MREG grid (nearest)")
        labels = np.rint(lbl_res.get_fdata()).astype(np.int16, copy=False)

    # Save
    sp.clusters_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(sp.clusters_dir) / deriv_name(
        sp.sub, "MREG", klass, "clusters", "mask"
    )

    if not out_path.exists() or overwrite:
        hdr = ref_img.header.copy()
        hdr.set_data_dtype(np.int16)
        nib.save(nib.Nifti1Image(labels, ref_aff, hdr), str(out_path))
        print(f"[clusters] Saved → {out_path}")
    else:
        print(f"[clusters] [SKIP] Exists: {out_path.name}")

    return out_path


def analyze_binned(sp, labels_path, *, klass: str, bands=None, overwrite=False):
    """
    Compute per-cluster band means for a class (arteries/veins/pvs), run
    ANOVA/Kruskal, and save class-specific CSV + figure.

    Parameters
    ----------
    sp : SubjectPaths
        Must provide `bandmaps_dir`, `stats_dir`, `figures_dir`.
    labels_path : str | Path
        Path to integer label NIfTI in MREG space. Negative values ignored.
    klass : {'arteries', 'veins', 'pvs'}
        Anatomical class token used in output filenames.
    bands : dict[str, tuple[float, float]] | None
        Frequency bands (Hz). If None, uses `BANDS_DEFAULT`.
    overwrite : bool, default False
        If False, skip when both outputs already exist.

    Returns
    -------
    (Path, Path)
        `(csv_path, fig_path)`.

    Files written
    -------------
    - stats/
      `sub-<ID>_space-MREG_class-<klass>_desc-binned_stats.csv`
      (semicolon-delimited with columns: band;F;p_anova;H;p_kruskal)
    - figures/
      `sub-<ID>_space-MREG_class-<klass>_desc-binned_bandpower.png`

    Warnings
    --------
    - Statistics computed only if ≥2 non-empty groups; otherwise NaNs recorded.
    """
    if bands is None:
        bands = BANDS_DEFAULT

    labels_img = nib.load(str(labels_path))
    labels_data = labels_img.get_fdata().astype(int)
    cluster_ids = np.unique(labels_data[~np.isnan(labels_data)])
    cluster_ids = cluster_ids[cluster_ids >= 0]
    cluster_ids = sorted(cluster_ids)

    sp.stats_dir.mkdir(parents=True, exist_ok=True)
    sp.figures_dir.mkdir(parents=True, exist_ok=True)

    stats_path = (
        sp.stats_dir
        / f"{sp.sub}_space-MREG_class-{klass}_desc-binned_stats.csv"
    )
    fig_path = (
        sp.figures_dir
        / f"{sp.sub}_space-MREG_class-{klass}_desc-binned_bandpower.png"
    )

    if stats_path.exists() and fig_path.exists() and not overwrite:
        print(f"[binned] Exists → {klass}; skipping.")
        return stats_path, fig_path

    cluster_means = {}
    stat_rows = []

    for band in bands:
        band_file = (
            sp.bandmaps_dir
            / f"{sp.sub}_space-MREG_band-{band}_desc-power_map.nii.gz"
        )
        if not band_file.exists():
            warnings.warn(f"[binned] Band map not found for '{band}'; skipping.")
            continue

        band_data = nib.load(str(band_file)).get_fdata()
        means = []
        samples = []

        for c in cluster_ids:
            vals = band_data[labels_data == c]
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                means.append(np.nan)
            else:
                means.append(np.nanmean(vals))
                samples.append(vals.ravel())

        cluster_means[band] = means

        if len(samples) > 1:
            try:
                F, p_anova = f_oneway(*samples)
            except Exception:
                F, p_anova = np.nan, np.nan
            try:
                H, p_kw = kruskal(*samples)
            except Exception:
                H, p_kw = np.nan, np.nan
        else:
            F, p_anova, H, p_kw = np.nan, np.nan, np.nan, np.nan

        stat_rows.append((band, F, p_anova, H, p_kw))

    # Save stats CSV (semicolon for EU Excel)
    with open(stats_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(["band", "F", "p_anova", "H", "p_kruskal"])
        for row in stat_rows:
            writer.writerow(row)
    print(f"[binned] Saved → {stats_path}")

    # Plot band power by cluster
    if not fig_path.exists() or overwrite:
        plt.figure(figsize=(8, 5))
        for band, means in cluster_means.items():
            plt.plot(cluster_ids, means, marker="o", label=band)
        plt.xlabel("Distance Cluster")
        plt.ylabel("Mean Band Power (a.u.)")
        plt.title(f"Band Power by Distance Cluster ({klass})")
        plt.legend()
        plt.grid(True)
        plt.savefig(str(fig_path))
        plt.close()
    else:
        print(f"[binned] [SKIP] Exists for class '{klass}': {fig_path.name}")

    return stats_path, fig_path


# -------------------------------------------------------------
# Spectral summaries by cluster
# -------------------------------------------------------------
def cluster_spectra(sp, labels_path, *, klass: str, max_hz=2.0, overwrite=False):
    """
    Compute mean amplitude spectrum per distance cluster for a class and
    write NPZ + figure.

    Parameters
    ----------
    sp : SubjectPaths
        Must provide `func_mreg_bold`, `spectra_dir`, `figures_dir`.
    labels_path : str | Path
        Path to cluster labels NIfTI (int) in MREG space.
    klass : {'arteries', 'veins', 'pvs'}
        Anatomical class token.
    max_hz : float, default 2.0
        Upper x-axis limit in Hz for the figure.
    overwrite : bool, default False
        If False, skip when outputs exist.

    Returns
    -------
    (Path, Path)
        `(npz_path, fig_path)`.

    Files written
    -------------
    - spectra/
      `sub-<ID>_space-MREG_class-<klass>_desc-cluster_spectra.npz`
      (arrays: `freqs`, `spectra`, `cluster_ids`)
    - figures/
      `sub-<ID>_space-MREG_class-<klass>_desc-cluster_spectra.png`

    Notes
    -----
    - Uses the same rFFT pipeline as band maps (|rFFT|/N after demean).
    """
    labels_data = nib.load(str(labels_path)).get_fdata().astype(int)
    cluster_ids = np.unique(labels_data[~np.isnan(labels_data)])
    cluster_ids = cluster_ids[cluster_ids >= 0]
    cluster_ids = sorted(cluster_ids)

    amp, freqs, _, _ = _load_mreg_data(sp)

    spectra_list = []
    valid_clusters = []
    for c in cluster_ids:
        mask = labels_data.ravel() == c
        if not np.any(mask):
            continue
        spec = amp[mask].mean(axis=0)
        spectra_list.append(spec)
        valid_clusters.append(c)

    if spectra_list:
        spectra_arr = np.vstack(spectra_list)
    else:
        spectra_arr = np.empty((0, freqs.size))

    sp.spectra_dir.mkdir(parents=True, exist_ok=True)
    npz_path = (
        sp.spectra_dir
        / f"{sp.sub}_space-MREG_class-{klass}_desc-cluster_spectra.npz"
    )
    if not npz_path.exists() or overwrite:
        np.savez_compressed(
            str(npz_path),
            freqs=freqs,
            spectra=spectra_arr,
            cluster_ids=np.array(valid_clusters),
        )
        print(f"[spectra] Saved → {npz_path}")
    else:
        print(f"[spectra] [SKIP] Exists: {npz_path.name}")

    sp.figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = (
        sp.figures_dir
        / f"{sp.sub}_space-MREG_class-{klass}_desc-cluster_spectra.png"
    )
    if not fig_path.exists() or overwrite:
        plt.figure(figsize=(8, 6))
        for c, spec in zip(valid_clusters, spectra_list):
            plt.plot(freqs, spec, label=f"Cluster {int(c)}")
        plt.xlim(0, max_hz)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (a.u.)")
        plt.title(f"Full Spectrum by Distance Cluster ({klass})")
        plt.legend()
        plt.grid(True)
        plt.savefig(str(fig_path))
        print(f"[spectra] Saved → {fig_path}")
        plt.close()
    else:
        print(f"[spectra] [SKIP] Exists: {fig_path.name}")

    return npz_path, fig_path


# -------------------------------------------------------------
# Continuous regression vs distance
# -------------------------------------------------------------

def analyze_continuous(
    sp,
    dist_map_path: Path,
    bands=None,
    mask_path: Path = None,
    ref_curve=None,
    ref_label="ref",
    *,
    overwrite: bool = False,
    max_points_plot=200000,
) -> dict:
    
    """
    Fit continuous regressions of log1p(band power) ~ distance (mm) for each
    band, writing per-band figures and a class-tagged CSV.

    Parameters
    ----------
    sp : SubjectPaths
        Must provide `bandmaps_dir`, `stats_dir`, `figures_dir`, `sub`.
    dist_map_path : str | Path
        3D distance map NIfTI in **MREG** space (float, mm). Filename must
        contain class token ("arteries", "veins", "pvs").
    bands : dict[str, tuple[float, float]] | list[str] | None
        Bands to analyze; defaults to keys of `BANDS_DEFAULT` if None.
    mask_path : str | Path | None
        Optional binary mask NIfTI (same grid/affine) to restrict voxels.
    ref_curve : callable | None
        Optional function `ref_curve(x)` for overlay.
    ref_label : str, default "ref"
        Legend label for the reference curve.
    overwrite : bool, default False
        If False, skip when CSV exists (and return parsed stats).
    max_points_plot : int, default 200000
        Max voxels sampled for fitting/plotting.

    Returns
    -------
    dict
        Summary with keys: `"class"` and `"per_band"` (slope/SE/p/R²/n).

    Files written
    -------------
    - stats/
      `sub-<ID>_space_MREG_class-<CLASS>-desc-continuous_stats.csv` (semicolon CSV)
      columns: band;slope;intercept;r;p;stderr;n
    - figures/
      `sub-<ID>_space-MREG_class-<CLASS>_band-<BAND>-desc-continuous.png`

    Assumptions / Preconditions
    ---------------------------
    - Distances are in mm; band maps share the same MREG grid/affine.

    Warnings
    --------
    - Raises on shape/affine mismatch between band map and distance map.
    - NaNs are removed before fitting; downsampling applied for large N.
    - Continuous CSV is read/written with semicolon delimiter for Excel compatibility.

    Notes
    -----
    - Linear stats are from OLS; robust Huber fit stabilizes visualization.
    """
    # Default band names
    try:
        from .analysis import BANDS_DEFAULT
    except ImportError:
        BANDS_DEFAULT = {
            "cardiac": (0.80, 1.20),
            "respiratory": (0.20, 0.30),
            "LF": (0.027, 0.073),
            "VLF": (0.010, 0.027),
        }
    if bands is None:
        bands = list(BANDS_DEFAULT.keys())
    else:
        bands = list(bands)

    # Determine class name from distance map filename
    class_name = None
    fname = dist_map_path.name.lower()
    for cls in ("arteries", "veins", "pvs"):
        if cls in fname:
            class_name = cls
            break
    if class_name is None:
        raise ValueError(f"Unable to determine vessel class from {dist_map_path}")

    # Output CSV path
    out_csv = sp.stats_dir / f"{sp.sub}_space-MREG_class-{class_name}-desc-continuous_stats.csv"
    if out_csv.exists() and not overwrite:
        # Load existing CSV into result_dict and return
        result_dict = {"class": class_name, "per_band": {}}
        with open(out_csv, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                band = row["band"]
                # r is correlation; convert to R²
                r_val = float(row["r"])
                result_dict["per_band"][band] = {
                    "beta": float(row["slope"]),
                    "se": float(row["stderr"]),
                    "p": float(row["p"]),
                    "r2": r_val**2,
                    "n": int(row["n"]),
                }
        return result_dict

    # Load distance map (mm)
    dist_img = nib.load(str(dist_map_path))
    dist = dist_img.get_fdata().astype(np.float32)

    # Load mask if provided
    if mask_path is not None and mask_path.exists():
        mask_img = nib.load(str(mask_path))
        mask = mask_img.get_fdata().astype(bool)
    else:
        mask = np.ones_like(dist, dtype=bool)

    # Prepare output container
    result_dict = {"class": class_name, "per_band": {}}
    stats_rows = []

    # Iterate over bands
    for band in bands:
        band_path = (
            sp.bandmaps_dir
            / f"{sp.sub}_space-MREG_band-{band}_desc-power_map.nii.gz"
        )
        if not band_path.exists():
            print(f"[continuous] [WARN] Missing band map for '{band}'; skipping")
            continue
        bm_img = nib.load(str(band_path))
        bandmap = bm_img.get_fdata().astype(np.float32)

        # Validate grid/affine
        if (bm_img.shape != dist_img.shape) or (
            not np.allclose(bm_img.affine, dist_img.affine)
        ):
            raise ValueError(
                "Shape/affine mismatch:\n"
                f"  dist: {dist_img.shape}\n"
                f"  band: {bm_img.shape}\n"
                f"{band_path} vs {dist_map_path}"
            )

        # Mask NaNs and optional mask
        valid_mask = np.isfinite(dist) & np.isfinite(bandmap) & (mask.astype(bool))
        d_flat = dist[valid_mask].ravel().astype(np.float32)
        p_flat = bandmap[valid_mask].ravel().astype(np.float32)
        N = d_flat.size
        if N == 0:
            print(f"[continuous] [WARN] No valid voxels for band '{band}'; skipping")
            continue

        # Downsample for stability/plotting
        if N > max_points_plot:
            np.random.seed(0)
            idx = np.random.choice(N, size=max_points_plot, replace=False)
            d_sub = d_flat[idx]
            p_sub = p_flat[idx]
        else:
            d_sub = d_flat
            p_sub = p_flat

        # Transform + prepare data
        y = np.log1p(p_sub)  # stabilize skew
        x = d_sub.astype(np.float32)

        # Ensure finite + enough unique x
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        n = x.size
        uniq_x = np.unique(x).size

        if n < 10 or uniq_x < 3:
            print(f"[continuous] [WARN] {class_name}/{band}: too few points (n={n}, unique_x={uniq_x}); skipping")
            continue

        # Pick a safe spline df based on data
        df_spline = int(min(6, max(4, uniq_x - 1, 3)))
        df_spline = min(df_spline, n - 2)

        # Fit both models (robust linear + spline OLS)
        lin = _fit_linear_huber(x, y)
        spl = _fit_spline_ols(x, y, df=df_spline, degree=3)

        # Density heatmap (distance × log-power)
        xmax = np.percentile(x, 99.5)
        xbins = np.linspace(0, xmax, 100)
        y_lo, y_hi = np.percentile(y, [1, 99])
        ybins = np.linspace(y_lo, y_hi, 100)
        H, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(
            H.T,
            origin="lower",
            cmap="viridis",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
        )
        ax.set_xlabel("Distance to vessel (mm)")
        ax.set_ylabel("log1p(Band power)")
        ax.set_title(f"{sp.sub} – {class_name}, band = {band}")
        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])

        # Prediction grid for overlays
        x_line = np.linspace(x.min(), x.max(), 300)

        # Linear line
        X_line_lin = sm.add_constant(x_line)
        y_lin = lin["ols"].predict(X_line_lin)

        # Spline line
        Xg_spl = dmatrix(
            f"bs(x, df={df_spline}, degree=3, include_intercept=True)",
            {"x": x_line},
            return_type="dataframe",
        )
        y_spl = spl["ols"].predict(Xg_spl)

        ax.plot(x_line, y_lin, lw=2, label="Linear")
        ax.plot(x_line, y_spl, lw=2, ls="--", label=f"Spline (df={df_spline})")

        # Optional reference overlay
        if ref_curve is not None:
            y_ref = ref_curve(x_line)
            ax.plot(x_line, y_ref, color="k", ls="--", lw=1.5, label=ref_label)
        ax.legend()

        # Save figure
        fig_path = (
            sp.figures_dir
            / f"{sp.sub}_space-MREG_class-{class_name}_band-{band}-desc-continuous.png"
        )
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        print(f"[continuous] Saved → {fig_path}")
        plt.close(fig)

        # Record stats (linear OLS)
        r_corr = linregress(x, y).rvalue
        stats_rows.append(
            [
                band,
                lin["slope"],
                lin["intercept"],
                r_corr,
                lin["p"],
                lin["stderr"],
                int(N),
            ]
        )

        result_dict["per_band"][band] = {
            "beta": lin["slope"],
            "se": lin["stderr"],
            "p": lin["p"],
            "r2": lin["r2"],
            "n": int(N),
        }

    # Write CSV (semicolon for EU Excel)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["band", "slope", "intercept", "r", "p", "stderr", "n"])
        writer.writerows(stats_rows)
    print(f"[continuous] Saved → {out_csv}")

    return result_dict


# -------------------------------------------------------------
# Radii estimation / fitting / QC
# -------------------------------------------------------------
def analyze_radius_vs_power(
    sp: SubjectPaths,
    klass: str,
    bands: dict,
    band_paths: dict[str, Path] | None = None,
    mask_path: Path | None = None,
    overwrite: bool = False,
    max_points_plot: int = 200_000,
    atol: float = ATOL_DEFAULT,
    mask_threshold: float = MASK_THR_DEFAULT,
):
    """
    Regress log1p(band power) ~ radius_mm at centerline voxels in MREG space.

    Parameters
    ----------
    sp : SubjectPaths
        Must provide `radii_dir`, `bandmaps_dir`, `stats_dir`, `figures_dir`.
    klass : {'arteries', 'veins', 'pvs'}
        Vascular class token used in output filenames.
    bands : dict
        Mapping `{band_name: (fmin, fmax)}`; iteration order defines plots.
    band_paths : dict[str, Path] | None
        Optional mapping of band name to NIfTI path; otherwise discovered.
    mask_path : Path | None
        Optional brain mask to confine analysis; snapped (nearest) if needed.
    overwrite : bool, default False
        If False, skip when figure/CSV exist (not enforced here).
    max_points_plot : int, default 200_000
        Downsample cap for scatter plotting.
    atol : float, default 1e-3
        Absolute tolerance for affine equality when aligning the optional brain
        mask (and any bandmap resampling) to the radii grid.
    mask_threshold : float, default 0.5
        Threshold applied to the (possibly resampled) brain mask; values >
        mask_threshold are treated as inside brain.

    Returns
    -------
    (Path | None, Path | None)
        `(csv_path, fig_path)` if successful, else `None`.

    Files written
    -------------
    - stats/
      `sub-<ID>_space-MREG_class-<klass>_desc-radius_vs_power.csv` (semicolon CSV)
    - figures/
      `sub-<ID>_space-MREG_class-<klass>_desc-radius_vs_power.png`

    Assumptions / Preconditions
    ---------------------------
    - Radii map exists at:
      `radii/sub-<ID>_space-MREG_class-<klass>_desc-radius_map.nii.gz`.
    - Centerline voxels have `radius > 0`; failed fits marked ≤0 downstream.

    Warnings
    --------
    - Affine/shape mismatches are corrected for band maps via resampling
      (linear, order=1) to the radii grid solely for plotting/regression.
    - Brain mask thresholding uses `mask_threshold`; resampling/snap decisions
      use `atol`. Band maps are resampled (linear) only when needed for grid match.
    """
    # paths
    rad_path = Path(sp.radii_dir) / deriv_name(sp.sub, "MREG", klass, "radius", "map")
    if not rad_path.exists():
        print(f"[radius] [SKIP] No MREG radii for {klass}: {rad_path.name}")
        return None

    # load radii & ref
    rad_img = nib.load(str(rad_path))
    rad = np.asarray(rad_img.get_fdata(), dtype=np.float32)
    ref_shape, ref_aff = rad_img.shape, rad_img.affine

    # brain mask (optional)
    brain = None
    if mask_path is not None and Path(mask_path).exists():
        m = nib.load(str(mask_path))
        if m.shape != ref_shape or not np.allclose(m.affine, ref_aff, atol=atol):
            m = resample_from_to(m, (ref_shape, ref_aff), order=0)
        brain = (m.get_fdata() > float(mask_threshold)).astype(bool) # brain = m.get_fdata().astype(bool)

    # centerline voxels with valid radii
    valid = rad > 0
    if brain is not None:
        valid &= brain
    if not np.any(valid):
        print(f"[radius] [SKIP] No valid in-brain radii for {klass}")
        return None

    # Prepare output CSV rows
    rows = [
        (
            "band",
            "n",
            "slope",
            "intercept",
            "p_value",
            "r_value",
            "r_squared",
            "spearman_r",
            "spearman_p",
            "mean_radius_mm",
        )
    ]
    fig, axs = plt.subplots(1, len(bands), figsize=(4.2 * len(bands), 4.0), squeeze=False)
    axs = axs[0]

    # gather indices once (for faster sampling)
    ii, jj, kk = np.where(valid)
    radii_vec = rad[ii, jj, kk]

    for bidx, (band_name, _) in enumerate(bands.items()):
        # locate band map
        if band_paths and band_name in band_paths:
            bpath = Path(band_paths[band_name])
        else:
            # fallback: look for a file that contains band name in desc
            cand = list(
                Path(sp.bandmaps_dir).glob(
                    f"{sp.sub}_space-MREG_*desc*{band_name}*_map.nii.gz"
                )
            )
            if not cand:
                print(f"[radius] [SKIP] Band '{band_name}' map not found")
                continue
            bpath = cand[0]

        bimg = nib.load(str(bpath))
        bdat = np.asarray(bimg.get_fdata(), dtype=np.float32)

        # align band map to radii grid if needed
        if bimg.shape != ref_shape or not np.allclose(bimg.affine, ref_aff, atol=1e-3):
            bimg = resample_from_to(bimg, (ref_shape, ref_aff), order=1)
            bdat = np.asarray(bimg.get_fdata(), dtype=np.float32)

        # sample band values at centerline voxels
        power_vec = bdat[ii, jj, kk]
        # mask out NaNs from brain-masked bandmaps
        good = np.isfinite(power_vec)
        x = radii_vec[good]
        y = power_vec[good]
        n = x.size
        if n < 10:
            print(f"[radius] [SKIP] Too few points for {band_name} ({n})")
            continue

        # regression on log1p(power)
        y_log = np.log1p(y)
        lr = linregress(x, y_log)  # slope, intercept, rvalue, pvalue, stderr
        spearman_r, spearman_p = spearmanr(x, y, nan_policy="omit")

        rows.append(
            (
                band_name,
                int(n),
                float(lr.slope),
                float(lr.intercept),
                float(lr.pvalue),
                float(lr.rvalue),
                float(lr.rvalue**2),
                float(spearman_r),
                float(spearman_p),
                float(np.mean(x)),
            )
        )

        # plot (downsample for speed)
        ax = axs[bidx]
        if n > max_points_plot:
            sel = np.random.default_rng(0).choice(n, size=max_points_plot, replace=False)
            xx, yy = x[sel], y[sel]
        else:
            xx, yy = x, y
        ax.scatter(xx, yy, s=2, alpha=0.3)
        # trendline in original power units (exp of log fit minus 1)
        xs = np.linspace(xx.min(), xx.max(), 200)
        ys = np.exp(lr.intercept + lr.slope * xs) - 1.0
        ax.plot(xs, ys, linewidth=2)
        ax.set_title(band_name)
        ax.set_xlabel("Radius (mm)")
        ax.set_ylabel("Band power")

    # save CSV + figure
    csv_path = (
        Path(sp.stats_dir)
        / f"{sp.sub}_space-MREG_class-{klass}_desc-radius_vs_power.csv"
    )
    fig_path = (
        Path(sp.figures_dir)
        / f"{sp.sub}_space-MREG_class-{klass}_desc-radius_vs_power.png"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    # write semicolon CSV (standardized)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        for r in rows:
            w.writerow(r)
    print(f"[radius] Saved → {fig_path}")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    print(f"[radius] Saved → {fig_path}")
    plt.close(fig)

    return csv_path, fig_path


# -------------------------------------------------------------
# Utilities (model fitting)
# -------------------------------------------------------------
def _fit_linear_huber(x, y):
    """
    Robust + OLS linear model: `y ~ x`.

    Parameters
    ----------
    x, y : ndarray, shape (n,), dtype=float32/float64
        Finite vectors after preprocessing.

    Returns
    -------
    dict
        Keys: `type`, `slope`, `intercept`, `stderr`, `p`, `r2`, `aic`,
        `ols` (statsmodels fit), and `yhat` (Huber predictions).

    Notes
    -----
    - Huber regression stabilizes against outliers; OLS provides inference.
    """
    # robust fit for line
    huber = HuberRegressor().fit(x.reshape(-1, 1), y)
    yhat_rl = huber.predict(x.reshape(-1, 1))

    # OLS for stats (p, SE, CI, AIC)
    Xc = sm.add_constant(x)
    ols = sm.OLS(y, Xc).fit()
    return {
        "type": "linear",
        "slope": float(ols.params[1]),
        "intercept": float(ols.params[0]),
        "stderr": float(ols.bse[1]),
        "p": float(ols.pvalues[1]),
        "r2": float(ols.rsquared),
        "aic": float(ols.aic),
        "ols": ols,  # keep for CI/prediction on a grid
        "yhat": yhat_rl,  # robust-predicted (visual line is stable)
    }


def _fit_spline_ols(x, y, df: float, degree=3):
    """
    Cubic regression spline OLS via patsy basis (`bs`).

    Parameters
    ----------
    x, y : ndarray, shape (n,), dtype=float32/float64
        Finite vectors after preprocessing.
    df : float
        Approximate degrees of freedom for the spline basis.
    degree : int, default 3
        Polynomial degree for each spline segment.

    Returns
    -------
    dict
        Keys: `type`, `df`, `degree`, `aic`, `r2`, `ols`, `X_design`.
    """
    Xs = dmatrix(
        f"bs(x, df={df}, degree={degree}, include_intercept=True)",
        {"x": x},
        return_type="dataframe",
    )
    ols = sm.OLS(y, Xs).fit()
    return {
        "type": "spline",
        "df": df,
        "degree": degree,
        "aic": float(ols.aic),
        "r2": float(ols.rsquared),
        "ols": ols,  # for CI/prediction
        "X_design": Xs,  # cache (optional)
    }
