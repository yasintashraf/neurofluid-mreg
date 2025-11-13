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
3. Build distance-based clusters from a distance map (MREG or MNI grid).
4. Analyze binned means across clusters (ANOVA/Kruskal) and plot.
5. Summarize mean spectra per cluster and plot up to `max_hz`.
6. Fit continuous regressions of log1p(band power) vs distance and plot.
7. Regress log1p(band power) vs radii at centerline voxels (MREG or MNI).

Inputs / Outputs
----------------
Inputs
    - `SubjectPaths` (BIDS roots/IDs and derivative dirs).
    - 4D MREG BOLD (`sp.func_mreg_bold` + JSON sidecar with TR).
    - Bandpower maps in space-MREG or space-MNI, depending on analysis.
    - Distance maps (mm) in MREG or MNI space.
    - Optional brain masks and radius maps (mm).

Outputs
    - NIfTI band maps and mean-amplitude maps (MREG).
    - Integer cluster masks (same space as distance map).
    - NPZ spectra summaries.
    - CSV statistics (binned, continuous, radius vs power).
    - PNG figures for QC and summaries.

Files written
-------------
- bandmaps/
  `sub-<ID>_space-MREG_band-<BAND>_desc-power_map.nii.gz`
  `sub-<ID>_space-MREG_desc-meanamp_map.nii.gz`
  `sub-<ID>_space-MREG_freq-<FREQ>_desc-amp_map.nii.gz`
- clusters/
  `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-clusters_mask.nii.gz`
- spectra/
  `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-cluster_spectra.npz`
- stats/
  `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-binned_stats.csv`
  `sub-<ID>_space-<SPACE>_class-<CLASS>-desc-continuous_stats.csv`
  `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-radius_vs_power.csv`
  (all CSVs are semicolon-delimited)
- figures/
  `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-binned_bandpower.png`
  `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-cluster_spectra.png`
  `sub-<ID>_space-<SPACE>_class-<CLASS>_band-<BAND>-desc-continuous.png`
  `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-radius_vs_power.png`

Assumptions / Preconditions
---------------------------
- Space: All time-series–driven analyses operate on the **native MREG grid**.
  Cluster and distance maps may be in MREG or MNI; space is inferred from the
  filename and used only for naming. When required, labels/masks are snapped
  to the MREG grid with nearest-neighbor interpolation for time-series
  sampling.
- TR: Read from the JSON sidecar (`"RepetitionTime"`, seconds). Header 4th
  zoom is used as a last resort.
- Shapes/dtypes: Band maps and mean-amplitude maps are float32; cluster labels
  are int16; spectra NPZ contain float arrays; CSVs are semicolon-delimited.
- BIDS naming: Uses `sub-<ID>_space-<SPACE>_class-<CLASS>_...` stems for NIfTI
  and CSV derivatives wherever class-specific outputs are written.

Warnings
--------
- Nearest-neighbor snapping preserves labels but may alias boundaries when
  resampling cluster or mask images.
- NaNs may be introduced (e.g., outside brain masks) and must be handled by
  downstream consumers.
- No multiple-comparison control is applied to ANOVA/Kruskal p-values.
- Bandpower maps represent **sums of amplitude** across FFT bins within each
  band; they are not power spectral density estimates.

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
def _infer_space_from_path(path: Path, default: str = "MREG") -> str:
    """
    Infer imaging space token ("MNI" or "MREG") from a file path.

    Parameters
    ----------
    path : pathlib.Path
        Path whose string representation may contain `_space-MNI` or
        `_space-MREG`.
    default : str, optional
        Fallback space token when no explicit match is found. Default is
        `"MREG"`.

    Returns
    -------
    str
        `"MNI"`, `"MREG"`, or `default` when no explicit token is detected.

    Warnings
    --------
    - This is a simple substring-based heuristic; if filenames diverge from
      the `space-<SPACE>` convention, `default` is returned.
    """
    s = str(path)
    if "space-MNI" in s or "_space-MNI" in s:
        return "MNI"
    if "space-MREG" in s or "_space-MREG" in s:
        return "MREG"
    return default


# -------------------------------------------------------------
# Preprocessing (demean, rFFT amplitude)
# -------------------------------------------------------------
def _load_mreg_data(sp):
    """
    Load 4D MREG and compute voxelwise rFFT amplitude spectra (|rFFT|/N).

    Parameters
    ----------
    sp : SubjectPaths
        Must provide `func_mreg_bold` pointing to a MREG BOLD NIfTI with an
        adjacent JSON sidecar containing `"RepetitionTime"` in seconds. If
        detrended/motionrealigned derivatives exist, they are preferred.

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
    - TR (s) is read from the JSON sidecar if present; header 4th zoom
      (seconds) is used as a fallback.
    - Data are loaded as float32 and demeaned along the time axis per voxel.

    Warnings
    --------
    - If multiple candidate MREG files exist, the first existing path in the
      internal preference list is used.
    - No temporal filtering is applied here beyond demeaning.
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
        TR = float(img.header.get_zooms()[3])  # safe fallback (seconds)

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
        Provides derivative directories (e.g., `bandmaps_dir`) and MREG
        paths (`mreg_dir`, `func_mreg_bold`).
    tr : float
        Ignored. TR is read from the JSON sidecar in `_load_mreg_data`.
        Preserved for backward compatibility.
    bands : dict[str, tuple[float, float]] or None
        Mapping from band name to `(low_hz, high_hz)` inclusive range.
        If None, `BANDS_DEFAULT` is used.
    mask_path : pathlib.Path or None
        Optional brain mask NIfTI in **MREG** space. If the mask grid/affine
        does not match the MREG reference within `atol`, it is snapped with
        nearest-neighbor interpolation.
    overwrite : bool, default False
        If False, existing band/mean maps are not recomputed.
    atol : float, default 1e-3
        Absolute tolerance for affine equality when checking mask alignment
        to the MREG grid.
    mask_threshold : float, default 0.5
        Threshold applied to the (possibly resampled) brain mask to create a
        boolean mask; values > mask_threshold are treated as inside brain.

    Returns
    -------
    dict[str, Path]
        Mapping `{band_name: band_map_path}` for all successfully written
        bands.

    Files written
    -------------
    - bandmaps/
      `sub-<ID>_space-MREG_band-<BAND>_desc-power_map.nii.gz` (float32)
      `sub-<ID>_space-MREG_desc-meanamp_map.nii.gz` (float32)

    Assumptions / Preconditions
    ---------------------------
    - Space: Outputs are on the MREG grid. If a mask is off-grid, it is
      snapped with nearest-neighbor.
    - Values represent amplitude sums (arbitrary units) across FFT bins
      within each band.

    Warnings
    --------
    - NaNs may be introduced outside the brain if `mask_path` is provided;
      downstream consumers should handle NaNs explicitly.
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
        Must provide `func_mreg_bold`, `mreg_dir`, and `bandmaps_dir`.
    freq_hz : float
        Frequency of interest in Hz; the nearest rFFT bin is used.
    overwrite : bool, default False
        If False, the map is not recomputed when an existing file is found.

    Returns
    -------
    pathlib.Path
        Path to
        `sub-<ID>_space-MREG_freq-{freq_hz:.3f}_desc-amp_map.nii.gz`.

    Files written
    -------------
    - bandmaps/
      `sub-<ID>_space-MREG_freq-{freq_hz:.3f}_desc-amp_map.nii.gz`
      (float32; amplitude |rFFT|/N).

    Assumptions / Preconditions
    ---------------------------
    - Amplitude is |rFFT|/N after voxelwise demean.
    - TR and frequency axis are derived by `_load_mreg_data`.
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
def make_distance_clusters(
    sp,
    dist_map_path,
    klass: str,
    bins=None,
    overwrite=False,
    *,
    atol: float = ATOL_DEFAULT,
):
    """
    Create distance-based cluster labels per class and write a class-tagged
    integer mask on the same grid as the distance map.

    Parameters
    ----------
    sp : SubjectPaths
        Provides `sub` and `clusters_dir`. The distance-map grid/affine is
        used as the reference for outputs.
    dist_map_path : str or pathlib.Path
        Path to a 3D distance NIfTI (float, mm). The file name is inspected
        for a space token (e.g., `space-MNI` or `space-MREG`) which is used
        in output filenames.
    klass : {'arteries', 'veins', 'pvs'}
        Vascular class token used in output filenames.
    bins : sequence[float or str] or None
        Monotonic bin edges (mm). If the last edge is `"max"`, it is
        replaced by the image maximum. If None, `DIST_BINS_DEFAULT`
        is used.
    overwrite : bool, default False
        If False, labels are not recomputed when the output file already
        exists.
    atol : float, default 1e-3
        Absolute tolerance for affine equality when verifying grids against
        other images (reserved for future extension).

    Returns
    -------
    pathlib.Path
        Path to
        `sub-<ID>_space-<SPACE>_class-<klass>_desc-clusters_mask.nii.gz`.

    Files written
    -------------
    - clusters/
      `sub-<ID>_space-<SPACE>_class-<klass>_desc-clusters_mask.nii.gz`
      (int16; -1 for invalid/NaN distance).

    Assumptions / Preconditions
    ---------------------------
    - Distance map values are in mm. NaNs are allowed and digitized to `-1`
      via `labels = -1` wherever distance is non-finite.

    Warnings
    --------
    - Bin edges are normalized and deduplicated; invalid or non-monotonic
      specifications raise `ValueError`.
    """
    if bins is None:
        bins = DIST_BINS_DEFAULT

    dist_path = Path(dist_map_path)
    if not dist_path.exists():
        raise FileNotFoundError(f"Distance map not found: {dist_path}")

    space = _infer_space_from_path(dist_path, default="MREG")

    # Load distance map; treat its grid as the anchor
    dist_img = nib.load(str(dist_path))
    dist = np.asarray(dist_img.get_fdata(), dtype=np.float32)
    ref_aff = dist_img.affine

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
            arr = arr[arr <= (vmax + 1e-6)]
            if arr.size == 0 or arr[-1] < vmax:
                arr = np.concatenate([arr, [np.float32(vmax)]])
        if arr.size < 2 or np.any(np.diff(arr) <= 0):
            raise ValueError(f"distance bin edges invalid: {arr}")
        return arr

    numeric_bins = _normalize_bins(bins, finite_max)

    # Digitize → labels (bin index), NaNs to -1
    valid = np.isfinite(dist)
    labels = np.full(dist.shape, -1, dtype=np.int16)
    if np.any(valid):
        labels[valid] = (
            np.digitize(dist[valid], numeric_bins, right=True).astype(np.int16) - 1
        )

    # Save on the SAME grid as the distance map
    sp.clusters_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(sp.clusters_dir) / deriv_name(
        sp.sub, space, klass, "clusters", "mask"
    )
    if not out_path.exists() or overwrite:
        hdr = dist_img.header.copy()
        hdr.set_data_dtype(np.int16)
        nib.save(nib.Nifti1Image(labels, ref_aff, hdr), str(out_path))
        print(f"[clusters] Saved → {out_path}")
    else:
        print(f"[clusters] [SKIP] Exists: {out_path.name}")

    return out_path


def analyze_binned(sp, labels_path, *, klass: str, bands=None, overwrite=False):
    """
    Compute per-cluster band means for a class (arteries/veins/pvs), run
    ANOVA/Kruskal, and save a class-specific CSV plus figure.

    Parameters
    ----------
    sp : SubjectPaths
        Must provide `bandmaps_dir`, `stats_dir`, and `figures_dir`.
    labels_path : str or pathlib.Path
        Path to an integer label NIfTI in MREG or MNI space. Negative values
        are ignored. The space token is inferred from the filename and used
        in output names.
    klass : {'arteries', 'veins', 'pvs'}
        Anatomical class token used in output filenames.
    bands : dict[str, tuple[float, float]] or None
        Frequency bands (Hz). If None, `BANDS_DEFAULT` is used.
    overwrite : bool, default False
        If False, skip computation when both CSV and figure already exist.

    Returns
    -------
    (pathlib.Path, pathlib.Path)
        `(csv_path, fig_path)` for the binned stats CSV and PNG figure.

    Files written
    -------------
    - stats/
      `sub-<ID>_space-<SPACE>_class-<klass>_desc-binned_stats.csv`
      (semicolon-delimited with columns: band;F;p_anova;H;p_kruskal)
    - figures/
      `sub-<ID>_space-<SPACE>_class-<klass>_desc-binned_bandpower.png`

    Warnings
    --------
    - Statistics are computed only when ≥2 non-empty groups exist; otherwise
      NaNs are recorded in the CSV.
    - Band maps are resampled linearly to the label grid when shape/affine
      mismatches occur.
    """
    if bands is None:
        bands = BANDS_DEFAULT

    space = _infer_space_from_path(Path(labels_path), default="MREG")

    if bands is None:
        bands = BANDS_DEFAULT

    labels_img = nib.load(str(labels_path))
    labels_data = labels_img.get_fdata().astype(int)
    cluster_ids = np.unique(labels_data[np.isfinite(labels_data)])
    cluster_ids = [int(c) for c in cluster_ids if c >= 0]
    cluster_ids = sorted(cluster_ids)

    sp.stats_dir.mkdir(parents=True, exist_ok=True)
    sp.figures_dir.mkdir(parents=True, exist_ok=True)

    stats_path = (
        sp.stats_dir
        / f"{sp.sub}_space-{space}_class-{klass}_desc-binned_stats.csv"
    )
    fig_path = (
        sp.figures_dir
        / f"{sp.sub}_space-{space}_class-{klass}_desc-binned_bandpower.png"
    )

    if stats_path.exists() and fig_path.exists() and not overwrite:
        print(f"[binned] Exists → {klass} ({space}); skipping.")
        return stats_path, fig_path

    cluster_means = {}
    stat_rows = []

    for band in bands:
        band_file = (
            sp.bandmaps_dir
            / f"{sp.sub}_space-{space}_band-{band}_desc-power_map.nii.gz"
        )
        if not band_file.exists():
            warnings.warn(
                f"[binned] Band map not found for '{band}' in space {space}; skipping."
            )
            continue

        # Align band map to labels grid if needed
        band_img = nib.load(str(band_file))
        band_dat = band_img.get_fdata()
        if (band_img.shape != labels_img.shape) or (
            not np.allclose(band_img.affine, labels_img.affine, atol=ATOL_DEFAULT)
        ):
            band_img = resample_from_to(
                band_img, (labels_img.shape, labels_img.affine), order=1
            )
            band_dat = band_img.get_fdata()

        means = []
        samples = []
        for c in cluster_ids:
            vals = band_dat[labels_data == c]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                means.append(np.nan)
            else:
                means.append(float(np.nanmean(vals)))
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

    # CSV (semicolon for EU Excel)
    with open(stats_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(["band", "F", "p_anova", "H", "p_kruskal"])
        for row in stat_rows:
            writer.writerow(row)
    print(f"[binned] Saved → {stats_path}")

    # Figure
    if not fig_path.exists() or overwrite:
        plt.figure(figsize=(8, 5))
        for band, means in cluster_means.items():
            plt.plot(cluster_ids, means, marker="o", label=band)
        plt.xlabel("Distance Cluster")
        plt.ylabel("Mean Band Power (a.u.)")
        plt.title(f"Band Power by Distance Cluster ({klass}, {space})")
        plt.legend()
        plt.grid(True)
        plt.savefig(str(fig_path))
        plt.close()
        print(f"[binned] Saved → {fig_path}")
    else:
        print(f"[binned] [SKIP] Exists: {fig_path.name}")

    return stats_path, fig_path


# -------------------------------------------------------------
# Spectral summaries by cluster
# -------------------------------------------------------------
def cluster_spectra(sp, labels_path, *, klass: str, max_hz=2.0, overwrite=False):
    """
    Compute mean amplitude spectrum per distance cluster for a class and
    write an NPZ file plus a figure.

    Labels may be in MNI or MREG space. When labels are not on the MREG
    grid, they are snapped (nearest-neighbor) onto the MREG reference grid
    for time-series sampling. Output files are tagged with the **original**
    label space (e.g., `space-MNI`).

    Parameters
    ----------
    sp : SubjectPaths
        Must provide `mreg_dir`, `spectra_dir`, `figures_dir`, and MREG
        3D mean map (`sub-<ID>_space-MREG_class-brain_desc-mean_map.nii.gz`).
    labels_path : str or pathlib.Path
        Path to an integer label NIfTI with non-negative cluster IDs. The
        space token is inferred from the filename and used in output names.
    klass : {'arteries', 'veins', 'pvs'}
        Vascular class token used in output filenames.
    max_hz : float, optional
        Upper frequency limit (Hz) for plotting. Default is 2.0.
    overwrite : bool, default False
        If False, skip NPZ/figure creation when they already exist.

    Returns
    -------
    (pathlib.Path, pathlib.Path)
        `(npz_path, fig_path)` for the spectra NPZ and PNG figure.

    Files written
    -------------
    - spectra/
      `sub-<ID>_space-<SPACE>_class-<klass>_desc-cluster_spectra.npz`
      (fields: `freqs`, `spectra`, `cluster_ids`)
    - figures/
      `sub-<ID>_space-<SPACE>_class-<klass>_desc-cluster_spectra.png`

    Assumptions / Preconditions
    ---------------------------
    - Cluster labels are integer-valued; negative IDs are ignored.
    - MREG spectra are computed via `_load_mreg_data` (demeaned, |rFFT|/N).

    Warnings
    --------
    - When labels are resampled to the MREG grid, nearest-neighbor
      interpolation is used to preserve integer cluster IDs.
    """
    labels_path = Path(labels_path)
    label_space = _infer_space_from_path(labels_path, default="MREG")  # e.g., "MNI" or "MREG"

    # Load labels
    lab_img = nib.load(str(labels_path))

    # Load MREG reference (mean map defines target grid/affine)
    mreg_mean = Path(sp.mreg_dir) / deriv_name(sp.sub, "MREG", "brain", "mean", "map")
    mreg_ref = nib.load(str(mreg_mean))
    ref_shape, ref_aff = mreg_ref.shape, mreg_ref.affine

    # If labels are not already on MREG grid, snap them (NN) for sampling
    if (lab_img.shape != ref_shape) or (
        not np.allclose(lab_img.affine, ref_aff, atol=ATOL_DEFAULT)
    ):
        print(f"[spectra] Resampling labels from {label_space} → MREG grid (nearest).")
        lab_img = resample_from_to(lab_img, (ref_shape, ref_aff), order=0)

    labels_data = lab_img.get_fdata().astype(int)
    cluster_ids = np.unique(labels_data[np.isfinite(labels_data)])
    cluster_ids = [int(c) for c in cluster_ids if c >= 0]
    cluster_ids = sorted(cluster_ids)

    # Load MREG time-series (demeaned, |rFFT|/N amplitude and freqs)
    amp, freqs, _, _ = _load_mreg_data(sp)

    spectra_list, valid_clusters = [], []
    flat_labels = labels_data.ravel()
    for c in cluster_ids:
        mask = flat_labels == c
        if not np.any(mask):
            continue
        spec = amp[mask].mean(axis=0)
        spectra_list.append(spec)
        valid_clusters.append(c)

    spectra_arr = (
        np.vstack(spectra_list) if spectra_list else np.empty((0, freqs.size))
    )
    out_space = label_space  # keep original label space for filenames

    # --- Save NPZ ---
    sp.spectra_dir.mkdir(parents=True, exist_ok=True)
    npz_path = (
        sp.spectra_dir
        / f"{sp.sub}_space-{out_space}_class-{klass}_desc-cluster_spectra.npz"
    )
    if (not npz_path.exists()) or overwrite:
        np.savez_compressed(
            str(npz_path),
            freqs=freqs,
            spectra=spectra_arr,
            cluster_ids=np.array(valid_clusters),
        )
        print(f"[spectra] Saved → {npz_path}")
    else:
        print(f"[spectra] [SKIP] Exists: {npz_path.name}")

    # --- Save Figure ---
    sp.figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = (
        sp.figures_dir
        / f"{sp.sub}_space-{out_space}_class-{klass}_desc-cluster_spectra.png"
    )
    if (not fig_path.exists()) or overwrite:
        plt.figure(figsize=(8, 6))
        for c, spec in zip(valid_clusters, spectra_list):
            plt.plot(freqs, spec, label=f"Cluster {int(c)}")
        plt.xlim(0, max_hz)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (a.u.)")
        # Subtitle currently assumes label-space/MREG sampling pairing
        plt.title(
            f"Full Spectrum by Distance Cluster ({klass}, {out_space})\n"
            f"labels in MNI; spectra sampled on MREG grid"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
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
        Must provide `bandmaps_dir`, `stats_dir`, `figures_dir`, and `sub`.
    dist_map_path : str or pathlib.Path
        3D distance map NIfTI in **MREG or MNI** space (float, mm). The space
        token is inferred from the filename and used in output names. The
        class token must appear in the filename ("arteries", "veins", "pvs").
    bands : dict[str, tuple[float, float]] | list[str] | None
        Bands to analyze. If None, defaults to the keys of `BANDS_DEFAULT`.
        If a list of strings is provided, these are treated as band names
        matching existing bandmaps.
    mask_path : str or pathlib.Path or None
        Optional binary mask NIfTI (same grid/affine as `dist_map_path` or
        resampled linearly) to restrict voxels.
    ref_curve : callable or None
        Optional function `ref_curve(x)` returning a reference curve for
        overlay on the distance vs log1p(power) plots.
    ref_label : str, default "ref"
        Legend label for the reference curve.
    overwrite : bool, default False
        If False, skip when the CSV already exists (and return parsed stats).
    max_points_plot : int, default 200000
        Maximum number of voxels sampled for fitting/plotting. When exceeded,
        a random subset is used for performance.

    Returns
    -------
    dict
        Summary with keys:
        - `"class"`: class token.
        - `"per_band"`: dict mapping band name to a dict with fields
          `beta`, `se`, `p`, `r2`, and `n`.

    Files written
    -------------
    - stats/
      `sub-<ID>_space-<SPACE>_class-<CLASS>-desc-continuous_stats.csv`
      (semicolon CSV; columns: band;slope;intercept;r;p;stderr;n)
    - figures/
      `sub-<ID>_space-<SPACE>_class-<CLASS>_band-<BAND>-desc-continuous.png`

    Assumptions / Preconditions
    ---------------------------
    - Distances are in mm; band maps share the same grid/affine as the
      distance map or are resampled linearly to that grid.
    - Bandpower is represented by amplitude-sum maps produced by
      `compute_bandpower_maps`.

    Warnings
    --------
    - Raises `ValueError` when the class token cannot be inferred from the
      distance-map filename.
    - Raises on shape/affine mismatch only indirectly (through resampling
      logic); NaNs are removed before fitting, and downsampling is applied
      for large N.
    - Continuous CSV is read/written with semicolon delimiter for Excel
      compatibility.

    Notes
    -----
    - Linear statistics are derived from OLS; a robust Huber fit is used to
      stabilize the visual overlay.
    """
    # Default bands
    if bands is None:
        bands = list(BANDS_DEFAULT.keys())
    else:
        bands = list(bands)

    # Space from distance file name
    dist_map_path = Path(dist_map_path)
    space = _infer_space_from_path(dist_map_path, default="MREG")

    # Class token from filename
    class_name = None
    fname = dist_map_path.name.lower()
    for cls in ("arteries", "veins", "pvs"):
        if cls in fname:
            class_name = cls
            break
    if class_name is None:
        raise ValueError(f"Unable to determine vessel class from {dist_map_path}")

    # Output CSV (space-tagged)
    out_csv = (
        sp.stats_dir
        / f"{sp.sub}_space-{space}_class-{class_name}-desc-continuous_stats.csv"
    )
    if out_csv.exists() and not overwrite:
        result_dict = {"class": class_name, "per_band": {}}
        with open(out_csv, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                band = row["band"]
                r_val = float(row["r"])
                result_dict["per_band"][band] = {
                    "beta": float(row["slope"]),
                    "se": float(row["stderr"]),
                    "p": float(row["p"]),
                    "r2": r_val**2,
                    "n": int(row["n"]),
                }
        return result_dict

    # Load distance map
    dist_img = nib.load(str(dist_map_path))
    dist = dist_img.get_fdata().astype(np.float32)

    # Optional mask (snap to dist grid if needed)
    if mask_path is not None and Path(mask_path).exists():
        mask_img = nib.load(str(mask_path))
        if (mask_img.shape != dist_img.shape) or (
            not np.allclose(mask_img.affine, dist_img.affine, atol=ATOL_DEFAULT)
        ):
            mask_img = resample_from_to(
                mask_img, (dist_img.shape, dist_img.affine), order=0
            )
        mask = mask_img.get_fdata().astype(bool)
    else:
        mask = np.ones_like(dist, dtype=bool)

    result_dict = {"class": class_name, "per_band": {}}
    stats_rows = []

    for band in bands:
        band_path = (
            sp.bandmaps_dir
            / f"{sp.sub}_space-{space}_band-{band}_desc-power_map.nii.gz"
        )
        if not band_path.exists():
            print(
                f"[continuous] [WARN] Missing band map for '{band}' in {space}; skipping"
            )
            continue

        bm_img = nib.load(str(band_path))
        bandmap = bm_img.get_fdata().astype(np.float32)

        # Align band map to dist grid when needed
        if (bm_img.shape != dist_img.shape) or (
            not np.allclose(bm_img.affine, dist_img.affine, atol=ATOL_DEFAULT)
        ):
            bm_img = resample_from_to(
                bm_img, (dist_img.shape, dist_img.affine), order=1
            )
            bandmap = bm_img.get_fdata().astype(np.float32)

        # Valid voxels
        valid_mask = np.isfinite(dist) & np.isfinite(bandmap) & mask
        d_flat = dist[valid_mask].ravel().astype(np.float32)
        p_flat = bandmap[valid_mask].ravel().astype(np.float32)
        N = d_flat.size
        if N == 0:
            print(
                f"[continuous] [WARN] No valid voxels for band '{band}'; skipping"
            )
            continue

        # Downsample for stability/plot speed
        if N > max_points_plot:
            rng = np.random.default_rng(0)
            idx = rng.choice(N, size=max_points_plot, replace=False)
            d_sub = d_flat[idx]
            p_sub = p_flat[idx]
        else:
            d_sub = d_flat
            p_sub = p_flat

        # Transform for regression
        x = d_sub.astype(np.float32)
        y = np.log1p(p_sub).astype(np.float32)

        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        n = x.size
        uniq_x = np.unique(x).size
        if n < 10 or uniq_x < 3:
            print(
                f"[continuous] [WARN] {class_name}/{band} ({space}): too few points "
                f"(n={n}, unique_x={uniq_x}); skipping"
            )
            continue

        # Robust + OLS linear (no spline)
        lin = _fit_linear_huber(x, y)

        # Plot: distance-density and linear line
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
        ax.set_title(f"{sp.sub} – {class_name}, band = {band} ({space})")
        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])

        # Linear overlay
        x_line = np.linspace(x.min(), x.max(), 300)
        X_line_lin = sm.add_constant(x_line)
        y_lin = lin["ols"].predict(X_line_lin)
        ax.plot(x_line, y_lin, lw=2, label="Linear")

        # Optional reference overlay
        if ref_curve is not None:
            y_ref = ref_curve(x_line)
            ax.plot(
                x_line, y_ref, color="k", ls="--", lw=1.5, label=ref_label
            )

        ax.legend()

        fig_path = (
            sp.figures_dir
            / f"{sp.sub}_space-{space}_class-{class_name}_band-{band}-desc-continuous.png"
        )
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        print(f"[continuous] Saved → {fig_path}")
        plt.close(fig)

        # Record stats (linear)
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
    *,
    # NEW: let caller point to a radii file or choose its space
    radii_path: Path | None = None,
    radii_space: str = "MREG",   # keeps old default behavior
):
    """
    Regress log1p(band power) on radius (mm) at centerline voxels.

    This function samples bandpower maps at voxels with valid radii on the grid
    defined by a radius map (MREG or MNI), fits linear models
    `log1p(band power) ~ radius_mm` per band, and writes a CSV summary and a
    multi-panel figure (one panel per band).

    Parameters
    ----------
    sp : SubjectPaths
        Subject-scoped paths and identifiers (e.g., `radii_dir`, `bandmaps_dir`,
        `stats_dir`, `figures_dir`, `sub`).
    klass : {'arteries', 'veins', 'pvs'}
        Vascular class token used in filenames and log messages.
    bands : dict
        Mapping from band name to band definition (e.g., `(fmin, fmax)` in Hz).
        Only the keys are used here; they must match the band map filenames and
        any provided `band_paths`.
    band_paths : dict[str, pathlib.Path] or None, optional
        Optional explicit mapping from band name to band map path. When not
        provided for a given band, a canonical filename is constructed as
        `sub-<ID>_space-<SPACE>_band-<BAND>_desc-power_map.nii.gz` in
        `sp.bandmaps_dir`, where `<SPACE>` is `radii_space`.
    mask_path : pathlib.Path or None, optional
        Optional brain mask NIfTI. If provided, the mask is resampled (nearest)
        onto the radii grid when shape/affine mismatch exceeds `atol`, and only
        voxels inside the mask are included in the analysis.
    overwrite : bool, optional
        If False (default) and both the CSV and figure already exist, the
        function skips computation and returns the existing paths.
    max_points_plot : int, optional
        Maximum number of points used for scatter plotting per band. If more
        valid voxels are present, a random subset of size `max_points_plot` is
        drawn for visualization; regression is always performed on all points.
    atol : float, optional
        Absolute tolerance for affine equality when checking alignment between
        the brain mask and radii grid, and between band maps and radii grid.
    mask_threshold : float, optional
        Threshold applied to the (possibly resampled) mask; values strictly
        greater than `mask_threshold` are treated as in-brain (`True`).
    radii_path : pathlib.Path or None, optional
        Optional explicit path to a radius map NIfTI. If None, a canonical
        filename is constructed as
        `sub-<ID>_space-<radii_space>_class-<klass>_desc-radius_map.nii.gz`
        under `sp.radii_dir`.
    radii_space : str, optional
        Space tag for the radii map and derived outputs (e.g., `"MREG"` or
        `"MNI"`). When `radii_path` is provided, `radii_space` is updated by
        `_infer_space_from_path` if a space token can be inferred from the
        filename. Default is `"MREG"` (legacy behavior).

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path] or None
        `(csv_path, fig_path)` if analysis is performed or existing outputs are
        found; None if radii are missing or no valid in-brain radii are
        available.

    Files written
    -------------
    - stats/
      `sub-<ID>_space-<SPACE>_class-<klass>_desc-radius_vs_power.csv`
      (semicolon-delimited with columns:
      `band;n;slope;intercept;p_value;r_value;r_squared;spearman_r;spearman_p;mean_radius_mm`)
    - figures/
      `sub-<ID>_space-<SPACE>_class-<klass>_desc-radius_vs_power.png`
      (one scatter + regression curve per band, panels arranged horizontally)

    Assumptions / Preconditions
    ---------------------------
    - Radii map:
      - Defined on a single 3D grid (MREG or MNI) with affine `ref_aff` and
        shape `ref_shape`.
      - Encodes radius in millimeters; valid centerline voxels have `rad > 0`
        and failed fits are ≤ 0 (typically -1).
    - Band maps:
      - Stored as float NIfTIs in the same space as `radii_space` or
        resample-able to that grid; amplitude units are arbitrary (a.u.).
      - One band map per key in `bands`.
    - Mask:
      - If provided, is a brain mask (0/1-like) that can be resampled to the
        radii grid via nearest-neighbor interpolation.

    Warnings
    --------
    - Band maps that do not align with the radii grid within `atol` are
      resampled with **linear** interpolation (order=1), which may slightly
      smooth values.
    - The brain mask, when provided, is resampled with **nearest-neighbor**
      interpolation to preserve binary labels.
    - At least 10 valid samples are required per band; bands with fewer valid
      points are skipped.
    - The CSV uses a semicolon delimiter for spreadsheet compatibility.

    Notes
    -----
    - Linear regressions are performed on `(radius_mm, log1p(band power))`,
      but the scatter plots and regression curves are displayed in raw band
      power units on the y-axis.
    - Pearson statistics (slope, intercept, r, p, stderr) are computed via
      `scipy.stats.linregress`, while Spearman rank correlation is computed
      on `(radius_mm, band power)` using `scipy.stats.spearmanr`.
    """
    # --- resolve radii path & space ---
    if radii_path is None:
        rad_path = Path(sp.radii_dir) / deriv_name(sp.sub, radii_space, klass, "radius", "map")
    else:
        rad_path = Path(radii_path)
        # infer space tag from filename if possible
        radii_space = _infer_space_from_path(rad_path, default=radii_space)

    if not rad_path.exists():
        print(f"[radius] [SKIP] No radii for {klass} in {radii_space}: {rad_path.name}")
        return None

    # --- output targets (space-tagged) ---
    csv_path = Path(sp.stats_dir)   / f"{sp.sub}_space-{radii_space}_class-{klass}_desc-radius_vs_power.csv"
    fig_path = Path(sp.figures_dir) / f"{sp.sub}_space-{radii_space}_class-{klass}_desc-radius_vs_power.png"

    if csv_path.exists() and fig_path.exists() and not overwrite:
        print(f"[radius] [SKIP] Exists (CSV+FIG) for {klass} in {radii_space}: {csv_path.name}, {fig_path.name}")
        return csv_path, fig_path

    # --- load radii ---
    rad_img = nib.load(str(rad_path))
    rad = np.asarray(rad_img.get_fdata(), dtype=np.float32)
    ref_shape, ref_aff = rad_img.shape, rad_img.affine

    # --- optional brain mask (snap to radii grid) ---
    brain = None
    if mask_path is not None and Path(mask_path).exists():
        m = nib.load(str(mask_path))
        if (m.shape != ref_shape) or (not np.allclose(m.affine, ref_aff, atol=atol)):
            m = resample_from_to(m, (ref_shape, ref_aff), order=0)
        brain = (m.get_fdata() > float(mask_threshold)).astype(bool)

    # --- valid centerline points ---
    valid = rad > 0
    if brain is not None:
        valid &= brain
    if not np.any(valid):
        print(f"[radius] [SKIP] No valid in-brain radii for {klass} in {radii_space}")
        return None

    ii, jj, kk = np.where(valid)
    radii_vec = rad[ii, jj, kk]

    # --- set up plotting ---
    rows = [("band","n","slope","intercept","p_value","r_value","r_squared",
             "spearman_r","spearman_p","mean_radius_mm")]
    fig, axs = plt.subplots(1, len(bands), figsize=(4.2 * len(bands), 4.0), squeeze=False)
    axs = axs[0]

    # --- loop bands ---
    for bidx, (band_name, _) in enumerate(bands.items()):
        # 1) resolve band map (prefer explicit, else discover space-matched)
        if band_paths and band_name in band_paths:
            bpath = Path(band_paths[band_name])
        else:
            # canonical name:
            candidate = Path(sp.bandmaps_dir) / f"{sp.sub}_space-{radii_space}_band-{band_name}_desc-power_map.nii.gz"
            if candidate.exists():
                bpath = candidate
            else:
                # fallback glob (more permissive)
                hits = list(Path(sp.bandmaps_dir).glob(
                    f"{sp.sub}_space-{radii_space}_*{band_name}*power_map.nii.gz"))
                if not hits:
                    print(f"[radius] [SKIP] Band '{band_name}' map not found in space {radii_space}")
                    continue
                bpath = hits[0]

        bimg = nib.load(str(bpath))
        bdat = np.asarray(bimg.get_fdata(), dtype=np.float32)

        # 2) align band map to radii grid
        if (bimg.shape != ref_shape) or (not np.allclose(bimg.affine, ref_aff, atol=atol)):
            bimg = resample_from_to(bimg, (ref_shape, ref_aff), order=1)
            bdat = np.asarray(bimg.get_fdata(), dtype=np.float32)

        # 3) sample band values at centerline
        power_vec = bdat[ii, jj, kk]
        good = np.isfinite(power_vec)
        x = radii_vec[good]
        y = power_vec[good]
        n = x.size
        if n < 10:
            print(f"[radius] [SKIP] Too few points for {band_name} ({n}) in {radii_space}")
            continue

        # 4) regression (log1p power); Spearman on raw (keep as before)
        y_log = np.log1p(y)
        lr = linregress(x, y_log)  # slope, intercept, rvalue, pvalue, stderr
        spearman_r, spearman_p = spearmanr(x, y, nan_policy="omit")

        rows.append((
            band_name, int(n), float(lr.slope), float(lr.intercept),
            float(lr.pvalue), float(lr.rvalue), float(lr.rvalue**2),
            float(spearman_r), float(spearman_p), float(np.mean(x))
        ))

        # 5) plot
        ax = axs[bidx]
        if n > max_points_plot:
            sel = np.random.default_rng(0).choice(n, size=max_points_plot, replace=False)
            xx, yy = x[sel], y[sel]
        else:
            xx, yy = x, y
        ax.scatter(xx, yy, s=2, alpha=0.3)
        xs = np.linspace(xx.min(), xx.max(), 200)
        ys = np.exp(lr.intercept + lr.slope * xs) - 1.0
        ax.plot(xs, ys, linewidth=2)
        ax.set_title(band_name)
        ax.set_xlabel("Radius (mm)")
        ax.set_ylabel("Band power")

    # --- write outputs (respect overwrite) ---
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        for r in rows:
            w.writerow(r)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"[radius] Saved CSV → {csv_path}")
    print(f"[radius] Saved FIG → {fig_path}")
    return csv_path, fig_path


def _fit_linear_huber(x, y):
    """
    Fit a robust and OLS linear model `y ~ x` and return summary stats.

    A Huber regressor is used to obtain a robust line for visualization,
    while an ordinary least squares (OLS) model provides inferential
    quantities such as p-values, standard errors, and AIC.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Predictor values (e.g., distance or radius), finite after any
        preprocessing.
    y : ndarray, shape (n,)
        Response values (e.g., log1p(band power)), finite after any
        preprocessing.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``type`` : str
            Model type (always ``"linear"``).
        - ``slope`` : float
            OLS slope coefficient.
        - ``intercept`` : float
            OLS intercept.
        - ``stderr`` : float
            Standard error of the slope (OLS).
        - ``p`` : float
            Two-sided p-value for the slope (OLS).
        - ``r2`` : float
            Coefficient of determination (R²) from OLS.
        - ``aic`` : float
            Akaike Information Criterion for the OLS fit.
        - ``ols`` : statsmodels.regression.linear_model.RegressionResults
            The fitted OLS model object.
        - ``yhat`` : ndarray, shape (n,)
            Robust-predicted values from the Huber regressor.

    Notes
    -----
    - No intercept centering is applied beyond the implicit constant term in
      the OLS design matrix (`sm.add_constant`).
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
