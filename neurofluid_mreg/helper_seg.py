# SPDX-License-Identifier: MIT
"""
helper_seg.py
--------------
Segmentation helpers for Neurofluid–MREG: array-only utilities used by `seg.py`
and related wrappers. This module provides I/O helpers, preprocessing routines
for MRV/R2*, Frangi-based vesselness, thresholding/post-processing, and compact
end-to-end segmentation entry points that return arrays (no resampling here).

Pipeline steps
--------------
1. Preprocessing
   - MRV/R2*: normalize → percentile window → light Gaussian
   - hT2w (PVS): window to [0, 1] → band-threshold → morphology (holes/size)
2. Vesselness / feature filtering (Frangi, dual-bank fusion)
3. Thresholding / post-processing / skeletonization (Lee 3D)
4. I/O helpers (load/save NIfTI, BIDS-like path construction)

Inputs / Outputs
----------------
Inputs  : NIfTI volumes (via paths) and/or ndarrays (float), no resampling.
Outputs : ndarrays (vesselness float, masks/skeletons bool) and filenames
          (for `make_output_paths`); optional NIfTI writing via `save_nifti`.

Files written
-------------
- None directly, except `save_nifti(out_path)` as provided by the caller.
- Filenames created by `make_output_paths` follow BIDS-like tokens:
  `sub-<ID>[_ses-<LABEL>]_space-<SPACE>_class-<CLASS>_desc-<DESC>_<SUFFIX>.nii.gz`

Assumptions / Preconditions
---------------------------
- Spaces: Functions here operate purely on arrays in **image space**. No
  resampling or space changes are performed.
- Shapes/dtypes: Processes 2D/3D arrays; typical dtype=float32 for intensity
  operations. Vesselness is normalized to [0, 1] where stated.
- BIDS naming: `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-<DESC>_<SUFFIX>.nii.gz`

Warnings
--------
- Affines are not checked in array operations here; NIfTI I/O passes through
  the provided affine/header unchanged.
- Skeletonization uses `skimage.morphology.skeletonize(method="lee")`, which
  supports **true 3D** thinning when given a 3D mask.

Public API
----------
- load_nifti, save_nifti, make_output_paths
- preprocess_mrv_for_vesselness, pvs_preprocess_hT2w
- combine_echoes_te_weighted, compute_r2star_map
- frangi_vesselness, segment_pvs_frangi3d
- intensity_gate, threshold_vesselness, iterative_hysteresis
- segment_vessels, segment_veins_dual_scale
"""

from __future__ import annotations

import numpy as np
import re
import nibabel as nib
from pathlib import Path
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import frangi, apply_hysteresis_threshold, threshold_multiotsu
from skimage.morphology import ( white_tophat, black_tophat, ball, disk, 
    remove_small_objects, skeletonize, )
from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_fill_holes 
from skimage.measure import label
import SimpleITK as sitk
from .transforms import register_t1_to_hT2w
from typing import Tuple



# -------------------------------------------------------------
# I/O HELPERS (BIDS naming, paths)
# -------------------------------------------------------------
def load_nifti(path: Path):
    """
    Load a NIfTI image.

    Parameters
    ----------
    path : Path
        Path to a `.nii`/`.nii.gz` file.

    Returns
    -------
    tuple
        (data, affine, header) where
        - data : ndarray, shape (X, Y, Z) or (X, Y, Z, T)
        - affine : ndarray, shape (4, 4)
        - header : nibabel header

    Files written
    -------------
    - None.

    Assumptions / Preconditions
    ---------------------------
    - File exists and is a readable NIfTI.
    - Data are read with nibabel defaults (often float64 from `get_fdata()`).

    Warnings
    --------
    - Dtype is determined by nibabel; downstream code usually casts to float32.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or is unreadable.
    """
    nii = nib.load(str(path))
    return nii.get_fdata(), nii.affine, nii.header


def save_nifti(data: np.ndarray, affine, header, out_path: Path):
    """
    Save a numpy array as a NIfTI image.

    Parameters
    ----------
    data : ndarray
        Image data to write. Dtype is preserved unless changed upstream.
    affine : ndarray, shape (4, 4)
        Affine matrix to store.
    header : nibabel header
        Header to attach to the output file.
    out_path : Path
        Destination path (directories must exist).

    Returns
    -------
    None

    Files written
    -------------
    - As provided by `out_path`, typically under:
      derivatives/neurofluid-mreg/sub-<ID>/.../*.nii.gz

    Assumptions / Preconditions
    ---------------------------
    - Parent directory exists (this function does not create parents).

    Raises
    ------
    OSError
        If writing fails (permissions, disk issues).
    """
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, str(out_path))
    print(f"[seg] Saved → {Path(out_path).resolve()}")



def make_output_paths(
    input_path: Path,
    out_dir: Path,
    class_name: str,
    *,
    sub_override: str | None = None,
    space_override: str | None = None,
):
    """
    Construct BIDS-like output paths for vesselness, skeleton, and main mask.

    Parameters
    ----------
    input_path : Path
        Input NIfTI path. Tokens `sub-*`, optional `ses-*`, and `space-*`
        are parsed from the filename; if `space-*` is absent, pass
        `space_override`.
    out_dir : Path
        Destination directory (created if missing).
    class_name : str
        One of {'arteries', 'veins', 'pvs'}.
    sub_override : str or None, optional
        Override for the `sub-*` token (e.g., pass `sp.sub`).
    space_override : str or None, optional
        Override for the `space-*` token (e.g., "TOF", "MRV", "hT2w").

    Returns
    -------
    tuple[Path, Path, Path]
        (vesselness_file, skeleton_file, mask_file), filenames only.

    Files written
    -------------
    - None (filenames only). Patterns:
      `sub-<ID>[_ses-<LABEL>]_space-<SPACE>_class-<CLASS>_desc-vesselness_map.nii.gz`
      `sub-<ID>[_ses-<LABEL>]_space-<SPACE>_class-<CLASS>_desc-skeleton_mask.nii.gz`
      `sub-<ID>[_ses-<LABEL>]_space-<SPACE>_class-<CLASS>_desc-main_mask.nii.gz`

    Assumptions / Preconditions
    ---------------------------
    - `out_dir` exists or is creatable.
    - A `space-*` token must be present, or `space_override` is provided.

    Raises
    ------
    ValueError
        If no `space-*` token is found and `space_override` is not given.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = Path(input_path).name

    # Extract subject/session/space from filename or path
    sub_part = None
    ses_part = None
    space_part = None

    for token in fname.split("_"):
        if token.startswith("sub-"):
            sub_part = token
        elif token.startswith("ses-"):
            ses_part = token
        elif token.startswith("space-"):
            space_part = token

    if sub_override:
        sub_part = sub_override
    if space_override:
        space_part = f"space-{space_override}"

    if sub_part is None:
        for part in Path(input_path).parts:
            if part.startswith("sub-"):
                sub_part = part
                break

    if ses_part is None:
        for part in Path(input_path).parts:
            if part.startswith("ses-"):
                ses_part = part
                break

    # Do not default to MNI; require explicit space or override.
    if space_part is None:
        raise ValueError(
            "make_output_paths: space-<...> not found; pass space_override='TOF'/'MRV'/'hT2w'"
        )

    base_parts = [sub_part if sub_part else "sub-unknown"]
    if ses_part:
        base_parts.append(ses_part)
    base_parts.append(space_part)
    base_parts.append(f"class-{class_name}")
    base = "_".join(base_parts)

    vesselness_file = out_dir / f"{base}_desc-vesselness_map.nii.gz"
    skeleton_file = out_dir / f"{base}_desc-skeleton_mask.nii.gz"
    mask_file = out_dir / f"{base}_desc-main_mask.nii.gz"
    return vesselness_file, skeleton_file, mask_file



# -------------------------------------------------------------
# MRV PREPROCESSING (denoise, normalize, CLAHE)
# -------------------------------------------------------------
def preprocess_mrv_for_vesselness(
    vol: np.ndarray,
    *,
    # Pre-filter (gentle denoise that preserves ridges)
    pre_gaussian_sigma: float = 0.6,  # try 0.6–0.7 for 0.5 mm iso
    # “Auto contrast” (ITK-SNAP-like percentiles on the whole volume)
    do_auto_window: bool = True,
    auto_p_low: float = 0.5,
    auto_p_high: float = 99.5,
    # Post-filter (one choice only → Gaussian)
    post_gaussian_sigma: float = 0.5,  # light smooth on vesselness input
    threshold_frac: float | None = None,
    remove_small_min_vox: int = 0,
    remove_small_conn: int = 1,
) -> np.ndarray:
    """
    Preprocess MRV/R2*-like volumes for vesselness computation (float [0, 1]).

    Parameters
    ----------
    vol : ndarray
        Input magnitude-like volume; 2D/3D, arbitrary dtype.
    pre_gaussian_sigma : float, optional
        Standard deviation for pre-denoise Gaussian filter. Set 0 to disable.
        Default: 0.6.
    do_auto_window : bool, optional
        If True, percentile window the volume using ``auto_p_low/high`` and
        rescale to [0, 1]. Default: True.
    auto_p_low, auto_p_high : float, optional
        Low/high percentiles for auto windowing. Default: 0.5, 99.5.
    post_gaussian_sigma : float, optional
        Standard deviation for post-filter Gaussian smoothing. Set 0 to
        disable. Default: 0.5.
    threshold_frac : float or None, optional
        If set, keep voxels ``>= threshold_frac`` and zero the rest.
    remove_small_min_vox : int, optional
        Minimum connected-component size (voxels) to keep after thresholding.
        Default: 0 (no removal).
    remove_small_conn : int, optional
        Connectivity for small-object removal (1, 2, or 3 for 3D). Default: 1.

    Returns
    -------
    ndarray
        Float32 array in [0, 1], same shape as input.

    Files written
    -------------
    - None.

    Assumptions / Preconditions
    ---------------------------
    - Operates purely in image space (no resampling).
    - Input values can be arbitrary scale; normalization is handled internally.

    Warnings
    --------
    - Aggressive percentile windowing can amplify noise and truncate bright
      structures.
    - Gaussian smoothing may broaden thin vessels if set too high.
    """
    v = vol.astype(np.float32, copy=False)

    # Normalize to [0, 1] for stable behavior
    vmin, vmax = float(v.min()), float(v.max())
    if vmax > vmin:
        v = (v - vmin) / (vmax - vmin)
    v = np.clip(v, 0, 1)

    # 1) Gaussian pre-denoise
    if pre_gaussian_sigma and pre_gaussian_sigma > 0:
        v = gaussian_filter(v, sigma=float(pre_gaussian_sigma)).astype(np.float32)

    # 2) Auto window (percentile clip → rescale to [0, 1])
    if do_auto_window:
        lo, hi = np.percentile(v, [float(auto_p_low), float(auto_p_high)])
        if hi > lo:
            v = np.clip(v, lo, hi)
            v = (v - lo) / (hi - lo)
        v = np.clip(v, 0, 1)

    # 4) Gaussian post-filter (kept over median to avoid ridge flattening)
    if post_gaussian_sigma and post_gaussian_sigma > 0:
        v = gaussian_filter(v, sigma=float(post_gaussian_sigma)).astype(np.float32)

    # 5) Optional: threshold mask + remove small objects → reapply mask
    if threshold_frac is not None:
        thr = float(threshold_frac)
        mask = v >= thr
        if remove_small_min_vox and remove_small_min_vox > 0:
            mask = remove_small_objects(
                mask,
                min_size=int(remove_small_min_vox),
                connectivity=int(remove_small_conn),
            )
        # keep float image and greyscale detail; just zero out rejected voxels
        v = (v * mask.astype(np.float32, copy=False)).astype(np.float32, copy=False)
        v = np.clip(v, 0, 1)

    return np.clip(v, 0, 1).astype(np.float32)


def intensity_gate(
    img: np.ndarray,
    *,
    method: str = "fraction_max",   # {"fraction_max","fraction_p99","percentile"}
    fraction: float = 0.20,         # for fraction_* methods (0..1)
    percentile: float = 98.5,       # for method="percentile" (0..100)
    robust_p: float = 99.0,         # cap for "fraction_p99"
    brain_mask: np.ndarray | None = None,
    remove_small_min_vox: int = 0,
    connectivity: int = 2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Gate intensities by keeping only the brightest voxels and zeroing the rest.

    Parameters
    ----------
    img : ndarray
        Input image; arbitrary dtype cast to float32.
    method : {'fraction_max', 'fraction_p99', 'percentile'}, optional
        Thresholding scheme:
        - 'fraction_max': ``thr = fraction * max(img)``
        - 'fraction_p99': ``thr = fraction * P(robust_p)`` (falls back to max)
        - 'percentile' : ``thr = P(percentile)``
        Default: 'fraction_max'.
    fraction : float, optional
        Fraction in [0, 1] for 'fraction_*' methods. Default: 0.20.
    percentile : float, optional
        Percentile in [0, 100] for 'percentile' method. Default: 98.5.
    robust_p : float, optional
        Percentile used to estimate the robust maximum for 'fraction_p99'.
        Default: 99.0.
    brain_mask : ndarray or None, optional
        If provided, compute statistics over ``img[brain_mask]`` only.
    remove_small_min_vox : int, optional
        Remove connected components smaller than this (voxels). Default: 0.
    connectivity : int, optional
        Connectivity used for small-object removal. Default: 2.

    Returns
    -------
    mask : ndarray, dtype=bool
        Binary mask of kept voxels.
    masked_img : ndarray, dtype=float32
        ``img * mask`` with rejected voxels set to 0.
    thr_value : float
        Absolute threshold applied to ``img``.

    Files written
    -------------
    - None.

    Assumptions / Preconditions
    ---------------------------
    - Input contains finite values; NaNs are not handled explicitly.

    Warnings
    --------
    - Very low dynamic range may collapse the mask to all-zeros or all-ones.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported options.

    Notes
    -----
    - When ``brain_mask`` is provided, threshold statistics are computed on
      the masked region only; gating is still applied to the full image.
    """
    v = img.astype(np.float32, copy=False)

    # Choose statistic domain (whole image or brain)
    sample = v if brain_mask is None else v[brain_mask]

    if method == "fraction_max":
        vmax = float(sample.max())
        thr = fraction * vmax
    elif method == "fraction_p99":
        p99 = float(np.percentile(sample, robust_p)) if sample.size else 0.0
        thr = fraction * (p99 if p99 > 0 else float(sample.max()))
    elif method == "percentile":
        thr = float(np.percentile(sample, percentile)) if sample.size else 0.0
    else:
        raise ValueError(f"Unknown method: {method}")

    mask = v >= thr

    if remove_small_min_vox and remove_small_min_vox > 0:
        mask = remove_small_objects(
            mask, min_size=int(remove_small_min_vox), connectivity=int(connectivity)
        )

    masked = (v * mask.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    return mask, masked, float(thr)



def combine_echoes_te_weighted(
    echo_images: list[np.ndarray],
    echo_times: list[float],
    brain_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Combine multi-echo magnitude images using a TE-weighted geometric mean.

    Parameters
    ----------
    echo_images : list of ndarray
        Magnitude images for each echo; same shape; arbitrary dtype.
    echo_times : list of float
        Echo times (ms or s; consistent units).
    brain_mask : ndarray, optional
        Boolean mask for z-score normalization region; if None, use positive
        voxels.

    Returns
    -------
    ndarray
        Combined image (float32), same shape as inputs, z-score normalized
        within `brain_mask` (or positive voxels if None).

    Files written
    -------------
    - None.

    Assumptions / Preconditions
    ---------------------------
    - `len(echo_images) == len(echo_times) >= 1`.
    - Positive intensities expected for log/exp stability.

    Raises
    ------
    ValueError
        If `brain_mask` shape mismatches the images.
    """
    n_echoes = len(echo_images)
    assert (
        n_echoes == len(echo_times) and n_echoes > 0
    ), "Number of echo images must match number of echo times and be at least 1."
    if n_echoes == 1:
        combined = echo_images[0].astype(np.float32)
        return combined

    echo_times = np.array(echo_times, dtype=float)
    echo_images = [img.astype(np.float32) for img in echo_images]

    # Compute weights for each echo based on TE fraction of total
    te_sum = echo_times.sum()
    weights = echo_times / te_sum

    # Weighted geometric mean: exp(sum(w_i * log(I_i)))
    log_images = []
    for img in echo_images:
        log_images.append(np.log(np.clip(img, 1e-6, None)))
    log_images = np.stack(log_images, axis=0)
    weighted_log = np.tensordot(weights.astype(np.float32), log_images, axes=(0, 0))
    combined = np.exp(weighted_log).astype(np.float32)

    # Z-score normalization within brain region
    if brain_mask is not None:
        bm = brain_mask.astype(bool)
        if bm.shape != combined.shape:
            raise ValueError(
                f"brain_mask shape {bm.shape} != image shape {combined.shape}"
            )
        brain_vals = combined[bm]
    else:
        brain_vals = combined[combined > 0]

    if brain_vals.size > 0:
        mean_val = brain_vals.mean()
        std_val = brain_vals.std()
        if std_val > 0:
            combined = (combined - mean_val) / std_val

    return combined.astype(np.float32)


def compute_r2star_map(
    echo_images: list[np.ndarray],
    echo_times: list[float],
    brain_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Compute R2* (1/T2*) from multi-echo magnitudes via mono-exponential fit.

    Parameters
    ----------
    echo_images : list of ndarray
        Magnitude images for each echo; same shape; arbitrary dtype.
    echo_times : list of float
        Echo times in consistent units (e.g., ms). Output units are inverse of
        TE units (e.g., 1/ms if TE in ms).
    brain_mask : ndarray, optional
        Boolean mask restricting the fit; voxels outside set to 0.

    Returns
    -------
    ndarray
        R2* map (float32), same shape as inputs. Non-brain/low-signal voxels
        set to 0.

    Files written
    -------------
    - None.

    Assumptions / Preconditions
    ---------------------------
    - At least two echoes (`len(echo_images) == len(echo_times) >= 2`).
    - Positive intensities; log is stabilized by clipping at 1e-6.

    Warnings
    --------
    - If all TEs are identical, the fit degenerates and returns zeros.

    Notes
    -----
    - Linear least squares on log-intensity vs TE; slope = -R2*.
    """
    n_echoes = len(echo_images)
    assert (
        n_echoes == len(echo_times) and n_echoes >= 2
    ), "R2* computation requires at least two echoes."
    echo_times = np.array(echo_times, dtype=float)

    r2star_map = np.zeros_like(echo_images[0], dtype=np.float32)

    images_stack = np.stack(
        [img.astype(np.float32) for img in echo_images], axis=0
    )
    log_images_stack = np.log(np.clip(images_stack, 1e-6, None))

    if brain_mask is not None:
        mask = brain_mask.astype(bool)
    else:
        mask = (images_stack > 0).any(axis=0)

    if not np.any(mask):
        return r2star_map

    N = float(n_echoes)
    sum_TE = echo_times.sum()
    sum_TE2 = (echo_times**2).sum()

    log_vals = log_images_stack[:, mask]
    sum_log = log_vals.sum(axis=0)
    sum_TE_log = (echo_times[:, None] * log_vals).sum(axis=0)

    denom = (N * sum_TE2 - (sum_TE**2))
    if denom == 0:
        slopes = np.zeros_like(sum_log)
    else:
        slopes = (N * sum_TE_log - sum_TE * sum_log) / denom

    r2_vals = -slopes
    r2_vals = np.maximum(r2_vals, 0.0)

    r2star_map[mask] = r2_vals.astype(np.float32)
    return r2star_map


# -------------------------------------------------------------
# PVS PREPROCESSING (hT2w-specific; window→band-threshold→morph)
# -------------------------------------------------------------
def pvs_preprocess_hT2w(
    hT2w_nii: Path,
    *,
    # Finalized choices (defaults reflect what you said you use)
    clip_low: float = 1.0,        # percentile
    clip_high: float = 99.0,      # percentile
    thr_low: float = 0.15,        # keep-band lower on [0,1]
    thr_high: float = 0.60,       # keep-band upper on [0,1]
    fill_holes_on: bool = True,
    min_size_vox: int = 30,       # drop CCs smaller than this
    max_size_vox: int = 5000,     # drop CCs larger than this
    # QC artifacts (optional)
    write_artifacts: bool = False,
    out_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, nib.Nifti1Header]:
    """
    Preprocess heavy T2-weighted (hT2w) data for PVS via windowing, band
    thresholding, and compact morphology.

    Parameters
    ----------
    hT2w_nii : Path
        Path to hT2w NIfTI. Affine and header are propagated unchanged.
    clip_low, clip_high : float, optional
        Percentiles used for robust windowing, then rescaled to [0, 1].
    thr_low, thr_high : float, optional
        Absolute keep band on the windowed image: voxels with
        ``thr_low ≤ I ≤ thr_high`` are kept.
    fill_holes_on : bool, optional
        If True, fill 3D holes inside each connected component (does not
        connect separate components). Default: True.
    min_size_vox : int, optional
        Remove connected components smaller than this size (in voxels).
        Default: 30.
    max_size_vox : int, optional
        Remove connected components larger than this size (in voxels).
        Default: 5000.
    write_artifacts : bool, optional
        If True, write the windowed image (masked) and the coarse mask to
        ``out_dir / "masks"``. Default: False.
    out_dir : Path or None, optional
        Destination directory when ``write_artifacts=True``.

    Returns
    -------
    pre_img : ndarray, dtype=float32
        Windowed image in [0, 1] with background suppressed by the final
        mask (i.e., ``pre_img = windowed * mask``).
    mask : ndarray, dtype=uint8
        Final binary mask after keep band and morphology ({0, 1}).
    affine : ndarray, shape (4, 4)
        Affine copied from the input NIfTI.
    header : nib.Nifti1Header
        Header copied from the input NIfTI.

    Files written
    -------------
    sub-*_space-hT2w_class-pvs_desc-preproc_map.nii.gz
    sub-*_space-hT2w_class-pvs_desc-coarsemask_mask.nii.gz
        Written under ``out_dir / "masks"`` when ``write_artifacts=True``.

    Assumptions / Preconditions
    ---------------------------
    - Input is an hT2w volume with PVS appearing bright.
    - ``load_nifti`` returns (image, affine, header).
    - Size thresholds are specified in **voxels** (not mm³).

    Warnings
    --------
    - Percentile clipping compresses extremes; choose percentiles with care.
    - Size-based filtering depends on voxel resolution; adjust for anisotropy.
    - ``write_artifacts=True`` requires ``out_dir`` (asserts otherwise).

    Notes
    -----
    - Windowing: clip to ``[clip_low, clip_high]`` percentiles, then scale to
      [0, 1] (with a small epsilon to avoid divide-by-zero).
    - Keep-band aims to isolate mid-gray intensities typical of PVS on hT2w.
    """
    # ---- load
    img, aff, hdr = load_nifti(hT2w_nii)  # expects float + header/affine
    data = img.astype(np.float32, copy=False)

    # ---- robust windowing to [0,1]
    p_lo, p_hi = np.percentile(data, (float(clip_low), float(clip_high)))
    data = np.clip(data, p_lo, p_hi)
    if p_hi > p_lo:
        data = (data - p_lo) / (p_hi - p_lo + 1e-8)
    else:
        data = np.zeros_like(data, dtype=np.float32)
    data = data.astype(np.float32, copy=False)

    # ---- two-sided absolute threshold (keep-band on [0,1])
    lo = float(min(thr_low, thr_high))
    hi = float(max(thr_low, thr_high))
    mask = (data >= lo) & (data <= hi)

    if fill_holes_on and mask.any():
        mask = binary_fill_holes(mask)

    # remove small CCs
    if int(min_size_vox) > 0 and mask.any():
        mask = remove_small_objects(mask, min_size=int(min_size_vox), connectivity=3)

    # remove large CCs
    if int(max_size_vox) > 0 and mask.any():
        lab = label(mask, connectivity=3)
        if lab.max() > 0:
            sizes = np.bincount(lab.ravel())
            # labels to drop (excluding background 0)
            drop = np.where(sizes > int(max_size_vox))[0]
            drop = drop[drop != 0]
            if drop.size:
                # zero-out dropped labels
                m = lab.copy()
                for did in drop:
                    m[lab == did] = 0
                mask = m > 0
        del lab

    mask_u8 = mask.astype(np.uint8, copy=False)

    # ---- suppress background only; do NOT re-normalize after masking
    pre_img = (data * mask_u8).astype(np.float32, copy=False)
    # ---- optional artifacts
    if write_artifacts:
        assert out_dir is not None, "out_dir required when write_artifacts=True"
        out_dir = Path(out_dir)

        # If caller already passed .../masks, write there; else create a 'masks' child
        masks_dir = out_dir if out_dir.name == "masks" else (out_dir / "masks")
        masks_dir.mkdir(parents=True, exist_ok=True)

        # derive BIDS-ish tokens from input name
        stem_parts = []
        txt = str(hT2w_nii)
        m_sub = re.search(r"(sub-[A-Za-z0-9][A-Za-z0-9_]+)", txt)
        m_ses = re.search(r"(ses-[A-Za-z0-9]+)", txt)
        if m_sub:
            stem_parts.append(m_sub.group(1))
        if m_ses:
            stem_parts.append(m_ses.group(1))
        stem = "_".join(stem_parts) + ("_" if stem_parts else "")

        preproc_path = masks_dir / f"{stem}space-hT2w_class-pvs_desc-preproc_map.nii.gz"
        coarse_path  = masks_dir / f"{stem}space-hT2w_class-pvs_desc-coarsemask_mask.nii.gz"
        save_nifti(pre_img.astype(np.float32, copy=False), aff, hdr, preproc_path)
        print(f"[pvs_preproc] Saved → {preproc_path}")
        save_nifti(mask_u8, aff, hdr, coarse_path)
        print(f"[pvs_preproc] Saved → {coarse_path}")
    return pre_img, mask_u8, aff, hdr




# -------------------------------------------------------------
# EPC FUSION HELPERS (assumes images are already aligned)
# -------------------------------------------------------------
def epc_fuse_t1_t2(
    t1_img: np.ndarray,
    t2_img: np.ndarray,
    brain_mask: np.ndarray | None = None,
    invert_ratio: bool = True,
) -> np.ndarray:
    """
    Compute EPC-style ratio and (optionally) invert so PVS become bright.

    Parameters
    ----------
    t1_img : ndarray
        T1 image aligned to hT2w space; float32 recommended.
    t2_img : ndarray
        Preprocessed hT2w image; float32 in [0, 1].
    brain_mask : ndarray or None, optional
        Boolean mask; when provided, background is set to 0 before ratio.
    invert_ratio : bool, optional
        If True, return 1 / (T1/T2 + eps). Default: True.

    Returns
    -------
    ndarray
        EPC image, float32 in [0, 1], same shape as inputs.

    Files written
    -------------
    - None.

    Assumptions / Preconditions
    ---------------------------
    - `t1_img` and `t2_img` have identical shapes and are co-registered.
    - Inputs are scaled to [0, 1] or consistent intensity ranges.

    Warnings
    --------
    - Division by very small values is stabilized by `eps=1e-6`.

    Notes
    -----
    - Robust rescaling (0.5–99.5 pct) is applied to the masked-positive
      region prior to final normalization to [0, 1].
    """
    eps = 1e-6
    R = t1_img.astype(np.float32) / (t2_img.astype(np.float32) + eps)
    EPC = 1.0 / (R + eps) if invert_ratio else R
    if brain_mask is not None:
        EPC = EPC * (brain_mask.astype(np.float32))
    vals = EPC[(EPC > 0)] if brain_mask is None else EPC[brain_mask > 0]
    if vals.size > 50:
        lo, hi = np.percentile(vals, (0.5, 99.5))
        EPC = np.clip(EPC, lo, hi)
        EPC = (EPC - lo) / (hi - lo + 1e-8)
    return EPC.astype(np.float32)


# -------------------------------------------------------------
# VESSELNESS / FEATURE FILTERING
# -------------------------------------------------------------
def _compute_sorted_hessian_eigenvalues(
    image: np.ndarray, sigma: float, mode: str = "reflect", cval: float = 0.0
):
    """
    Compute Hessian eigenvalues (abs-sorted) at a given scale.

    Parameters
    ----------
    image : ndarray
        2D or 3D input image (float).
    sigma : float
        Standard deviation for Gaussian derivatives.
    mode : str, optional
        Padding mode for derivatives. Default: 'reflect'.
    cval : float, optional
        Constant value when `mode='constant'`.

    Returns
    -------
    ndarray
        Eigenvalues array with eigenvalue axis first (K, *image.shape), sorted
        by absolute value.
    """
    eigvals = hessian_matrix_eigvals(
        hessian_matrix(
            image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True
        )
    )
    eigvals = np.take_along_axis(
        eigvals, np.abs(eigvals).argsort(axis=0), axis=0
    )
    return eigvals


def frangi_vesselness(
    image: np.ndarray,
    scale_min: float = 0.5,
    scale_max: float = 3.0,
    scale_step: float = 0.5,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 15,
    black_ridges: bool = False,
    select_gamma_auto: bool = False,
    auto_gamma_fraction: float = 0.5,
    mode: str = "reflect",
    cval: float = 0.0,
):
    """
    Compute a Frangi vesselness map (bright or dark ridges).

    Parameters
    ----------
    image : ndarray
        2D or 3D image; recommended normalized to [0, 1].
    scale_min, scale_max, scale_step : float
        Sigma range for multi-scale Frangi.
    alpha, beta, gamma : float
        Frangi parameters. If `select_gamma_auto=True`, `gamma` is scaled
        from Hessian eigenvalues at the smallest sigma.
    black_ridges : bool, optional
        If True, detect dark ridges (e.g., veins in magnitude MRV).
    select_gamma_auto : bool, optional
        If True, set `gamma = auto_gamma_fraction * max(|eigvals|)` at the
        smallest sigma.
    auto_gamma_fraction : float, optional
        Fraction used when `select_gamma_auto=True`.
    mode : str, optional
        Padding mode for derivatives.
    cval : float, optional
        Constant value when `mode='constant'`.

    Returns
    -------
    ndarray
        Vesselness map, same shape as `image`. Scale is consistent with
        scikit-image's `frangi`; callers often normalize to [0, 1].

    Files written
    -------------
    - None.

    Assumptions / Preconditions
    ---------------------------
    - No resampling; operates in image space.

    Notes
    -----
    - Auto-γ uses the absolute maximum of Hessian eigenvalues at the
      smallest sigma for stability.
    """
    sigmas = np.arange(
        float(scale_min), float(scale_max) + 1e-6, float(scale_step)
    )
    if select_gamma_auto and sigmas.size > 0:
        eigvals = _compute_sorted_hessian_eigenvalues(
            image, sigmas[0], mode=mode, cval=cval
        )
        lam = np.max(np.abs(eigvals))
        if lam > 0:
            gamma = float(auto_gamma_fraction * lam)

    vesselness = frangi(
        image,
        sigmas=tuple(sigmas.tolist()),
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
    )
    return vesselness


# -------------------------------------------------------------
# THRESHOLDING / POST-PROCESSING / SKELETONIZATION
# -------------------------------------------------------------
def threshold_vesselness(vesselness: np.ndarray, thresh_frac: float = 0.5):
    """
    Threshold a vesselness map by a fraction of its maximum.

    Parameters
    ----------
    vesselness : ndarray
        Vesselness image.
    thresh_frac : float, optional
        Threshold as a fraction of `vesselness.max()`. If <= 0, returns
        `vesselness > 0`.

    Returns
    -------
    ndarray (bool)
        Binary mask.

    Files written
    -------------
    - None.
    """
    if thresh_frac <= 0:
        return vesselness > 0
    thr_val = float(vesselness.max() * thresh_frac)
    return vesselness > thr_val


def iterative_hysteresis(
    vesselness: np.ndarray,
    n_iter: int = 1,
    kappa: float = 0.1,
    pruning_cutoff: int = 0,
    prevent_leaking: bool = True,
):
    """
    Iterative hysteresis thresholding to boost weak vessels.

    Parameters
    ----------
    vesselness : ndarray
        Vesselness image in [0, 1].
    n_iter : int, optional
        Number of hysteresis iterations.
    kappa : float, optional
        Additive boost applied within the detected mask each iteration.
    pruning_cutoff : int, optional
        Minimum object size (voxels/pixels) for pruning.
    prevent_leaking : bool, optional
        If True, perform final 3D skeletonization (Lee) and re-threshold.

    Returns
    -------
    ndarray (bool)
        Binary mask after hysteresis and cleanup.

    Files written
    -------------
    - None.

    Assumptions / Preconditions
    ---------------------------
    - Otsu (3 classes) applied to positive voxels each iteration.

    Notes
    -----
    - Skeletonization uses `skeletonize(..., method="lee")` which supports
      true 3D thinning when the input is 3D.
    """
    img = vesselness.copy()
    mask = None
    for _ in range(max(1, n_iter)):
        if img[img > 0].size == 0:
            break
        thr = threshold_multiotsu(img[img > 0], classes=3)
        mask = apply_hysteresis_threshold(img, thr[0], thr[1])
        mask = remove_small_objects(mask, pruning_cutoff, connectivity=2)
        img = np.clip(img + kappa * mask, 0, 1)
    if prevent_leaking and mask is not None:
        centerline = skeletonize(mask, method="lee")
        img = np.clip(vesselness + centerline, 0, 1)
        thr = threshold_multiotsu(img[img > 0], classes=3)
        mask = apply_hysteresis_threshold(img, thr[0], thr[-1])
        mask = remove_small_objects(mask, pruning_cutoff, connectivity=2)
    return mask.astype(bool) if mask is not None else np.zeros_like(
        vesselness, dtype=bool
    )


# -------------------------------------------------------------
# END-TO-END SEGMENTATION WRAPPERS
# -------------------------------------------------------------
def segment_vessels(
    image: np.ndarray,
    scale_min: float = 0.5,
    scale_max: float = 6.0,
    scale_step: float = 0.5,
    alpha: float = 0.6,
    beta: float = 0.5,
    gamma: float = 0.02,
    black_ridges: bool = False,
    select_gamma_auto: bool = False,
    auto_gamma_fraction: float = 0.5,
    threshold_frac: float = 0.2,
    min_size: int = 25,
    n_iter: int = 1,
    kappa: float = 0.1,
    prevent_leaking: bool = True,
    do_tophat: bool = False,
    tophat_size: int = None,
    do_dilate: bool = False,
    dilate_rad: int = 1,
):
    """
    Segment vessels using Frangi + optional morphology and hysteresis.

    Parameters
    ----------
    image : ndarray
        2D/3D image; normalized internally to [0, 1].
    scale_min, scale_max, scale_step : float
        Sigma range for Frangi.
    alpha, beta, gamma : float
        Frangi parameters. `select_gamma_auto` can adapt γ from Hessian stats.
    black_ridges : bool, optional
        If True, detect dark ridges; otherwise bright.
    select_gamma_auto : bool, optional
        Enable auto-γ (fraction of |eigs| at smallest sigma).
    auto_gamma_fraction : float, optional
        Fraction for auto-γ scaling.
    threshold_frac : float, optional
        Fraction of max vesselness for simple thresholding when `n_iter == 1`.
    min_size : int, optional
        Minimum object size to retain.
    n_iter : int, optional
        Hysteresis iterations; if >1, Otsu-based iterative scheme is used.
    kappa : float, optional
        Additive vesselness boost during hysteresis.
    prevent_leaking : bool, optional
        If True, 3D Lee skeletonization + re-threshold step.
    do_tophat : bool, optional
        Apply white/black top-hat (black for dark ridges, then flip to bright).
    tophat_size : int or None, optional
        Structural element radius; defaults to ~scale_max if None.
    do_dilate : bool, optional
        Optional dilation of the final mask.
    dilate_rad : int, optional
        Radius for structuring element used in dilation.

    Returns
    -------
    tuple of ndarray
        (vesselness float32 in [0, 1], mask bool, skeleton bool).

    Files written
    -------------
    - None.

    Assumptions / Preconditions
    ---------------------------
    - Pure image-space processing; no resampling.

    Warnings
    --------
    - Overly large `tophat_size` can suppress broader structures.
    """
    # 1) Normalize to [0,1]
    image = image.astype(float)
    imin, imax = image.min(), image.max()
    if imax > imin:
        image = (image - imin) / (imax - imin)
    image = np.clip(image, 0, 1)

    # 2) Optional top-hat (white for arteries; black for veins then flip)
    if do_tophat:
        if tophat_size is None:
            tophat_size = int(scale_max + 0.5)  # ~2× largest radius (vox)
        selem = ball(tophat_size) if image.ndim == 3 else disk(tophat_size)
        if black_ridges:
            image = black_tophat(image, footprint=selem)
            black_ridges = False  # treat as bright ridges after black top-hat
        else:
            image = white_tophat(image, footprint=selem)
        if image.max() > 0:
            image /= image.max()

    # 3) Frangi (multi-scale)
    vesselness = frangi_vesselness(
        image,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_step=scale_step,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
        select_gamma_auto=select_gamma_auto,
        auto_gamma_fraction=auto_gamma_fraction,
    )
    if vesselness.max() > 0:
        vesselness /= vesselness.max()

    # 4) Thresholding / hysteresis
    if n_iter > 1:
        mask = iterative_hysteresis(
            vesselness,
            n_iter=n_iter,
            kappa=kappa,
            pruning_cutoff=min_size,
            prevent_leaking=prevent_leaking,
        )
    else:
        mask = threshold_vesselness(vesselness, thresh_frac=threshold_frac)
        if min_size and min_size > 0:
            mask = remove_small_objects(mask, min_size=min_size, connectivity=2)

    # 5) Optional dilation
    if do_dilate and mask.any():
        struct = ball(dilate_rad) if mask.ndim == 3 else disk(dilate_rad)
        mask = ndimage.binary_dilation(mask, structure=struct)

    # 6) Skeleton (true 3D Lee for 3D input)
    skeleton = skeletonize(mask, method="lee")

    return (
        vesselness.astype(np.float32),
        mask.astype(bool),
        skeleton.astype(bool),
    )

def segment_pvs_frangi3d(
    image: np.ndarray,
    *,
    header,                                 # NIfTI header (for voxel sizes)
    # ---- Frangi in PHYSICAL mm (converted to vox for processing) ----
    sigma_min_mm: float = 0.2,
    sigma_max_mm: float = 1.2,
    sigma_step_mm: float = 0.2,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 15.0,
    # ---- Threshold & cleanup ----
    threshold_value: float = 0.60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 3D Frangi vesselness for PVS using mm-scale parameterization.

    Parameters
    ----------
    image : ndarray
        Preprocessed magnitude-like image; 2D/3D. Cast to float32 internally.
    header : nib.Nifti1Header or object
        Header providing ``get_zooms()`` to read voxel sizes (mm).
    sigma_min_mm, sigma_max_mm, sigma_step_mm : float
        Sigma grid in millimeters; converted to voxel units via header zooms.
    alpha, beta : float, optional
        Frangi sensitivity to plate-/blob-like responses. Default: 0.5 each.
    gamma : float, optional
        Nominal Frangi contrast parameter. Currently overridden by a fixed
        internal value (see Notes). Default: 15.0.
    threshold_value : float, optional
        Fraction of the vesselness maximum used for binarization
        (``thresh_frac``). Default: 0.60.

    Returns
    -------
    vesselness : ndarray, dtype=float32
        Vesselness map in [0, 1], same shape as ``image``.
    mask : ndarray, dtype=bool
        Binary mask after simple fraction-of-max thresholding.
    skeleton : ndarray, dtype=bool
        3D Lee skeleton of ``mask`` in the native grid.

    Files written
    -------------
    - None.

    Assumptions / Preconditions
    ---------------------------
    - No spatial resampling is persisted; computations are in voxel space.
    - ``header.get_zooms()`` returns voxel sizes (mm); the first element is
      used for mm→vox conversion here.

    Warnings
    --------
    - Using only the first voxel size for mm→vox conversion may be suboptimal
      for anisotropic data.
    - ``gamma`` argument is currently unused (overridden by a fixed value).

    Notes
    -----
    - mm→vox conversion computes a voxel-scale sigma grid and forwards it to
      ``frangi_vesselness``.
    - ``select_gamma_auto`` is disabled; a fixed gamma within [10, 25] is used.
    - Skeletonization uses the Lee 3D method on the thresholded mask.
    """
    img = np.asarray(image, dtype=np.float32)

    # ----- voxel sizes (mm) and iso resample plan -----
    if hasattr(header, "get_zooms"):
        vx_mm = float(header.get_zooms()[0])
    else:
        vx_mm = 1.0

    # --- mm → vox with same guards as GUI ---
    smin_v = max(0.05, float(sigma_min_mm / vx_mm))
    smax_v = max(smin_v + 1e-6, float(sigma_max_mm / vx_mm))
    sstep_v = max(0.01, float(sigma_step_mm / vx_mm))

    # (optional: for debugging/verification)
    _sig_gui = np.arange(smin_v, smax_v + 1e-6, sstep_v)

    def p99_norm(a: np.ndarray, *, positive_only: bool = True) -> np.ndarray:
        """
        Normalize to the 99th percentile (robust). Clips to [0,1].
        If positive_only=True, computes the percentile over a>0 voxels.
        """
        a = a.astype(np.float32, copy=False)
        sample = a[a > 0] if positive_only else a[np.isfinite(a)]
        if sample.size:
            p99 = float(np.percentile(sample, 99.0))
            if p99 > 0:
                a = a / p99
        np.clip(a, 0.0, 1.0, out=a)
        return a

    GAMMA_MIN, GAMMA_MAX = 10.0, 25.0
    gamma_fixed = GAMMA_MIN + 0.75 * (GAMMA_MAX - GAMMA_MIN)

    vess_iso = frangi_vesselness(
        img.astype(np.float32, copy=False),
        scale_min=smin_v,
        scale_max=smax_v,
        scale_step=sstep_v,
        alpha=alpha,
        beta=beta,
        gamma=float(gamma_fixed),
        black_ridges=False,
        select_gamma_auto=False,  # keep fixed unless enabling auto-γ
    ).astype(np.float32)

    # after computing vess_iso
    vesselness = p99_norm(np.clip(vess_iso, 0.0, None))

    mask = threshold_vesselness(vess_iso, thresh_frac=float(threshold_value))
    # Skeleton on native grid (3D Lee)
    skeleton_mask = skeletonize(mask.astype(bool), method="lee")

    return vesselness.astype(np.float32), mask.astype(bool), skeleton_mask.astype(bool)



# -------------------------------------------------------------
# UTILITIES (N4, robust scaling)
# -------------------------------------------------------------
def _n4_on_numpy(arr_xyz: np.ndarray) -> np.ndarray:
    """
    Apply SimpleITK N4 bias correction to a numpy volume.

    Parameters
    ----------
    arr_xyz : ndarray
        Input in (x, y, z) order, float32 preferred.

    Returns
    -------
    ndarray
        Bias-corrected array in (x, y, z), float32.

    Raises
    ------
    RuntimeError
        If SimpleITK is not available.

    Notes
    -----
    - SimpleITK expects (z, y, x) ordering; arrays are transposed in/out.
    - A coarse Otsu mask is used to guide N4.
    """
    if sitk is None:
        raise RuntimeError(
            "SimpleITK not available; set do_n4=False or install SimpleITK."
        )
    # NIfTI uses (x,y,z); SimpleITK expects (z,y,x)
    img_itk = sitk.GetImageFromArray(
        np.transpose(arr_xyz.astype(np.float32), (2, 1, 0))
    )
    msk = sitk.OtsuThreshold(img_itk, 0, 1, 200)
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    corr = n4.Execute(img_itk, msk)
    out_zyx = sitk.GetArrayFromImage(corr).astype(np.float32)
    return np.transpose(out_zyx, (2, 1, 0))  # back to (x,y,z)


def _robust_scale_to_01(arr: np.ndarray, pct=(1.0, 99.0)) -> np.ndarray:
    """
    Robustly scale a volume to [0, 1] using percentile clipping.

    Parameters
    ----------
    arr : ndarray
        Input array; arbitrary dtype.
    pct : tuple of float, optional
        (low, high) percentiles for clipping. Default: (1.0, 99.0).

    Returns
    -------
    ndarray
        Float32 array with values scaled to [0, 1]; same shape as input.

    Assumptions / Preconditions
    ---------------------------
    - Input is finite (NaNs/Infs are not handled explicitly).

    Warnings
    --------
    - Extremely low dynamic range after clipping may yield near-constant output.

    Notes
    -----
    - If the count of positive voxels is small, falls back to using all voxels.
    - Adds a small epsilon in the denominator to avoid division by zero.
    """
    a = arr.astype(np.float32, copy=False)
    nz = a[a > 0]
    if nz.size < 100:
        nz = a.reshape(-1)
    lo, hi = np.percentile(nz, pct)
    if hi <= lo:
        lo, hi = float(np.min(nz)), float(np.max(nz) + 1e-8)
    a = np.clip(a, lo, hi)
    a = (a - lo) / (hi - lo + 1e-8)
    return a.astype(np.float32)


# -----------------------------------------------------------------
# One-call entry: T1 light-preproc → DIPY register → EPC → (optional save)
# -----------------------------------------------------------------

def epc_from_t1_and_hT2w(
    t1_path: Path,
    hT2w_arr: np.ndarray,
    hT2w_aff: np.ndarray,
    *,
    out_fused_path: Path | None = None,
    out_xfm_path: Path | None = None,     # NEW: save affine used for registration
    do_n4: bool = True,
    robust_pct: tuple[float, float] = (1.0, 99.0),
    reg_mode: str = "rigid_affine",
    use_nonlin: bool = False,
    brain_mask: np.ndarray | None = None,
    invert_ratio: bool = True,
) -> np.ndarray:
    """
    Light-preprocess T1, register T1→hT2w, and EPC-fuse.

    Parameters
    ----------
    t1_path : Path
        Path to a T1 NIfTI (raw or denoised).
    hT2w_arr : ndarray
        Preprocessed hT2w array in [0, 1].
    hT2w_aff : ndarray
        4×4 affine for hT2w (used when writing fused output).
    out_fused_path : Path or None, optional
        If provided, write the EPC image as NIfTI (affine = `hT2w_aff`).
    out_xfm_path : Path or None, optional
        If provided, save the affine matrix used for registration as text.
    do_n4 : bool, optional
        Apply N4 to T1 before registration. Default: True.
    robust_pct : tuple of float, optional
        Percentiles for robust scaling of T1 to [0, 1]. Default: (1.0, 99.0).
    reg_mode : str, optional
        Registration mode key understood by `register_t1_to_hT2w`.
    use_nonlin : bool, optional
        If True, enable non-linear refinement (if supported by `register_*`).
    brain_mask : ndarray or None, optional
        Optional mask applied during EPC (background suppressed).
    invert_ratio : bool, optional
        If True, return inverted ratio so PVS are bright. Default: True.

    Returns
    -------
    ndarray
        EPC image, float32 in [0, 1], same shape as `hT2w_arr`.

    Files written
    -------------
    out_fused_path
        Fused EPC NIfTI (if provided).
    out_xfm_path
        Registration affine (text file, if provided).

    Assumptions / Preconditions
    ---------------------------
    - `hT2w_arr` is preprocessed and scaled to [0, 1].
    - `register_t1_to_hT2w` returns `(registered_array, affine_used)`.

    Raises
    ------
    RuntimeError
        If the registered T1 and `hT2w_arr` shapes do not match.

    Notes
    -----
    - N4 is performed via `_n4_on_numpy`; robust scaling uses `_robust_scale_to_01`.
    - The saved transform is whatever `register_t1_to_hT2w` reports as `aff_used`.
    """
    t1_nib = nib.load(str(t1_path))
    t1_arr = t1_nib.get_fdata().astype(np.float32)
    t1_aff = t1_nib.affine

    hT2w_arr = np.clip(hT2w_arr.astype(np.float32, copy=False), 0.0, 1.0)

    # Light T1 preprocessing
    if do_n4:
        t1_arr = _n4_on_numpy(t1_arr)
    t1_arr = _robust_scale_to_01(t1_arr, pct=robust_pct)

    # Register T1 → hT2w
    t1_reg_arr, aff_used = register_t1_to_hT2w(
        moving=t1_arr, moving_aff=t1_aff,
        static=hT2w_arr, static_aff=hT2w_aff,
        reg_mode=reg_mode,
        use_nonlin=use_nonlin,
    )

    if t1_reg_arr.shape != hT2w_arr.shape:
        raise RuntimeError(
            f"Registration mismatch: T1_reg {t1_reg_arr.shape} vs hT2w {hT2w_arr.shape}"
        )

    # EPC fusion
    epc = epc_fuse_t1_t2(
        t1_reg_arr, hT2w_arr,
        brain_mask=brain_mask,
        invert_ratio=invert_ratio,
    ).astype(np.float32)

    # Optional writes
    if out_fused_path is not None:
        out_fused_path = Path(out_fused_path)
        out_fused_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(epc, hT2w_aff), str(out_fused_path))
        print(f"[epc] Saved → {out_fused_path}")

    if out_xfm_path is not None:
        out_xfm_path = Path(out_xfm_path)
        out_xfm_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(str(out_xfm_path), aff_used, fmt="%.8f")
        print(f"[epc] Saved → {out_xfm_path}")


    return epc
