# SPDX-License-Identifier: MIT
"""
distance.py
-----------
Distance-map generation on the MREG grid for downstream distance-clustered
spectral analysis.

This module standardizes computation of Euclidean distance transforms (EDT) in
millimeters from class masks (e.g., arteries) anchored to a 3D MREG reference
image. Masks are snapped with **nearest-neighbor** resampling to the MREG grid
(labels preserved). Outputs are float32 NIfTI files with the reference affine.

New in this version
-------------------
In addition to computing the EDT after snapping a mask to the MREG grid, this
module can compute the EDT in the **native mask space** and then compose a
single affine (native→T1→MREG) to resample **once** onto the MREG grid using
linear interpolation (float). This avoids two resamples and preserves native
voxel geometry for the distance computation.

Pipeline steps
--------------
1. Resolve an MREG reference image (meanamp → mean → ref → fallback picker).
2. Snap class masks to the reference grid (nearest-neighbor, labels).
   2a. Alternatively: compute EDT in native space and resample once to MREG
       through the composed (native→T1→MREG) affine.
3. Compute Euclidean distance transform in **mm** using header zooms.
4. Optionally confine distances to an MREG brain mask (set outside to NaN).

Inputs / Outputs
----------------
Inputs  : Class masks in (ideally) MREG space; optionally native-space masks
          plus transforms; an optional brain mask in MREG space; SubjectPaths
          for BIDS-derivatives roots and references.
Outputs : Distance maps saved as float32 NIfTI images on the MREG grid.

Files written
-------------
- derivatives/neurofluid-mreg/sub-<ID>/distmaps/
  sub-<ID>_space-MREG_class-<CLASS>_desc-dist_map.nii.gz

Assumptions / Preconditions
---------------------------
- Spaces: Operates in **MREG** space. If a source mask is not on the MREG grid,
  it is resampled with **nearest-neighbor** to the reference image, or the EDT
  is computed in native space and resampled once to MREG if transforms are
  provided.
- Affines: If affines differ, continue in image space after snapping to the
  reference (no smoothing; labels preserved) unless the native-first route is
  taken (linear interpolation, one resample).
- Shapes/dtypes: Output `float32`, shape equals MREG reference shape.
- BIDS naming: `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-<DESC>_<SUFFIX>.nii.gz`.

Warnings
--------
- Nearest-neighbor snapping may introduce staircase effects at label borders.
- If a brain mask is provided, voxels outside are set to **NaN** (ignored by
  downstream statistics); consumers must handle NaNs explicitly.
- Distance outside the mask is distance to the nearest mask voxel (standard EDT).

Public API
----------
- distance_map
- generate_distance_maps
"""

from __future__ import annotations
from pathlib import Path
import nibabel as nib, numpy as np
from scipy.ndimage import distance_transform_edt
from .io import deriv_name
from nibabel.processing import resample_from_to
from .mreg import _pick_mreg_ref_path
from .transforms import apply_affine_to_ref

# Which native space + transform key to use per class
_CLASS_TO_NATIVE = {
    "arteries": ("TOF", "tof_to_t1"),
    "veins": ("MRV", "mrv_to_t1"),
    "pvs": ("HT2W", "ht2w_to_t1"),
}
# Policy: compute EDT on the analysis grid (MREG). Do NOT resample distance maps.
NATIVE_FIRST_ALLOWED = False  # set True only for quick QC, not for analysis



# -------------------------------------------------------------
# I/O helpers (BIDS naming, paths)
# -------------------------------------------------------------
def _get_mreg_ref_img(sp):
    """
    Resolve a 3D MREG reference image for anchoring outputs.

    Preference order
    ----------------
    1) `sp.mreg_meanamp_path`
    2) `sp.mreg_mean_path`
    3) `sp.mreg_ref_path`
    4) Fallback: `_pick_mreg_ref_path(sp)` from `.mreg`

    Returns
    -------
    nib.Nifti1Image
        Reference image whose shape and affine define the MREG grid.

    Assumptions / Preconditions
    ---------------------------
    - Attributes (if present) contain valid NIfTI paths.
    - First existing path in the preference order is used.

    Warnings
    --------
    - No validation of image content beyond successful load.
    """
    for attr in ("mreg_meanamp_path", "mreg_mean_path", "mreg_ref_path"):
        p = getattr(sp, attr, None)
        if p and Path(p).exists():
            return nib.load(str(p))
    return nib.load(str(_pick_mreg_ref_path(sp)))


# -------------------------------------------------------------
# Registration / native-space transforms
# -------------------------------------------------------------
def _distance_map_native_to_mreg(
    sp,
    xfm,
    native_mask_path: Path,
    native_to_t1_key: str,
    out_path: Path,
) -> Path:
    """
    Compute EDT in the native mask grid and resample once to MREG.

    The native→MREG mapping is formed by composing the provided transforms
    (native→T1 and T1→MREG). The native-space float distance map is then
    resampled to the MREG grid in a single step using linear interpolation.

    Parameters
    ----------
    sp : SubjectPaths
        Subject structure providing access to MREG references and output roots.
    xfm : Mapping[str, str] or dict
        Lookup table from transform key to text file path of a 4×4 affine.
        Must at least contain:
        - `native_to_t1_key` (passed here)
        - `'t1_to_mreg'`
    native_mask_path : Path
        Path to a binary native-space mask NIfTI.
    native_to_t1_key : str
        Key within `xfm` for the native→T1 4×4 affine (text file).
    out_path : Path
        Destination path for the MREG-space distance map (float32).

    Returns
    -------
    Path
        Saved distance map path (float32, MREG grid).

    Files written
    -------------
    - `<out_path>` : final distance map (float32).
    - Temporary files with prefixes `tmp_native_dist_` and `_ref_mreg_` are
      created next to `<out_path>` and removed at the end.

    Assumptions / Preconditions
    ---------------------------
    - Transform files pointed to by `xfm[...]` are readable 4×4 matrices in
      text format (`np.loadtxt`).
    - Native mask is binary or thresholdable to boolean; header zooms encode mm.
    - The MREG reference header/affine define the target grid.

    Warnings
    --------
    - Linear interpolation of distances may slightly reduce sharp peaks.
    - Numerical round-off may yield tiny negative values; these are clamped to
      0.0 before saving.
    - Accuracy depends on the correctness of the provided affines and headers.

    Raises
    ------
    FileNotFoundError
        If the native mask or any required transform file cannot be loaded.
    ValueError
        If a transform file does not contain a valid 4×4 affine.
    RuntimeError
        If resampling through `apply_affine_to_ref` fails.

    Notes
    -----
    - EDT is computed in native space using native voxel sizes (mm). A single
      resample to MREG avoids stacking resampling errors.
    """

    # 1) EDT in native (preserves mm units via native zooms)
    mask_img = nib.load(str(native_mask_path))
    mask = mask_img.get_fdata() > 0.5
    zooms = mask_img.header.get_zooms()[:3]
    dist_native = distance_transform_edt(~mask, sampling=zooms).astype(np.float32)

    # Persist a tiny temp NIfTI in the same folder (so apply_affine_to_ref can
    # read paths)
    tmp_native = out_path.parent / ("tmp_native_dist_" + out_path.name)
    nib.save(
        nib.Nifti1Image(dist_native, mask_img.affine, mask_img.header),
        str(tmp_native),
    )

    # 2) Compose native→MREG = (T1→MREG) · (native→T1)
    A_native_to_t1 = np.loadtxt(xfm.get(native_to_t1_key))
    A_t1_to_mreg = np.loadtxt(xfm.get("t1_to_mreg"))
    A_native_to_mreg = A_t1_to_mreg @ A_native_to_t1
    print("[dist] native→MREG composed (T1→MREG · native→T1)")

    # 3) One-shot resample to MREG grid (linear for float distances)
    mreg_ref_img = _get_mreg_ref_img(sp)
    mreg_ref_path = out_path.parent / ("_ref_mreg_" + out_path.name)  # tiny stub path
    nib.save(
        nib.Nifti1Image(
            np.zeros(mreg_ref_img.shape, dtype=np.float32),
            mreg_ref_img.affine,
            mreg_ref_img.header,
        ),
        str(mreg_ref_path),
    )
    apply_affine_to_ref(
        image_path=tmp_native,
        affine=A_native_to_mreg,
        ref_img_path=mreg_ref_path,
        out_path=out_path,
        interpolation="linear",
    )

    # Cleanup temp files
    try:
        tmp_native.unlink(missing_ok=True)
        mreg_ref_path.unlink(missing_ok=True)
    except Exception:
        pass

    # 4) Clamp tiny negatives (numerical) to 0
    out_img = nib.load(str(out_path))
    arr = np.asarray(out_img.get_fdata(), dtype=np.float32)
    arr[arr < 0] = 0.0
    nib.save(nib.Nifti1Image(arr, out_img.affine, out_img.header), str(out_path))
    print(f"[dist] Saved → {out_path}")
    return out_path


# -------------------------------------------------------------
# Thresholding / post-processing / skeletonization
# (Distance-map computation and writing)
# -------------------------------------------------------------
def distance_map(
    sp,
    mask_path: Path,
    out_path: Path,
    *,
    xfm=None,  # deprecated; ignored
    native_to_t1_key: str | None = None,  # deprecated; ignored
) -> Path:
    """
    Compute a Euclidean distance transform (EDT, mm) and save on the MREG grid.

    If `xfm` and `native_to_t1_key` are provided and the input mask is **not**
    on the MREG grid, the EDT is computed in the mask's native space and then
    resampled **once** to the MREG grid through the composed (native→T1→MREG)
    affine using linear interpolation. Otherwise, the mask is snapped to MREG
    with nearest-neighbor and EDT is computed on the MREG grid.

    Parameters
    ----------
    sp : SubjectPaths
        Provides access to optional MREG reference paths and output roots.
    mask_path : Path
        Path to a binary mask NIfTI. If its grid/affine differ from the MREG
        reference, it is snapped with **nearest-neighbor** unless transforms
        are provided (native-first route).
    out_path : Path
        Destination path for the distance map NIfTI.
    xfm : Any, deprecated
        Ignored. The previous "native-first EDT then resample once" route has
        been removed in favor of "register-to-MREG then EDT".
    native_to_t1_key : str or None, deprecated
        Ignored; see `xfm`.

    Returns
    -------
    Path
        Saved distance map path (float32, MREG grid).

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/distmaps/
      sub-<ID>_space-MREG_class-<CLASS>_desc-dist_map.nii.gz

    Assumptions / Preconditions
    ---------------------------
    - Spaces: Operates on/saves to **MREG** space.
    - If mask and MREG reference mismatch, mask is snapped (nearest-neighbor).
    - Input mask is binary or thresholdable to boolean.

    Warnings
    --------
    - Nearest-neighbor snapping preserves labels but may alias boundaries.
    - Small negative values should not occur (no linear resampling of distances).

    Raises
    ------
    FileNotFoundError
        If `mask_path` or resolved reference path cannot be loaded.

    Notes
    -----
    - Distances are in **millimeters** from the MREG header zooms.
    - Output dtype is float32; header/affine copied from the MREG reference.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ref_img = _get_mreg_ref_img(sp)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If we got transforms AND the mask is not on the MREG grid,
    # do the native-first route.
    ref_img = _get_mreg_ref_img(sp)
    try:
        src_img = nib.load(str(mask_path))
    except Exception as e:
        raise FileNotFoundError(f"Cannot load mask: {mask_path} :: {e}")

    if NATIVE_FIRST_ALLOWED and (xfm is not None) and (native_to_t1_key is not None) and (
    src_img.shape != ref_img.shape
    or not np.allclose(src_img.affine, ref_img.affine, atol=1e-3)):
        print("[dist] Route: native EDT → one-shot resample to MREG")
        return _distance_map_native_to_mreg(
            sp, xfm, mask_path, native_to_t1_key, out_path
        )

    ref_shape, ref_aff = ref_img.shape, ref_img.affine

    # Snap mask to MREG grid (nearest; labels preserved)
    if src_img.shape != ref_shape or not np.allclose(
        src_img.affine, ref_aff, atol=1e-3
    ):
        src_img = resample_from_to(src_img, (ref_shape, ref_aff), order=0)
        print("[dist] [WARN] Source mask snapped to MREG grid (nearest)")

    mask = src_img.get_fdata().astype(bool, copy=False)
    zooms = ref_img.header.get_zooms()[:3]
    dist = distance_transform_edt(~mask, sampling=zooms).astype(np.float32, copy=False)

    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(dist, ref_aff, hdr), str(out_path))
    print(f"[dist] Saved → {out_path}")
    return out_path


def generate_distance_maps(
    sp,
    classes: tuple[str, ...] = ("arteries",),
    mask_path: Path | None = None,
    overwrite: bool = False,
    xfm=None,
) -> None:
    """
    Build class-wise distance maps **on the MREG grid** (EDT computed in MREG).

    For each vascular class, the mask is ensured to be in MREG space. If an
    MREG-space mask already exists, it is used directly. If only a native-space
    mask exists, it is resampled to MREG with nearest-neighbor, and the
    Euclidean distance transform (EDT) is then computed on the MREG grid.
    The distance image is never resampled.

    Parameters
    ----------
    sp : SubjectPaths
        Provides subject context and directories. Must expose:
        - `sp.sub` (str), `sp.masks_dir` (Path), `sp.distmaps_dir` (Path)
    classes : tuple[str, ...], default ("arteries",)
        Vascular classes to process (e.g., ("arteries", "veins", "pvs")). For
        each class, masks are expected at:
            <masks_dir>/sub-<ID>_space-<NATIVE>_class-<CLASS>_desc-main_mask.nii.gz
            <masks_dir>/sub-<ID>_space-MREG_class-<CLASS>_desc-main_mask.nii.gz
        The distance map is written to:
            <distmaps_dir>/
            sub-<ID>_space-MREG_class-<CLASS>_desc-dist_map.nii.gz
        (All filenames produced via `deriv_name`.)
    mask_path : Path or None, optional
        If provided, path to a brain mask in **MREG space**. Applied *after*
        distance computation; voxels outside are set to **NaN** (ignored later).
        If grid/affine differ, the mask is resampled with nearest-neighbor.
    overwrite : bool, default False
        If False, skip classes whose output already exists.
    xfm : Any, deprecated
        Ignored. Present only for backward compatibility; distance is always
        computed in MREG space.

    Returns
    -------
    None

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/distmaps/
      sub-<ID>_space-MREG_class-<CLASS>_desc-dist_map.nii.gz  (float32, NaNs
      outside brain if `mask_path` provided)

    Assumptions / Preconditions
    ---------------------------
    - Class masks exist in `masks_dir` and are binary or thresholdable.
    - Spaces: If a class mask is not in MREG, it will be **snapped to MREG**;
      EDT is **always** computed on the MREG grid.
    - Affines: Any mismatch resolved by nearest-neighbor resampling (MREG route)
      or a single linear resample (native-first route).

    Warnings
    --------
    - Nearest-neighbor snapping preserves labels but may alias boundaries.    
    - NaNs are introduced when brain-masking; downstream consumers must handle
      them (e.g., nan-aware statistics).
    - Existing outputs are skipped unless `overwrite=True`.

    Notes
    -----
    - Distances are in **mm** according to the MREG header zooms.
    """
    Path(sp.distmaps_dir).mkdir(parents=True, exist_ok=True)

    for klass in classes:
        # Prefer NATIVE mask if it exists; else fall back to MREG mask
        native_space, native_key = _CLASS_TO_NATIVE.get(klass, (None, None))
        mask_native = None
        if native_space is not None:
            candidate = (
                Path(sp.masks_dir)
                / deriv_name(sp.sub, native_space, klass, "main", "mask")
            )
            if candidate.exists():
                mask_native = candidate

        mask_mreg = (
            Path(sp.masks_dir) / deriv_name(sp.sub, "MREG", klass, "main", "mask")
        )
        out_map = Path(sp.distmaps_dir) / deriv_name(sp.sub, "MREG", klass, "dist", "map")

        if mask_mreg.exists():
            # Already on MREG grid → EDT in MREG (best)
            distance_map(sp, mask_mreg, out_map)
        elif mask_native is not None:
            # Fallback: snap native mask to MREG, then EDT in MREG (no distance resampling)
            distance_map(sp, mask_native, out_map)  # xfm=None → safe route
        else:
            print(f"[dist] [SKIP] No mask for {klass} (native or MREG)")
            continue

        # Optionally confine to brain (set outside voxels to NaN)
        if mask_path is not None and out_map.exists():
            try:
                dist_img = nib.load(str(out_map))
                dist = np.asarray(dist_img.get_fdata(), dtype=np.float32)

                brain_img = nib.load(str(mask_path))
                brain = brain_img.get_fdata().astype(bool)

                # Align brain mask to the distance-map grid if needed (nearest)
                if (brain.shape != dist.shape) or (
                    not np.allclose(brain_img.affine, dist_img.affine, atol=1e-3)
                ):
                    brain_res = resample_from_to(
                        brain_img, (dist_img.shape, dist_img.affine), order=0
                    )
                    print("[dist] [WARN] Brain mask snapped to distance grid")
                    brain = brain_res.get_fdata() > 0.5

                dist[~brain] = np.nan  # outside brain ignored by downstream stats
                out_hdr = dist_img.header.copy()
                out_hdr.set_data_dtype(np.float32)
                nib.save(nib.Nifti1Image(dist, dist_img.affine, out_hdr), str(out_map))
                print(f"[dist] Saved → {out_map}")
                print("[dist] Applied brain mask (NaN outside)")
            except Exception as e:
                print(f"[dist] Brain masking skipped (error): {e}")
