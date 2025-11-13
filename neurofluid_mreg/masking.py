# SPDX-License-Identifier: MIT
"""
masking.py
----------
Brain masking utilities for Neurofluid–MREG.

This module provides mean-EPI/mean-BOLD brain masking on native fMRI/MREG
data and a single subject-level brain mask on a canonical MNI grid derived
from the subject's T1 image. Affines and headers are preserved on write;
registrations/resampling are delegated to `warp_t1_to_mni_once` and nilearn
masking routines.

Pipeline steps
--------------
1. For 4D EPI/BOLD (e.g., MREG), compute a lightly smoothed temporal mean
   and derive an EPI mask via nilearn `compute_epi_mask`.
2. For T1, warp the subject T1 to MNI (via `warp_t1_to_mni_once`) and
   derive a whole-brain mask on the MNI grid using nilearn
   `compute_brain_mask`.
3. Reuse the canonical MNI brain mask across downstream analyses (e.g.,
   distance and continuous proximity analyses).

Inputs / Outputs
----------------
Inputs
    - Native-space NIfTI images:
      * 4D EPI/BOLD (MREG) volumes for EPI masking.
      * 3D structural T1 volumes for MNI-based masking.
    - `SubjectPaths` and `TransformBook` (for MNI warps) when computing
      the canonical MNI brain mask.
Outputs
    - Paths to written mask NIfTIs (uint8); no arrays are returned.

Files written
-------------
- derivatives/neurofluid-mreg/sub-<ID>/masks/
  - sub-<ID>_space-<SPACE>_desc-brain_mask.nii.gz
    (space is typically MREG for EPI masks or MNI for the canonical mask)

Assumptions / Preconditions
---------------------------
- Spaces: T1, MREG, and MNI as indicated by filenames and calling context.
- Affines: The MNI mask is computed on a canonical MNI grid (typically
  1 mm isotropic); projections from T1 to MNI are performed by
  `warp_t1_to_mni_once`.
- Shapes/dtypes: Masks are written as uint8 (0/1); intermediate arrays are
  float32. 4D inputs have shape (X, Y, Z, T); masks are 3D (X, Y, Z).
- BIDS naming: Filenames follow
  `sub-<ID>_space-<SPACE>_desc-brain_mask.nii.gz`.

Warnings
--------
- EPI masking thresholds (`lower_cutoff`, `upper_cutoff`) are fixed in this
  module and may require tuning for unusual sequence settings or contrasts.
- The MNI brain mask uses nilearn defaults for `compute_brain_mask`; mask
  extent can vary with intensity scaling and registration quality.
- Units for MNI voxel sizes are assumed to be millimeters; this is standard
  for the MNI152 template but should be confirmed if using custom templates.

Public API
----------
- mask_brain
- compute_mni_brain_mask_once
"""

# neurofluid_mreg/masking.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import nibabel as nib

from scipy.ndimage import gaussian_filter
from nilearn.masking import compute_brain_mask, compute_epi_mask
from neurofluid_mreg.transforms import warp_t1_to_mni_once


# -------------------------------------------------------------
# Masking (MREG/T1/MNI paths)
# -------------------------------------------------------------
def mask_brain(
    image: Path, out_path: Path, overwrite: bool = False, assume_mni: bool = False
) -> Path:
    """
    Create a brain mask from a 4D EPI/BOLD or 3D T1 image and write it to disk.

    For 4D EPI/BOLD inputs (e.g., MREG), this function computes a temporal
    mean after light Gaussian smoothing (σ=1 voxel) and then calls nilearn
    `compute_epi_mask` to derive a brain mask. For 3D T1 inputs, nilearn
    `compute_brain_mask` is allowed only when the image is already on or
    near an MNI-like grid (`assume_mni=True`); otherwise a RuntimeError is
    raised to force the T1→MNI masking pathway.

    Parameters
    ----------
    image : pathlib.Path
        Path to the input NIfTI image. For EPI masking, a 4D BOLD/MREG
        volume (X, Y, Z, T). For T1 masking, a 3D structural T1 image.
        Units are dictated by the input header; no resampling is performed
        here beyond the internal nilearn operations.
    out_path : pathlib.Path
        Destination path for the mask NIfTI (uint8). Parent directories are
        created if missing.
    overwrite : bool, optional
        If False and `out_path` already exists, the existing mask is reused
        and the function returns immediately. Default is False.
    assume_mni : bool, optional
        If True for a 3D T1 input, call `compute_brain_mask` directly on
        the T1 (assumed already on/near the MNI grid). If False for 3D T1,
        a RuntimeError is raised. Default is False.

    Returns
    -------
    pathlib.Path
        Path to the written mask NIfTI (uint8).

    Files written
    -------------
    - Typically under: derivatives/neurofluid-mreg/sub-<ID>/masks/
      `sub-<ID>_space-<SPACE>_desc-brain_mask.nii.gz`
      (the exact filename is `out_path`).

    Assumptions / Preconditions
    ---------------------------
    - EPI: temporal mean is a reasonable surrogate for structural contrast
      for masking; background is near zero and signal is positive.
    - T1: direct nilearn masking is used only when `assume_mni=True` and
      the image has already been appropriately registered to an MNI-like
      grid.

    Warnings
    --------
    - EPI masking uses fixed parameters:
      `lower_cutoff=0.005`, `upper_cutoff=0.8`, `connected=True`,
      `opening=0`, `exclude_zeros=True`. These may require adjustment for
      atypical acquisition protocols.
    - For 4D inputs, Gaussian smoothing uses `sigma=1.0` in voxel units,
      applied independently to each timepoint.

    Raises
    ------
    RuntimeError
        If a 3D T1 volume is provided with `assume_mni=False`, instructing
        the caller to use `compute_mni_brain_mask_once` instead.
    """
    image = Path(image)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        print(f"[mask] [SKIP] Exists: {out_path.name}")
        return out_path

    img = nib.load(str(image))
    if len(img.shape) == 4 and img.shape[3] > 1:
        data = img.get_fdata(dtype=np.float32)

        # --- Light Gaussian smoothing (σ=1 voxel per timepoint) ---
        for t in range(data.shape[3]):
            data[..., t] = gaussian_filter(data[..., t], sigma=1.0)

        # Temporal mean (EPI reference)
        mean3d = np.nanmean(data, axis=3)
        epi_mean_img = nib.Nifti1Image(mean3d, img.affine, img.header)

        # EPI mask (nilearn)
        print("[mask] EPI mean computed; running nilearn.compute_epi_mask")
        mask_img = compute_epi_mask(
            epi_mean_img,
            lower_cutoff=0.005,
            upper_cutoff=0.8,
            connected=True,
            opening=0,
            exclude_zeros=True,
        )
        out = (mask_img.get_fdata() > 0.5).astype(np.uint8)
        nib.save(
            nib.Nifti1Image(out, epi_mean_img.affine, epi_mean_img.header),
            str(out_path),
        )
        print(f"[mask] Saved → {out_path}")
        return out_path

    # 3D (T1) path: ONLY use nilearn when already ~MNI
    if assume_mni:
        print("[mask] T1 assumed MNI; running nilearn.compute_brain_mask")
        mask_img = compute_brain_mask(img, mask_type="whole-brain")
        out = (mask_img.get_fdata() > 0.5).astype(np.uint8)
        nib.save(nib.Nifti1Image(out, img.affine, img.header), str(out_path))
        print(f"[mask] Saved → {out_path}")
        return out_path

    raise RuntimeError(
        "3D T1 provided but assume_mni=False. "
        "Use compute_mni_brain_mask_once() via warp_t1_to_mni_once()."
    )


# -------------------------------------------------------------
# Canonical MNI brain mask (single subject-level mask)
# -------------------------------------------------------------
def compute_mni_brain_mask_once(
    sp,
    xfm,  # TransformBook
    *,
    t1_path: Path,
    overwrite: bool = False,
    threshold: float = 0.5,
    opening: int | bool = 2,
    mask_type: str = "whole-brain",
) -> Path:
    """
    Compute a single subject-level brain mask on the T1-in-MNI space.

    This function warps the subject's T1 image to a canonical MNI template
    using `warp_t1_to_mni_once` (linear registration) and then computes a
    whole-brain mask on that MNI-aligned T1 using nilearn `compute_brain_mask`.
    The resulting mask is written once to disk and reused by downstream
    analyses that require a subject-level MNI brain mask.

    Parameters
    ----------
    sp
        SubjectPaths-like object providing `sub` and `masks_dir` attributes
        for BIDS-style naming and output directory resolution.
    xfm
        TransformBook-like object required by `warp_t1_to_mni_once` to
        manage T1→MNI registration and caching of the MNI-aligned T1.
    t1_path : pathlib.Path
        Path to the subject's structural T1 image in native space. This is
        passed directly to `warp_t1_to_mni_once` without autodiscovery.
    overwrite : bool, optional
        If False and the MNI brain mask already exists, the existing file is
        reused and the function returns immediately. Default is False.
    threshold : float, optional
        Threshold parameter forwarded to nilearn `compute_brain_mask`
        (fraction of robust max). Default is 0.5.
    opening : int or bool, optional
        Morphological opening parameter forwarded to `compute_brain_mask`.
        Default is 2 (nilearn default-like).
    mask_type : str, optional
        Mask type passed to `compute_brain_mask`. Default is "whole-brain".

    Returns
    -------
    pathlib.Path
        Path to the subject-level MNI brain mask NIfTI:
        `sub-<ID>_space-MNI_desc-brain_mask.nii.gz`.

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/masks/
      sub-<ID>_space-MNI_desc-brain_mask.nii.gz

    Assumptions / Preconditions
    ---------------------------
    - `warp_t1_to_mni_once` successfully produces a T1-in-MNI NIfTI volume
      whose header zooms are in mm on a canonical MNI grid (typically 1 mm).
    - `sp.masks_dir` exists or can be created; `sp.sub` matches the subject
      ID used in other derivatives.
    - The T1 image at `t1_path` is structurally suitable for standard
      whole-brain masking (e.g., no severe artifacts).

    Warnings
    --------
    - Units for the MNI mask are inherited from the MNI template used inside
      `warp_t1_to_mni_once` and are assumed to be millimeters; verify if a
      non-standard template is introduced.
    - `threshold`, `opening`, and `mask_type` can substantially alter the
      extent of the mask; defaults are geared toward whole-brain coverage
      and may need tuning for specific applications.

    Raises
    ------
    RuntimeError
        Propagated if `warp_t1_to_mni_once` fails internally (e.g., missing
        reference template or registration issues).
    """
    sub = sp.sub
    masks_dir = Path(sp.masks_dir)
    out_mask = masks_dir / f"{sub}_space-MNI_desc-brain_mask.nii.gz"
    if out_mask.exists() and not overwrite:
        return out_mask

    t1_mni_path = warp_t1_to_mni_once(sp, xfm, t1_path=t1_path, overwrite=False)
    t1_mni_img = nib.load(str(t1_mni_path))

    mask_img = compute_brain_mask(
        t1_mni_img,
        threshold=threshold,
        connected=True,
        opening=opening,
        mask_type=mask_type,
    )
    nib.save(mask_img, str(out_mask))
    print(f"[mask] Saved subject MNI brain mask → {out_mask}")
    return out_mask
