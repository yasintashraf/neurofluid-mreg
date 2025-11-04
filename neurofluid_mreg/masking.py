# SPDX-License-Identifier: MIT
"""
masking.py
----------
Brain masking utilities for Neurofluid–MREG. Provides EPI/mean-BOLD masking,
T1→MNI masking with nilearn on a canonical grid, and projection of masks back
to T1 and MREG using label-safe warps. Affines/headers are preserved on write;
this module performs registrations/resampling as needed for masking only.

Pipeline steps
--------------
1. Mean-BOLD (EPI) mask in native space (nilearn `compute_epi_mask`)
2. T1→MNI registration (rigid prealign + SyN) or external warp usage
3. Whole-brain mask on canonical MNI grid (nilearn `compute_brain_mask`)
4. Project mask MNI→T1 and T1→MREG using **nearest** interpolation

Inputs / Outputs
----------------
Inputs  : Paths to native-space NIfTI images (EPI/T1/MREG), optional transform
          provider `xfm` (supports mapping cache, `warp_image`, `warp_labels`).
Outputs : Paths to written mask NIfTIs (uint8); no arrays returned here.

Files written
-------------
- derivatives/neurofluid-mreg/sub-<ID>/masks/
  - sub-<ID>_space-MNI_desc-brain_mask.nii.gz
  - sub-<ID>_space-T1_desc-brain_mask.nii.gz
  - sub-<ID>_space-MREG_desc-brain_mask.nii.gz

Assumptions / Preconditions
---------------------------
- Spaces: T1/MREG/MNI as indicated by filenames or arguments.
- Affines: For MNI masking, data are placed on a canonical 1 mm MNI grid;
  projections back to T1/MREG use label-safe nearest interpolation.
- Shapes/dtypes: Masks written as uint8; intermediate arrays float32.
- BIDS naming: Filenames follow `sub-<ID>_space-<SPACE>_desc-brain_mask.nii.gz`.

Warnings
--------
- Resampling outside FOV or large affine mismatches may trim edges.
- EPI masking thresholds (`lower_cutoff`, `upper_cutoff`) are fixed here
  and may require tuning for atypical contrasts.

Public API
----------
- mask_brain
- compute_t1_mask_via_mni_and_project
"""

# neurofluid_mreg/masking.py
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import nibabel as nib

from scipy.ndimage import gaussian_filter
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_template
from nilearn.masking import compute_brain_mask, compute_epi_mask

from dipy.align.imaffine import (
    transform_centers_of_mass,
    AffineRegistration,
    MutualInformationMetric,
)
from dipy.align.transforms import RigidTransform3D
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric


# -------------------------------------------------------------
# Masking (MREG/T1/MNI paths)
# -------------------------------------------------------------
def mask_brain(
    image: Path, out_path: Path, overwrite: bool = False, assume_mni: bool = False
) -> Path:
    """
    Create a brain mask and write it to disk.

    For 4D EPI/BOLD inputs, computes a temporal mean after light Gaussian
    smoothing (σ=1 voxel per timepoint) and runs nilearn `compute_epi_mask`.
    For 3D T1 inputs, nilearn `compute_brain_mask` is allowed only when the
    image is already ~MNI (`assume_mni=True`); otherwise a RuntimeError is
    raised to force the T1→MNI pathway.

    Parameters
    ----------
    image : pathlib.Path
        Input NIfTI path in native space (3D T1 or 4D EPI/BOLD).
    out_path : pathlib.Path
        Destination mask path; parent directories are created if missing.
    overwrite : bool, optional
        If False and `out_path` exists, returns immediately. Default: False.
    assume_mni : bool, optional
        If True for 3D T1, run `compute_brain_mask` directly on T1 (assumed
        on/near MNI grid). Default: False.

    Returns
    -------
    pathlib.Path
        Path to the written mask NIfTI (uint8).

    Files written
    -------------
    - Typically under: derivatives/neurofluid-mreg/sub-<ID>/masks/
      `sub-<ID>_space-<SPACE>_desc-brain_mask.nii.gz` (exact file is `out_path`).

    Assumptions / Preconditions
    ---------------------------
    - EPI: temporal mean is meaningful for masking; background near zero.
    - T1: direct nilearn masking only when `assume_mni=True`.

    Warnings
    --------
    - Fixed EPI parameters: `lower_cutoff=0.005`, `upper_cutoff=0.8`,
      `connected=True`, `opening=0`, `exclude_zeros=True`.

    Raises
    ------
    RuntimeError
        If a 3D T1 is provided with `assume_mni=False`.
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
        nib.save(nib.Nifti1Image(out, epi_mean_img.affine, epi_mean_img.header),str(out_path),)
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
        "3D T1 provided but assume_mni=False. Use compute_t1_mask_via_mni_and_project()."
    )


# -------------------------------------------------------------
# Applying transforms (registration / resampling)
# -------------------------------------------------------------
def _register_t1_to_mni_inline(
    t1_img: nib.Nifti1Image, level_iters=(50, 20, 10)
) -> tuple[nib.Nifti1Image, nib.Nifti1Image, object]:
    """
    Register T1→MNI (1 mm) with rigid prealignment + SyN (CC).

    Performs CoM alignment, short rigid optimization with MI, then SyN with a
    cross-correlation metric. Returns the T1 warped to the canonical MNI grid,
    along with the MNI reference image and the deformation mapping.

    Parameters
    ----------
    t1_img : nibabel.Nifti1Image
        Moving T1 image in native space.
    level_iters : tuple of int, optional
        SyN multi-resolution iterations (coarse→fine). Default: (50, 20, 10).

    Returns
    -------
    tuple
        (t1_on_mni_img, mni_ref_img, mapping_or_None)

    Files written
    -------------
    - None.

    Assumptions / Preconditions
    ---------------------------
    - Uses `nilearn.datasets.load_mni152_template(resolution=1)` as fixed grid.
    - DIPY registration available (`AffineRegistration`, `SyN`).

    Warnings
    --------
    - Computationally expensive; used as a fallback when external warps are
      unavailable.

    Notes
    -----
    - Prealignment via centers-of-mass + rigid MI; SyN(CC) refines deformation.
    """
    mni_img = load_mni152_template(resolution=1)
    moving = t1_img.get_fdata().astype(np.float32)
    static = mni_img.get_fdata().astype(np.float32)

    print("[xfm] Prealign T1→MNI (CoM + rigid MI)")
    # CoM + short rigid prealignment
    pre = transform_centers_of_mass(
        static, mni_img.affine, moving, t1_img.affine
    ).affine
    mi = MutualInformationMetric(nbins=32)
    affreg = AffineRegistration(metric=mi, level_iters=[1000, 100, 10])
    rigid = RigidTransform3D()
    opt = affreg.optimize(
        static,
        moving,
        rigid,
        None,
        static_grid2world=mni_img.affine,
        moving_grid2world=t1_img.affine,
        starting_affine=pre,
    )
    pre = opt.affine

    # SyN (CC)
    print(f"[xfm] SyN(CC) with level iters = {list(level_iters)}")
    sdr = SymmetricDiffeomorphicRegistration(CCMetric(3), level_iters=list(level_iters))
    mapping = sdr.optimize(
        static=static,
        moving=moving,
        static_grid2world=mni_img.affine,
        moving_grid2world=t1_img.affine,
        prealign=pre,
    )
    moved = mapping.transform(moving, interpolation="linear")
    t1_mni_img = nib.Nifti1Image(moved.astype(np.float32, copy=False), mni_img.affine, mni_img.header)
    print("[xfm] Inline T1→MNI complete (canonical grid)")
    return t1_mni_img, mni_img, mapping


# -------------------------------------------------------------
# Masking (MREG/T1/MNI paths)
# -------------------------------------------------------------
def compute_t1_mask_via_mni_and_project(
    *,
    sub: str,
    t1_path: Path,
    mreg_ref_path: Path,
    xfm,  # your TransformBook
    masks_dir: Path,
    overwrite: bool = False,
    level_iters=(50, 20, 10),
) -> Dict[str, str]:
    """
    Compute a whole-brain mask on canonical MNI and project to T1 and MREG.

    Workflow
    --------
    1) Put T1 on a canonical MNI 1 mm grid (prefer cached mapping; else
       `xfm.warp_image`; else inline registration fallback).
    2) Run `nilearn.masking.compute_brain_mask` on that canonical grid.
    3) Project mask to T1 and then to MREG using **nearest** interpolation
       (label-safe) via `xfm.warp_labels`.
    4) Save 3 masks using simple f-string names consistent with earlier code.

    Parameters
    ----------
    sub : str
        Subject token (e.g., "sub-xh33_x107").
    t1_path : pathlib.Path
        Native-space T1 NIfTI path (moving).
    mreg_ref_path : pathlib.Path
        MREG reference NIfTI path (target for final projection).
    xfm : object
        Transform provider. May expose:
        - `mni_ref_path` (optional canonical MNI reference),
        - `_mapping_cache` (optional deformation cache),
        - `warp_image(moving_img, reference_space, out_path, chain, interpolation)`,
        - `warp_labels(moving_img, reference_img, out_path, chain, interpolation)`.
    masks_dir : pathlib.Path
        Output directory for mask files (created if missing).
    overwrite : bool, optional
        Overwrite existing masks. Default: False.
    level_iters : tuple of int, optional
        Inline SyN iterations if fallback registration is used.

    Returns
    -------
    dict
        {"MNI": <path>, "T1": <path>, "MREG": <path>} — stringified paths.

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/masks/
      - sub-<ID>_space-MNI_desc-brain_mask.nii.gz
      - sub-<ID>_space-T1_desc-brain_mask.nii.gz
      - sub-<ID>_space-MREG_desc-brain_mask.nii.gz

    Assumptions / Preconditions
    ---------------------------
    - The canonical MNI reference is either provided via `xfm.mni_ref_path`
      or loaded from nilearn (1 mm).
    - Masks are binarized with threshold > 0.5 and saved as uint8.
    - Projections use **nearest** interpolation to preserve labels.

    Warnings
    --------
    - If `warp_image` fails or is unavailable, a compute-heavy fallback
      registration is used.
    - Grid is enforced via `resample_to_img` to match the canonical MNI
      reference exactly (shape/affine).

    Notes
    -----
    - Operates in the target image space when affines differ; label warps are
      discrete to avoid interpolation artifacts.
    """
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Filenames (retain the original #2 style)
    p_mni = masks_dir / f"{sub}_space-MNI_desc-brain_mask.nii.gz"
    p_t1 = masks_dir / f"{sub}_space-T1_desc-brain_mask.nii.gz"
    p_mreg = masks_dir / f"{sub}_space-MREG_desc-brain_mask.nii.gz"

    # Canonical MNI reference — single grid for masking
    if getattr(xfm, "mni_ref_path", None):
        mni_ref_img = nib.load(str(xfm.mni_ref_path))
    else:
        mni_ref_img = load_mni152_template(resolution=1)

    # 1) T1 on MNI grid (cache -> warp_image -> inline reg)
    t1_img = nib.load(str(t1_path))
    t1_on_mni_img = None

    mapping = getattr(xfm, "_mapping_cache", None)
    if mapping is not None:
        print("[xfm] Using cached T1→MNI deformation")
        moved = mapping.transform(t1_img.get_fdata().astype(np.float32),interpolation="linear",)
        t1_on_mni_img = nib.Nifti1Image(moved.astype(np.float32, copy=False),mni_ref_img.affine,mni_ref_img.header,)
    else:
        tmp_t1_on_mni = masks_dir / f"{sub}_space-MNI_desc-t1onmni_temp.nii.gz"
        try:
            if hasattr(xfm, "warp_image"):
                print("[xfm] Using xfm.warp_image for T1→MNI")
                xfm.warp_image(
                    moving_img=str(t1_path),
                    reference_space="MNI",
                    out_path=str(tmp_t1_on_mni),
                    chain=("T1", "MNI"),
                    interpolation="linear",
                )
                t1_on_mni_img = nib.load(str(tmp_t1_on_mni))
            else:
                raise AttributeError("xfm.warp_image not available")
        except Exception:
            # Fallback identical in spirit to your original
            # Expected to return (img_on_mni, mni_like, mapping_or_none)
            print("[xfm] [WARN] warp_image unavailable/failed; running inline registration")
            t1_on_mni_img, _, _ = _register_t1_to_mni_inline(
                t1_img, level_iters=level_iters
            )
        finally:
            try:
                if "tmp_t1_on_mni" in locals() and tmp_t1_on_mni.exists():
                    tmp_t1_on_mni.unlink(missing_ok=True)
            except Exception:
                pass

    # Enforce exact grid match to canonical MNI reference (critical)
    if (t1_on_mni_img.shape != mni_ref_img.shape
        or not np.allclose(t1_on_mni_img.affine, mni_ref_img.affine, atol=1e-3)):
        t1_on_mni_img = resample_to_img(t1_on_mni_img, mni_ref_img, interpolation="linear")
        print("[xfm] [WARN] Resampled T1-on-MNI to canonical MNI grid")


    # 2) Compute whole-brain mask on canonical MNI grid
    mask_mni_img = compute_brain_mask(t1_on_mni_img, mask_type="whole-brain")
    mask_mni_bin = (mask_mni_img.get_fdata() > 0.5).astype(np.uint8)
    mask_mni_img = nib.Nifti1Image(
        mask_mni_bin, t1_on_mni_img.affine, mni_ref_img.header
    )
    mask_mni_img.set_data_dtype(np.uint8)
    if (not p_mni.exists()) or overwrite:
        nib.save(mask_mni_img, str(p_mni))
        print(f"[mask] Saved → {p_mni}")
    else:
        print(f"[mask] [SKIP] Exists: {p_mni.name}")

    # 3) Project MNI -> T1 (label-safe)
    if (not p_t1.exists()) or overwrite:
        xfm.warp_labels(
            moving_img=str(p_mni),
            reference_img=str(t1_path),
            out_path=str(p_t1),
            chain=("MNI", "T1"),
            interpolation="nearest",
        )
        _img = nib.load(str(p_t1))
        _dat = (_img.get_fdata() > 0.5).astype(np.uint8)
        nib.save(nib.Nifti1Image(_dat, _img.affine, _img.header), str(p_t1))
        print(f"[mask] Saved → {p_t1}")
    else:
        print(f"[mask] [SKIP] Exists: {p_t1.name}")

    # 4) Project T1 -> MREG (label-safe)
    if (not p_mreg.exists()) or overwrite:
        xfm.warp_labels(
            moving_img=str(p_t1),
            reference_img=str(mreg_ref_path),
            out_path=str(p_mreg),
            chain=("T1", "MREG"),
            interpolation="nearest",
        )
        _img = nib.load(str(p_mreg))
        _dat = (_img.get_fdata() > 0.5).astype(np.uint8)
        nib.save(nib.Nifti1Image(_dat, _img.affine, _img.header), str(p_mreg))
        print(f"[mask] Saved → {p_mreg}")
    else:
        print(f"[mask] [SKIP] Exists: {p_mreg.name}")

    return {"MNI": str(p_mni), "T1": str(p_t1), "MREG": str(p_mreg)}
