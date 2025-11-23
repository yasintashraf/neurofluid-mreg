# SPDX-License-Identifier: MIT
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
# pyright: reportUndefinedVariable=false
"""
transforms.py
-------------
Unified registration and resampling utilities for Neurofluid–MREG.

This module provides small, deterministic helpers to estimate affine/nonlinear
transforms among native modalities (TOF/MRV/hT2w), T1, and MNI, and to apply
those transforms for image/label resampling. It also includes a lightweight
bookkeeping class (`TransformBook`) and a minimal wrapper to denoise MP2RAGE.

Pipeline steps
--------------
1. Affine registration: SOURCE → T1 (DIPY, MI; translation → rigid → affine).
2. Nonlinear registration: T1 → MNI (DIPY SyN, CC; optional).
3. Resampling helpers: apply affine (auto world/voxel detection) to a reference grid; compose affine+warp.
4. Bookkeeping: save/read transform files; chain/invert; label-safe warps.

Inputs / Outputs
----------------
Inputs  : NIfTI volumes in native spaces (TOF/MRV/hT2w), T1, MREG mean; optional MNI.
Outputs : Transform text files (.txt, 4×4),Matrices may be in world→world or voxel→voxel space; functions auto–normalize.
          warp fields (NIfTI, vector fields),and resampled image/label NIfTI as requested by callers.

Files written
-------------
- derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-<TAG>toT1.txt
- derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-T1toMNI_warp.nii.gz
- derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-MNItoT1_warp.nii.gz
- derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_space-MNI_desc-T1w_in-MNI_map.nii.gz
- derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-T1toMREG.txt
- derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-MREGMEANtoMNI.txt
- Additional resampled artifacts per caller (e.g., masks/radii) using
  `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-<DESC>_<SUFFIX>.nii.gz`.

Assumptions / Preconditions
---------------------------
- Spaces: inputs are in their native spaces (TOF/MRV/hT2w/MREG) or T1; MNI is
  resolved to MNI152 1 mm when needed. Affines may be world→world or voxel→voxel; 
  functions pick the correct convention by center-matching in world space.
- Shapes/dtypes: images are float32/float64 on load; outputs are float32 unless
  a label is requested (nearest resampling downstream).
- BIDS naming: filenames follow `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-<DESC>_<SUFFIX>.nii.gz`
  for NIfTI artifacts; transforms are saved as `_xfm-*.txt` / `_warp.nii.gz`.

Warnings
--------
- If a requested MNI template is unavailable, nonlinear registration is skipped
  gracefully and only affine outputs are produced.
- Resampling outside FOV returns zeros; labels are resampled with nearest-neighbor.
- If affine matrices use an unexpected convention, the module selects the closest 
  valid interpretation (world or voxel) based on center alignment.

Public API
----------
- register_t1_to_hT2w
- register_to_t1
- register_t1_to_mni
- apply_affine_to_ref
- apply_affine_then_warp_to_mni
- TransformBook
- warp_radii_to_mreg
- run_mp2rage_denoise
- push_to_mni_float
- push_to_mni_label
- warp_t1_to_mni_once
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple
import os
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

# DIPY imports for affine + nonlinear registration
from dipy.align.imaffine import (
    AffineRegistration,
    MutualInformationMetric,
    AffineMap,
    transform_centers_of_mass)
from dipy.align.transforms import (
    TranslationTransform3D,
    RigidTransform3D,
    AffineTransform3D)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
# Local application imports
from neurofluid_mreg.mp2rage_denoise import mp2rage_denoise
from .io import SubjectPaths, deriv_name

# -------------------------------------------------------------
# Utilities (logging, checks)
# -------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    """Create directory `p` and parents if missing (idempotent)."""
    p.mkdir(parents=True, exist_ok=True)

def _save_affine_txt(affine: np.ndarray, out_txt: Path) -> Path:
    """
    Save a 4×4 affine matrix to a plain-text file.

    Parameters
    ----------
    affine : ndarray, shape (4, 4), dtype=float64
        Affine matrix (will be saved with high precision).
    out_txt : pathlib.Path
        Output text path. Parent folders are created if missing.

    Returns
    -------
    pathlib.Path
        The `out_txt` path (for chaining).
    """
    out_txt = Path(out_txt)
    _ensure_dir(out_txt.parent)
    np.savetxt(out_txt, affine, fmt="%.8f")
    print(f"[xfm] Saved → {out_txt}")
    return out_txt

def _load_affine_txt(txt_path) -> np.ndarray:
    """
    Load a 4×4 affine matrix from a plain-text file.

    Parameters
    ----------
    txt_path : str or pathlib.Path
        Path to text file created by `_save_affine_txt`.

    Returns
    -------
    ndarray, shape (4, 4), dtype=float64
        Loaded affine matrix.
    """
    return np.loadtxt(str(txt_path))

def _resolve_mni_path() -> Path:
    """
    Resolve a canonical MNI152 1 mm template path.

    Returns
    -------
    pathlib.Path
        Cached path under `~/.cache/neurofluid_mreg/MNI152_T1_1mm.nii.gz`.

    Notes
    -----
    Uses `nilearn.datasets.load_mni152_template` and writes a local cached copy.
    """
    from nilearn.datasets import load_mni152_template
    out = Path(os.path.expanduser("~")) / ".cache" / "neurofluid_mreg" / "MNI152_T1_1mm.nii.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        img = load_mni152_template(resolution=1)  # returns Nifti1Image
        nib.save(img, str(out))
        print(f"[xfm] Cached MNI template → {out}")
    return out


# -------------------------------------------------------------
# Core registration functions
# -------------------------------------------------------------
def register_t1_to_hT2w(
    moving: np.ndarray, moving_aff: np.ndarray,
    static: np.ndarray, static_aff: np.ndarray,
    *,
    reg_mode: str = "rigid_affine",
    use_nonlin: bool = False,
    mi_bins: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Register T1 (moving) to heavy-T2w (static) image using mutual-information
    affine registration, with optional SyN nonlinear refinement.

    Parameters
    ----------
    moving : ndarray, shape (X, Y, Z), dtype=float32 or float64
        T1-weighted (UNI or denoised) volume to be aligned.
    moving_aff : ndarray, shape (4, 4)
        Affine of the moving image.
    static : ndarray, shape (X, Y, Z), dtype=float32 or float64
        Heavy-T2w target image defining the output grid.
    static_aff : ndarray, shape (4, 4)
        Affine of the static image.
    reg_mode : {"rigid", "rigid_affine"}, optional
        Registration mode. If `"rigid_affine"`, runs translation→rigid→affine
        stages; otherwise stops after the rigid transform.
    use_nonlin : bool, optional
        If True, perform an additional SyN (symmetric diffeomorphic)
        refinement using cross-correlation metric after affine alignment.
    mi_bins : int, optional
        Number of histogram bins for mutual information metric.

    Returns
    -------
    moved : ndarray, shape identical to `static`, dtype=float32
        Moving (T1) volume resampled to the static grid.
    affine_used : ndarray, shape (4, 4)
        Final affine matrix used before optional SyN refinement.

    Files written
    -------------
    - None (returns arrays only; caller may save to
      `sub-<ID>_space-hT2w_class-T1_desc-registered_map.nii.gz`).

    Assumptions / Preconditions
    ---------------------------
    - Input images are bias-corrected, brain-masked, and roughly aligned.
    - Operates entirely in image space using DIPY transforms.
    - Output shape must exactly match `static`; raises if mismatch.

    Warnings
    --------
    - If `use_nonlin=True`, nonlinear stage may slightly smooth intensity edges.
    - Output intensities are clipped to [0, 1] and NaNs/inf replaced with 0/1.

    Raises
    ------
    RuntimeError
        If final resampled volume shape differs from `static`.

    Notes
    -----
    - Uses DIPY’s mutual-information metric with three-level pyramid.
    - Typical runtime: 10–30 min (depends on image size and `use_nonlin` flag).
    """

    com = transform_centers_of_mass(static, static_aff, moving, moving_aff)

    metric = MutualInformationMetric(nbins=mi_bins, sampling_proportion=0.3)
    level_iters = [1000, 100, 10]  # tuned elsewhere; logic unchanged
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(
        metric=metric,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors,
    )
    print(f"[xfm] Affine pyramid (T1→hT2w): iters={level_iters}, sigmas={sigmas}, factors={factors}")
    trans = affreg.optimize(
        static, moving,
        TranslationTransform3D(), params0=None,
        static_grid2world=static_aff, moving_grid2world=moving_aff,
        starting_affine=com.affine,
    )

    rigid = affreg.optimize(
        static, moving,
        RigidTransform3D(), params0=None,
        static_grid2world=static_aff, moving_grid2world=moving_aff,
        starting_affine=trans.affine,
    )

    best_map = rigid
    if reg_mode.lower() in {"rigid_affine", "affine"}:
        best_map = affreg.optimize(
            static, moving,
            AffineTransform3D(), params0=None,
            static_grid2world=static_aff, moving_grid2world=moving_aff,
            starting_affine=rigid.affine,
        )

    moved = best_map.transform(moving).astype(np.float32)

    if use_nonlin:
        syn_metric = CCMetric(3)
        sdr = SymmetricDiffeomorphicRegistration(syn_metric, [10, 10, 5])
        print("[xfm] SyN pyramid (T1→hT2w): iters=[10, 10, 5]")
        mapping = sdr.optimize(
            static, moving,
            static_affine=static_aff, moving_affine=moving_aff,
            starting_affine=best_map.affine,
        )
        moved = mapping.transform(moving).astype(np.float32)
        affine_used = best_map.affine  # record the affine stage used pre-SyN
    else:
        affine_used = best_map.affine

    moved = np.nan_to_num(moved, nan=0.0, posinf=1.0, neginf=0.0)
    moved = np.clip(moved, 0.0, 1.0).astype(np.float32)

    if moved.shape != static.shape:
        raise RuntimeError(f"DIPY resample mismatch: {moved.shape} vs {static.shape}")

    return moved, affine_used

def register_to_t1(
    src_img_path: Path,
    ref_t1_path: Path,
    out_dir: Path,
    *,
    sub_id: str,
    modality_tag: str,
    allow_4d_mean: bool = True,
    overwrite: bool = False, 
) -> Path:
    """
    Register a source image to T1 with MI-based affine (translation→rigid→affine).

    Parameters
    ----------
    src_img_path : pathlib.Path
        Moving image (TOF/MRV/hT2w/MREG). If 4D and `allow_4d_mean=True`,
        the temporal mean is used.
    ref_t1_path : pathlib.Path
        Fixed T1 image (denoised MP2RAGE recommended).
    out_dir : pathlib.Path
        Output directory for transforms (anat derivatives).
    sub_id : str
        Subject identifier including `sub-` prefix.
    modality_tag : str
        Token for filename tag (e.g., "TOF", "MRV", "hT2w", "MREG").
    allow_4d_mean : bool, default=True
        If True and `src_img_path` is 4D, average over time before registration.

    Returns
    -------
    pathlib.Path
        Path to `sub-<ID>_xfm-<TAG>toT1.txt` (4×4 affine).

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-<TAG>toT1.txt

    Assumptions / Preconditions
    ---------------------------
    - Inputs are 3D (or 4D averaged) and share valid NIfTI affines.
    - Uses DIPY with 32-bin mutual information.

    Raises
    ------
    ValueError
        If a 4D source is provided and `allow_4d_mean=False`.

    Notes
    -----
    - The saved 4×4 is a world→world affine (mm) returned by DIPY’s optimizer.
    -  Downstream resampling always forces the output to the requested reference grid.    
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # Load images
    moving_img = nib.load(str(src_img_path))
    fixed_img = nib.load(str(ref_t1_path))
    moving = moving_img.get_fdata().astype(np.float32)
    static = fixed_img.get_fdata().astype(np.float32)

    # Reduce 4D to temporal mean if requested
    if moving.ndim == 4:
        if not allow_4d_mean:
            raise ValueError("4D input provided but allow_4d_mean=False")
        moving = moving.mean(axis=3).astype(np.float32)
        modality_tag = "MREGMEAN" if modality_tag.upper().startswith("MREG") else modality_tag
        print(f"[xfm] {modality_tag}: used 4D temporal mean for registration")

    # Output path now that tag is final
    out_txt = out_dir / f"{sub_id}_xfm-{modality_tag}toT1.txt"

    # --- skip if present and valid ---
    if out_txt.exists() and not overwrite:
        try:
            A = np.loadtxt(out_txt)
            if A.shape == (4, 4):
                print(f"[xfm] [SKIP] Exists: {out_txt.name}")
                return out_txt
            else:
                print(f"[xfm] [WARN] Bad shape in {out_txt.name} → recompute")
        except Exception as e:
            print(f"[xfm] [WARN] Could not read {out_txt.name} ({e}) → recompute")

    # Initial COM alignment
    com = transform_centers_of_mass(static, fixed_img.affine, moving, moving_img.affine)

    # MI metric (good for cross-contrast)
    mi = MutualInformationMetric(nbins=32, sampling_proportion=None)

    # Multires pyramid
    level_iters = [1000, 100, 10]  # tuned elsewhere; logic unchanged
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    areg = AffineRegistration(metric=mi, level_iters=level_iters, sigmas=sigmas, factors=factors)
    print(f"[xfm] Affine pyramid ({modality_tag}→T1): iters={level_iters}, sigmas={sigmas}, factors={factors}")

    # Stage 1: translation
    xform = TranslationTransform3D()
    opt = areg.optimize(
        static, moving, xform, None,
        static_grid2world=fixed_img.affine,
        moving_grid2world=moving_img.affine,
        starting_affine=com.affine,
    )

    # Stage 2: rigid
    xform = RigidTransform3D()
    opt = areg.optimize(
        static, moving, xform, None,
        static_grid2world=fixed_img.affine,
        moving_grid2world=moving_img.affine,
        starting_affine=opt.affine,
    )

    # Stage 3: full affine
    xform = AffineTransform3D()
    opt = areg.optimize(
        static, moving, xform, None,
        static_grid2world=fixed_img.affine,
        moving_grid2world=moving_img.affine,
        starting_affine=opt.affine,
    )

    # Save 4×4 matrix
    _save_affine_txt(opt.affine, out_txt)
    return out_txt

def register_t1_to_mni(
    t1_path: Path,
    mni_path: Optional[Path],
    out_dir: Path,
    *,
    sub_id: str,
    level_iters: Tuple[int, int, int] = (1000,100,10), 
) -> Tuple["DiffeomorphicMap", Path, Path]:
    """
    Nonlinear SyN (CC metric) registration of T1 to MNI152 1 mm.

    Parameters
    ----------
    t1_path : pathlib.Path
        Moving T1 image (native T1 space).
    mni_path : pathlib.Path or None
        MNI reference. If None, a canonical template is resolved.
    out_dir : pathlib.Path
        Output directory for warp fields (anat derivatives).
    sub_id : str
        Subject identifier including `sub-` prefix.
    level_iters : tuple of int, default=(1000, 100, 10)
        Pyramid iterations for SyN.

    Returns
    -------
    (mapping, fwd_path, inv_path) : tuple
        `mapping` is a DIPY `DiffeomorphicMap`; paths point to saved warps.

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-T1toMNI_warp.nii.gz
    - derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-MNItoT1_warp.nii.gz

    Assumptions / Preconditions
    ---------------------------
    - Inputs are 3D volumes with valid affines. A short rigid MI pre-alignment
      is performed before SyN.

    Warnings
    --------
    - If exporting vector fields fails, a compressed `.npz` with error info is
      written; returned paths may not exist on disk.
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # Always resolve canonical MNI if not provided/doesn't exist
    mni_path = _resolve_mni_path()

    t1_img  = nib.load(str(t1_path))
    mni_img = nib.load(str(mni_path))
    moving  = t1_img.get_fdata().astype(np.float32)   # T1 (moving)
    static  = mni_img.get_fdata().astype(np.float32)  # MNI (fixed)

    # --------- improved pre-alignment ----------
    # 1) translation via centers of mass
    pre = transform_centers_of_mass(static, mni_img.affine, moving, t1_img.affine).affine

    # 2) short rigid MI optimization (handles rotation/scale better than COM alone)
    mi = MutualInformationMetric(nbins=32)
    level_iters = [1000, 100, 10]
    affreg = AffineRegistration(metric=mi, level_iters=level_iters)
    print(f"[xfm] Pre-rigid pyramid (T1→MNI): iters={level_iters}")
    rigid = RigidTransform3D()
    opt = affreg.optimize(
        static, moving, rigid, None,
        static_grid2world=mni_img.affine,
        moving_grid2world=t1_img.affine,
        starting_affine=pre,
    )
    pre = opt.affine  # feed this into SyN

    # --------- SyN (CC)
    sdr = SymmetricDiffeomorphicRegistration(CCMetric(3), level_iters=list(level_iters))
    print(f"[xfm] SyN pyramid (T1→MNI): iters={list(level_iters)}")
    mapping = sdr.optimize(
        static=static,
        moving=moving,
        static_grid2world=mni_img.affine,
        moving_grid2world=t1_img.affine,
        prealign=pre,
    )

    # Save deformation fields (defined on STATIC/MNI grid)
    fwd_path = out_dir / f"{sub_id}_xfm-T1toMNI_warp.nii.gz"
    inv_path = out_dir / f"{sub_id}_xfm-MNItoT1_warp.nii.gz"
    try:
        fwd = mapping.get_forward_field().astype(np.float32)   # T1→MNI
        inv = mapping.get_backward_field().astype(np.float32)  # MNI→T1
        nib.save(nib.Nifti1Image(fwd, mni_img.affine), str(fwd_path))
        print(f"[xfm] Saved → {fwd_path}")
        nib.save(nib.Nifti1Image(inv, mni_img.affine), str(inv_path))
        print(f"[xfm] Saved → {inv_path}")
    except Exception as e:
        np.savez_compressed(out_dir / f"{sub_id}_xfm-T1MNI_mapping_export_error.npz", info=str(e))
        print(f"[xfm] [WARN] Failed to export warps; wrote error NPZ for {sub_id}")

    return mapping, fwd_path, inv_path


# -------------------------------------------------------------
# Applying transforms (resampling)
# -------------------------------------------------------------
def apply_affine_to_ref(
    image_path: Path,
    affine,                      # Path | np.ndarray (4x4)
    ref_img_path: Path,
    out_path: Path,
    *,
    interpolation: str = "linear",  # "linear" for images, "nearest" for labels
) -> Path:
    """
    Apply a 4×4 transform and resample a moving image onto a reference grid.

    This implementation uses `nibabel.processing.resample_from_to` and
    automatically disambiguates whether the 4×4 matrix is given in
    world→world or voxel→voxel (index→index) coordinates, choosing the
    convention that best aligns the image centers in world space.

    Parameters
    ----------
    image_path : pathlib.Path
        Moving image path (3D or 4D).
    affine : pathlib.Path or ndarray, shape (4, 4)
        4×4 transform relating the moving and reference images.
        If a path, the matrix is loaded via `_load_affine_txt`.
        The matrix may be:
        - world→world (moving→reference), or
        - voxel→voxel (moving indices→reference indices),
        and the function will pick the interpretation that maps the moving
        image center closest to the reference center in world coordinates.
    ref_img_path : pathlib.Path
        Reference image defining the target grid (shape and affine).
    out_path : pathlib.Path
        Output NIfTI path.
    interpolation : {"linear", "nearest"}, default="linear"
        Interpolation mode used by `resample_from_to`;
        use `"nearest"` for label images.

    Returns
    -------
    pathlib.Path
        `out_path`.

    Files written
    -------------
    - Output NIfTI at `out_path` with:
      - data resampled onto `ref_img_path`'s grid, and
      - affine equal to the reference image affine.

    Assumptions / Preconditions
    ---------------------------
    - The 4×4 transform is either world→world or voxel→voxel between the
      moving and reference images; other conventions are not handled.
    - For 4D inputs, all volumes are resampled with the same spatial
      transform (handled internally by `resample_from_to`).
    - Data are cast to float32 before resampling.
    - The moving header is reused with an updated affine (`A_best @ mov.affine`)
      purely to encode the chosen transform for resampling; the final output
      geometry is taken from the reference image.
    """

    # --- load moving & reference ---
    mov = nib.load(str(image_path))
    ref = nib.load(str(ref_img_path))

    # --- load 4x4 ---
    A = _load_affine_txt(Path(affine)) if isinstance(affine, (str, Path)) else np.asarray(affine, dtype=np.float32)
    if A.shape != (4, 4):
        raise ValueError(f"affine must be 4x4, got {A.shape}")

    # --- build both interpretations, pick the one that maps centers best ---
    def center_world(img):
        i, j, k = (np.array(img.shape[:3]) - 1) / 2.0
        return img.affine @ np.array([i, j, k, 1.0])

    c_mov = center_world(mov)[:3]
    c_ref = center_world(ref)[:3]

    # Option A: txt is world->world
    A_world = A
    dA = np.linalg.norm((A_world @ np.append(c_mov, 1))[:3] - c_ref)

    # Option B: txt is voxel->voxel  (convert to world->world)
    A_vox2world = ref.affine @ A @ np.linalg.inv(mov.affine)
    dB = np.linalg.norm((A_vox2world @ np.append(c_mov, 1))[:3] - c_ref)

    A_best = A_world if dA <= dB else A_vox2world

    # --- embed A_best into moving header, then force-resample onto ref grid ---
    order = 0 if interpolation == "nearest" else 1
    mov_prime = nib.Nifti1Image(mov.get_fdata().astype("float32"), A_best @ mov.affine, mov.header)
    out_img = resample_from_to(mov_prime, (ref.shape, ref.affine), order=order)

    nib.save(out_img, str(out_path))
    print(f"[xfm] Saved → {out_path}")
    return Path(out_path)

def apply_affine_then_warp_to_mni(
    image_path: Path,
    affine_txt: Path,
    mapping: "DiffeomorphicMap",
    mni_path: Path,
    out_path: Path,
    *,
    is_label: bool = False,
) -> Path:
    """
    Apply a src→T1 affine and a T1→MNI nonlinear warp, and resample to MNI space.

    The 4×4 affine is automatically interpreted as either world→world or
    voxel→voxel (index→index) between the moving image and the T1 grid used
    to estimate the warp. The selected interpretation is the one that maps
    the moving image center closest to the T1 center in world space. The
    image is first resampled onto the T1 grid using
    `nibabel.processing.resample_from_to`, then warped to MNI using the
    provided `DiffeomorphicMap`.

    Parameters
    ----------
    image_path : pathlib.Path
        Moving image path (3D or 4D).
    affine_txt : pathlib.Path
        4×4 transform text file relating the moving image to the T1 grid.
        The matrix may be:
        - world→world (moving→T1), or
        - voxel→voxel (moving indices→T1 indices).
        The function tries both interpretations and chooses the one that
        aligns the moving and T1 centers best in world coordinates.
    mapping : DiffeomorphicMap
        T1→MNI nonlinear warp. Must expose:
        - `domain_shape`       : T1 grid shape (3D),
        - `domain_grid2world`  : T1 grid→world affine,
        and a `.transform(data, interpolation=...)` method that maps data on
        the T1 grid into MNI space.
    mni_path : pathlib.Path
        MNI template image; its grid (shape+affine) and header are used for
        the output.
    out_path : pathlib.Path
        Output NIfTI path.
    is_label : bool, default=False
        If True, use nearest-neighbor interpolation at both the affine
        (resample_to_T1) and nonlinear (T1→MNI) steps. If False, use linear
        interpolation for both.

    Returns
    -------
    pathlib.Path
        `out_path`.

    Files written
    -------------
    - Output NIfTI at `out_path` in MNI space, using:
      - the MNI template grid and affine, and
      - the MNI template header (copied from `mni_path`).

    Assumptions / Preconditions
    ---------------------------
    - `affine_txt` encodes a valid 4×4 transform between the moving image
      and the T1 grid used as the warp domain, either in world→world or
      voxel→voxel convention.
    - `mapping.domain_shape` and `mapping.domain_grid2world` correspond to
      the T1 grid on which the warp was estimated.
    - For 4D inputs, the spatial transform is the same for all volumes; the
      affine resampling is handled by `resample_from_to`, and the nonlinear
      warp is applied to the full 3D/4D array.
    - Data are cast to float32 before resampling/warping.
    - Output data live on the MNI grid defined by `mni_path`, and the output
      header is taken directly from the MNI template.
    """
    
    mov = nib.load(str(image_path))
    mni = nib.load(str(mni_path))

    # mapping.domain_* = T1 grid used to ESTIMATE the warp
    t1_shape = mapping.domain_shape
    t1_aff   = mapping.domain_grid2world

    # ---- load and normalize affine to WORLD->WORLD (same trick as above) ----
    A = _load_affine_txt(affine_txt)

    def center_world(shape, aff):
        i, j, k = (np.array(shape) - 1) / 2.0
        return aff @ np.array([i, j, k, 1.0])

    c_mov = center_world(mov.shape[:3], mov.affine)[:3]
    c_t1  = center_world(t1_shape,        t1_aff)[:3]

    A_world = A
    dA = np.linalg.norm((A_world @ np.append(c_mov, 1))[:3] - c_t1)
    A_vox2world = t1_aff @ A @ np.linalg.inv(mov.affine)
    dB = np.linalg.norm((A_vox2world @ np.append(c_mov, 1))[:3] - c_t1)
    A_best = A_world if dA <= dB else A_vox2world

    # ---- Step 1: embed A_best and resample onto the T1 grid (the warp domain) ----
    order = 0 if is_label else 1
    mov_prime = nib.Nifti1Image(mov.get_fdata().astype("float32"), A_best @ mov.affine, mov.header)
    # Use a tuple (shape, affine) so we don't need a real T1 file here
    mov_in_t1 = resample_from_to(mov_prime, (t1_shape, t1_aff), order=order)

    # ---- Step 2: apply nonlinear T1->MNI warp on the T1-grid data ----
    data_t1  = mov_in_t1.get_fdata().astype("float32")
    data_mni = mapping.transform(data_t1, interpolation="nearest" if is_label else "linear")

    # ---- Save with the exact MNI header (affine+header) ----
    out_img = nib.Nifti1Image(data_mni.astype("float32"), mni.affine, mni.header)
    nib.save(out_img, str(out_path))
    print(f"[xfm] Saved → {out_path}")
    return Path(out_path)

# -------------------------------------------------------------
# Transform bookkeeping (compose chains, inversion, etc.)
# -------------------------------------------------------------
@dataclass
class SubjectPaths:
    """
    Minimal paths container for registration bookkeeping.

    Attributes
    ----------
    sub : str
        Subject ID including the 'sub-' prefix (e.g., 'sub-xh33_x107').
    derivatives_root : pathlib.Path
        Root of derivatives (e.g., Path('derivatives')).

    Notes
    -----
    The property `anat_dir` points to:
    `derivatives/neurofluid-mreg/<SUB>/anat`.
    """
    sub: str                 # e.g., "sub-xh33_x107"  (already includes "sub-")
    derivatives_root: Path   # e.g., Path("derivatives")

    @property
    def anat_dir(self) -> Path:
        """Return `derivatives/neurofluid-mreg/<SUB>/anat`."""
        return Path(self.derivatives_root) / "neurofluid-mreg" / self.sub / "anat"

@dataclass
class TransformBook:
    """
    Bookkeeping for core transform estimation and filenames.

    Parameters
    ----------
    sp : SubjectPaths
        Must expose `sub` and **`transforms_dir`** (or `anat_dir` if aliased)
        where transform files are saved.

    Attributes
    ----------
    paths : dict
        Saved artifacts keyed by:
        - 'tof_to_t1'            : 4×4 txt
        - 'mrv_to_t1'            : 4×4 txt (if MRV provided)
        - 'hT2w_to_t1'           : 4×4 txt (if hT2w provided)
        - 'mregmean_to_t1'       : 4×4 txt
        - 't1_to_mni_warp'       : NIfTI (vector field; if MNI available)
        - 'mni_to_t1_warp'       : NIfTI (vector field; if MNI available)
        - 't1_to_mreg'           : 4×4 txt (inverse of mregmean→t1)
        - 'mregmean_to_mni_chain': txt with 2-line chain descriptor (if warps exist)

    Notes
    -----
    Filenames start with the subject ID (`sub-...`) and use `_xfm-<TAG><EXT>`.
    Affine files are saved as plain text (4×4) with high precision.
    """
    sp: SubjectPaths
    paths: Dict[str, Path] = field(default_factory=dict)

    def _xfm_path(self, tag: str, ext: str) -> Path:
        """
        Build an output path under the subject's transforms directory.

        Returns
        -------
        pathlib.Path
            `derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-<tag><ext>`.
        """
        return Path(self.sp.transforms_dir) / f"{self.sp.sub}_xfm-{tag}{ext}"

    def estimate_and_save_core_transforms(
        self,
        t1_denoised_path: Path,
        tof_path: Path,
        mreg_mean_path: Path,
        *,
        mrv_path: Optional[Path] = None,
        hT2w_path: Optional[Path] = None,
        mni_path: Optional[Path] = None,
    ) -> None:
        """
        Estimate TOF/MRV/hT2w/MREGMEAN → T1 affines; optionally T1 ↔ MNI warps.

        Parameters
        ----------
        t1_denoised_path : pathlib.Path
            Denoised T1 (fixed image).
        tof_path : pathlib.Path
            TOF image to register to T1.
        mreg_mean_path : pathlib.Path
            Mean MREG (3D) for MREGMEAN→T1.
        mrv_path : pathlib.Path or None
            Optional MRV image for MRV→T1.
        hT2w_path : pathlib.Path or None
            Optional heavy-T2w image for hT2w→T1.
        mni_path : pathlib.Path or None
            Optional MNI path; if None, a template is resolved.

        Files written
        -------------
        - derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-TOFtoT1.txt
        - derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-MRVtoT1.txt
        - derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-hT2wtoT1.txt
        - derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-MREGMEANtoT1.txt
        - derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-T1toMNI_warp.nii.gz
        - derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-MNItoT1_warp.nii.gz
        - derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-T1toMREG.txt
        - derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_xfm-MREGMEANtoMNI.txt

        Assumptions / Preconditions
        ---------------------------
        - `sp.transforms_dir` is writable. Affines/warps are saved there.
        - Nonlinear registration is attempted only if an MNI template is available.
        """
        _ensure_dir(Path(self.sp.transforms_dir))  # no behavioral changes

        # ---- src→T1 affines ----
        self.paths["tof_to_t1"] = register_to_t1(
            tof_path, t1_denoised_path, Path(self.sp.transforms_dir),
            sub_id=self.sp.sub, modality_tag="TOF"
        )

        if mrv_path is not None:
            self.paths["mrv_to_t1"] = register_to_t1(
                mrv_path, t1_denoised_path, Path(self.sp.transforms_dir),
                sub_id=self.sp.sub, modality_tag="MRV"
            )

        if hT2w_path is not None:
            self.paths["hT2w_to_t1"] = register_to_t1(
                hT2w_path, t1_denoised_path, Path(self.sp.transforms_dir),
                sub_id=self.sp.sub, modality_tag="hT2w"
            )

        self.paths["mregmean_to_t1"] = register_to_t1(
            mreg_mean_path, t1_denoised_path, Path(self.sp.transforms_dir),
            sub_id=self.sp.sub, modality_tag="MREGMEAN"
        )

        # ---- T1→MNI nonlinear (optional; only if we have a template) ----
        self._mapping_cache = None
        self.mni_ref_path = None

        # If caller didn't pass an MNI path, try to resolve one automatically
        if mni_path is None:
            try:
                mni_path = _resolve_mni_path()
                print(f"[xfm] Using resolved MNI template: {mni_path}")
            except Exception as e:
                print(f"[xfm] No MNI template available ({e}); skipping MNI warps.")

        if mni_path is not None and Path(mni_path).exists():
            mapping, fwd_warp, inv_warp = register_t1_to_mni(
                t1_denoised_path, mni_path, Path(self.sp.transforms_dir), sub_id=self.sp.sub
            )
            self.paths["t1_to_mni_warp"] = fwd_warp
            self.paths["mni_to_t1_warp"] = inv_warp
            self._mapping_cache = mapping
            self.mni_ref_path = Path(mni_path)
        else:
            print("[xfm] Skipping MNI: template path missing or not found.")

        # ---- Derived: T1→MREG (inverse of MREGMEAN→T1) ----
        A = _load_affine_txt(self.paths["mregmean_to_t1"])
        Ainv = np.linalg.inv(A)
        t1_to_mreg = self._xfm_path("T1toMREG", ".txt")
        _save_affine_txt(Ainv, t1_to_mreg)
        self.paths["t1_to_mreg"] = t1_to_mreg

        # ---- Convenience chain descriptor: MREGMEAN→MNI ----
        if "t1_to_mni_warp" in self.paths:
            chain_txt = self._xfm_path("MREGMEANtoMNI", ".txt")
            with open(chain_txt, "w", encoding="utf-8") as f:
                f.write(str(self.paths["mregmean_to_t1"]) + "\n")
                f.write(str(self.paths["t1_to_mni_warp"]) + "\n")
            self.paths["mregmean_to_mni_chain"] = chain_txt

    def get(self, key: str):
        """
        Retrieve a saved transform path by key.

        Parameters
        ----------
        key : str
            Name in `self.paths` (see class docstring for common keys).

        Returns
        -------
        pathlib.Path
            Path to the transform artifact.

        Raises
        ------
        KeyError
            If the key is not present.
        """
        try:
            return self.paths[key]
        except KeyError:
            raise KeyError(f"Transform '{key}' not found. Available: {list(self.paths.keys())}")

    def warp_labels(
        self,
        *,
        moving_img: str,
        reference_img: str,
        out_path: str,
        chain: tuple = ("MNI", "T1"),
        interpolation: str = "nearest",
    ) -> None:
        """
        Label-safe resampling using existing transforms.

        Parameters
        ----------
        moving_img : str
            Path to moving label image (NIfTI).
        reference_img : str
            Path to target reference image (NIfTI).
        out_path : str
            Output NIfTI path (written).
        chain : tuple, default=("MNI", "T1")
            Supported chains:
              - ("MNI","T1")
                Inverse T1↔MNI map → snap MNI labels to the requested T1 grid.
              - ("T1","MREG")
                4×4 T1→MREG affine baked into header, then resampled to MREG.
              - ("TOF","MNI"), ("MRV","MNI"), ("hT2w","MNI")
                Compose (src→T1 affine) with stored T1↔MNI warp via
                `apply_affine_then_warp_to_mni` (nearest for labels, linear for
                images depending on `interpolation`).
        interpolation : {"nearest","linear"}, default="nearest"
            Resampling order; labels should use "nearest".

        Files written
        -------------
        - Output label NIfTI at `out_path` (space chosen by chain target).

        Raises
        ------
        RuntimeError
            If a required affine or T1↔MNI mapping is missing.
        NotImplementedError
            If the `chain` is unsupported.
        """

        order = 0 if interpolation == "nearest" else 1
        mov = nib.load(moving_img)
        ref = nib.load(reference_img)

        if chain == ("MNI", "T1"):
            if getattr(self, "_mapping_cache", None) is None:
                raise RuntimeError("No T1↔MNI mapping cached. Did you skip T1→MNI registration?")
            mapping = self._mapping_cache  # domain: T1, codomain: MNI

            # If the moving mask grid doesn't match the exact MNI template we used, align it first
            if getattr(self, "mni_ref_path", None):
                mni_ref = nib.load(str(self.mni_ref_path))
                if (mov.shape != mni_ref.shape) or not np.allclose(mov.affine, mni_ref.affine, atol=1e-3):
                    mov = resample_from_to(mov, (mni_ref.shape, mni_ref.affine), order=0)

            data = np.asanyarray(mov.get_fdata(), dtype=np.float32)
            # Transform codomain (MNI) → domain (T1)
            try:
                out_data = mapping.transform_inverse(data, interpolation="nearest" if order == 0 else "linear")
            except TypeError:
                out_data = mapping.transform_inverse(data)

            out_img = nib.Nifti1Image(out_data, mapping.domain_grid2world, ref.header.copy())

            # If this T1 grid (from mapping) differs from the requested T1 reference, resample once more (affine-only)
            if (out_img.shape != ref.shape) or not np.allclose(out_img.affine, ref.affine, atol=1e-3):
                out_img = resample_from_to(out_img, (ref.shape, ref.affine), order=order)

            nib.save(out_img, out_path)
            return

        if chain == ("T1", "MREG"):
            # Saved 4×4 T1→MREG
            A_t1_to_mreg = np.loadtxt(self.get("t1_to_mreg"))

            # Embed the T1→MREG affine into the moving image's affine,
            # so that resample_from_to uses the desired mapping implicitly.
            mov_data = mov.get_fdata()
            mov_hdr  = mov.header.copy()
            mov_prime_affine = A_t1_to_mreg @ mov.affine
            mov_prime = nib.Nifti1Image(mov_data, mov_prime_affine, mov_hdr)

            # Now a plain resample_to the reference grid is enough (nearest for labels)
            out_img = resample_from_to(mov_prime, ref, order=order)

            nib.save(out_img, out_path)
            return
        
        if chain[1] == "MNI" and chain[0] in ("TOF", "MRV", "hT2w"):
            if getattr(self, "_mapping_cache", None) is None or getattr(self, "mni_ref_path", None) is None:
                raise RuntimeError("No T1↔MNI mapping cached. Did you skip T1→MNI registration?")

            # Normalize source token 
            src_token = chain[0]
            src_key_map = {
                "TOF":  "tof_to_t1",
                "MRV":  "mrv_to_t1",
                "hT2w": "hT2w_to_t1",
            }
            src_key = src_key_map[src_token]

            try:
                affine_txt = self.get(src_key)
            except KeyError:
                # This is what you already see for missing MRV/hT2w affines
                raise RuntimeError(f"Missing affine '{src_key}'. Available: {list(self.paths.keys())}")

            # nearest => labels, linear => images
            is_label = (interpolation == "nearest")

            apply_affine_then_warp_to_mni(
                image_path=Path(moving_img),
                affine_txt=Path(affine_txt),
                mapping=self._mapping_cache,
                mni_path=self.mni_ref_path,   # enforce the same MNI grid used for registration
                out_path=Path(out_path),
                is_label=is_label,
            )
            return
        raise NotImplementedError(f"Unsupported chain {chain}")


# -------------------------------------------------------------
# Radii estimation / fitting / QC (warping only in this module)
# -------------------------------------------------------------
def warp_radii_to_mreg(
    sp: SubjectPaths,
    xfm,  # TransformBook
    *,
    mreg_ref_path: Path,
    t1_ref_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Warp native-space radius maps (TOF/MRV/hT2w) → MREG via T1.

    Parameters
    ----------
    sp : SubjectPaths
        Subject-scoped paths/naming helper.
    xfm : TransformBook
        Provides native→T1 and T1→MREG transforms.
    mreg_ref_path : pathlib.Path
        MREG reference image (grid target).
    t1_ref_path : pathlib.Path or None, default=None
        T1 reference image; falls back to `sp.anat_t1w` if None.
    overwrite : bool, default=False
        If False, skip outputs that already exist.

    Returns
    -------
    dict
        Mapping `{klass: out_path}` for artifacts produced.

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/radii/sub-<ID>_space-MREG_class-<CLASS>_desc-radius_map.nii.gz

    Assumptions / Preconditions
    ---------------------------
    - Native radii maps must exist under `radii/` per class; nearest resampling
      is used for label-like radii maps.
    """
    out: dict[str, Path] = {}

    # Resolve reference images (3D)
    mreg_ref_img = nib.load(str(mreg_ref_path))
    if t1_ref_path is None:
        # fallback: subject's original T1
        t1_ref_path = Path(sp.anat_t1w)
    t1_ref_img = nib.load(str(t1_ref_path))

    # mapping per class
    mapping = {
        "arteries": ("TOF",  "tof_to_t1"),
        "veins":    ("MRV",  "mrv_to_t1"),
        "pvs":      ("hT2w", "hT2w_to_t1"),
    }

    def _apply_native_to_t1(native_img: Path, akey: str, out_t1_img: Path) -> Path:
        # Use existing apply_affine_to_ref with NEAREST (label-like)
        A = xfm.get(akey)  # path to 4x4 txt
        return apply_affine_to_ref(
            image_path=native_img,
            affine=A,
            ref_img_path=Path(t1_ref_path),
            out_path=out_t1_img,
            interpolation="nearest",
        )

    def _apply_t1_to_mreg(t1_img: Path, out_mreg_img: Path) -> Path:
        # Use TransformBook.warp_labels with chain ("T1","MREG")
        xfm.warp_labels(
            moving_img=str(t1_img),
            reference_img=str(mreg_ref_path),
            out_path=str(out_mreg_img),
            chain=("T1", "MREG"),
            interpolation="nearest",
        )
        return out_mreg_img

    for klass, (space, key_native_to_t1) in mapping.items():
        src_native = Path(sp.radii_dir) / deriv_name(sp.sub, space, klass, "radius", "map")
        if not src_native.exists():
            # nothing to warp for this class
            continue

        out_mreg = Path(sp.radii_dir) / deriv_name(sp.sub, "MREG", klass, "radius", "map")
        if out_mreg.exists() and not overwrite:
            out[klass] = out_mreg
            continue

        # temp file in T1 geometry
        tmp_t1 = Path(sp.radii_dir) / deriv_name(sp.sub, "T1", klass, "radius", "map")

        # Step 1: native → T1
        try:
            _apply_native_to_t1(src_native, key_native_to_t1, tmp_t1)
        except KeyError as e:
            # Give a helpful error listing available keys
            available = getattr(xfm, "paths", {}).keys()
            print(f"[radii-warp] Missing transform key '{key_native_to_t1}'. Available: {list(available)}")
            continue

        # Step 2: T1 → MREG
        _apply_t1_to_mreg(tmp_t1, out_mreg)
        out[klass] = out_mreg

        # optional: clean temp
        try:
            tmp_t1.unlink(missing_ok=True)
        except Exception:
            pass

        # final sanity: ensure exact MREG grid
        m = nib.load(str(out_mreg))
        if (m.shape != mreg_ref_img.shape) or (not np.allclose(m.affine, mreg_ref_img.affine, atol=1e-3)):
            # fallback snap (shouldn't happen if transforms are correct)
            snapped = resample_from_to(m, (mreg_ref_img.shape, mreg_ref_img.affine), order=0)
            nib.save(snapped, str(out_mreg))

        print(f"[radii-warp] Saved: {out_mreg.name}")

    if not out:
        print("[radii-warp] Nothing to warp (no native radii found or all already up-to-date).")
    return out


# -------------------------------------------------------------
# Preprocessing (denoise, normalize, CLAHE)
# -------------------------------------------------------------
def run_mp2rage_denoise(uni_path: Path,
                        inv1_path: Path,
                        inv2_path: Path,
                        out_dir: Path,
                        sub_id: str,
                        lamb: float = 10.0,
                        corner_width: int = 11) -> Path:
    """
    Denoise MP2RAGE and save a denoised T1 under anat derivatives.

    Parameters
    ----------
    uni_path : pathlib.Path
        MP2RAGE UNI image.
    inv1_path : pathlib.Path
        MP2RAGE INV1 image.
    inv2_path : pathlib.Path
        MP2RAGE INV2 image.
    out_dir : pathlib.Path
        Output anat derivatives directory.
    sub_id : str
        Subject identifier including `sub-` prefix.
    lamb : float, default=10.0
        Regularization factor.
    corner_width : int, default=11
        Corner cube width for noise estimate.

    Returns
    -------
    pathlib.Path
        Path to `<out_dir>/<sub_id>_T1w_denoised.nii.gz`.

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/anat/sub-<ID>_T1w_denoised.nii.gz
    """
    out_path = out_dir / f"{sub_id}_T1w_denoised.nii.gz"
    if out_path.exists():
        print(f"[mp2rage] [SKIP] Exists: {out_path.name}")
        return out_path

    mp2rage_denoise(uni_path, inv1_path, inv2_path,
                    out_path=out_path,
                    lamb=lamb,
                    corner_width=corner_width)
    print(f"[mp2rage] Saved → {out_path}")  # keep your existing style here
    return out_path


def push_to_mni_float(
    image_path: Path,
    *,
    src_to_t1_txt: Path,
    xfm: TransformBook,
    out_path: Path,
) -> Path:
    """
    Resample a source image to MNI via (src→T1 affine) + (T1→MNI warp), float.

    Parameters
    ----------
    image_path : pathlib.Path
        Path to the source image in its native space (e.g., TOF/MRV/hT2w/MREG).
    src_to_t1_txt : pathlib.Path
        4×4 affine text file describing the src→T1 mapping (as produced by
        `register_to_t1`).
    xfm : TransformBook
        Must expose a cached T1↔MNI mapping (`_mapping_cache`) and, ideally,
        `mni_ref_path`. If `mni_ref_path` is missing, a canonical MNI template
        is resolved via `_resolve_mni_path`.
    out_path : pathlib.Path
        Output NIfTI path in MNI space (caller controls BIDS naming).

    Returns
    -------
    pathlib.Path
        `out_path` of the written MNI-space image (float32).

    Files written
    -------------
    - Output NIfTI at `out_path` in space-MNI (float32).

    Assumptions / Preconditions
    ---------------------------
    - `xfm._mapping_cache` was created by `register_t1_to_mni` and matches the
      T1 used to derive `src_to_t1_txt`.
    - Linear interpolation is appropriate for the quantity being resampled
      (e.g., band-power maps, intensities).

    Warnings
    --------
    - If the T1 used in `src_to_t1_txt` does not match the T1 used for the
      T1↔MNI mapping, misalignment may occur.

    Raises
    ------
    RuntimeError
        If `xfm._mapping_cache` is missing (indirectly via
        `apply_affine_then_warp_to_mni`).
    """
    mni_ref = xfm.mni_ref_path if getattr(xfm, "mni_ref_path", None) else _resolve_mni_path()
    return apply_affine_then_warp_to_mni(
        image_path=image_path,
        affine_txt=src_to_t1_txt,
        mapping=xfm._mapping_cache,
        mni_path=mni_ref,
        out_path=out_path,
        is_label=False,  # linear
    )


def push_to_mni_label(
    image_path: Path,
    *,
    src_to_t1_txt: Path,
    xfm: TransformBook,
    out_path: Path,
) -> Path:
    """
    Resample a label-like image to MNI via (src→T1 affine) + (T1→MNI warp).

    Parameters
    ----------
    image_path : pathlib.Path
        Path to the source label/segmentation in its native space.
    src_to_t1_txt : pathlib.Path
        4×4 affine text file describing the src→T1 mapping (as produced by
        `register_to_t1`).
    xfm : TransformBook
        Must expose a cached T1↔MNI mapping (`_mapping_cache`) and, ideally,
        `mni_ref_path`. If `mni_ref_path` is missing, a canonical MNI template
        is resolved via `_resolve_mni_path`.
    out_path : pathlib.Path
        Output NIfTI path in MNI space (caller controls BIDS naming).

    Returns
    -------
    pathlib.Path
        `out_path` of the written MNI-space label (uint8/float-like array).

    Files written
    -------------
    - Output NIfTI at `out_path` in space-MNI (nearest-neighbor resampling).

    Assumptions / Preconditions
    ---------------------------
    - `xfm._mapping_cache` was created by `register_t1_to_mni` and matches the
      T1 used to derive `src_to_t1_txt`.
    - Nearest-neighbor interpolation is required to preserve label integrity.

    Warnings
    --------
    - If the T1 used in `src_to_t1_txt` does not match the T1 used for the
      T1↔MNI mapping, label boundaries may be misaligned.

    Raises
    ------
    RuntimeError
        If `xfm._mapping_cache` is missing (indirectly via
        `apply_affine_then_warp_to_mni`).
    """
    mni_ref = xfm.mni_ref_path if getattr(xfm, "mni_ref_path", None) else _resolve_mni_path()
    return apply_affine_then_warp_to_mni(
        image_path=image_path,
        affine_txt=src_to_t1_txt,
        mapping=xfm._mapping_cache,
        mni_path=mni_ref,
        out_path=out_path,
        is_label=True,   # nearest
    )


# --- NEW: warp subject T1 → MNI (once), then compute reusable brain mask ---

def warp_t1_to_mni_once(
    sp: SubjectPaths,
    xfm: TransformBook,
    *,
    t1_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Warp subject T1 to MNI (linear) for a reusable subject-level MNI brain mask.

    Parameters
    ----------
    sp : SubjectPaths
        Subject-scoped paths helper (expects `sub` or `subQ`, `anat_out`).
    xfm : TransformBook
        Must carry a valid T1↔MNI mapping (`_mapping_cache`) and MNI reference
        path (`mni_ref_path`) as produced by `estimate_and_save_core_transforms`.
    t1_path : pathlib.Path or None, optional
        T1 image to warp. If None, callers should pass the same denoised T1
        that was used for registration; the function assumes an existing path.
    overwrite : bool, optional
        If False (default), return an existing MNI-space T1 if present.

    Returns
    -------
    pathlib.Path
        Path to `sub-<ID>_space-MNI_desc-T1w_in-MNI_map.nii.gz`.

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/anat/
      sub-<ID>_space-MNI_desc-T1w_in-MNI_map.nii.gz

    Assumptions / Preconditions
    ---------------------------
    - `xfm._mapping_cache` and `xfm.mni_ref_path` were created beforehand via
      `estimate_and_save_core_transforms` and correspond to `t1_path`.
    - Linear interpolation is sufficient for deriving a subject-level brain
      mask or similar scalar maps.

    Raises
    ------
    FileNotFoundError
        If `t1_path` does not exist.
    RuntimeError
        If the T1↔MNI mapping is missing in `xfm`.

    Notes
    -----
    - This helper is designed to be called once per subject; downstream code
      can re-use the saved T1-in-MNI map to derive masks or QC overlays.
    """
    # Use the same subject ID convention you use elsewhere; adjust if you really need subQ
    sub = getattr(sp, "subQ", None) or sp.sub

    anat_out = Path(sp.anat_out)
    anat_out.mkdir(parents=True, exist_ok=True)

    # If caller explicitly supplied a T1, trust it (and validate)
    t1_path = Path(t1_path)
    if not t1_path.exists():
        raise FileNotFoundError(f"[mask] Provided T1 path does not exist: {t1_path}")

    out_t1_mni = anat_out / f"{sub}_space-MNI_desc-T1w_in-MNI_map.nii.gz"
    if out_t1_mni.exists() and not overwrite:
        return out_t1_mni

    mapping = getattr(xfm, "_mapping_cache", None)
    if mapping is None:
        raise RuntimeError(
            "[mask] T1↔MNI mapping missing. Run estimate_and_save_core_transforms first."
        )

    # Load and warp
    t1_img = nib.load(str(t1_path))
    t1_dat = t1_img.get_fdata().astype(np.float32)

    # Transform into MNI space
    t1_mni = mapping.transform(t1_dat, interpolation="linear")

    # Get MNI reference
    mni_ref_path = getattr(xfm, "mni_ref_path", None)
    if mni_ref_path is None:
        mni_ref_path = _resolve_mni_path()
    mni_ref = nib.load(str(mni_ref_path))

    nib.save(
        nib.Nifti1Image(t1_mni, mni_ref.affine, mni_ref.header),
        str(out_t1_mni),
    )
    print(f"[mask] Saved subject T1 in MNI → {out_t1_mni}")
    return out_t1_mni
