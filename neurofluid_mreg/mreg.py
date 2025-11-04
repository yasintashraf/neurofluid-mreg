# SPDX-License-Identifier: MIT
"""
mreg.py
-------
Single-subject MREG preprocessing and label warping for a BIDS-first pipeline.
This module keeps your algorithmic choices intact (NiPy Realign4d, voxelwise
polynomial detrend, DIPY staged affine for mean MREG→T1, linear interpolation
for images, nearest for labels) and standardizes derivatives layout/I-O.

Pipeline steps
--------------
1. Realign + detrend the 4D MREG time series (native MREG space)
2. Compute a 3D temporal mean on the MREG grid
3. Register mean MREG → T1 (COM → translation → rigid → affine; MI metric)
4. Apply the mean→T1 affine to the full 4D series (linear)
5. Warp native-space masks (TOF/MRV/HT2w) → MREG (and optionally → MNI)

Inputs / Outputs
----------------
Inputs  : Subject-scoped paths via `SubjectPaths`; optional transforms via
          `TransformBook`.
Outputs : NIfTIs (float32 for images; uint8 for masks) and small text/JSON/TSV
          auxiliaries written under `derivatives/neurofluid-mreg/sub-<ID>/`.

Files written
-------------
- mreg/<sub>_space-MREG_class-brain_desc-motionrealigned_bold.nii.gz
- mreg/<sub>_space-MREG_class-brain_desc-detrended_bold.nii.gz
- mreg/<sub>_space-MREG_class-brain_desc-mean_map.nii.gz
- mreg/<sub>_space-T1_class-brain_desc-registered_bold.nii.gz
- qc/<sub>_desc-motion_params.tsv
- qc/<sub>_desc-preproc_params.json
- masks/<sub>_space-MREG_class-<CLASS>_desc-main_mask.nii.gz
- masks/<sub>_space-MNI_class-<CLASS>_desc-main_mask.nii.gz (optional)
Where filenames follow:
`sub-<ID>_space-<SPACE>_class-<CLASS>_desc-<DESC>_<SUFFIX>.nii.gz`

Assumptions / Preconditions
---------------------------
- Spaces: All operations occur in image space of the chosen reference. Apply
  transforms/resampling only where stated. Affine mismatches are handled by
  explicit resampling against the reference image.
- Shapes/dtypes: 4D MREG (X, Y, Z, T). Images saved float32; masks saved uint8.
- TR: Pulled from NIfTI header (`header.get_zooms()[3]`) for Realign4d.
- BIDS naming: Use `SubjectPaths.sub` (already includes "sub-") unless noted.

Warnings
--------
- NiPy may return resampled realignment outputs as 5D `(X, Y, Z, 1, T)`; these
  are squeezed to 4D. Non-finite samples are replaced as documented below.
- `apply_mean_xfm_to_full_mreg` constructs `sub-<ID>` from `sp.subject` while
  other functions use `sp.sub`; ensure consistency in your inputs/config.

Public API
----------
- realign_and_detrend_mreg
- compute_mreg_mean
- estimate_mregmean_to_t1
- apply_mean_xfm_to_full_mreg
- warp_masks_single_shot
"""

import json
from pathlib import Path
import numpy as np
import nibabel as nib
from nipy.algorithms.registration import Realign4d
from nibabel.processing import resample_from_to
from dipy.align.imaffine import (
    AffineRegistration,
    MutualInformationMetric,
    transform_centers_of_mass,
)
from dipy.align.transforms import (
    RigidTransform3D,
    AffineTransform3D,
    TranslationTransform3D,
)
from .transforms import (
    TransformBook,
    apply_affine_to_ref,
    apply_affine_then_warp_to_mni,
)
from .io import SubjectPaths, deriv_name
from typing import Union


# -------------------------------------------------------------
# Preprocessing (denoise, normalize, detrend)
# -------------------------------------------------------------
def realign_and_detrend_mreg(sp: SubjectPaths, overwrite: bool = False) -> None:
    """
    Motion-correct and polynomial-detrend the subject's MREG 4D series.

    Parameters
    ----------
    sp : SubjectPaths
        Provides `func_mreg_bold` (input 4D), `mreg_dir` (outputs), and `sub`.
    overwrite : bool, optional
        If False, skip when both realigned and detrended outputs already exist.

    Returns
    -------
    None

    Files written
    -------------
    - mreg/sub-<ID>_space-MREG_class-brain_desc-motionrealigned_bold.nii.gz
    - mreg/sub-<ID>_space-MREG_class-brain_desc-detrended_bold.nii.gz
    - qc/sub-<ID>_desc-motion_params.tsv
    - qc/sub-<ID>_desc-preproc_params.json

    Assumptions / Preconditions
    ---------------------------
    - TR is read from header (`header.get_zooms()[3]`).
    - Realign4d operates in the native MREG image space.

    Warnings
    --------
    - NiPy may yield `(X, Y, Z, 1, T)`; a squeeze to 4D is performed.
    - Non-finite samples are replaced per voxel using temporal mean; all-NaN
      voxels become 0.0 (recorded in JSON).

    Notes
    -----
    - Detrend order fixed at 3 (voxelwise polynomial).
    - Images saved as float32; motion TSV columns: dx, dy, dz (mm),
      rx, ry, rz (rad).
    """
    mreg_dir = Path(sp.mreg_dir)
    mreg_dir.mkdir(parents=True, exist_ok=True)

    subj = sp.sub
    mreg_4d = Path(sp.func_mreg_bold)

    realign_fname = f"{subj}_space-MREG_class-brain_desc-motionrealigned_bold.nii.gz"
    detrend_fname = f"{subj}_space-MREG_class-brain_desc-detrended_bold.nii.gz"
    out_realigned = mreg_dir / realign_fname
    out_detrended = mreg_dir / detrend_fname

    qc_dir = mreg_dir.parent / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    motion_tsv = qc_dir / f"{subj}_desc-motion_params.tsv"
    params_json = qc_dir / f"{subj}_desc-preproc_params.json"

    realign_exists = out_realigned.exists()
    detrend_exists = out_detrended.exists()
    if realign_exists and detrend_exists and not overwrite:
        pprint(f"[mreg] [SKIP] Realign/detrend exist for {subj}")
        return

    img = nib.load(str(mreg_4d))
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    nt = data.shape[3]
    ref_vol = nt // 2

    if not realign_exists or overwrite:
        try:
            zooms = header.get_zooms()
            tr = zooms[3] if len(zooms) >= 4 and zooms[3] > 0 else 1.0
            realigner = Realign4d(img, tr=tr)
            realigner.estimate(refscan=ref_vol)
            aligned_vols = realigner.resample()

            mc_data = np.stack([vol.get_fdata() for vol in aligned_vols], axis=3)
            # Normalize to 4D: squeeze a singleton 4th dim if present (X,Y,Z,1,T) → (X,Y,Z,T)
            if mc_data.ndim == 5 and mc_data.shape[3] == 1:
                old_shape = mc_data.shape
                mc_data = np.squeeze(mc_data, axis=3)
                print(f"[mreg] Squeezed realigned data 5D→4D: {old_shape} → {mc_data.shape}")
            elif mc_data.ndim != 4:
                raise ValueError(f"[mreg] Realigned data should be 4D, got {mc_data.shape}")
            # --- Sanitize non-finite values before saving/using realigned 4D ---
            mc_data = mc_data.astype(np.float32, copy=False)
            n_nonfinite = np.count_nonzero(~np.isfinite(mc_data))
            if n_nonfinite:
                # Fill NaNs/±inf per-voxel with that voxel's temporal mean; all-NaN voxels → 0.0
                with np.errstate(invalid="ignore"):
                    voxel_means = np.nanmean(mc_data, axis=3, keepdims=True)  # (X,Y,Z,1)
                voxel_means = np.where(np.isfinite(voxel_means), voxel_means, 0.0)
                bad = ~np.isfinite(mc_data)
                mc_data[bad] = np.broadcast_to(voxel_means, mc_data.shape)[bad]
                print(f"[mreg] Replaced {n_nonfinite} non-finite samples in realigned 4D")

            mc_img = nib.Nifti1Image(mc_data.astype(np.float32), affine, header)
            nib.save(mc_img, str(out_realigned))
            print(f"[mreg] Saved → {out_realigned}")

            # Extract transforms if available
            xforms_runs = getattr(realigner, "_transforms", None)
            if xforms_runs and len(xforms_runs) > 0:
                run_xforms = xforms_runs[0]
                affs = np.array([xf.as_affine() for xf in run_xforms])

                motions = np.zeros((nt, 6), dtype=float)
                for i, A in enumerate(affs):
                    # translations (mm)
                    motions[i, 0:3] = A[0:3, 3]
                    # rotations (rad)
                    motions[i, 3] = np.arctan2(A[2, 1], A[2, 2])  # rx
                    motions[i, 4] = np.arctan2(-A[2, 0], np.hypot(A[2, 1], A[2, 2]))  # ry
                    motions[i, 5] = np.arctan2(A[1, 0], A[0, 0])  # rz

                header_txt = "dx\tdy\tdz\trx\try\trz"
                np.savetxt(motion_tsv, motions, header=header_txt, comments="")
                print(f"[mreg] Saved motion params TSV: {motion_tsv}")
            else:
                print("[mreg] [WARN] Realign4d transforms unavailable; skipping motion TSV")

            params = {
                "realign_ref_volume": int(ref_vol),
                "detrend_order": 3,
                "realign_nonfinite_replaced": int(n_nonfinite) if "n_nonfinite" in locals() else 0,
            }
            with open(params_json, "w") as f:
                json.dump(params, f)
            print(f"[mreg] Saved → {params_json}")
        except Exception as e:
            print(f"[mreg] [ERROR] Realignment failed for {subj}: {e}")
            return
    else:
        print(f"[mreg] Using existing realigned → {out_realigned}")
        mc_img = nib.load(str(out_realigned))
        mc_data = mc_img.get_fdata()

    if not detrend_exists or overwrite:

        def detrend_data_4d(data4d, degree=3):
            nx, ny, nz, nt = data4d.shape
            x = np.arange(nt, dtype=np.float32)
            ts = data4d.reshape(-1, nt).T  # (nt, nvox)

            # --- Sanitize non-finite per voxel before fitting ---
            bad_any = ~np.isfinite(ts)
            if bad_any.any():
                with np.errstate(invalid="ignore"):
                    col_means = np.nanmean(ts, axis=0)  # per-voxel mean over time
                col_means = np.where(np.isfinite(col_means), col_means, 0.0)
                ts[bad_any] = np.take(col_means, np.where(bad_any)[1])
                print(f"[mreg] Detrend: replaced {bad_any.sum()} non-finite samples")

            # Design matrix for polynomial detrend
            X = np.vstack([x ** i for i in range(degree + 1)]).T  # (nt, degree+1)
            coef = np.linalg.lstsq(X, ts, rcond=None)[0]  # (degree+1, nvox)
            trend = X.dot(coef)  # (nt, nvox)
            detrended = ts - trend  # (nt, nvox)
            data_dt = detrended.T.reshape(nx, ny, nz, nt)
            return data_dt

        try:
            data_dt = detrend_data_4d(mc_data.astype(np.float32), degree=3)
            dt_img = nib.Nifti1Image(data_dt.astype(np.float32), affine, header)
            nib.save(dt_img, str(out_detrended))
            print(f"[mreg] Saved detrended 4D: {out_detrended}")
        except Exception as e:
            print(f"[mreg] [ERROR] Detrending failed for {subj}: {e}")
    else:
        print(f"[mreg] [SKIP] Exists: {out_detrended.name}")


def compute_mreg_mean(mreg_4d: Path, sp, overwrite: bool = False) -> Path:
    """
    Compute and save the temporal mean (float32) from a subject's MREG 4D series.

    Parameters
    ----------
    mreg_4d : pathlib.Path
        Raw 4D MREG path (used if detrended/realigned are absent).
    sp : SubjectPaths
        Provides `mreg_dir` (outputs) and `sub`.
    overwrite : bool, optional
        If False, skip when the mean image already exists.

    Returns
    -------
    pathlib.Path
        Path to `mreg/sub-<ID>_space-MREG_class-brain_desc-mean_map.nii.gz`.

    Files written
    -------------
    - mreg/sub-<ID>_space-MREG_class-brain_desc-mean_map.nii.gz

    Assumptions / Preconditions
    ---------------------------
    - Data are 4D (or 5D with a singleton axis to be squeezed).
    - Operates in image space of the chosen source (no resampling).

    Warnings
    --------
    - Any existing header scaling is cleared when possible to avoid surprises.
    """
    mreg_dir = Path(sp.mreg_dir)
    mreg_dir.mkdir(parents=True, exist_ok=True)
    subj = sp.sub

    detrended = mreg_dir / f"{subj}_space-MREG_class-brain_desc-detrended_bold.nii.gz"
    realigned = mreg_dir / f"{subj}_space-MREG_class-brain_desc-motionrealigned_bold.nii.gz"
    source_4d = realigned if realigned.exists() else Path(mreg_4d)

    if not source_4d.exists():
        raise FileNotFoundError(f"[mreg] Source 4D not found: {source_4d}")

    mean_path = mreg_dir / f"{subj}_space-MREG_class-brain_desc-mean_map.nii.gz"
    if mean_path.exists() and not overwrite:
        print(f"[mreg] [SKIP] Exists: {mean_path.name}")
        return mean_path

    img = nib.load(str(source_4d))
    data = img.get_fdata()

    # Handle accidental 5D, then enforce 4D
    if data.ndim == 5:
        if data.shape[3] == 1:
            data = np.squeeze(data, axis=3)
            print(f"[mreg] Squeezed 5D→4D on axis 3: now {data.shape}")
        elif data.shape[4] == 1:
            data = np.squeeze(data, axis=4)
            print(f"[mreg] Squeezed 5D→4D on axis 4: now {data.shape}")
        else:
            raise ValueError(f"[mreg] Unsupported 5D shape without singleton: {data.shape}")

    if data.ndim != 4:
        raise ValueError(f"[mreg] Expected 4D after squeeze, got {data.shape}")

    mean_data = data.mean(axis=3).astype(np.float32, copy=False)

    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)
    try:
        hdr.set_slope_inter(1.0, 0.0)
    except Exception:
        pass

    mean_img = nib.Nifti1Image(mean_data, img.affine, hdr)
    nib.save(mean_img, str(mean_path))
    print(f"[mreg] Saved temporal mean → {mean_path}")

    return mean_path


# -------------------------------------------------------------
# Transform bookkeeping (compose chains, inversion, etc.)
# -------------------------------------------------------------
def _save_affine_matrix(M: np.ndarray, out_path: Union[str, Path]) -> None:
    """
    Save a 4×4 affine matrix based on output suffix.

    Parameters
    ----------
    M : ndarray
        4×4 affine matrix (cast to float64).
    out_path : str or pathlib.Path
        Target path. Behavior:
        - `.npy`  → NumPy binary
        - `.txt`/`.tsv`/`.csv` → ASCII with '%.10f'
        - other → ASCII as given + companion `.npy`

    Returns
    -------
    None

    Files written
    -------------
    - `<out_path>` (and optional companion `.npy`).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    M64 = M.astype(np.float64, copy=False)
    suf = out_path.suffix.lower()
    if suf == ".npy":
        np.save(str(out_path), M64)
        print(f"[xfm] Saved → {out_path}")
    elif suf in (".txt", ".tsv", ".csv"):
        np.savetxt(str(out_path), M64, fmt="%.10f")
        print(f"[xfm] Saved → {out_path}")
    else:
        np.savetxt(str(out_path), M64, fmt="%.10f")
        print(f"[xfm] Saved → {out_path}")
        np.save(str(out_path.with_suffix(out_path.suffix + ".npy")), M64)
        print(f"[xfm] Saved → {out_path} (and {out_path.with_suffix(out_path.suffix + '.npy')})")


def _load_affine_matrix(path: Path) -> np.ndarray:
    """
    Load a 4×4 affine matrix from disk.

    Parameters
    ----------
    path : pathlib.Path
        `.npy` loads with `np.load`; others via `np.loadtxt`.

    Returns
    -------
    ndarray
        4×4 (float64) affine matrix.

    Raises
    ------
    ValueError
        If the array is not shape (4, 4).
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        M = np.load(str(path))
    else:
        M = np.loadtxt(str(path))
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"[mreg] Expected a 4x4 matrix in {path}, got {M.shape}")
    return M


# -------------------------------------------------------------
# Applying transforms (registration / resampling)
# -------------------------------------------------------------
def estimate_mregmean_to_t1(
    mreg_mean: Path,
    t1: Path,
    xfm: "TransformBook",
    *,
    overwrite: bool = False,
) -> None:
    """
    Estimate affine mapping mean MREG (moving) → T1 (static) and save both
    forward and inverse matrices via `TransformBook` targets.

    Registration pipeline
    ---------------------
    Center-of-mass → Translation(3D) → Rigid(3D) → Affine(3D)
    - Metric: Mutual Information (32 bins)
    - Pyramid: level_iters=[1000, 200, 50], sigmas=[3.0, 1.0, 0.0], factors=[4,2,1]

    Parameters
    ----------
    mreg_mean : pathlib.Path
        3D mean MREG image (moving).
    t1 : pathlib.Path
        3D T1 image (static/reference).
    xfm : TransformBook
        Must provide targets:
        - xfm["mregmean_to_t1"] (forward 4×4 path)
        - xfm["t1_to_mreg"]     (inverse 4×4 path)
    overwrite : bool, optional
        If False, skip when both targets already exist.

    Returns
    -------
    None

    Files written
    -------------
    - `<xfm['mregmean_to_t1']>` forward 4×4
    - `<xfm['t1_to_mreg']>` inverse 4×4

    Assumptions / Preconditions
    ---------------------------
    - Both inputs are 3D images; registration in image space.

    Raises
    ------
    KeyError
        If required `TransformBook` keys are missing.
    ValueError
        If moving/static are not 3D.
    """
    try:
        out_aff_path = Path(xfm["mregmean_to_t1"])
        out_inv_path = Path(xfm["t1_to_mreg"])
    except Exception as e:
        raise KeyError(
            "[mreg] TransformBook must provide 'mregmean_to_t1' and 't1_to_mreg' targets"
        ) from e

    if out_aff_path.exists() and out_inv_path.exists() and not overwrite:
        print(
            f"[mreg] Affines exist → skip (overwrite=False): "
            f"{out_aff_path.name}, {out_inv_path.name}"
        )
        return

    mov_img = nib.load(str(mreg_mean))
    ref_img = nib.load(str(t1))
    moving = mov_img.get_fdata().astype(np.float32, copy=False)
    static = ref_img.get_fdata().astype(np.float32, copy=False)
    if moving.ndim != 3 or static.ndim != 3:
        raise ValueError(
            f"[mreg] Expect 3D inputs; got moving={moving.shape}, static={static.shape}"
        )

    moving_g2w = mov_img.affine
    static_g2w = ref_img.affine

    # Initial center-of-mass alignment
    com = transform_centers_of_mass(static, static_g2w, moving, moving_g2w)

    # AffineRegistration setup
    metric = MutualInformationMetric(nbins=32)
    areg = AffineRegistration(
        metric=metric, level_iters=[1000, 200, 50], sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1]
    )

    # Stage 1: translation
    xform = TranslationTransform3D()
    opt = areg.optimize(
        static,
        moving,
        xform,
        None,
        static_grid2world=static_g2w,
        moving_grid2world=moving_g2w,
        starting_affine=com.affine,
    )

    # Stage 2: rigid
    xform = RigidTransform3D()
    opt = areg.optimize(
        static,
        moving,
        xform,
        None,
        static_grid2world=static_g2w,
        moving_grid2world=moving_g2w,
        starting_affine=opt.affine,
    )

    # Stage 3: full affine
    xform = AffineTransform3D()
    opt = areg.optimize(
        static,
        moving,
        xform,
        None,
        static_grid2world=static_g2w,
        moving_grid2world=moving_g2w,
        starting_affine=opt.affine,
    )

    A_mreg_to_t1 = opt.affine.astype(np.float64, copy=False)
    A_t1_to_mreg = np.linalg.inv(A_mreg_to_t1)

    _save_affine_matrix(A_mreg_to_t1, out_aff_path)
    _save_affine_matrix(A_t1_to_mreg, out_inv_path)

    xfm["mregmean_to_t1"] = str(out_aff_path)
    xfm["t1_to_mreg"] = str(out_inv_path)

    print(f"[mreg] Saved MREGmean→T1 affine → {out_aff_path}")
    print(f"[mreg] Saved T1→MREG inverse  → {out_inv_path}")


def apply_mean_xfm_to_full_mreg(
    sp: SubjectPaths,
    xfm: "TransformBook",
    *,
    overwrite: bool = False,
    also_mni: bool = False,
) -> None:
    """
    Apply the pre-estimated mean→T1 affine to the full 4D MREG and write a
    T1-space 4D (linear interpolation, float32).

    Parameters
    ----------
    sp : SubjectPaths
        Provides `mreg_dir`, `func_dir`, `anat_dir`, and identifiers.
    xfm : TransformBook
        Must contain the forward affine path under `"mregmean_to_t1"`.
    overwrite : bool, optional
        If False, skip when the T1-space output already exists.
    also_mni : bool, optional
        Placeholder flag; MNI export not implemented here.

    Returns
    -------
    None

    Files written
    -------------
    - mreg/sub-<ID>_space-T1_class-brain_desc-registered_bold.nii.gz

    Assumptions / Preconditions
    ---------------------------
    - Source 4D comes from detrended → realigned → raw precedence.
    - T1 reference is discovered from common candidates under `anat/`.
    - Operates in image space; linear interpolation.

    Raises
    ------
    FileNotFoundError
        If the source 4D, T1, or the forward affine is missing.
    ValueError
        If the source is not 4D after loading.
    """
    subj_tag = sp.sub #subj_tag = f"sub-{sp.subject}"
    mreg_dir = Path(sp.mreg_dir)
    mreg_dir.mkdir(parents=True, exist_ok=True)

    detrended = mreg_dir / deriv_name(subj_tag, "MREG", "brain", "detrended", "bold")
    realigned = mreg_dir / deriv_name(
        subj_tag, "MREG", "brain", "motionrealigned", "bold"
    )
    raw = Path(sp.func_dir) / f"{subj_tag}_task-mreg_bold.nii.gz"

    if detrended.exists():
        src_4d = detrended
    elif realigned.exists():
        src_4d = realigned
    else:
        src_4d = raw
    if not src_4d.exists():
        raise FileNotFoundError(f"[mreg] No MREG 4D found: {src_4d}")

    out_t1_4d = mreg_dir / deriv_name(subj_tag, "T1w", "brain", "registered", "bold")
    if out_t1_4d.exists() and not overwrite:
        print(f"[mreg] [SKIP] Registered 4D exists → skip: {out_t1_4d.name}")
        return

    # locate T1
    t1_candidates = [
        getattr(sp, "t1_path", None),
        Path(sp.anat_dir) / f"{subj_tag}_T1w.nii.gz",
        Path(sp.anat_dir) / f"{subj_tag}_desc-denoised_T1w.nii.gz",
    ]
    t1_path = next((p for p in t1_candidates if p and Path(p).exists()), None)
    if t1_path is None:
        raise FileNotFoundError("[mreg] Could not locate T1 reference")

    src_img = nib.load(str(src_4d))
    src_data = src_img.get_fdata().astype(np.float32, copy=False)
    if src_data.ndim != 4:
        raise ValueError(f"[mreg] Expected 4D MREG, got {src_data.shape}")

    t1_img = nib.load(str(t1_path))

    # load affine MREGmean→T1
    aff_path = Path(xfm["mregmean_to_t1"])
    if not aff_path.exists():
        raise FileNotFoundError(f"[mreg] Affine not found: {aff_path}")
    A_mreg_to_t1 = _load_affine_matrix(aff_path)

    out_vols = []
    for t in range(src_data.shape[3]):
        vol = src_data[..., t]
        moved = apply_affine_to_ref(
            vol, src_img.affine, t1_img, A_mreg_to_t1, interp="linear"
        )
        if isinstance(moved, nib.Nifti1Image):
            moved_data = moved.get_fdata().astype(np.float32, copy=False)
        else:
            moved_data = np.asarray(moved, dtype=np.float32)
        out_vols.append(moved_data)

    out_4d = np.stack(out_vols, axis=3).astype(np.float32, copy=False)
    nib.save(nib.Nifti1Image(out_4d, t1_img.affine, t1_img.header), str(out_t1_4d))
    print(f"[mreg] Saved T1-space registered 4D → {out_t1_4d}")

    if also_mni:
        print("[mreg] also_mni=True requested, MNI export not implemented here.")


# -------------------------------------------------------------
# I/O helpers (BIDS naming, paths)
# -------------------------------------------------------------
def _pick_mreg_ref_path(sp: SubjectPaths) -> Path:
    """
    Select a reference image on the MREG grid for alignment/warping.

    Preference order
    ----------------
    1) bandmaps/<sub>_space-MREG_desc-meanamp_map.nii.gz
    2) mreg/<sub>_space-MREG_class-brain_desc-mean_map.nii.gz
    3) mreg/<sub>_space-MREG_class-brain_desc-detrended_bold.nii.gz
    4) mreg/<sub>_space-MREG_class-brain_desc-motionrealigned_bold.nii.gz
    5) sp.func_mreg_bold  (raw 4D)

    Parameters
    ----------
    sp : SubjectPaths
        Subject-scoped paths.

    Returns
    -------
    pathlib.Path
        Path to the first existing candidate (3D or 4D).

    Files written
    -------------
    - None.

    Raises
    ------
    FileNotFoundError
        If none of the candidates exist.

    Notes
    -----
    - Callers typically use `shape[:3]` and `affine` to define the target grid.
    """
    sub = sp.sub
    band_dir = Path(sp.bandmaps_dir)
    mreg_dir = Path(sp.mreg_dir)

    meanamp = band_dir / f"{sub}_space-MREG_desc-meanamp_map.nii.gz"
    if meanamp.exists():
        return meanamp

    mean = mreg_dir / deriv_name(sub, "MREG", "brain", "mean", "map")
    if mean.exists():
        return mean

    detr = mreg_dir / deriv_name(sub, "MREG", "brain", "detrended", "bold")
    real = mreg_dir / deriv_name(sub, "MREG", "brain", "motionrealigned", "bold")
    for p in (detr, real, Path(sp.func_mreg_bold)):
        if Path(p).exists():
            return Path(p)

    raise FileNotFoundError("No MREG reference found.")


# -------------------------------------------------------------
# Masking (MREG/T1/MNI paths)
# -------------------------------------------------------------
def warp_masks_single_shot(
    sp: SubjectPaths,
    xfm: TransformBook,
    also_mni: bool = True,
    clip_with_mreg_mask: Path | None = None,
) -> None:
    """
    Warp native-space masks → MREG (and optionally → MNI) in one pass.

    Sources & transforms
    --------------------
    Masks (under `sp.masks_dir`, produced elsewhere, e.g., `seg.py`):
      - TOF arteries : `*_space-TOF_class-arteries_desc-main_mask.nii.gz`
      - MRV veins    : `*_space-MRV_class-veins_desc-main_mask.nii.gz`
      - HT2w PVS     : `*_space-HT2w_class-pvs_desc-main_mask.nii.gz`
    Required affine keys in `xfm`:
      - `"tof_to_t1"`, `"mrv_to_t1"`, `"ht2w_to_t1"` (scan→T1)
      - `"t1_to_mreg"` (T1→MREG)

    Parameters
    ----------
    sp : SubjectPaths
        Provides `sub`, `masks_dir`, and access to the MREG reference via
        `_pick_mreg_ref_path`.
    xfm : TransformBook
        Mapping that yields file paths to 4×4 text matrices. Optionally:
        `xfm.mni_ref_path` (MNI reference), `xfm._mapping_cache` (deformable
        mapping for MNI export).
    also_mni : bool, optional
        If True and MNI resources exist, write MNI-space masks as well.
    clip_with_mreg_mask : pathlib.Path or None, optional
        If provided, a binary **MREG-space** brain mask used to AND-clip each
        warped label before saving.

    Returns
    -------
    None

    Files written
    -------------
    - masks/sub-<ID>_space-MREG_class-arteries_desc-main_mask.nii.gz
    - masks/sub-<ID>_space-MREG_class-veins_desc-main_mask.nii.gz
    - masks/sub-<ID>_space-MREG_class-pvs_desc-main_mask.nii.gz
    - masks/sub-<ID>_space-MNI_class-<klass>_desc-main_mask.nii.gz (optional)

    Assumptions / Preconditions
    ---------------------------
    - Composed affine scan→MREG is `(T1→MREG) @ (scan→T1)`.
    - Resampling target is the MREG grid from `_pick_mreg_ref_path`.
    - Label resampling uses nearest-neighbor; masks are thresholded > 0.5.

    Warnings
    --------
    - Missing masks or missing transforms are skipped with a warning.
    - Output shapes are validated against the MREG grid (mismatch → RuntimeError).
    """
    sub = sp.sub
    masks_dir = Path(sp.masks_dir)

    # native masks produced by seg.py
    art_native = masks_dir / deriv_name(sub, "TOF", "arteries", "main", "mask")
    vein_mrv = masks_dir / deriv_name(sub, "MRV", "veins", "main", "mask")
    pvs_native = masks_dir / deriv_name(sub, "hT2w", "pvs", "main", "mask")

    # MREG reference grid (use mean if present, else the 4D bold)
    mreg_ref = _pick_mreg_ref_path(sp)

    # Optional: load the EPI brain mask on the MREG grid (once)
    epi_mask_bool = None
    if clip_with_mreg_mask is not None and Path(clip_with_mreg_mask).exists():
        ref_img = nib.load(str(mreg_ref))
        ref_shape = ref_img.shape[:3]
        ref_aff = ref_img.affine

        epi_img = nib.load(str(clip_with_mreg_mask))
        # Align mask to the exact MREG grid if needed (nearest)
        if (epi_img.shape[:3] != ref_shape) or (
            not np.allclose(epi_img.affine, ref_aff, atol=1e-3)
        ):
            
            epi_img = resample_from_to(epi_img, (ref_shape, ref_aff), order=0)
            print("[warp] [WARN] EPI brain mask snapped to MREG grid ")
        epi_mask_bool = epi_img.get_fdata() > 0.5
    else:
        ref_img = None
        ref_shape = ref_aff = None

    def _to_mreg(native_mask: Path, src_to_t1_key: str, klass: str):
        """
        Warp a native-space label/mask to the MREG grid using precomputed
        transforms (scan→T1, T1→MREG). Optionally write MNI-space outputs.
        """

        if not native_mask or not Path(native_mask).exists():
            print(f"[warp] [SKIP] Missing native {klass} mask")
            return

        out_mreg = masks_dir / deriv_name(sub, "MREG", klass, "main", "mask")
        out_mreg.parent.mkdir(parents=True, exist_ok=True)

        # Compose SCAN→MREG
        try:
            A_src_to_t1 = np.loadtxt(str(xfm.get(src_to_t1_key)))
            A_t1_to_mreg = np.loadtxt(str(xfm.get("t1_to_mreg")))
        except Exception as e:
            raise RuntimeError(f"[warp] Required transform missing: {e}")

        if A_src_to_t1.shape != (4, 4) or A_t1_to_mreg.shape != (4, 4):
            raise ValueError("[warp] Expected 4x4 affines for src→T1 or T1→MREG.")

        A_src_to_mreg = A_t1_to_mreg @ A_src_to_t1

        moving = nib.load(str(native_mask))

        # Reuse cached MREG reference when available
        if "ref_img" in locals() and ref_img is not None:
            _ref_img = ref_img
            _ref_shape = ref_shape
            _ref_aff = ref_aff
        else:
            _ref_img = nib.load(str(mreg_ref))
            _ref_shape = _ref_img.shape[:3]
            _ref_aff = _ref_img.affine

        # Bake composed transform into moving header
        moving_affine = A_src_to_mreg @ moving.affine
        moving_img = nib.Nifti1Image(
            moving.get_fdata(), moving_affine, moving.header.copy()
        )

        # Resample to MREG grid (nearest for labels)
        resampled = resample_from_to(moving_img, (_ref_shape, _ref_aff), order=0)
        resampled_data = (resampled.get_fdata() > 0.5).astype(np.uint8, copy=False)

        # Optional clipping by EPI/MREG brain mask
        if "epi_mask_bool" in locals() and epi_mask_bool is not None:
            resampled_data = (
                resampled_data.astype(bool) & epi_mask_bool
            ).astype(np.uint8, copy=False)

        # Save as uint8 on MREG grid
        hdr = _ref_img.header.copy()
        hdr.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(resampled_data, _ref_aff, hdr), str(out_mreg))

        # Grid sanity-check
        if resampled_data.shape != tuple(_ref_shape):
            raise RuntimeError(
                f"[warp] {klass} mask not on MREG grid: got {resampled_data.shape}, want {_ref_shape}"
            )

        print(f"[warp] Saved {klass} mask in MREG space → {out_mreg}")

        # Optional MNI output
        if "also_mni" in locals() and also_mni:
            if getattr(xfm, "mni_ref_path", None) and getattr(xfm, "_mapping_cache", None):
                out_mni = masks_dir / deriv_name(sub, "MNI", klass, "main", "mask")
                out_mni.parent.mkdir(parents=True, exist_ok=True)
                apply_affine_then_warp_to_mni(
                    native_mask,
                    xfm.get(src_to_t1_key),
                    xfm._mapping_cache,
                    xfm.mni_ref_path,
                    out_mni,
                    is_label=True,
                )
                print(f"[warp] Saved {klass} mask in MNI space → {out_mni}")
            else:
                print(
                    "[warp] also_mni=True but no MNI template/mapping; skipping MNI outputs."
                )

    # arteries (TOF)
    if art_native.exists():
        _to_mreg(art_native, "tof_to_t1", "arteries")

    def _has_xfm(key: str) -> bool:
        try:
            xfm.get(key)
            return True
        except KeyError:
            return False

    # veins
    if vein_mrv.exists() and _has_xfm("mrv_to_t1"):
        _to_mreg(vein_mrv, "mrv_to_t1", "veins")
    else:
        print("[warp] [SKIP] No MRV→T1 transform or mask; skipping veins")

    # pvs (hT2w)
    if pvs_native.exists() and _has_xfm("ht2w_to_t1"):
        _to_mreg(pvs_native, "ht2w_to_t1", "pvs")
    else:
        print("[warp] [SKIP] No HT2w→T1 transform or mask; skipping pvs")