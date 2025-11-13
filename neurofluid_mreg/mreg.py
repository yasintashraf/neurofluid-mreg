# SPDX-License-Identifier: MIT
"""
mreg.py
-------
Single-subject MREG preprocessing and band-power export for a BIDS-first
pipeline. This module keeps your algorithmic choices intact (NiPy Realign4d,
voxelwise polynomial detrend, DIPY staged affine for mean MREG→T1, linear
interpolation for images) and standardizes derivatives layout/I/O.

Pipeline steps
--------------
1. Realign + detrend the 4D MREG time series (native MREG space)
2. Compute a 3D temporal mean on the MREG grid
3. Register mean MREG → T1 (COM → translation → rigid → affine; MI metric)
4. Apply the mean→T1 affine to the full 4D series (linear)
5. Export MREG-space band-power maps (and mean amplitude) to MNI space

Inputs / Outputs
----------------
Inputs  : Subject-scoped paths via `SubjectPaths`; optional transforms via
          `TransformBook`.
Outputs : NIfTIs (float32 for images) and small text/JSON/TSV auxiliaries
          written under `derivatives/neurofluid-mreg/sub-<ID>/`.

Files written
-------------
- mreg/<sub>_space-MREG_class-brain_desc-motionrealigned_bold.nii.gz
- mreg/<sub>_space-MREG_class-brain_desc-detrended_bold.nii.gz
- mreg/<sub>_space-MREG_class-brain_desc-mean_map.nii.gz
- mreg/<sub>_space-T1_class-brain_desc-registered_bold.nii.gz
- bandmaps/<sub>_space-MNI_band-<BAND>_desc-power_map.nii.gz
- bandmaps/<sub>_space-MNI_desc-meanamp_map.nii.gz
- qc/<sub>_desc-motion_params.tsv
- qc/<sub>_desc-preproc_params.json

Where filenames follow:
`sub-<ID>_space-<SPACE>_class-<CLASS>_desc-<DESC>_<SUFFIX>.nii.gz`

Assumptions / Preconditions
---------------------------
- Spaces: All operations occur in image space of the chosen reference. Apply
  transforms/resampling only where stated. Affine mismatches are handled by
  explicit resampling against the reference image.
- Shapes/dtypes: 4D MREG (X, Y, Z, T). Images saved float32.
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
- export_bandpower_to_mni
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
        print(f"[mreg] [SKIP] Realign/detrend exist for {subj}")
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
# Band power export (MREG → MNI)
# -------------------------------------------------------------



def export_bandpower_to_mni(
    sp,
    xfm,
    *,
    t1_path: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Parameters
    ----------
    sp : SubjectPaths
        Subject context (expects `sub`, `bandmaps_dir`).
    xfm : TransformBook
        Must provide:
          - 4×4 text affine 't1_to_mreg' (or 'mregmean_to_t1')
          - `_mapping_cache` : saved T1↔MNI deformation with `.transform(...)`
          - `mni_ref_path`   : canonical MNI reference NIfTI path
    t1_path : pathlib.Path
        Path to the **T1 image you want to use** (e.g., denoised T1).
    overwrite : bool, optional
        If False (default), skip writing when the MNI outputs already exist.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping of band token (and 'meanamp') to written MNI NIfTI paths.

    Files written
    -------------
    - bandmaps/sub-<ID>_space-MNI_band-<BAND>_desc-power_map.nii.gz
    - bandmaps/sub-<ID>_space-MNI_desc-meanamp_map.nii.gz

    Assumptions / Preconditions
    ---------------------------
    - Input maps exist on the **MREG grid**:
      `*_space-MREG_band-*_desc-power_map.nii.gz` and (optionally)
      `*_space-MREG_desc-meanamp_map.nii.gz`.
    - The T1 used here (`t1_path`) matches the saved T1↔MNI mapping in `xfm`.
    - Affine used is **MREG→T1** (prefer explicit; else inverse of T1→MREG).
    - Linear interpolation is used for all resampling; outputs are float32 on
      the MNI reference grid/affine.

    Warnings
    --------
    - If T1↔MNI mapping does not correspond to `t1_path`, misalignment can
      occur (ensure both belong to the same subject & T1 version).
    - Repeated resampling may smooth band-power values slightly; if this is
      critical, consider exporting once from the highest-fidelity source.

    Raises
    ------
    FileNotFoundError
        If `t1_path` is missing or no MREG band maps are found.
    RuntimeError
        If `xfm._mapping_cache` or required affines are missing.

    Notes
    -----
    - Outputs are saved as float32 and adopt the MNI reference affine/header.
    - The **band token** is preserved verbatim from source filenames.

    """
    sub = sp.sub
    out_paths: dict[str, Path] = {}

    bandmaps_dir = Path(sp.bandmaps_dir)
    bandmaps_dir.mkdir(parents=True, exist_ok=True)

    have_any = any(bandmaps_dir.glob(f"{sub}_space-MREG_band-*_desc-power_map.nii.gz")) \
               or (bandmaps_dir / f"{sub}_space-MREG_desc-meanamp_map.nii.gz").exists()
    if not have_any:
        print("[export→mni] No MREG-space band/meanamp maps found; nothing to export.")
        return {}

    # --- Load T1 that the caller decided (already denoised or not)
    if not Path(t1_path).exists():
        raise FileNotFoundError(f"[export→mni] t1_path does not exist: {t1_path}")
    t1_img = nib.load(str(t1_path))

    # --- MNI reference + T1→MNI mapping
    if not getattr(xfm, "mni_ref_path", None):
        raise RuntimeError("[export→mni] Missing xfm.mni_ref_path.")
    mni_ref_img = nib.load(str(xfm.mni_ref_path))

    mapping = getattr(xfm, "_mapping_cache", None)
    if mapping is None:
        raise RuntimeError("[export→mni] Missing T1↔MNI mapping (_mapping_cache).")

    # --- MREG→T1 affine (prefer explicit, else inverse of T1→MREG)
    try:
        A_mreg_to_t1 = np.loadtxt(str(xfm.get("mregmean_to_t1")))
    except Exception:
        try:
            A_t1_to_mreg = np.loadtxt(str(xfm.get("t1_to_mreg")))
            A_mreg_to_t1 = np.linalg.inv(A_t1_to_mreg)
        except Exception as e:
            raise RuntimeError(f"[export→mni] Could not form MREG→T1 affine: {e}")

    def _to_mni_float(src_path: Path, out_path: Path):
        """Affine MREG→T1 (linear), then deform T1→MNI (linear), save float32."""
        src_img = nib.load(str(src_path))

        # Bake MREG→T1 into the header, then resample to T1 grid (linear)
        baked = nib.Nifti1Image(
            np.asarray(src_img.get_fdata(), dtype=np.float32),
            A_mreg_to_t1 @ src_img.affine,
            src_img.header.copy(),
        )
        on_t1 = resample_from_to(baked, (t1_img.shape, t1_img.affine), order=1)

        # Deform to MNI with saved mapping (linear)
        moved = mapping.transform(
            np.asarray(on_t1.get_fdata(), dtype=np.float32), interpolation="linear"
        )
        out_img = nib.Nifti1Image(
            moved.astype(np.float32, copy=False),
            mni_ref_img.affine,
            mni_ref_img.header,
        )
        out_img.header.set_data_dtype(np.float32)

        if overwrite or (not out_path.exists()):
            nib.save(out_img, str(out_path))
            print(f"[export→mni] Saved → {out_path}")
        else:
            print(f"[export→mni] [SKIP] Exists: {out_path.name}")

    # --- Mean-amplitude map
    mean_src = bandmaps_dir / f"{sub}_space-MREG_desc-meanamp_map.nii.gz"
    mean_out = bandmaps_dir / f"{sub}_space-MNI_desc-meanamp_map.nii.gz"
    if mean_src.exists():
        _to_mni_float(mean_src, mean_out)
        out_paths["meanamp"] = mean_out

    # --- Band maps (discover whatever you wrote on MREG)
    for src in sorted(bandmaps_dir.glob(f"{sub}_space-MREG_band-*_desc-power_map.nii.gz")):
        band_token = src.name.split("_band-")[1].split("_")[0]  # preserve original name
        out = bandmaps_dir / f"{sub}_space-MNI_band-{band_token}_desc-power_map.nii.gz"
        _to_mni_float(src, out)
        out_paths[band_token] = out

    return out_paths
