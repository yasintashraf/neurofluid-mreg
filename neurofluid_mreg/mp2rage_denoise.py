# SPDX-License-Identifier: MIT
"""
mp2rage_denoise.py
------------------
Robust MP2RAGE combination (Pure-Python) inspired by José P. Marques'
RobustCombination routine. Implements polarity correction, quadratic INV1
refinement, background-noise regularization, and safe I/O.

Pipeline steps
--------------
1. Load UNI/INV1/INV2 (no resampling; image space)
2. Correct INV1 polarity to match phase-sensitive UNI
3. Solve quadratic per voxel to refine INV1
4. Estimate background noise from INV2 corners; compute β = (λσ)^2
5. Robust combination → save (optionally 12-bit) and JSON sidecar

Inputs / Outputs
----------------
Inputs  : NIfTI UNI/INV1/INV2 volumes (same shape, native space).
Outputs : Robust UNI (float32 by default, or 12-bit uint16 if flagged).

Files written
-------------
- derivatives/neurofluid-mreg/sub-<ID>/anat/
  - sub-<ID>_space-T1w_class-brain_desc-mp2rageRobust_map.nii.gz
  - sub-<ID>_space-T1w_class-brain_desc-mp2rageRobust_map.json

Assumptions / Preconditions
---------------------------
- UNI/INV1/INV2 are co-registered and share shape/affine; processing occurs
  strictly in **image space** (no resampling).
- Intensities for UNI may be 12-bit centered (≈[-0.5, 0.5]) or float.

Warnings
--------
- Sidecar propagation uses `Path.with_suffix('.json')`; outputs ending with
  `.nii.gz` will yield a `.nii.json` filename (adjust upstream if needed).
- Background σ is estimated from INV2 8-corner patches via MAD; atypical
  background may bias β.

Public API
----------
- load_nifti
- save_nifti
- robust_mp2rage
- mp2rage_denoise
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter  # only if you later want optional smoothing


# -------------------------------------------------------------
# I/O helpers (NIfTI load/save)
# -------------------------------------------------------------
def load_nifti(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Header, np.ndarray]:
    """
    Load a NIfTI image.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to `.nii`/`.nii.gz`.

    Returns
    -------
    tuple
        (data, header, affine):
        - data : ndarray, dtype=float64, shape (X, Y, Z[ ,T])
        - header : nib.Nifti1Header (copied)
        - affine : ndarray, shape (4, 4)

    Files written
    -------------
    - None.

    Assumptions / Preconditions
    ---------------------------
    - Operates in image space; no resampling.
    """
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float64)
    return data, img.header.copy(), img.affine


def save_nifti(
    data: np.ndarray,
    header: nib.Nifti1Header,
    out_path: str | Path,
    as_12bit: bool = False,
    affine: np.ndarray | None = None,
) -> None:
    """
    Save an array as NIfTI (float32 or 12-bit uint16).

    Parameters
    ----------
    data : ndarray
        Image to write; NaN/Inf are converted to 0.
    header : nib.Nifti1Header
        Header to attach to the output.
    out_path : str or pathlib.Path
        Destination filename.
    as_12bit : bool, optional
        If True, map data in ~[-0.5, 0.5] to 12-bit [0, 4095] uint16 via
        round(4095*(x+0.5)); else write float32.
    affine : ndarray or None, optional
        4×4 affine; if None, use `header.get_best_affine()`.

    Returns
    -------
    None

    Files written
    -------------
    - `<out_path>` (NIfTI).

    Warnings
    --------
    - 12-bit scaling assumes input already centered around zero.
    """
    if as_12bit:
        data_safe = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data_scaled = np.round(4095 * (data_safe + 0.5))
        data_scaled = np.clip(data_scaled, 0, 4095).astype(np.uint16)
        data_to_save = data_scaled
        header.set_data_dtype(np.uint16)
    else:
        data_to_save = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
        header.set_data_dtype(np.float32)

    if affine is None:
        affine = header.get_best_affine()

    nib.save(nib.Nifti1Image(data_to_save, affine, header), str(out_path))
    print(f"[mp2rage] Saved → {out_path}")



# -------------------------------------------------------------
# Core maths
# -------------------------------------------------------------
def _correct_inv1_polarity(uni: np.ndarray, inv1: np.ndarray) -> np.ndarray:
    """
    Match INV1 sign to phase-sensitive UNI.

    Parameters
    ----------
    uni : ndarray
        UNI volume (float).
    inv1 : ndarray
        INV1 volume (float), same shape as `uni`.

    Returns
    -------
    ndarray
        INV1 with polarity corrected to match `uni`.

    Assumptions / Preconditions
    ---------------------------
    - Same shape and alignment as `uni`.
    """
    return np.sign(uni) * inv1


def _solve_inv1_quadratic(
    uni: np.ndarray, inv1: np.ndarray, inv2: np.ndarray
) -> np.ndarray:
    """
    Refine INV1 via quadratic root selection to match measured INV1.

    The relationship (Marques & O’Brien) is:
        uni = (inv1 * inv2 − β) / (inv1^2 + inv2^2 + 2β)
    with β=0 when deriving the quadratic in inv1. Solving
        a*inv1^2 + b*inv1 + c = 0
    where a = −uni, b = inv2, c = −inv2^2 * uni,
    two roots are obtained; pick the one closer to the measured INV1.

    Parameters
    ----------
    uni, inv1, inv2 : ndarray
        Input volumes (float), same shape.

    Returns
    -------
    ndarray
        Refined INV1 estimate, same shape.

    Warnings
    --------
    - Discriminant is clamped at 0 to avoid small negative values from
      numerical error.
    """
    a = -uni
    b = inv2
    c = -inv2**2 * uni

    discr = np.maximum(b**2 - 4 * a * c, 0.0)
    sqrt_discr = np.sqrt(discr)

    root_pos = (-b + sqrt_discr) / (2 * a)
    root_neg = (-b - sqrt_discr) / (2 * a)

    use_neg = np.abs(inv1 - root_neg) < np.abs(inv1 - root_pos)
    inv1_refined = np.where(use_neg, root_neg, root_pos)
    return inv1_refined


def _estimate_noise(
    inv2: np.ndarray, corner_width: int = 11, lamb: float = 10.0
) -> float:
    """
    Estimate background σ from 8 corner cubes of INV2 (MAD-based).

    Parameters
    ----------
    inv2 : ndarray
        INV2 volume (float).
    corner_width : int, optional
        Width of each corner cube (voxels).
    lamb : float, optional
        Regularization factor (λ); returned σ is scaled by λ.

    Returns
    -------
    float
        λ·σ estimate (≥ 1e-8).

    Notes
    -----
    - σ is approximated as 1.4826 * MAD of |background|.
    """
    cw = corner_width
    X, Y, Z = inv2.shape
    corners = [
        inv2[0:cw, 0:cw, 0:cw],
        inv2[-cw:X, 0:cw, 0:cw],
        inv2[0:cw, -cw:Y, 0:cw],
        inv2[0:cw, 0:cw, -cw:Z],
        inv2[-cw:X, -cw:Y, 0:cw],
        inv2[-cw:X, 0:cw, -cw:Z],
        inv2[0:cw, -cw:Y, -cw:Z],
        inv2[-cw:X, -cw:Y, -cw:Z],
    ]
    bg = np.abs(np.concatenate([c.ravel() for c in corners]))
    med = np.median(bg)
    mad = np.median(np.abs(bg - med))
    sigma = 1.4826 * mad
    return max(lamb * sigma, 1e-8)


def robust_mp2rage(
    uni: np.ndarray,
    inv1: np.ndarray,
    inv2: np.ndarray,
    lamb: float = 10.0,
    corner_width: int = 11,
) -> np.ndarray:
    """
    Compute robust, background-suppressed UNI.

    Parameters
    ----------
    uni, inv1, inv2 : ndarray
        Input volumes (float64), same shape.
    lamb : float, optional
        Regularization factor λ (a.k.a. `multiplyingFactor`).
    corner_width : int, optional
        Corner width for background σ estimation from INV2.

    Returns
    -------
    tuple
        (robust, was_12bit)
        - robust : ndarray, float32, same shape as inputs.
        - was_12bit : bool, True if UNI appeared 12-bit centered and was
          treated as such for scaling on save.

    Files written
    -------------
    - None (see `mp2rage_denoise` for saving).

    Assumptions / Preconditions
    ---------------------------
    - UNI/INV1/INV2 are co-registered (same shape/affine).

    Warnings
    --------
    - If denom is zero, outputs 0 for those voxels (safe divide).

    Notes
    -----
    - β = (λ·σ)^2, where σ is estimated via MAD from INV2 corners.
    - Robust formula:
        robust = (inv1_refined*inv2 - β) / (inv1_refined^2 + inv2^2 + 2β)
    """
    # 1) Possibly convert 12-bit to −0.5…0.5
    as_12bit = False
    if uni.min() >= 0 and uni.max() >= 0.51:
        uni = (uni - uni.max() / 2.0) / uni.max()
        as_12bit = True

    # 2) Fix INV1 polarity
    inv1_signed = _correct_inv1_polarity(uni, inv1)

    # 3) Quadratic refinement
    inv1_refined = _solve_inv1_quadratic(uni, inv1_signed, inv2)

    # 4) Noise estimate
    sigma_scaled = _estimate_noise(inv2, corner_width, lamb)
    beta = sigma_scaled**2

    # 5) Robust combination
    numer = inv1_refined * inv2 - beta
    denom = inv1_refined**2 + inv2**2 + 2 * beta
    robust = np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)

    # Original was robust = numer / denom
    return robust.astype(np.float32), as_12bit


# -------------------------------------------------------------
# High-level wrapper
# -------------------------------------------------------------
def mp2rage_denoise(
    uni_path: str | Path,
    inv1_path: str | Path,
    inv2_path: str | Path,
    out_path: str | Path | None = None,
    lamb: float = 10.0,
    corner_width: int = 11,
) -> np.ndarray:
    """
    Run robust MP2RAGE denoising and optionally write outputs/sidecar.

    Parameters
    ----------
    uni_path, inv1_path, inv2_path : str or pathlib.Path
        Paths to input NIfTI files (same shape/affine).
    out_path : str or pathlib.Path or None, optional
        If provided, write robust UNI to this path (float32 or 12-bit).
    lamb : float, optional
        Regularization factor λ.
    corner_width : int, optional
        Corner width (voxels) for σ estimation.

    Returns
    -------
    ndarray
        Robust UNI (float32), regardless of saving mode.

    Files written
    -------------
    - `<out_path>` (NIfTI).
    - Sidecar JSON `<out_path with .json suffix>` (if an input UNI sidecar
      exists), enriched with provenance.

    Assumptions / Preconditions
    ---------------------------
    - Inputs are co-registered; processing in image space.

    Warnings
    --------
    - When `out_path` ends with `.nii.gz`, the sidecar name becomes
      `<name>.nii.json` due to `Path.with_suffix('.json')`.
    """
    uni, uni_hdr, uni_aff = load_nifti(uni_path)
    inv1, _, _ = load_nifti(inv1_path)
    inv2, _, _ = load_nifti(inv2_path)

    robust, was_12bit = robust_mp2rage(
        uni, inv1, inv2, lamb=lamb, corner_width=corner_width
    )

    if out_path:
        save_nifti(robust, uni_hdr, out_path, as_12bit=was_12bit, affine=uni_aff)
        _write_json_sidecar(uni_path, inv1_path, inv2_path, out_path, lamb)

    return robust


def _write_json_sidecar(
    uni_path: str | Path,
    inv1_path: str | Path,
    inv2_path: str | Path,
    out_path: str | Path,
    lamb: float,
) -> None:
    """
    Propagate and enrich BIDS JSON sidecar (if UNI sidecar exists).

    Parameters
    ----------
    uni_path, inv1_path, inv2_path : str or pathlib.Path
        Input file paths.
    out_path : str or pathlib.Path
        Output robust UNI path (used to derive JSON sidecar path).
    lamb : float
        Regularization factor λ written to metadata.

    Returns
    -------
    None

    Files written
    -------------
    - `<out_path with .json suffix>` JSON sidecar (if UNI sidecar existed).

    Warnings
    --------
    - Uses `Path.with_suffix('.json')`; with `.nii.gz` inputs/outputs this
      creates `.nii.json` filenames.
    """
    in_json = Path(uni_path).with_suffix(".json")
    if not in_json.exists():
        print("[mp2rage] [SKIP] No UNI sidecar found; JSON not written")
        return

    with open(in_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta.update(
        {
            "BasedOn": [str(uni_path), str(inv1_path), str(inv2_path)],
            "SeriesDescription": f"{meta.get('ProtocolName','')}_MP2RAGE_denoised_background",
            "NoiseRegularization": lamb,
        }
    )

    out_json = Path(out_path).with_suffix(".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[mp2rage] Saved → {out_json}")



# -------------------------------------------------------------
# Command-line interface
# -------------------------------------------------------------
def _parse_cli() -> argparse.Namespace:
    """
    Parse command-line arguments for robust MP2RAGE.

    Returns
    -------
    argparse.Namespace
        Parsed arguments: uni, inv1, inv2, out, lamb, corner_width.
    """
    p = argparse.ArgumentParser(description="MP2RAGE background-noise suppression")
    p.add_argument("--uni", required=True, help="UNI NIfTI file")
    p.add_argument("--inv1", required=True, help="INV1 NIfTI file")
    p.add_argument("--inv2", required=True, help="INV2 NIfTI file")
    p.add_argument("--out", required=True, help="output robust UNI NIfTI")
    p.add_argument(
        "--lambda",
        dest="lamb",
        type=float,
        default=10.0,
        help="regularisation factor (default 10)",
    )
    p.add_argument(
        "--corner-width",
        dest="corner_width",
        type=int,
        default=11,
        help="corner cube width (default 11)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    mp2rage_denoise(args.uni, args.inv1, args.inv2, args.out, args.lamb, args.corner_width)
    print("[mp2rage] DONE")

