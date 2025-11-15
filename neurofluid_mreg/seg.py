# SPDX-License-Identifier: MIT
"""
seg.py
------
Neurofluid–MREG segmentation wrappers that invoke existing array-level helpers
to produce BIDS-derivative outputs in **native space** (no resampling). Each
wrapper reads a NIfTI, runs the fixed pipeline (Frangi/hysteresis/morphology/
skeletonization), and writes three artifacts: vesselness, skeleton, and mask.

Pipeline steps
--------------
1. Load native-space NIfTI (affine/header preserved on write)
2. Run modality-appropriate segmentation (parameters fixed by wrapper)
3. Save artifacts with BIDS-like filenames (space enforced per modality)

Inputs / Outputs
----------------
Inputs  : Paths to native-space NIfTI volumes (TOF, MRV, hT2w).
Outputs : Dict per call: {"vesselness": Path, "skeleton": Path, "mask": Path}.

Files written
-------------
- derivatives/neurofluid-mreg/sub-<ID>/masks/
  - sub-<ID>[_ses-<LABEL>]_space-<SPACE>_class-<CLASS>_desc-vesselness_map.nii.gz
  - sub-<ID>[_ses-<LABEL>]_space-<SPACE>_class-<CLASS>_desc-skeleton_mask.nii.gz
  - sub-<ID>[_ses-<LABEL>]_space-<SPACE>_class-<CLASS>_desc-main_mask.nii.gz
  
- derivatives/neurofluid-mreg/sub-<ID>/masks/
  - sub-<ID>[_ses-<LABEL>]_space-hT2w_class-pvs_desc-preproc_map.nii.gz
  - sub-<ID>[_ses-<LABEL>]_space-hT2w_class-brain_desc-coarsemask_mask.nii.gz
  - sub-<ID>[_ses-<LABEL>]_space-MRV_class-veins_desc-prefrangi_image.nii.gz
- derivatives/neurofluid-mreg/sub-<ID>/epc/  (only if `use_epc=True`)
  - sub-<ID>[_ses-<LABEL>]_space-hT2w_class-pvs_desc-epc_map.nii.gz
  - sub-<ID>[_ses-<LABEL>]_from-T1_to-hT2w_mode-affine_xfm.txt


Assumptions / Preconditions
---------------------------
- Spaces: Functions run strictly in **native voxel space**; no resampling.
- Affines: Outputs reuse the loaded affine/header unchanged.
- Shapes/dtypes: Vesselness saved as float32; masks/skeletons saved as uint8.
- BIDS naming: Filenames reuse `sub-*`, optional `ses-*`; `space-*` is enforced
  per modality via `space_override` ("TOF", "MRV", "hT2w").
- PVS outputs are always written on the native hT2w grid/affine; EPC performs
  T1→hT2w registration internally but does not change the output space.
- When `use_epc=True`, `t1_path` must belong to the same subject; MI-based
  registration refines any residual misalignment.

Warnings
--------
- Inputs should already be skull-stripped / preprocessed as required upstream.
- If input filenames lack `space-*`, wrappers still enforce the correct space
  token via `space_override` when constructing outputs.
- QC PNG writing in `veins_mrv` requires matplotlib and writes to `out_dir/qc/`.
- If `use_epc=True` but `t1_path` is missing/invalid, `pvs_hT2w` falls back to
  the hT2w driver and continues.
- EPC and small-scale Frangi may shift contrast; downstream thresholds may need
  retuning. Preprocessing uses robust percentile windowing and may clip tails.
- All wrappers support `overwrite`: when False (default), processing is skipped if
  the expected outputs already exist (early check).


Public API
----------
- arteries_tof
- veins_mrv
- pvs_hT2w
"""

import re
import numpy as np
from pathlib import Path
import json
from scipy import ndimage
from .helper_seg import (
    load_nifti,
    save_nifti,
    segment_vessels,
    make_output_paths,
    compute_r2star_map,
    combine_echoes_te_weighted,
    frangi_vesselness,
    threshold_vesselness,
    iterative_hysteresis,
    preprocess_mrv_for_vesselness,
    intensity_gate,
    pvs_preprocess_hT2w,
    epc_from_t1_and_hT2w,
    segment_pvs_frangi3d
)
from skimage.morphology import (
    remove_small_objects,
    ball,
    disk,
    black_tophat,
    white_tophat,
    skeletonize,
)

# -------------------------------------------------------------
# Vessel segmentation wrappers (native space, BIDS I/O)
# -------------------------------------------------------------


def arteries_tof(
    tof_nii: Path,
    out_dir: Path,
    do_dilate: bool = True,
    dilate_rad: int = 1,
    *,
    select_gamma_auto: bool = True,
    auto_gamma_fraction: float = 0.5,
    scale_min: float = 0.5,
    scale_max: float = 6.0,
    scale_step: float = 0.5,
    alpha: float = 0.5,
    beta: float = 0.6,
    gamma: float = 0.018,
    threshold_frac: float = 0.2,
    min_size: int = 25,
    n_iter: int = 2,
    kappa: float = 0.1,
    prevent_leaking: bool = True,
    do_tophat: bool = True,
    tophat_size: int | None = None,
    overwrite: bool = False, 
) -> dict:
    """
    Segment cerebral arteries from TOF MRI and write vesselness, mask, skeleton.

    Parameters
    ----------
    tof_nii : pathlib.Path
        Native-space TOF NIfTI path. BIDS entities (e.g., `sub-*`, `space-TOF`)
        are reused to build derivative filenames.
    out_dir : pathlib.Path
        Subject derivatives directory where outputs are written.
    do_dilate : bool, optional
        If True, dilate the binary mask.
    dilate_rad : int, optional
        Ball radius (vox) for dilation when `do_dilate=True`.
    select_gamma_auto, auto_gamma_fraction : float or bool, optional
        Enable and scale auto-γ selection from Hessian stats.
    scale_min, scale_max, scale_step : float, optional
        Frangi sigma range (vox).
    alpha, beta, gamma : float, optional
        Frangi parameters.
    threshold_frac : float, optional
        Simple max-fraction threshold used when `n_iter == 1`.
    min_size : int, optional
        Minimum object size to retain (vox).
    n_iter, kappa, prevent_leaking : int/float/bool, optional
        Iterative hysteresis parameters; when `n_iter > 1`, Otsu-driven
        hysteresis with leak prevention (3D Lee skeleton re-threshold).
    do_tophat : bool, optional
        Apply white top-hat (arteries expected bright).
    tophat_size : int or None, optional
        Structural element radius; defaults to ~scale_max if None.
    overwrite : bool, optional
        If False (default), skip processing when the expected vesselness/skeleton/mask files already exist.

    Returns
    -------
    dict
        {"vesselness": Path, "skeleton": Path, "mask": Path} of written NIfTIs.

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/masks/
      - sub-<ID>[_ses-<LABEL>]_space-TOF_class-arteries_desc-vesselness_map.nii.gz
      - sub-<ID>[_ses-<LABEL>]_space-TOF_class-arteries_desc-skeleton_mask.nii.gz
      - sub-<ID>[_ses-<LABEL>]_space-TOF_class-arteries_desc-main_mask.nii.gz

    Assumptions / Preconditions
    ---------------------------
    - Processing in native voxel space; no resampling.
    - Vesselness saved float32; masks/skeletons saved uint8.

    Warnings
    --------
    - Excessive dilation may merge adjacent vessels undesirably.
    """
    print(f"[arteries] Input found → {tof_nii.name}")

    # Compute expected outputs early and skip if present
    vesselness_file, skeleton_file, mask_file = make_output_paths(
        tof_nii, out_dir, class_name="arteries", space_override="TOF"
    )
    if (not overwrite) and all(p.exists() for p in (vesselness_file, skeleton_file, mask_file)):
        print("[arteries] [SKIP] outputs exist (use overwrite=True to recompute)")
        return {"vesselness": vesselness_file, "skeleton": skeleton_file, "mask": mask_file}

    # Load image volume
    image, affine, header = load_nifti(tof_nii)

    # Segment arteries in native TOF space (bright ridges)
    vesselness, mask, skeleton = segment_vessels(
        image,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_step=scale_step,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=False,
        select_gamma_auto=select_gamma_auto,
        auto_gamma_fraction=auto_gamma_fraction,
        threshold_frac=threshold_frac,
        min_size=min_size,
        n_iter=n_iter,
        kappa=kappa,
        prevent_leaking=prevent_leaking,
        do_tophat=do_tophat,
        tophat_size=tophat_size,
        do_dilate=do_dilate,
        dilate_rad=dilate_rad,
    )

    # Save outputs (float32 for vesselness, uint8 for masks)
    save_nifti(vesselness, affine, header, vesselness_file)
    save_nifti(mask.astype(np.uint8), affine, header, mask_file)
    save_nifti(skeleton.astype(np.uint8), affine, header, skeleton_file)
    
    print("[arteries] DONE")
    return {
        "vesselness": vesselness_file,
        "skeleton": skeleton_file,
        "mask": mask_file,
    }


def veins_mrv(
    mrv_nii: Path,
    out_dir: Path,
    do_dilate: bool = False,
    dilate_rad: int = 1,
    *,
    # Fusion policy (prefers R2* → TE-weighted geometric mean → long-TE)
    mrv_fusion: str = "r2star",  # {"auto","r2star","geo_mean","longTE"}
    echo_times: list[float] | None = None, 
    # Vesselness (auto-γ supported inside frangi_vesselness)
    select_gamma_auto: bool = True,
    cortical_auto_gamma_fraction: float = 0.45,
    deep_auto_gamma_fraction: float = 0.25,    
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.02,
    # Dual-scale banks in mm (0.5 mm iso tuned) 
    cortical_sigmas_mm: tuple[float, ...] = (0.40, 0.50, 0.60), #PREVIOUS 0.35 0.50 0.60 0.70 0.80
    deep_sigmas_mm: tuple[float, ...] = (2.0,2.2, 2.4),
    bank_fusion: str = "max",  # "product" (stricter) | "max" (inclusive)
    # Hysteresis / cleanup
    threshold_frac: float = 0.2,
    min_size: int = 25,
    n_iter: int = 1,
    kappa: float = 0.1,
    prevent_leaking: bool = True,
    # Preprocessing
    do_tophat: bool = False,
    tophat_size: int | None = 0,  # black top-hat for dark-vein inputs
    r2_post_median=1, 
    overwrite: bool = False,
) -> dict:
    """
    Segment cerebral veins from MRV and write vesselness, mask, skeleton.

    Fusion policy
    -------------
    - If ≥2 echoes and {"auto","r2star"} → compute R2* (veins bright).
    - Else if ≥2 and {"auto","geo_mean"} → TE-weighted geometric mean
      (veins dark).
    - Else → single/long-TE magnitude (veins dark).

    Echo discovery
    --------------
    - 4D input: treat a small-sized axis (≤32) as echo axis (prefer last).
    - 3D input: search sibling files matching `...e1/e2/e3` next to `mrv_nii`.

    Parameters
    ----------
    mrv_nii : pathlib.Path
        Native-space MRV NIfTI path.
    out_dir : pathlib.Path
        Subject derivatives directory for outputs.
    do_dilate, dilate_rad : bool, int, optional
        Optional dilation parameters.
    mrv_fusion : {"auto","r2star","geo_mean","longTE"}, optional
        Strategy to build the input for vesselness.
    echo_times : list of float or None, optional
        Echo times (consistent units) if known; else inferred ordering 1..E.
    select_gamma_auto, cortical_auto_gamma_fraction, deep_auto_gamma_fraction : bool or float, optional
        Auto-γ configuration for Frangi (per-bank fractions).
    alpha, beta, gamma : float, optional
        Frangi parameters.
    cortical_sigmas_mm, deep_sigmas_mm : tuple of float, optional
        Sigma banks (mm) for cortical vs deep veins (converted to vox).
    bank_fusion : {"product","max"}, optional
        Fusion rule for per-bank vesselness.
    threshold_frac, min_size, n_iter, kappa, prevent_leaking : various, optional
        Thresholding and iterative hysteresis parameters.
    do_tophat, tophat_size : bool, int|None, optional
        Optional black/white top-hat before vesselness.
    r2_post_median : int, default 1
        Radius (voxels) for a single 3D median applied only to R2* fusion.    
    overwrite : bool, optional
        If False (default), skip processing when the expected vesselness/skeleton/mask files already exist.

    Returns
    -------
    dict
        {"vesselness": Path, "skeleton": Path, "mask": Path} of written NIfTIs.

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/masks/
      - sub-<ID>[_ses-<LABEL>]_space-MRV_class-veins_desc-vesselness_map.nii.gz
      - sub-<ID>[_ses-<LABEL>]_space-MRV_class-veins_desc-skeleton_mask.nii.gz
      - sub-<ID>[_ses-<LABEL>]_space-MRV_class-veins_desc-main_mask.nii.gz
      - sub-<ID>[_ses-<LABEL>]_space-MRV_class-veins_desc-prefrangi_image.nii.gz

    Assumptions / Preconditions
    ---------------------------
    - Processing in native voxel space; no resampling.
    - R2* computation requires ≥2 echoes; failures fall back gracefully.

    Warnings
    --------
    - "product" fusion is stricter and may suppress true positives if one bank
      is weak; "max" is more inclusive.
    """
    # -------------------------------------------------------------
    # Utilities (helpers local to this wrapper)
    # -------------------------------------------------------------
    print(f"[veins] Input found → {Path(mrv_nii).name if isinstance(mrv_nii, Path) else 'stack'}")

    vesselness_file, skeleton_file, mask_file = make_output_paths(
        mrv_nii, out_dir, class_name="veins", space_override="MRV")
    
    # Compute expected outputs early and skip if present
    if (not overwrite) and all(p.exists() for p in (vesselness_file, skeleton_file, mask_file)):
        print("[veins] [SKIP] outputs exist (use overwrite=True to recompute)")
        return {"vesselness": vesselness_file, "skeleton": skeleton_file, "mask": mask_file}

    echo_paths: list[Path] = []
    def _voxel_size_mm(aff: np.ndarray) -> float:
        vs = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))
        vs = float(np.mean(vs))
        return float(np.clip(vs, 0.2, 5.0))

    def _collect_echo_stack(p: Path) -> tuple[list[np.ndarray], list[float] | None, np.ndarray, any]:
        """Return ([echo_vols], tes_or_None, affine, header). Handles 4D or sibling '*e[1-9].nii*'."""
        img, aff, hdr = load_nifti(p)  # 3D or 4D
        if img.ndim == 4:
            # Prefer last axis as echo axis if plausible
            if img.shape[-1] <= 32:
                stack = [img[..., i] for i in range(img.shape[-1])]
            elif img.shape[0] <= 32:
                stack = [img[i, ...] for i in range(img.shape[0])]
            else:
                return [img], None, aff, hdr

            echo_paths[:] = [p] * len(stack)
            return stack, None, aff, hdr

        # Not 4D: sibling pattern "...e1/e2/e3"
        name = p.name
        # Strip suffixes (".nii.gz" or ".nii")
        if name.endswith(".nii.gz"):
            stem = name[:-7]
            suffix = ".nii.gz"
        elif name.endswith(".nii"):
            stem = name[:-4]
            suffix = ".nii"
        else:
            stem = p.stem
            suffix = "".join(p.suffixes) or ".nii.gz"

        m = re.search(r"^(.*?)(e[0-9]+)$", stem)
        base = m.group(1) if m else stem

        found = []
        found_paths = []
        for eidx in range(1, 10):
            cand = p.with_name(f"{base}e{eidx}{suffix}")
            if cand.exists():
                vol, _, _ = load_nifti(cand)
                found.append(vol)
                found_paths.append(cand)
        if len(found) >= 2:
            echo_paths[:] = found_paths
            # force later sidecar lookup by returning tes=None
            return found, None, aff, hdr
        else:
            return [img], None, aff, hdr

    def _normalize_01(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.float32)
        amin, amax = float(a.min()), float(a.max())
        if amax > amin:
            a = (a - amin) / (amax - amin)
        return np.clip(a, 0, 1)

    def _p99_norm_pos(v: np.ndarray) -> np.ndarray:
        pos = v[v > 0]
        if pos.size:
            p99 = float(np.percentile(pos, 99.0))
            if p99 > 0:
                v = np.clip(v / p99, 0, 1)
            else:
                m = float(v.max())
                if m > 0:
                    v = v / m
        return v
    
    def _sidecar_json_path(nifti_path):
        s = str(nifti_path)
        if s.endswith(".nii.gz"):
            return Path(s[:-7] + ".json")
        if s.endswith(".nii"):
            return Path(s[:-4] + ".json")
        return Path(s + ".json")

    def _echo_time_from_sidecar(nifti_path):
        jp = _sidecar_json_path(nifti_path)
        if not jp.exists():
            return None
        try:
            with open(jp, "r") as f:
                meta = json.load(f)
            et = meta.get("EchoTime", None)  # seconds
            return float(et) if et is not None else None
        except Exception:
            return None

    # -------------------------------------------------------------
    # Load echoes & fuse according to policy
    # -------------------------------------------------------------
    echo_vols, tes, affine, header = _collect_echo_stack(mrv_nii)
    nE = len(echo_vols)
    if tes is None:
        # Try sidecar JSONs first
        sidecar_tes = []
        all_paths_available = True
        for i, ev in enumerate(echo_vols):
            # prefer paths we captured in _collect_echo_stack
            p = echo_paths[i] if i < len(echo_paths) else None
            # fallback: if caller passed a list of echo file paths
            if p is None and isinstance(mrv_nii, (list, tuple)) and len(mrv_nii) == nE:
                p = mrv_nii[i]
            # last resort: an attached attribute on the array (unlikely)
            if p is None:
                p = getattr(ev, "path", None)
            if p is None:
                all_paths_available = False
                break
            et = _echo_time_from_sidecar(p)
            if et is None:
                all_paths_available = False
                break
            sidecar_tes.append(et)

        if all_paths_available and len(sidecar_tes) == nE:
            tes = sidecar_tes
        else:
            # final fallback
            tes = [8.0, 13.0, 21.0]


    # Prefer R2* if available/allowed
    bright_input = False
    fused = None
    if nE >= 2 and mrv_fusion in {"auto", "r2star"}:
        try:
            fused = compute_r2star_map(echo_vols, tes)
            fused = fused.astype(np.float32)
            if fused.max() > 0:
                fused /= fused.max()
            bright_input = True  # veins bright on R2*
            if r2_post_median and r2_post_median > 0:
                _size = 2 * int(r2_post_median) + 1
                fused = ndimage.median_filter(fused, size=_size).astype(np.float32, copy=False)
                # keep [0,1] semantics
                if fused.min() < 0.0 or fused.max() > 1.0:
                    np.clip(fused, 0.0, 1.0, out=fused)            
        except Exception:
            fused = None

    if fused is None and nE >= 2 and mrv_fusion in {"auto", "geo_mean"}:
        fused = combine_echoes_te_weighted(echo_vols, tes)
        fused = _normalize_01(fused)
        bright_input = False  # dark veins

    if fused is None:
        fused = echo_vols[-1].astype(np.float32)  # long-TE or only echo
        fused = _normalize_01(fused)
        bright_input = False  # dark veins

    fusion_tag = ("R2*" if (nE >= 2 and mrv_fusion in {"auto","r2star"} and bright_input) 
              else ("TE-weighted geo mean" if (nE >= 2 and mrv_fusion in {"auto","geo_mean"}) 
                    else "single/long-TE"))
    print(f"[veins] Fusion → {fusion_tag} (nE={nE}, bright_input={bright_input})")

    # -------------------------------------------------------------
    # Preprocessing (gentle MRV prep for vesselness)
    # -------------------------------------------------------------
    fused_proc = preprocess_mrv_for_vesselness(
        fused,
        pre_gaussian_sigma=0.3,  
        do_auto_window=True,
        auto_p_low=0.5,
        auto_p_high=94.0,
        post_gaussian_sigma=0.0,
        threshold_frac=0.2,          # ← threshold fractioning at 0.2
        remove_small_min_vox=40,     # ← min voxels
        remove_small_conn=2,         # ← connectivity

    )

    # -------------------------------------------------------------
    # Optional top-hat (black for dark veins → treat as bright later)
    # -------------------------------------------------------------
    if do_tophat:
        # Footprint radius (vox): 
        vs_mm = _voxel_size_mm(affine)
        deep_max_mm = float(max(deep_sigmas_mm)) if deep_sigmas_mm else 3.0
        rad_vox = int(tophat_size) if tophat_size is not None else int(deep_max_mm / vs_mm + 0.5)
        rad_vox = max(1, rad_vox)
        fp = ball(rad_vox) if fused.ndim == 3 else disk(rad_vox)
        if not bright_input:
            fused = black_tophat(fused_proc, footprint=fp)
            fused = _normalize_01(fused)
            bright_input = True  # flip to bright-ridge processing
        else:
            fused = white_tophat(fused_proc, footprint=fp)
            fused = _normalize_01(fused)

    # -------------------------------------------------------------
    # Dual-scale vesselness (cortical + deep banks) and fusion
    # -------------------------------------------------------------
    vs_mm = _voxel_size_mm(affine)
    cort_vox = tuple(max(s / vs_mm, 0.1) for s in cortical_sigmas_mm)
    deep_vox = tuple(max(s / vs_mm, 0.1) for s in deep_sigmas_mm)

    def _step(sigmas_vox: tuple[float, ...]) -> float:
        if len(sigmas_vox) <= 1:
            return 0.5
        diffs = np.diff(np.sort(sigmas_vox))
        return float(np.clip(diffs.min(), 0.3, 0.7))

    _, step_d = _step(cort_vox), _step(deep_vox)
    black_ridges = not bright_input

    # --- Dual-scale vesselness (cortical optional, deep required) ---

    pre_frangi_img = fused_proc.astype(np.float32)
    _, pre_frangi_img_deep,_ = intensity_gate(
    pre_frangi_img,                
    method="fraction_max",         
    fraction=0.85,                              
    remove_small_min_vox=150,       
    connectivity=2)

    # Always compute DEEP
    v_deep = frangi_vesselness(
        pre_frangi_img_deep,
        scale_min=min(deep_vox),
        scale_max=max(deep_vox),
        scale_step=step_d,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
        select_gamma_auto=select_gamma_auto,
        auto_gamma_fraction=deep_auto_gamma_fraction,  # keep your per-bank setting
    ).astype(np.float32)

    # Optionally compute CORTICAL (you can comment this whole block; fallback below works)
    v_cort = None
    # try:
    #     v_cort = frangi_vesselness(
    #         pre_frangi_img,
    #         scale_min=min(cort_vox),
    #         scale_max=max(cort_vox),
    #         scale_step=step_c,
    #         alpha=alpha,
    #         beta=beta,
    #         gamma=gamma,
    #         black_ridges=black_ridges,
    #         select_gamma_auto=select_gamma_auto,
    #         auto_gamma_fraction=cortical_auto_gamma_fraction,  # keep your per-bank setting
    #     ).astype(np.float32)
    # except Exception:
    #     # If cortical computation is removed or fails, we just skip it.
    #     v_cort = None

    # Normalize per-bank (only if present)
    v_deep = _p99_norm_pos(v_deep)
    if v_cort is not None:
        v_cort = _p99_norm_pos(v_cort)

    # Fuse: if cortical missing, use deep-only; else use selected fusion
    if v_cort is None:
        vesselness = v_deep
    else:
        if bank_fusion.lower() == "product":
            vesselness = v_cort * v_deep
        else:
            vesselness = np.maximum(v_cort, v_deep)

    # Final rescale
    if vesselness.max() > 0:
        vesselness = vesselness / vesselness.max()

    # -------------------------------------------------------------
    # Thresholding / hysteresis / cleanup
    # -------------------------------------------------------------
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

    # Optional dilation
    if do_dilate and mask.any():
        struct = ball(dilate_rad) if mask.ndim == 3 else disk(dilate_rad)
        mask = ndimage.binary_dilation(mask, structure=struct)

    # Skeleton (true 3D Lee for 3D input)
    skeleton = skeletonize(mask, method="lee")

    # -------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------
    pre_frangi_file = vesselness_file.with_name(
        vesselness_file.name.replace("desc-vesselness_map", "desc-prefrangi_image")
    )
    save_nifti(pre_frangi_img.astype(np.float32), affine, header, pre_frangi_file)
    save_nifti(vesselness.astype(np.float32), affine, header, vesselness_file)
    save_nifti(mask.astype(np.uint8), affine, header, mask_file)
    save_nifti(skeleton.astype(np.uint8), affine, header, skeleton_file)

    return {
        "vesselness": vesselness_file,
        "skeleton": skeleton_file,
        "mask": mask_file,
    }


def pvs_hT2w(
    hT2w_nii: Path,
    out_dir: Path,
    *,
    use_epc: bool = False,                 
    t1_path: Path | None = None,         
    t1_do_n4: bool = True,               
    t1_robust_pct: tuple[float,float] = (1.0, 99.0),
    write_preproc_artifacts: bool = True,  
    write_epc_artifacts: bool = True ,      
    overwrite: bool = False,
) -> dict:
    """
    Segment perivascular spaces (PVS) from heavy-T2w (hT2w) and write
    vesselness/skeleton/mask, with optional EPC driver built from T1.

    The default driver is a preprocessed hT2w volume (intensity windowing,
    gentle smoothing, coarse brain mask). If ``use_epc=True`` and ``t1_path``
    is provided, an EPC (Enhanced PVS Contrast) image is constructed by
    registering T1→hT2w (MI-based translation→rigid→affine), applying a light
    T1 preprocess (optional N4; robust percentile windowing), ratio/inversion,
    and using the EPC as the driver for small-scale 3D Frangi. Outputs are
    always written in **native hT2w space**.

    Parameters
    ----------
    hT2w_nii : pathlib.Path
        Native-space heavy-T2w NIfTI path (moving/driver is resampled nowhere;
        outputs preserve this grid/affine).
    out_dir : pathlib.Path
        Subject derivatives directory for outputs (typically `.../masks/`).
    use_epc : bool, optional
        If True, build an EPC driver from T1→hT2w and use it instead of the
        preprocessed hT2w image.
    t1_path : pathlib.Path or None, optional
        Path to subject T1 (raw or denoised). **Required** when
        ``use_epc=True``; otherwise ignored.
    t1_do_n4 : bool, optional
        Run a light N4 bias correction on T1 before forming EPC.
    t1_robust_pct : tuple of float, optional
        Percentile windowing for robust T1 intensity normalization (low, high).
    write_preproc_artifacts : bool, optional
        If True, write preprocessing artifacts for hT2w (preproc map, coarse
        brain mask) into `masks/`.
    write_epc_artifacts : bool, optional
        If True and EPC is used, write the EPC map and the T1→hT2w affine into
        an `epc/` sibling folder.
    overwrite : bool, optional
        If False (default), skip processing when the expected vesselness/skeleton/mask files already exist.

    Returns
    -------
    dict
        ``{"vesselness": Path, "skeleton": Path, "mask": Path}`` of written
        NIfTIs (all in **hT2w space**).

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/masks/
      - sub-<ID>[_ses-<LABEL>]_space-hT2w_class-pvs_desc-vesselness_map.nii.gz
      - sub-<ID>[_ses-<LABEL>]_space-hT2w_class-pvs_desc-skeleton_mask.nii.gz
      - sub-<ID>[_ses-<LABEL>]_space-hT2w_class-pvs_desc-main_mask.nii.gz
      - [optional, if ``write_preproc_artifacts``]
        - sub-<ID>[_ses-<LABEL>]_space-hT2w_class-pvs_desc-preproc_map.nii.gz
        - sub-<ID>[_ses-<LABEL>]_space-hT2w_class-brain_desc-coarsemask_mask.nii.gz
    - derivatives/neurofluid-mreg/sub-<ID>/epc/  (only if ``use_epc`` and
      ``write_epc_artifacts``)
      - sub-<ID>[_ses-<LABEL>]_space-hT2w_class-pvs_desc-epc_map.nii.gz
      - sub-<ID>[_ses-<LABEL>]_from-T1_to-hT2w_mode-affine_xfm.txt

    Assumptions / Preconditions
    ---------------------------
    - All operations occur in **native hT2w voxel space**; the EPC path performs
      a T1→hT2w registration internally but final segmentation outputs are on
      the hT2w grid.
    - When ``use_epc=True``, ``t1_path`` must refer to the *same subject* and
      roughly aligned anatomy; MI registration refines this.
    - Frangi is tuned for small PVS scales (σ ≈ 0.2–1.2 mm); hysteresis is off
      by default.

    Warnings
    --------
    - If ``use_epc=True`` but ``t1_path`` is missing/invalid, the function
      silently falls back to the hT2w driver.
    - EPC/nonlinear choices can alter contrast; downstream thresholds may need
      retuning.
    - Preprocessing may clip intensities outside robust percentile windows.

    Notes
    -----
    - Outputs: vesselness=float32; masks/skeleton=uint8 (binary).
    - PVS parameters and the EPC path are **provisional** and may be revised in
      future releases; keep `write_*_artifacts=True` for QC while iterating.
    """
    print(f"[pvs] Input found → {hT2w_nii.name}")
 
 
    vesselness_file, skeleton_file, mask_file = make_output_paths(
    hT2w_nii, out_dir, class_name="pvs", space_override="hT2w")

    # Compute expected outputs early and skip if present    
    if (not overwrite) and all(p.exists() for p in (vesselness_file, skeleton_file, mask_file)):
        print("[pvs] [SKIP] outputs exist (use overwrite=True to recompute)")
        return {"vesselness": vesselness_file, "skeleton": skeleton_file, "mask": mask_file}
 
    pre_img, pre_mask, affine, header = pvs_preprocess_hT2w(
        hT2w_nii,
        clip_low=1.0,
        clip_high=99.0,
        thr_low=0.15,
        thr_high=0.60,
        fill_holes_on=True,
        min_size_vox=30,
        max_size_vox=5000,
        write_artifacts=write_preproc_artifacts,
        out_dir=out_dir,
    )


    # --- choose driver image (default: preprocessed hT2w)
    driver_img = pre_img

    # If EPC requested and T1 provided, build EPC driver
    if use_epc:
        if t1_path is None or not Path(t1_path).exists():
            print("[pvs] [NOTE] use_epc=True but missing/invalid T1 → fallback to hT2w driver")
        else:
            # epc/ folder sibling to masks/
            epc_dir = Path(out_dir).parent / "epc"
            epc_dir.mkdir(parents=True, exist_ok=True)

            # BIDS-like stem
            txt = str(hT2w_nii)
            m_sub = re.search(r"(sub-[A-Za-z0-9]+)", txt)
            m_ses = re.search(r"(ses-[A-Za-z0-9]+)", txt)
            stem_parts = []
            if m_sub: stem_parts.append(m_sub.group(1))
            if m_ses: stem_parts.append(m_ses.group(1))
            stem = "_".join(stem_parts) + ("_" if stem_parts else "")

            epc_out = epc_dir / f"{stem}space-hT2w_class-pvs_desc-epc_map.nii.gz"
            xfm_out = epc_dir / f"{stem}from-T1_to-hT2w_mode-affine_xfm.txt"

            # Build EPC (register T1→hT2w inside; light T1 preprocess as needed)
            epc_img = epc_from_t1_and_hT2w(
                t1_path=Path(t1_path),
                hT2w_arr=pre_img,            # already [0,1]
                hT2w_aff=affine,
                out_fused_path=(epc_out if write_epc_artifacts else None),
                out_xfm_path=(xfm_out if write_epc_artifacts else None),
                do_n4=bool(t1_do_n4),        # this is ONLY for T1; fine to keep
                robust_pct=t1_robust_pct,
                reg_mode="rigid_affine",
                use_nonlin=False,
                brain_mask=None,             # << do NOT pass the mid-gray mask here
                invert_ratio=True,
            )

            driver_img = epc_img.astype(np.float32)
            print("[pvs] EPC driver enabled (T1→hT2w affine + EPC map written if requested)")

    # --- PVS-focused 3D Frangi ---
    vesselness, mask, skeleton = segment_pvs_frangi3d(
        driver_img,
        header=header,
        sigma_min_mm=0.9,
        sigma_max_mm=1.5,
        sigma_step_mm=0.2,
        alpha=0.5,
        beta=0.5,
        gamma=21.0,
        threshold_value=0.0,
        )

    # --- write exactly as before (BIDS-style paths)
    save_nifti(vesselness.astype(np.float32), affine, header, vesselness_file)
    save_nifti(mask.astype(np.uint8),        affine, header, mask_file)
    save_nifti(skeleton.astype(np.uint8),    affine, header, skeleton_file)

    return {
        "vesselness": vesselness_file,
        "skeleton": skeleton_file,
        "mask": mask_file,
    }

