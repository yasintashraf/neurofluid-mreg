# SPDX-License-Identifier: MIT
"""
run_pipeline.py
----------------
Pipeline orchestrator for Neurofluid–MREG (BIDS-first).

This module wires together native-space segmentation, MREG preprocessing,
core registrations, mask/radii warps, distance-map generation, and spectral
analysis. It standardizes folder layout and execution order; algorithmic
details are implemented in their respective modules.

Pipeline steps
--------------
1. Segmentation in native spaces (TOF/MRV/HT2w).
2. MREG preprocessing (realign, detrend) and temporal mean.
3. Core transforms (affine source→T1; optional SyN T1→MNI).
4. Warp masks (and radii) to the MREG grid (optional MNI exports).
5. Distance maps on the MREG grid.
6. Bandpower maps, clusters, spectra, and statistics/QC (MREG grid).
7. Continuous analyses (band power vs. distance, and radius vs. power).

Inputs / Outputs
----------------
Inputs  : `pipeline.yaml` via `PipelineConfig.from_yaml` (subject, paths, explicit
          filenames, bands, distance bins, flags).
Outputs : NIfTI/TSV/NPZ artifacts under
          `derivatives/neurofluid-mreg/sub-<ID>/...` according to each step.

Files written
-------------
- derivatives/neurofluid-mreg/sub-<ID>/masks/sub-<ID>_space-<SPACE>_class-<CLASS>_desc-<DESC>_<SUFFIX>.nii.gz
- derivatives/neurofluid-mreg/sub-<ID>/mreg/sub-<ID>_space-MREG_class-brain_desc-<DESC>_<SUFFIX>.nii.gz
- derivatives/neurofluid-mreg/sub-<ID>/distmaps/sub-<ID>_space-MREG_class-<CLASS>_desc-dist_map.nii.gz
- derivatives/neurofluid-mreg/sub-<ID>/bandmaps/... (bandpower maps)
- derivatives/neurofluid-mreg/sub-<ID>/clusters/... (distance-bin labels)
- derivatives/neurofluid-mreg/sub-<ID>/spectra/... (NPZ spectra)
- derivatives/neurofluid-mreg/sub-<ID>/stats/... (binned/continuous summaries)
- derivatives/neurofluid-mreg/sub-<ID>/figures/... (PNGs)

Assumptions / Preconditions
---------------------------
- Spaces: segmentation runs in native modality spaces; analysis runs on the
  native MREG grid after warping. Affines define geometry for resampling.
- Shapes/dtypes: image outputs are float32; label/mask outputs are uint8;
  spectra/stats use standard NumPy dtypes.
- BIDS naming: NIfTI outputs follow
  `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-<DESC>_<SUFFIX>.nii.gz`.

Warnings
--------
- TR is read from the MREG JSON sidecar by analysis routines (any `tr` argument
  is ignored and kept for compatibility).
- Optional inputs (MRV/HT2w) are conditional; missing data are skipped with
  warnings upstream.
- Most steps are idempotent and honor `overwrite`.

Public API
----------
- main
"""

from pathlib import Path

# -------------------------------------------------------------
# I/O helpers (BIDS naming, paths)
# -------------------------------------------------------------
from neurofluid_mreg.config import PipelineConfig
from neurofluid_mreg.io import (
    SubjectPaths,
    ensure_derivatives_layout,
    validate_required_inputs,
    deriv_name,
)

# -------------------------------------------------------------
# Masking (MREG/T1/MNI paths)
# -------------------------------------------------------------
from neurofluid_mreg.masking import mask_brain
from neurofluid_mreg.masking import compute_t1_mask_via_mni_and_project

# -------------------------------------------------------------
# Thresholding / post-processing / skeletonization (segmentation)
# -------------------------------------------------------------
from neurofluid_mreg.seg import arteries_tof, veins_mrv, pvs_ht2w

# -------------------------------------------------------------
# Radii estimation / fitting / QC
# -------------------------------------------------------------
from neurofluid_mreg.radii import (
    compute_radii_for_subject
    )

# -------------------------------------------------------------
# Transform bookkeeping (compose chains, inversion, etc.)
# -------------------------------------------------------------
from neurofluid_mreg.transforms import (
    TransformBook,
    run_mp2rage_denoise,
    warp_radii_to_mreg,
)

# -------------------------------------------------------------
# Preprocessing (MREG) and mask warps
# -------------------------------------------------------------
from neurofluid_mreg.mreg import (
    realign_and_detrend_mreg,
    compute_mreg_mean,
    warp_masks_single_shot,
    apply_mean_xfm_to_full_mreg,
)

# -------------------------------------------------------------
# Distance maps
# -------------------------------------------------------------
from neurofluid_mreg.distance import generate_distance_maps

# -------------------------------------------------------------
# Spectral analysis
# -------------------------------------------------------------
from neurofluid_mreg.analysis import (
    compute_bandpower_maps,
    make_distance_clusters,
    analyze_binned,
    cluster_spectra,
    frequency_map,
    analyze_continuous,
    analyze_radius_vs_power,
)

# -------------------------------------------------------------
# Utilities (defaults)
# -------------------------------------------------------------
from neurofluid_mreg.analysis import BANDS_DEFAULT, DIST_BINS_DEFAULT


def main():
    """
    Run the Neurofluid–MREG pipeline for a single subject (BIDS-first).

    Parameters
    ----------
    None

    Returns
    -------
    None

    Files written
    -------------
    - See module-level "Files written" section; artifacts are created in
      `derivatives/neurofluid-mreg/sub-<ID>/...` across `anat/`, `masks/`,
      `mreg/`, `bandmaps/`, `clusters/`, `spectra/`, `stats/`, `figures/`,
      `qc/`, and `manifest/`.

    Assumptions / Preconditions
    ---------------------------
    - `pipeline.yaml` is present in the CWD and contains explicit filenames.
    - Required inputs exist (T1w, TOF, MREG); validated by `validate_required_inputs`.
    - Optional steps (MRV/HT2w, radii, MNI) are executed only if configured and
      data are available.

    Warnings
    --------
    - `compute_bandpower_maps` reads TR from the MREG JSON sidecar regardless of
      the `tr` argument value (compatibility).
    """
    cfg = PipelineConfig.from_yaml(Path("pipeline.yaml"))
    sp = SubjectPaths(cfg)

    print(f"\n[PIPELINE] Subject: {sp.sub}")
    print(f"[PATHS] BIDS: {sp.bids_root}")
    print(f"[PATHS] Derivatives: {sp.deriv_root}\n")

    # I/O setup
    ensure_derivatives_layout(sp)
    validate_required_inputs(sp)

    if cfg.dry_run:
        print("[DRY-RUN] Stopping after I/O checks. Set dry_run:false to execute.")
        return

    # -------------------------
    # 1) SEGMENTATION (native)
    # -------------------------
    print("\n[STEP] Segmentation (native spaces)")
    if sp.anat_tof and Path(sp.anat_tof).exists():
        arteries_tof(Path(sp.anat_tof), Path(sp.masks_dir))
    else:
        print("[arteries] [SKIP] TOF not provided or missing")
    if sp.anat_mrv and Path(sp.anat_mrv).exists():
        veins_mrv(Path(sp.anat_mrv), Path(sp.masks_dir))
    else: 
        print("[veins] [SKIP] MRV not provided or missing")
    if sp.anat_heavy_t2w and Path(sp.anat_heavy_t2w).exists():
        pvs_ht2w(Path(sp.anat_heavy_t2w), Path(sp.masks_dir), use_epc=True, t1_path=None)
    else: 
        print("[pvs] [SKIP] HT2w/T2w not provided or missing")

    # 1b) Radii (native spaces) — toggled only by YAML: radii_enabled
    if getattr(cfg, "radii_enabled", False):
        print("\n[STEP] Centerline radii (native spaces)")
        ow = bool(getattr(cfg, "radii_overwrite", False))
        _r = compute_radii_for_subject(
            sp,
            classes=None,  # auto-detect available classes
            search_radius=2,
            overwrite=ow,
        )
        if not _r:
            print("[radii] Nothing computed (check images/masks/skeletons).")

    # -----------------------------------
    # 2) MREG PREPROC (realign + detrend)
    # -----------------------------------
    print("\n[STEP] MREG preprocessing (realign + detrend)")
    realign_and_detrend_mreg(sp, overwrite=False)

    # Resolve realigned 4D path (written by the step above)
    realigned_4d = Path(sp.mreg_dir) / deriv_name(
        sp.sub, "MREG", "brain", "motionrealigned", "bold"
    )

    # 2a) Mask from REALIGNED 4D (EPI mask) — final brain mask for downstream
    mreg_mask_epi = Path(sp.masks_dir) / deriv_name(
        sp.sub, "MREG", "brain", "brain", "mask"
    )
    mask_brain(realigned_4d, mreg_mask_epi, overwrite=False)
    mreg_brain_mask = mreg_mask_epi  # single source of truth for downstream steps
    print(f"[INFO] Using EPI mask (final): {mreg_brain_mask}")

    # 2b) Mean from REALIGNED 4D (for registration, QC)
    print("\n[STEP] Compute MREG mean (3D)")
    compute_mreg_mean(realigned_4d, sp, overwrite=False)

    # -----------------------------
    # 3) CORE TRANSFORMS (estimate)
    # -----------------------------
    print("\n[STEP] Estimate core transforms")
    xfm = TransformBook(sp)

    # Optional T1 denoise via MP2RAGE if you have INV1/INV2 and the helper
    t1_for_reg = Path(sp.anat_t1w)
    if run_mp2rage_denoise and sp.anat_inv1 and sp.anat_inv2:
        if Path(sp.anat_inv1).exists() and Path(sp.anat_inv2).exists():
            print("[INFO] Denoising T1 via MP2RAGE (INV1/INV2)")
            t1_for_reg = run_mp2rage_denoise(
                uni_path=Path(sp.anat_t1w),
                inv1_path=Path(sp.anat_inv1),
                inv2_path=Path(sp.anat_inv2),
                out_dir=Path(sp.anat_out),
                sub_id=sp.sub,
            )

    # Ensure core transforms exist and are saved
    mreg_mean_img = Path(sp.mreg_dir) / deriv_name(
        sp.sub, "MREG", "brain", "mean", "map"
    )
    xfm.estimate_and_save_core_transforms(
        t1_denoised_path=t1_for_reg,
        tof_path=Path(sp.anat_tof),
        mreg_mean_path=mreg_mean_img,
        mrv_path=(
            Path(sp.anat_mrv)
            if sp.anat_mrv and Path(sp.anat_mrv).exists()
            else None
        ),
        ht2w_path=(
            Path(sp.anat_heavy_t2w)
            if sp.anat_heavy_t2w and Path(sp.anat_heavy_t2w).exists()
            else None
        ),
        mni_path=None,
    )

    if getattr(cfg, "make_t1_bold_4d", False):
        apply_mean_xfm_to_full_mreg(
            sp,
            xfm,
            overwrite=False,
            also_mni=False,  # not implemented
        )

    print("\n[STEP] T1 masking via MNI → project to T1/MREG")
    compute_t1_mask_via_mni_and_project(
        sub=sp.sub,
        t1_path=Path(t1_for_reg),
        mreg_ref_path=Path(mreg_mean_img),
        xfm=xfm,
        masks_dir=Path(sp.masks_dir),
        overwrite=False,
    )

    # -----------------------------------------
    # 4) WARP MASKS → MREG (and optional → MNI)
    # -----------------------------------------
    print("\n[STEP] Warp masks → MREG/MNI")
    warp_masks_single_shot(
        sp, xfm, also_mni=cfg.make_mni, clip_with_mreg_mask=mreg_brain_mask
    )

    print("\n[STEP] Warp radii → MREG")
    warp_radii_to_mreg(
        sp,
        xfm,
        mreg_ref_path=mreg_mean_img,
        t1_ref_path=Path(t1_for_reg),
        overwrite=False,
    )

    # ----------------------------
    # 5) DISTANCE MAPS (MREG grid)
    # ----------------------------
    print("\n[STEP] Distance maps (MREG space)")
    Path(sp.distmaps_dir).mkdir(parents=True, exist_ok=True)
    generate_distance_maps(
        sp,
        classes=("arteries", "veins", "pvs"),
        mask_path=mreg_brain_mask,
        overwrite=True, xfm=xfm,)
    
    # dist_arteries = Path(sp.distmaps_dir) / deriv_name(
    #     sp.sub, "MREG", "arteries", "dist", "map"
    # )

    # ---------------------------------------------
    # 6) SPECTRAL BAND MAPS + CLUSTERS + STATS/QC
    # ---------------------------------------------
    print("\n[STEP] Spectral band maps")
    bands = cfg.bands or BANDS_DEFAULT
    band_paths = compute_bandpower_maps(
        sp, tr=None, bands=bands, mask_path=mreg_brain_mask, overwrite=True
    )

    print("\n[STEP] Distance clusters")
    cluster_paths = {}
    classes_to_cluster = ("arteries", "veins", "pvs")
    for klass in classes_to_cluster:
        dist_map = Path(sp.distmaps_dir) / deriv_name(sp.sub, "MREG", klass, "dist", "map")
        if dist_map.exists():
            out_path = make_distance_clusters(
                sp,
                dist_map_path=dist_map,
                klass=klass,
                bins=cfg.distance_bins or DIST_BINS_DEFAULT,
                overwrite=True,
            )
            cluster_paths[klass] = out_path
        else:
            print(f"[clusters] No distance map for {klass}; skipping.")

    print("\n[STEP] Binned stats + figures")
    for klass, labels_path in cluster_paths.items():
        analyze_binned(sp, labels_path, klass=klass, bands=bands, overwrite=True)

    print("\n[STEP] Cluster spectra + figures")
    for klass, labels_path in cluster_paths.items():
        cluster_spectra(sp, labels_path, klass=klass, max_hz=2.0, overwrite=True)

    # Optional: single-frequency map (e.g., cardiac 1.0 Hz)
    frequency_map(sp, freq_hz=1.0)

    # ---------------------------------------------
    # 7) CONTINUOUS: band power ~ distance (per class/band)
    # ---------------------------------------------
    print("\n[STEP] Continuous stats + figures (robust linear on log1p(power))")
    for klass in ("arteries", "veins", "pvs"):
        dist_map = Path(sp.distmaps_dir) / deriv_name(sp.sub, "MREG", klass, "dist", "map")
        if not dist_map.exists():
            print(f"[continuous] No distance map for {klass}; skipping.")
            continue
        analyze_continuous(
            sp,
            dist_map_path=dist_map,
            bands=bands,
            mask_path=mreg_brain_mask,
            overwrite=True,
        )

    print("\n[STEP] Radius vs Power")
    for klass in ("arteries", "veins", "pvs"):
        analyze_radius_vs_power(
            sp,
            klass,
            bands=bands,
            band_paths=band_paths,
            mask_path=mreg_brain_mask,
            overwrite=True,
        )

    print("\n[DONE] Pipeline finished.")


if __name__ == "__main__":
    main()
