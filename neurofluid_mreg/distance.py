# SPDX-License-Identifier: MIT
"""
distance.py
-----------
Distance-map generation in MNI space for downstream distance-clustered
spectral analysis.

This module standardizes computation of Euclidean distance transforms (EDT)
in millimeters from class masks (e.g., arteries, veins, perivascular spaces)
after warping them into a common MNI reference space via `TransformBook`.
Masks are warped with nearest-neighbor interpolation (labels preserved),
and the EDT is computed on the MNI grid using voxel sizes from the MNI
header. Outputs are float32 NIfTI files in MNI space used by cluster and
continuous proximity analyses.

Pipeline steps
--------------
1. Resolve native-space masks per class via `SubjectPaths` and `deriv_name`.
2. Warp each native mask to MNI with nearest-neighbor interpolation
   (chain: native space → T1 → MNI), and save the warped MNI mask for
   QC/reuse.
3. Compute the Euclidean distance transform (EDT) in MNI space using the
   first three header zooms (assumed to be in mm).
4. Save distance maps as float32 NIfTI images in the subject-level
   `distmaps/` directory in MNI space.

Inputs / Outputs
----------------
Inputs
    - Native-space binary masks for each class (arteries/veins/pvs) under
      `derivatives/neurofluid-mreg/sub-<ID>/masks/`.
    - A `TransformBook` instance providing native→T1 and T1→MNI transforms
      and an MNI reference image.
    - A `SubjectPaths` instance for naming and directory conventions.
Outputs
    - Per-class distance maps on the MNI grid as float32 NIfTI images.

Files written
-------------
- derivatives/neurofluid-mreg/sub-<ID>/distmaps/
  sub-<ID>_space-MNI_class-<CLASS>_desc-dist_map.nii.gz

Assumptions / Preconditions
---------------------------
- Spaces: Operates in **MNI** space. Native-space masks (TOF/MRV/hT2w) are
  warped to MNI before computing the EDT. No distance maps are computed
  directly on the MREG grid in this module.
- Affines: The MNI reference affine encodes voxel sizes (assumed mm). EDT
  uses the first three header zooms; downstream interpretation assumes,
  but does not enforce, millimeter units.
- Shapes/dtypes: Output images have shape equal to the MNI reference, with
  dtype float32. Warped masks are stored as uint8.
- BIDS naming: Distance maps use the pattern
  `sub-<ID>_space-MNI_class-<CLASS>_desc-dist_map.nii.gz`.

Warnings
--------
- If a required transform (e.g., `mrv_to_t1`) is missing, the corresponding
  class distance map is skipped and `None` may be returned for that class.
- Units of the distance map are inferred from header zooms and treated as
  millimeters; verify that the MNI reference uses mm (standard in NIfTI)
  when interpreting slopes or effect sizes.
- This module does not currently support an external brain mask; all
  voxels in the MNI field of view are included in the EDT.

Public API
----------
- distance_map_native_to_mni
- generate_distance_maps_mni
"""

from __future__ import annotations
from pathlib import Path
import nibabel as nib, numpy as np
from scipy.ndimage import distance_transform_edt
from .io import deriv_name
from .transforms import TransformBook

# Which native space + transform key to use per class
_CLASS_TO_NATIVE = {
    "arteries": ("TOF", "tof_to_t1"),
    "veins": ("MRV", "mrv_to_t1"),
    "pvs": ("hT2w", "hT2w_to_t1"),
}

# -------------------------------------------------------------
# Registration / native-space transforms
# -------------------------------------------------------------


def distance_map_native_to_mni(
    *,
    sp,
    xfm: TransformBook,
    klass: str,
    native_mask_path: Path,
    out_mni_path: Path,
    overwrite: bool = False,
    mask_threshold: float = 0.5,
) -> Path | None:
    """
    Compute the Euclidean distance transform (mm) in MNI space from a native mask.

    The native binary mask is first warped to MNI space using nearest-neighbor
    interpolation via `TransformBook.warp_labels`, then hard-binarized, and
    finally an EDT is computed on the MNI grid using voxel sizes from the MNI
    header. The result is saved as a float32 NIfTI file with the MNI affine.

    Parameters
    ----------
    sp : SubjectPaths
        Subject-specific paths and derivative directories, used to locate and
        name the MNI-space mask.
    xfm : TransformBook
        Transform registry providing native→T1 and T1→MNI chains. Must expose
        `mni_ref_path` and support `warp_labels(...)` with a chain that maps
        from the class native space to MNI.
    klass : {'arteries', 'veins', 'pvs'}
        Class label, used to look up the native-space key and to construct
        derivative filenames.
    native_mask_path : Path
        Path to the native-space binary mask NIfTI image for the class. Values
        are interpreted as foreground if greater than `mask_threshold`.
    out_mni_path : Path
        Destination path for the MNI-space float32 distance map NIfTI image.
    overwrite : bool, default=False
        If False and `out_mni_path` already exists, computation is skipped and
        the existing path is returned.
    mask_threshold : float, default=0.5
        Threshold applied after warping to MNI to re-binarize the mask. Values
        strictly greater than this threshold are treated as foreground.

    Returns
    -------
    Path or None
        `out_mni_path` on success. Returns None if a required transform is
        missing (e.g., missing affine for the given native space), in which
        case the corresponding class is skipped.

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/masks/
      sub-<ID>_space-MNI_class-<CLASS>_desc-main_mask.nii.gz
      (warped, binarized MNI mask for QC/reuse)
    - derivatives/neurofluid-mreg/sub-<ID>/distmaps/
      sub-<ID>_space-MNI_class-<CLASS>_desc-dist_map.nii.gz
      (float32 Euclidean distance map in mm)

    Assumptions / Preconditions
    ---------------------------
    - `xfm.mni_ref_path` points to a valid MNI reference NIfTI image whose
      header zooms encode voxel sizes in mm.
    - The native mask is registered to the native space implied by
      `_CLASS_TO_NATIVE[klass]`.
    - `native_mask_path` exists and is readable as a NIfTI image.

    Warnings
    --------
    - Units inferred from the MNI header zooms are assumed to be millimeters;
      verify mm vs voxel indices if using non-standard references.
    - If a "Missing affine" error arises inside `warp_labels`, the function
      prints a skip message and returns None; callers should handle this
      possibility when aggregating results.

    Raises
    ------
    RuntimeError
        If `xfm.mni_ref_path` is not set or if `warp_labels` fails for reasons
        other than missing affine transforms.
    """
    native_space, _ = _CLASS_TO_NATIVE[klass]

    out_mni_path.parent.mkdir(parents=True, exist_ok=True)

    # --- where the MNI mask belongs (derivatives/masks) ---
    mni_mask_path = Path(sp.masks_dir) / deriv_name(
        sp.sub, "MNI", klass, "main", "mask"
    )

    # Resolve MNI reference for warp_labels
    if getattr(xfm, "mni_ref_path", None):
        mni_ref_path = Path(xfm.mni_ref_path)
        _ = nib.load(str(mni_ref_path))  # ensure readable
    else:
        raise RuntimeError(
            "[dist→mni] xfm.mni_ref_path is required for label warps."
        )

    # 1) Warp native mask → MNI (NEAREST) and SAVE for QC/reuse
    if (not mni_mask_path.exists()) or overwrite:
        try:
            xfm.warp_labels(
                moving_img=str(native_mask_path),
                reference_img=str(mni_ref_path),
                out_path=str(mni_mask_path),
                chain=(native_space, "MNI"),
                interpolation="nearest",
            )
        except RuntimeError as e:
            # e.g. Missing affine 'mrv_to_t1' / 'ht2w_to_t1' when MRV/hT2w not present
            if "Missing affine" in str(e):
                print(
                    f"[dist→mni] [SKIP] No transform for {klass} "
                    f"({native_space}→T1); skipping distance map."
                )
                return None
            raise
        # hard-binarize to {0,1}
        _img = nib.load(str(mni_mask_path))
        _dat = (_img.get_fdata() > float(mask_threshold)).astype(np.uint8)
        nib.save(
            nib.Nifti1Image(_dat, _img.affine, _img.header), str(mni_mask_path)
        )
        print(f"[dist→mni] Saved warped {klass} mask in MNI → {mni_mask_path}")
    else:
        print(
            f"[dist→mni] [SKIP] MNI {klass} mask exists: {mni_mask_path.name}"
        )

    # 2) If distance map already exists and not overwriting, stop early
    if out_mni_path.exists() and not overwrite:
        print(f"[dist→mni] [SKIP] Exists: {out_mni_path.name}")
        return out_mni_path

    # 3) EDT on the MNI grid (use MNI voxel sizes)
    mni_mask_img = nib.load(str(mni_mask_path))
    mni_mask = mni_mask_img.get_fdata() > float(mask_threshold)
    zooms = mni_mask_img.header.get_zooms()[:3]
    dist_mni = distance_transform_edt(~mni_mask, sampling=zooms).astype(
        np.float32
    )

    hdr = mni_mask_img.header.copy()
    hdr.set_data_dtype(np.float32)
    nib.save(
        nib.Nifti1Image(dist_mni, mni_mask_img.affine, hdr), str(out_mni_path)
    )
    print(f"[dist→mni] Saved → {out_mni_path}")

    return out_mni_path


# -------------------------------------------------------------
# Distance-map computation and writing (MNI grid)
# -------------------------------------------------------------


def generate_distance_maps_mni(
    sp,
    xfm: TransformBook,
    classes: tuple[str, ...] = ("arteries", "veins", "pvs"),
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Generate MNI-space distance maps for each vascular class.

    For each requested class, this function locates the corresponding native
    mask, warps it to MNI via `distance_map_native_to_mni`, and writes a
    float32 EDT NIfTI distance map under the subject's `distmaps/` directory.
    Existing distance maps are reused unless `overwrite` is True.

    Parameters
    ----------
    sp : SubjectPaths
        Subject-specific paths and derivatives root used to find native masks
        and to construct output paths in `distmaps/`.
    xfm : TransformBook
        Transform registry providing native→T1 and T1→MNI chains, and an MNI
        reference image for warping and EDT computation.
    classes : tuple of {'arteries', 'veins', 'pvs'}, default=('arteries', 'veins', 'pvs')
        Sequence of class labels to process. Unknown labels are skipped with
        a log message.
    overwrite : bool, default=False
        If True, any existing MNI distance maps are recomputed; otherwise,
        the function reuses existing files and skips computation.

    Returns
    -------
    dict of str to Path or None
        Dictionary mapping each processed class name to the corresponding
        MNI distance-map path. If a class is skipped due to missing masks
        or transforms, the value may be None or the key may be absent.

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/distmaps/
      sub-<ID>_space-MNI_class-<CLASS>_desc-dist_map.nii.gz

    Assumptions / Preconditions
    ---------------------------
    - Native-space masks exist under `sp.masks_dir` with names constructed
      via `deriv_name(sp.sub, native_space, klass, "main", "mask")`.
    - Transform chains required by each class (e.g., TOF→T1→MNI) are present
      in `xfm`; missing transforms cause that class to be skipped.
    - The MNI reference in `xfm.mni_ref_path` has zooms in mm, used to define
      the distance units.

    Warnings
    --------
    - Units are assumed to be millimeters derived from the MNI header zooms;
    - If no native masks are found or all distance maps are already up to
      date (and `overwrite` is False), the returned dictionary may be empty
      and a summary message is printed.
    """
    out: dict[str, Path] = {}
    masks_dir = Path(sp.masks_dir)
    dist_dir = Path(sp.distmaps_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)

    for klass in classes:
        native_space, _ = _CLASS_TO_NATIVE.get(klass, (None, None))
        if native_space is None:
            print(f"[dist→mni] [SKIP] Unknown class: {klass}")
            continue

        native_mask = masks_dir / deriv_name(
            sp.sub, native_space, klass, "main", "mask"
        )
        if not native_mask.exists():
            print(
                f"[dist→mni] [SKIP] No native mask for {klass}: "
                f"{native_mask.name}"
            )
            continue

        out_mni = dist_dir / deriv_name(
            sp.sub, "MNI", klass, "dist", "map"
        )
        if out_mni.exists() and not overwrite:
            print(f"[dist→mni] [SKIP] Exists: {out_mni.name}")
            out[klass] = out_mni
            continue

        out[klass] = distance_map_native_to_mni(
            sp=sp,
            xfm=xfm,
            klass=klass,
            native_mask_path=native_mask,
            out_mni_path=out_mni,
        )

    if not out:
        print("[dist→mni] Nothing written (no native masks or all up-to-date).")
    return out
