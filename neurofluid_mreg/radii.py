# SPDX-License-Identifier: MIT
"""
radii.py
--------
Centerline-based vessel radius estimation for Neurofluid–MREG.

This module estimates vessel radii at skeleton voxels by fitting a 2D Gaussian
to cross-sectional intensity patches orthogonal to the local centerline
tangent. Radii are derived in voxel units and converted to millimeters using
image header zooms. Algorithmic behavior is unchanged; this file standardizes
docstrings, section banners, and BIDS-style I/O descriptions.

Pipeline steps
--------------
1. Build a centerline from a (binary or labeled) segmentation mask.
2. For each centerline voxel, estimate a local tangent via SVD on neighbors.
3. Sample an orthogonal 2D patch and fit a 2D Gaussian; convert FWHM → radius.
4. Write a radius map (values only on the skeleton) and an optional TSV with
   per-voxel fit statistics.

Inputs / Outputs
----------------
Inputs
    - Native-space anatomical image (TOF/MRV/hT2w) or a shared analysis grid
      (e.g., MNI) used for intensity sampling and voxel spacing.
    - Segmentation mask and optional skeleton mask on the same grid as the
      intensity image.
    - `SubjectPaths` for class- and space-specific input/output resolution.
Outputs
    - Float32 radius map (mm) with non-skeleton voxels = 0.0 and failed fits
      on skeleton voxels = -1.0.
    - Optional TSV with per-voxel radii and fit quality metrics.

Files written
-------------
- derivatives/neurofluid-mreg/sub-<ID>/radii/
  - sub-<ID>_space-<SPACE>_class-<CLASS>_desc-radius_map.nii.gz
  - sub-<ID>_space-<SPACE>_class-<CLASS>_desc-centerline_radii.tsv

Assumptions / Preconditions
---------------------------
- Spaces: By default operates in **native image space** per class
  (TOF/MRV/hT2w). `compute_radii_for_subject` can also compute radii on a
  shared grid such as MNI by setting `image_space`/`seg_space`.
- Affines: Intensity image and segmentation must share the same affine for a
  given run (enforced in `compute_centerline_radii`). If affines differ,
  resample upstream.
- Shapes/dtypes: 3D arrays; image data are loaded as float via nibabel
  `get_fdata`, masks are bool/int; output radius maps are float32.
- Voxel size: Derived from header zooms; this implementation uses the mean of
  the first three zooms and assumes they are expressed in millimeters.

Warnings
--------
- Skeletonization uses `skimage.morphology.skeletonize(method="lee")` and
  supports 3D when inputs are 3D; labeled masks are skeletonized per label.
- When radii are computed in MNI space, units still follow the MNI header
  zooms (assumed mm). Verify template units if non-standard references are
  used.
- The TSV writing function uses a minimal column schema; see
  `compute_centerline_radii` for details.

Public API
----------
- compute_centerline_radii
- compute_radii_for_subject
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Optional
import numpy as np
import nibabel as nib
import csv

from scipy.spatial import KDTree
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from skimage.morphology import skeletonize

from .io import SubjectPaths, deriv_name


# Fixed defaults (kept out of YAML on purpose)
_DEFAULT_CLASSES = ("arteries", "veins", "pvs")
_DEFAULT_SEARCH_RADIUS = 2  # voxels
_DEFAULT_OVERWRITE = False
_WRITE_TSV = True  # you asked to include TSV since it's not difficult
_NATIVE_SPACE = {"arteries": "TOF", "veins": "MRV", "pvs": "hT2W"}

# Map class -> (space token, SubjectPaths attribute for source image)
_CLASS_TO_SPACE_AND_IMG = {
    "arteries": ("TOF", "anat_tof"),
    "veins": ("MRV", "anat_mrv"),
    "pvs": ("hT2w", "anat_heavy_t2w"),
}


# -------------------------------------------------------------
# Radii estimation / fitting / QC
# -------------------------------------------------------------
def compute_centerline_radii(
    image_path: str | Path,
    seg_path: str | Path,
    output_path: str | Path,
    skeleton_path: str | Path | None = None,
    search_radius: int = _DEFAULT_SEARCH_RADIUS,
    overwrite: bool = _DEFAULT_OVERWRITE,
    out_tsv_path: Optional[str | Path] = None,
    write_tsv: bool = _WRITE_TSV,
    min_r2: Optional[float] = None,
) -> Path:
    """
    Estimate radii (mm) on vessel centerline voxels and write a radius map.

    For each skeleton voxel, a local tangent is estimated using an SVD on
    neighboring skeleton coordinates, a 2D orthogonal patch is sampled from
    the intensity image, and a 2D Gaussian is fitted. The geometric-mean
    FWHM is converted to radius in millimeters using the image voxel size.
    The output volume stores radii on skeleton voxels, 0.0 elsewhere, and
    -1.0 for failed fits at skeleton voxels.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Image used for intensity sampling and voxel spacing (e.g., native TOF,
        MRV, hT2w, or an MNI-space image). The first three header zooms are
        used to infer voxel size and are assumed to be in mm.
    seg_path : str or pathlib.Path
        Segmentation (binary or labeled) on the same grid as `image_path`.
        If `skeleton_path` is not provided, a skeleton is computed from this
        mask.
    output_path : str or pathlib.Path
        Target path for the radius map NIfTI (float32, mm units).
    skeleton_path : str or pathlib.Path or None, optional
        Precomputed skeleton mask on the same grid as `seg_path`. If None,
        a skeleton is computed: binary masks are skeletonized directly;
        labeled masks are skeletonized per label and OR-combined.
    search_radius : int, optional
        Half-size of the orthogonal patch in voxels. Also used as the
        neighborhood radius when estimating local tangents (with a minimum
        enforced at 3). Default is 2.
    overwrite : bool, optional
        If False and `output_path` exists (and TSV output is also satisfied),
        the computation is skipped. Default is False.
    out_tsv_path : str or pathlib.Path or None, optional
        If provided and `write_tsv=True`, per-voxel results are written to
        this TSV file with columns:
        `i, j, k, x_mm, y_mm, z_mm, radius_mm, fit_r2, fit_rmse`.
    write_tsv : bool, optional
        Enable/disable TSV writing. Default follows module constant
        `_WRITE_TSV`.
    min_r2 : float or None, optional
        Minimum acceptable coefficient of determination (`fit_r2`) for
        Gaussian fits. Fits with `r2 < min_r2` are treated as failures. If
        None (default), all fits passing geometric constraints are accepted.

    Returns
    -------
    pathlib.Path
        `output_path` of the written radius map (float32, mm).

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/radii/
      - sub-<ID>_space-<SPACE>_class-<CLASS>_desc-radius_map.nii.gz
      - sub-<ID>_space-<SPACE>_class-<CLASS>_desc-centerline_radii.tsv (optional)

    Assumptions / Preconditions
    ---------------------------
    - `image_path` and `seg_path` refer to 3D images on the same grid, with
      affines equal within 1e-5 (checked here). If not, resample upstream.
    - The voxel size is derived from `img_hdr.get_zooms()[:3]` and the mean
      value is used to convert voxel radii to millimeters. Units are assumed
      to be mm.

    Warnings
    --------
    - Skeletonization uses `skeletonize(..., method="lee")` and may be
      sensitive to segmentation topology.
    - The fitted radius in voxels is constrained to be `< search_radius`;
      larger fits are treated as failures to avoid extrapolation beyond the
      sampled patch.
    - TSV rows are written in append-like fashion (all in one call here),
      using the minimal schema described above.

    Raises
    ------
    RuntimeError
        If the image and segmentation affines differ, or if the skeleton is
        empty after skeletonization.
    """
    output_path = Path(output_path)
    tsv_exists = out_tsv_path and Path(out_tsv_path).exists()
    if output_path.exists() and (not overwrite) and (not write_tsv or tsv_exists):
        print(f"[radii] [SKIP] Exists: {output_path.name}")
        return output_path

    img_data, img_aff, img_hdr = _load_nii(image_path)
    seg_data, seg_aff, seg_hdr = _load_nii(seg_path)

    if not np.allclose(img_aff, seg_aff, atol=1e-5):
        raise RuntimeError(
            "[radii] image and seg affines differ; resample/alignment needed."
        )

    # Voxel size in mm (mean of spatial zooms).
    zooms_img = img_hdr.get_zooms()[:3]
    vxl_mm = float(np.mean(zooms_img))

    # Skeleton
    if skeleton_path is None:
        print("[radii] Skeletonizing segmentation (per-label if labeled)...")
        seg_int = np.asarray(seg_data, dtype=np.int32)
        skel = np.zeros_like(seg_int, dtype=bool)
        labels = np.unique(seg_int)
        if labels.size <= 2:  # binary mask
            skel |= skeletonize(seg_int > 0, method="lee")
        else:  # labeled vessels
            for lbl in labels:
                if lbl == 0:
                    continue
                skel |= skeletonize(seg_int == lbl, method="lee")
        skeleton_mask = skel
    else:
        skel_data, _, _ = _load_nii(skeleton_path)
        skeleton_mask = np.asarray(skel_data > 0, dtype=bool)

    coords = np.argwhere(skeleton_mask)
    if coords.size == 0:
        raise RuntimeError("[radii] Empty skeleton → nothing to compute.")

    out = np.zeros_like(img_data, dtype=np.float32)  # mm

    # TSV accumulators (only if requested)
    rows = [] if write_tsv else None
    affine = img_aff

    print(
        f"[radii] Fitting gaussians @ {coords.shape[0]} centerline voxels "
        f"(search_radius={search_radius})"
    )
    tree = KDTree(coords)
    for idx, ijk in enumerate(coords):
        c = ijk.astype(np.float32)
        # Local tangent + orthonormal axes from neighborhood SVD
        t, o1, o2 = _local_tangent_vector(
            c, coords, search_radius=max(3, search_radius), tree=tree
        )
        patch, gx, gy = _sample_cross_section(
            img_data, c, o1, o2, search_radius=search_radius
        )
        fit = _fit_gauss2d(patch, gx, gy)
        if fit and fit["radius_vox"] < (search_radius + 1e-6) and (
            (min_r2 is None) or (fit.get("r2", 1.0) >= min_r2)
        ):
            r_mm = fit["radius_vox"] * vxl_mm
            out[ijk[0], ijk[1], ijk[2]] = r_mm
            if rows is not None:
                # mm coordinates using affine
                xyz_mm = nib.affines.apply_affine(affine, ijk.astype(np.float64))
                rows.append(
                    [
                        int(ijk[0]),
                        int(ijk[1]),
                        int(ijk[2]),
                        float(xyz_mm[0]),
                        float(xyz_mm[1]),
                        float(xyz_mm[2]),
                        float(r_mm),
                        float(fit["r2"]),
                        float(fit["rmse"]),
                    ]
                )
        else:
            # Mark failed fit distinctly from background
            out[ijk[0], ijk[1], ijk[2]] = -1.0

    _save_nii(out, img_aff, img_hdr, output_path)

    # Write TSV if requested
    if rows is not None and out_tsv_path is not None:
        out_tsv_path = Path(out_tsv_path)
        out_tsv_path.parent.mkdir(parents=True, exist_ok=True)
        with out_tsv_path.open("w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(
                ["i", "j", "k", "x_mm", "y_mm", "z_mm", "radius_mm", "fit_r2", "fit_rmse"]
            )
            w.writerows(rows)
        print(f"[radii] Saved TSV → {out_tsv_path}")

    return output_path


def _local_tangent_vector(
    center_voxel: np.ndarray,
    skeleton_coords: np.ndarray,
    search_radius: int = 3,
    tree: Optional[KDTree] = None,
):
    """
    Compute local tangent and two orthonormal axes via neighborhood SVD.

    Parameters
    ----------
    center_voxel : ndarray, shape (3,), dtype=float32
        Centerline voxel coordinates (i, j, k) in voxel units.
    skeleton_coords : ndarray, shape (N, 3)
        All skeleton coordinates in voxel units.
    search_radius : int, optional
        Neighborhood radius in voxels for KD-tree query. Default is 3.
    tree : KDTree or None, optional
        Prebuilt KDTree over `skeleton_coords` for reuse. If None, a new
        KDTree is constructed internally.

    Returns
    -------
    tuple of ndarray
        `(t, o1, o2)` unit vectors (each shape (3,)). If fewer than 3
        neighbors are found, canonical axes are returned as a fallback.

    Notes
    -----
    - SVD is computed on mean-centered neighbor coordinates; the first
      right-singular vector is taken as the tangent direction.
    """
    if tree is None:
        tree = KDTree(skeleton_coords)
    idx = tree.query_ball_point(center_voxel, r=search_radius)
    neighbors = skeleton_coords[idx]
    if neighbors.shape[0] < 3:
        # Fallback basis
        return (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
    centered = neighbors - neighbors.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    t, o1, o2 = vh[0], vh[1], vh[2]
    # Normalize
    t = t / (np.linalg.norm(t) + 1e-8)
    o1 = o1 / (np.linalg.norm(o1) + 1e-8)
    o2 = o2 / (np.linalg.norm(o2) + 1e-8)
    return t, o1, o2


def _sample_cross_section(
    volume: np.ndarray,
    center: np.ndarray,
    ortho_v1: np.ndarray,
    ortho_v2: np.ndarray,
    search_radius: int = 7,
):
    """
    Sample a square patch in the plane spanned by `ortho_v1` and `ortho_v2`.

    Parameters
    ----------
    volume : ndarray, shape (X, Y, Z)
        Intensity volume. Interpolation order is 0 for boolean-like arrays,
        and 1 for all other dtypes.
    center : ndarray, shape (3,)
        Center voxel (i, j, k) in voxel coordinates (float).
    ortho_v1, ortho_v2 : ndarray, shape (3,)
        Orthonormal axes spanning the sampling plane in voxel units.
    search_radius : int, optional
        Half-size in voxels. The resulting patch has shape
        `(2*search_radius + 1, 2*search_radius + 1)`.

    Returns
    -------
    tuple
        `(patch, gx, gy)` where:
        - `patch` is a 2D array of sampled intensities with shape (H, W).
        - `gx`, `gy` are the integer coordinate grids used for fitting.
    """
    order = (
        0
        if (volume.dtype == np.bool_ or np.issubdtype(volume.dtype, np.bool_))
        else 1
    )
    grid = np.arange(-search_radius, search_radius + 1)
    gx, gy = np.meshgrid(grid, grid, indexing="xy")
    coords = center + gx[..., None] * ortho_v1 + gy[..., None] * ortho_v2
    coords = coords.reshape(-1, 3).T
    vals = map_coordinates(volume, coords, order=order, mode="constant", cval=0.0)
    return vals.reshape(gx.shape), gx, gy


def _gauss2d(coords, A, x0, y0, sx, sy, C):
    """
    Parametric 2D Gaussian used for cross-section fitting.

    Parameters
    ----------
    coords : tuple of ndarray
        `(x, y)` coordinate arrays (raveled grids).
    A : float
        Amplitude.
    x0, y0 : float
        Center coordinates.
    sx, sy : float
        Standard deviations along x and y.
    C : float
        Constant offset.

    Returns
    -------
    ndarray
        Flattened Gaussian values at the input coordinates.
    """
    x, y = coords
    return C + A * np.exp(
        -(((x - x0) ** 2) / (2 * sx**2) + ((y - y0) ** 2) / (2 * sy**2))
    )


def _fit_gauss2d(slice_2d: np.ndarray, gx: np.ndarray, gy: np.ndarray):
    """
    Fit a 2D Gaussian to a sampled cross-section.

    Parameters
    ----------
    slice_2d : ndarray
        Sampled intensity patch (2D).
    gx, gy : ndarray
        Integer coordinate grids corresponding to `slice_2d`.

    Returns
    -------
    dict or None
        On success, a dictionary with keys:
        `{"fwhm_x", "fwhm_y", "radius_vox", "r2", "rmse"}`.
        On failure, returns None.

    Notes
    -----
    - Initial parameters are derived from image statistics (max, mean).
    - The radius in voxel units is computed as
      `0.5 * sqrt(FWHM_x * FWHM_y)` (geometric-mean diameter / 2).
    - Fit quality is summarized via R² (`r2`) and RMSE (`rmse`).
    """
    x = gx.ravel()
    y = gy.ravel()
    z = slice_2d.ravel()
    if not np.isfinite(z).any() or z.max() <= 0:
        return None
    p0 = (float(z.max()), float(x.mean()), float(y.mean()), 1.0, 1.0, float(z.min()))
    bounds = (
        [0, -np.inf, -np.inf, 1e-3, 1e-3, -np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    )
    try:
        popt, _ = curve_fit(
            _gauss2d, (x, y), z, p0=p0, bounds=bounds, maxfev=5000
        )
        sx, sy = float(popt[3]), float(popt[4])
        fwhm_x = 2.355 * sx
        fwhm_y = 2.355 * sy
        radius_vox = 0.5 * np.sqrt(fwhm_x * fwhm_y)  # geometric mean diameter / 2
        # Fit quality
        zhat = _gauss2d((x, y), *popt)
        resid = z - zhat
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((z - np.mean(z)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        rmse = float(np.sqrt(ss_res / (z.size + 1e-12)))
        if not np.isfinite(radius_vox):
            return None
        return {
            "fwhm_x": fwhm_x,
            "fwhm_y": fwhm_y,
            "radius_vox": radius_vox,
            "r2": r2,
            "rmse": rmse,
        }
    except Exception:
        return None


def compute_radii_for_subject(
    sp: SubjectPaths,
    classes: Iterable[str] | None = None,
    search_radius: int = _DEFAULT_SEARCH_RADIUS,
    overwrite: bool = _DEFAULT_OVERWRITE,
    *,
    image_space: Optional[str] = None,
    seg_space: Optional[str] = None,
    image_override: Optional[Dict[str, Path]] = None,
    allow_on_the_fly_skeleton: bool = True,
) -> Dict[str, Path]:
    """
    Orchestrate vessel radii estimation for multiple classes and spaces.

    This function resolves the appropriate intensity image, segmentation, and
    skeleton for each vascular class and calls `compute_centerline_radii` on
    the chosen grid. By default, radii are computed in the native image
    space per class (TOF/MRV/hT2w). Optionally, a shared space such as MNI
    can be selected via `image_space`/`seg_space`.

    - When `image_space="MNI"` and `seg_space="MNI"`:
      * the segmentation comes from the MNI mask
        `sub-<ID>_space-MNI_class-<CLASS>_desc-main_mask.nii.gz`;
      * if no `image_override` is provided, this same mask is used as the
        intensity image for Gaussian fitting (mask-as-intensity).

    Parameters
    ----------
    sp : SubjectPaths
        Subject-scoped paths and identifiers (e.g., `sub`, `masks_dir`,
        `radii_dir`, and anatomical image attributes such as `anat_tof`).
    classes : iterable of str or None, optional
        Vascular classes to process (e.g., `("arteries", "veins", "pvs")`).
        If None, defaults to all known classes in `_DEFAULT_CLASSES`.
    search_radius : int, optional
        Half-size of the orthogonal patch in voxels used in
        `compute_centerline_radii`. Default is `_DEFAULT_SEARCH_RADIUS`.
    overwrite : bool, optional
        If True, recompute radius maps even when they already exist. Default
        is `_DEFAULT_OVERWRITE`.
    image_space : str or None, optional
        Target space token for the intensity image (e.g., `"TOF"`, `"MRV"`,
        `"hT2W"`, `"MNI"`). If None, the class-specific native space is used.
    seg_space : str or None, optional
        Target space token for segmentation/skeleton. If None, defaults to
        `image_space` (or the class-specific native space when `image_space`
        is None).
    image_override : dict or None, optional
        Optional mapping `klass -> Path` providing explicit images to use
        for intensity sampling when computing in a shared space (e.g., MNI).
        If provided, overrides the default choice for that class.
    allow_on_the_fly_skeleton : bool, optional
        If True, missing skeletons are computed from the segmentation; if
        False, classes without skeletons are skipped.

    Returns
    -------
    dict of str to pathlib.Path
        Mapping from each processed class to the corresponding radius-map
        NIfTI path. Classes that are skipped due to missing inputs or
        transforms will not appear in the dictionary.

    Files written
    -------------
    - derivatives/neurofluid-mreg/sub-<ID>/radii/
      sub-<ID>_space-<SPACE>_class-<CLASS>_desc-radius_map.nii.gz
      sub-<ID>_space-<SPACE>_class-<CLASS>_desc-centerline_radii.tsv

    Assumptions / Preconditions
    ---------------------------
    - `sp` exposes attributes for required anatomical images (e.g.,
      `anat_tof`, `anat_mrv`, `anat_heavy_t2w`) when computing in native
      spaces.
    - Segmentation and skeleton paths follow the `deriv_name` convention
      resolved by `_seg_and_skel_paths`.
    - When computing in MNI, the MNI masks and any override images use
      header zooms in mm so radii are expressed in mm.

    Warnings
    --------
    - If `image_space="MNI"` and `image_override` is not provided, the MNI
      segmentation mask is used as the intensity image for fitting, which
      may be suboptimal compared to using a continuous-valued anatomical
      image resampled to MNI.
    - Class names not present in `_CLASS_TO_SPACE_AND_IMG` are skipped with
      a warning and no radii maps are produced.
    """
    if classes is None:
        classes = _DEFAULT_CLASSES

    image_override = image_override or {}
    results: Dict[str, Path] = {}

    for klass in classes:
        if klass not in _CLASS_TO_SPACE_AND_IMG:
            print(f"[radii] [WARN] Unknown class: {klass}")
            continue

        # Class-specific native space and SubjectPaths attribute
        default_native_space, img_attr = _CLASS_TO_SPACE_AND_IMG[klass]

        # Grid on which radii will be computed for this class
        op_space = (image_space or default_native_space).upper()
        seg_op_space = (seg_space or op_space).upper()

        # --- resolve segmentation + skeleton in seg_op_space ---
        seg_path, skel_path = _seg_and_skel_paths(sp, seg_op_space, klass)
        if not seg_path.exists():
            print(
                f"[radii] [SKIP] Missing segmentation for {klass}@{seg_op_space}: "
                f"{seg_path.name}"
            )
            continue

        if skel_path.exists():
            skel_for_compute: Optional[Path] = skel_path
        else:
            if allow_on_the_fly_skeleton:
                skel_for_compute = None
                print(
                    f"[radii] {klass}@{seg_op_space}: skeleton not found → "
                    "will skeletonize from seg."
                )
            else:
                print(
                    f"[radii] [SKIP] Missing skeleton for {klass}@{seg_op_space}: "
                    f"{skel_path.name}"
                )
                continue

        # --- resolve intensity image in op_space ---
        if op_space == "MNI":
            # explicit override wins
            img_path = image_override.get(klass)
            if img_path is None:
                # default: use the MNI segmentation mask as intensity
                img_path = seg_path
                print(
                    f"[radii] {klass}@MNI: using MNI main_mask as intensity image "
                    f"for radii fitting."
                )
        else:
            # native spaces: TOF/MRV/hT2w from SubjectPaths
            img_path = getattr(sp, img_attr, None)

        if not img_path or not Path(img_path).exists():
            print(
                f"[radii] [SKIP] Missing source image for {klass}@{op_space}: "
                f"{img_path}"
            )
            continue

        # --- outputs reflect the grid we compute on (op_space) ---
        out_nii, out_tsv = _radii_out_paths(sp, op_space, klass)

        print(f"[radii] {klass} | space={op_space}")
        print(f"        img:  {img_path}")
        print(f"        seg:  {seg_path.name}")
        print(
            f"        skel: {skel_path.name if skel_for_compute is not None else '(on-the-fly)'}"
        )
        print(f"        out:  {out_nii.name}")
        if _WRITE_TSV:
            print(f"        tsv:  {out_tsv.name}")

        # --- compute radii map (and TSV if enabled) ---
        compute_centerline_radii(
            image_path=img_path,
            seg_path=seg_path,
            output_path=out_nii,
            skeleton_path=skel_for_compute,
            search_radius=search_radius,
            overwrite=overwrite,
            out_tsv_path=(out_tsv if _WRITE_TSV else None),
            write_tsv=_WRITE_TSV,
        )

        results[klass] = out_nii

    return results


# -------------------------------------------------------------
# I/O helpers (BIDS naming, paths)
# -------------------------------------------------------------
def _load_nii(path: str | Path):
    """
    Load a NIfTI and return data, affine, and header.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to `.nii`/`.nii.gz`.

    Returns
    -------
    tuple
        `(data, affine, header)` where:
        - `data` is a float array as returned by `img.get_fdata()` (i.e.,
          with any scale factors applied by nibabel).
        - `affine` is the 4x4 image affine.
        - `header` is the corresponding nibabel header.

    Files written
    -------------
    - None.

    Warnings
    --------
    - Because `get_fdata()` is used, the returned `data` is a floating-point
      representation with scaling applied; original on-disk dtypes are not
      preserved. If native integer dtypes are required, this helper would
      need to be adapted.
    """
    img = nib.load(str(path))
    data = img.get_fdata()
    return data, img.affine, img.header


def _save_nii(data: np.ndarray, affine, header, out_path: str | Path):
    """
    Save a float32 NIfTI volume at the provided location.

    Parameters
    ----------
    data : ndarray
        Volume to save; cast to float32 before writing.
    affine : ndarray, shape (4, 4)
        Image affine.
    header : nibabel header
        Header propagated to the output image (dtype is effectively
        overridden to float32).
    out_path : str or pathlib.Path
        Destination path. Parent directories are created if missing.

    Returns
    -------
    None

    Files written
    -------------
    - A NIfTI file at `out_path` with data stored as float32.

    Warnings
    --------
    - Any original dtype information in `header` is overridden by the
      float32 cast.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(data.astype(np.float32), affine, header)
    nib.save(out_img, str(out_path))
    print(f"[radii] Saved → {out_path}")


def _tsv_name(sub: str, space: str, klass: str) -> str:
    """
    Construct a standardized TSV filename for per-voxel radii.

    Parameters
    ----------
    sub : str
        Subject token used in filenames (typically the subject label, with
        or without the `sub-` prefix depending on `SubjectPaths`).
    space : str
        Space token (e.g., `TOF`, `MRV`, `hT2W`, `MNI`).
    klass : str
        Structure class (`arteries`, `veins`, `pvs`).

    Returns
    -------
    str
        `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-centerline_radii.tsv`
        (actual subject token prefixing is determined by the caller).
    """
    return f"{sub}_space-{space}_class-{klass}_desc-centerline_radii.tsv"


def _seg_and_skel_paths(sp: SubjectPaths, space: str, klass: str) -> tuple[Path, Path]:
    """
    Build paths to the segmentation and skeleton masks for a given class/space.

    Parameters
    ----------
    sp : SubjectPaths
        Subject-scoped directories and identifiers.
    space : str
        Space token (e.g., `TOF`, `MRV`, `hT2W`, `MNI`).
    klass : str
        Structure class (`arteries`, `veins`, `pvs`).

    Returns
    -------
    tuple of pathlib.Path
        `(seg_path, skel_path)` under `sp.masks_dir`.
    """
    seg_name = deriv_name(sp.sub, space, klass, "main", "mask")
    skel_name = deriv_name(sp.sub, space, klass, "skeleton", "mask")
    return (Path(sp.masks_dir) / seg_name, Path(sp.masks_dir) / skel_name)


def _radii_out_paths(sp: SubjectPaths, space: str, klass: str) -> tuple[Path, Path]:
    """
    Build output paths for the radii map and TSV for a class/space.

    Parameters
    ----------
    sp : SubjectPaths
        Subject-scoped directories and identifiers.
    space : str
        Space token (e.g., `TOF`, `MRV`, `hT2W`, `MNI`).
    klass : str
        Structure class (`arteries`, `veins`, `pvs`).

    Returns
    -------
    tuple of pathlib.Path
        `(nii_out, tsv_out)` under `sp.radii_dir`.
    """
    nii_name = deriv_name(sp.sub, space, klass, "radius", "map")
    tsv_name = _tsv_name(sp.sub, space, klass)
    return Path(sp.radii_dir) / nii_name, Path(sp.radii_dir) / tsv_name


# -------------------------------------------------------------
# Utilities (logging, checks)
# -------------------------------------------------------------
