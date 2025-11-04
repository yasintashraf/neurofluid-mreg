# SPDX-License-Identifier: MIT
"""
radii.py
--------
Estimate vessel radii at centerline voxels in **native space** (TOF/MRV/hT2w).
This module fits a small 2D Gaussian to cross-sectional patches orthogonal to
the local centerline tangent and reports radii in millimeters. Algorithmic
behavior is unchanged; this file standardizes docstrings, banners, and I/O
descriptions (BIDS-first).

Pipeline steps
--------------
1. Build centerline from a (binary or labeled) segmentation mask.
2. For each centerline voxel, estimate a local tangent via SVD on neighbors.
3. Sample an orthogonal 2D patch and fit a Gaussian; convert FWHM → radius.
4. Write a radius map (values only on skeleton) and an optional TSV.

Inputs / Outputs
----------------
Inputs  : Native-space anatomical image (TOF/MRV/hT2w), segmentation, and
          a centerline skeleton (or skeletonized on-the-fly).
Outputs : A float32 radius map (mm) with non-skeleton voxels = 0.0 and failed
          fits on skeleton voxels = -1.0; optional per-voxel TSV.

Files written
-------------
- derivatives/neurofluid-mreg/sub-<ID>/radii/
  - sub-<ID>_space-<SPACE>_class-<CLASS>_desc-radius_map.nii.gz
  - sub-<ID>_space-<SPACE>_class-<CLASS>_desc-centerline_radii.tsv

Assumptions / Preconditions
---------------------------
- Spaces: Operates in **native image space**; image and segmentation affines
  must match exactly (enforced). If affines differ, resample upstream.
- Shapes/dtypes: 3D arrays; image float (any); masks bool/int; output float32.
- Voxel size: Derived from header zooms; uses mean of spatial zooms (mm).

Warnings
--------
- Skeletonization uses `skimage.morphology.skeletonize(method="lee")` and
  supports 3D when inputs are 3D; labeled masks are skeletonized per-label.
- TSV writing follows a minimal schema; see notes in `compute_centerline_radii`.

Public API
----------
- compute_centerline_radii
- compute_radii_for_subject
- has_native_radii
- has_mreg_radii
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
    "pvs": ("hT2W", "anat_heavy_t2w"),
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

    For each skeleton voxel, a local tangent is estimated (SVD over neighbors),
    a 2D orthogonal patch is sampled from the **image**, and a 2D Gaussian is
    fitted. The geometric-mean FWHM is converted to radius in millimeters. The
    output volume stores radii on skeleton voxels, 0.0 elsewhere, and -1.0 for
    failed fits at skeleton voxels.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Native-space anatomical image (e.g., TOF, MRV, hT2w). Used for intensity
        sampling and voxel spacing (mm).
    seg_path : str or pathlib.Path
        Native-space segmentation (binary or labeled). If `skeleton_path` is
        not provided, a skeleton is computed from this mask.
    output_path : str or pathlib.Path
        Target path for the radius map (float32, mm).
    skeleton_path : str or pathlib.Path or None, optional
        Precomputed skeleton (same grid as `seg_path`). If None, a skeleton is
        computed: binary masks are skeletonized directly; labeled masks are
        skeletonized per-label and OR-combined.
    search_radius : int, optional
        Half-size of the orthogonal patch (voxels). Also used for the neighbor
        radius when estimating local tangents (min enforced at 3).
    overwrite : bool, optional
        If False and the output exists (and TSV is also satisfied), skip work.
    out_tsv_path : str or pathlib.Path or None, optional
        If provided and `write_tsv=True`, per-voxel results are written here as
        a TSV (`i,j,k,x_mm,y_mm,z_mm,radius_mm,fit_r2,fit_rmse`).
    write_tsv : bool, optional
        Enable/disable TSV writing (default: module constant `_WRITE_TSV`).
    min_r2 : float or None, optional
        Accept only Gaussian fits with coefficient of determination ≥ `min_r2`.
        If None, accept all fits.

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
    - `image_path` and `seg_path` are on the **same grid**; their affines must
      be equal within 1e-5 (checked). If not, resample upstream.
    - The volume is 3D and voxel spacings are read from the image header.

    Warnings
    --------
    - Skeletonization uses `skeletonize(..., method="lee")`.
    - Fitting radius is constrained to be `< search_radius`; larger fits are
      treated as failures to avoid out-of-patch extrapolation.
    - The TSV writer assumes rows are appended as sequences.

    Raises
    ------
    RuntimeError
        If image and segmentation affines differ, or if the skeleton is empty.
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
                rows.append([
                    int(ijk[0]), int(ijk[1]), int(ijk[2]),
                    float(xyz_mm[0]), float(xyz_mm[1]), float(xyz_mm[2]),
                    float(r_mm), float(fit["r2"]), float(fit["rmse"])])
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
        Centerline voxel coordinates (i, j, k).
    skeleton_coords : ndarray, shape (N, 3)
        All skeleton coordinates.
    search_radius : int, optional
        Neighborhood radius in voxels for KD-tree query (default 3).
    tree : KDTree or None, optional
        Prebuilt KDTree over `skeleton_coords` for reuse; built if None.

    Returns
    -------
    tuple of ndarray
        `(t, o1, o2)` unit vectors. If neighbors < 3, returns canonical axes.

    Notes
    -----
    - SVD is computed on mean-centered neighbor coordinates.
    """
    if tree is None:
        tree = KDTree(skeleton_coords)
    idx = tree.query_ball_point(center_voxel, r=search_radius)
    neighbors = skeleton_coords[idx]
    if neighbors.shape[0] < 3:
        # Fallback basis
        return (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
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
    Sample a square patch in the plane spanned by `ortho_v1`/`ortho_v2`.

    Parameters
    ----------
    volume : ndarray, shape (X, Y, Z)
        Intensity volume (float or bool). Order for interpolation is 0 for
        boolean-like arrays, else 1.
    center : ndarray, shape (3,)
        Center voxel (i, j, k) as float.
    ortho_v1, ortho_v2 : ndarray, shape (3,)
        Orthonormal axes spanning the sampling plane.
    search_radius : int, optional
        Half-size in voxels; patch shape is `(2*search_radius+1)^2`.

    Returns
    -------
    tuple
        `(patch, gx, gy)` where `patch` has shape `(H, W)` and `(gx, gy)` are the
        integer coordinate grids used for fitting.
    """
    order = 0 if (volume.dtype == np.bool_ or np.issubdtype(volume.dtype, np.bool_)) else 1
    grid = np.arange(-search_radius, search_radius + 1)
    gx, gy = np.meshgrid(grid, grid, indexing="xy")
    coords = center + gx[..., None] * ortho_v1 + gy[..., None] * ortho_v2
    coords = coords.reshape(-1, 3).T
    vals = map_coordinates(volume, coords, order=order, mode="constant", cval=0.0)
    return vals.reshape(gx.shape), gx, gy


def _gauss2d(coords, A, x0, y0, sx, sy, C):
    """Parametric 2D Gaussian used for cross-section fitting."""
    x, y = coords
    return C + A * np.exp(-(((x - x0) ** 2) / (2 * sx**2) + ((y - y0) ** 2) / (2 * sy**2)))


def _fit_gauss2d(slice_2d: np.ndarray, gx: np.ndarray, gy: np.ndarray):
    """
    Fit a 2D Gaussian to a sampled cross-section.

    Parameters
    ----------
    slice_2d : ndarray
        Sampled intensity patch.
    gx, gy : ndarray
        Integer coordinate grids corresponding to `slice_2d`.

    Returns
    -------
    dict or None
        On success: `{"fwhm_x", "fwhm_y", "radius_vox", "r2", "rmse"}`.
        On failure: None.

    Notes
    -----
    - Initial parameters are derived from simple statistics.
    - Radius (vox) is `0.5 * sqrt(FWHM_x * FWHM_y)`.
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
        popt, _ = curve_fit(_gauss2d, (x, y), z, p0=p0, bounds=bounds, maxfev=5000)
        sx, sy = float(popt[3]), float(popt[4])
        fwhm_x = 2.355 * sx
        fwhm_y = 2.355 * sy
        radius_vox = 0.5 * np.sqrt(fwhm_x * fwhm_y)  # geometric mean diameter / 2
        # Fit quality
        zhat = _gauss2d((x, y), *popt)
        resid = z - zhat
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((z - np.mean(z))**2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        rmse = float(np.sqrt(ss_res / (z.size + 1e-12)))
        if not np.isfinite(radius_vox):
            return None
        return {"fwhm_x": fwhm_x, "fwhm_y": fwhm_y, "radius_vox": radius_vox, "r2": r2, "rmse": rmse}
    except Exception:
        return None


def compute_radii_for_subject(
    sp: SubjectPaths,
    classes: Iterable[str] | None = None,
    search_radius: int = _DEFAULT_SEARCH_RADIUS,
    overwrite: bool = _DEFAULT_OVERWRITE,
) -> Dict[str, Path]:
    """
    Orchestrate native-space radii estimation for available classes.

    Parameters
    ----------
    sp : SubjectPaths
        Provides native-space image paths (TOF/MRV/hT2w), masks/skeletons, and
        output directories.
    classes : iterable of str or None, optional
        Subset of `{"arteries","veins","pvs"}`. If None, try all and skip
        missing inputs gracefully.
    search_radius : int, optional
        Half-size of the orthogonal patch (voxels) passed through to the fitter.
    overwrite : bool, optional
        If False, skip existing outputs when TSV status also satisfies write
        policy (see `compute_centerline_radii`).

    Returns
    -------
    dict
        Mapping `{class: Path}` for produced radius maps.

    Files written
    -------------
    - See `compute_centerline_radii` for per-class artifacts.

    Assumptions / Preconditions
    ---------------------------
    - Uses existing segmentation and skeleton files under `sp.masks_dir`.
    - Operates in native space per class (TOF/MRV/hT2w); no resampling here.

    Warnings
    --------
    - Missing inputs (image/seg/skeleton) cause a logged skip per class.
    """
    if classes is None:
        classes = _DEFAULT_CLASSES

    results: Dict[str, Path] = {}
    for klass in classes:
        if klass not in _CLASS_TO_SPACE_AND_IMG:
            print(f"[radii] [WARN] Unknown class: {klass}")
            continue
        space, img_attr = _CLASS_TO_SPACE_AND_IMG[klass]
        image_path = getattr(sp, img_attr, None)
        if not image_path or not Path(image_path).exists():
            print(f"[radii] [SKIP] Missing source image for {klass} ({img_attr})")
            continue

        seg_path, skel_path = _seg_and_skel_paths(sp, space, klass)
        if not seg_path.exists():
            print(f"[radii] [SKIP] Missing seg for {klass}@{space}: {seg_path.name}")
            continue
        if not skel_path.exists():
            print(f"[radii] [SKIP] Missing skeleton for {klass}@{space}: {skel_path.name}")
            continue

        out_nii, out_tsv = _radii_out_paths(sp, space, klass)
        print(f"[radii] {klass} | space={space}")
        print(f"        img:  {image_path}")
        print(f"        seg:  {seg_path.name}")
        print(f"        skel: {skel_path.name}")
        print(f"        out:  {out_nii.name}")
        if _WRITE_TSV:
            print(f"        tsv:  {out_tsv.name}")

        compute_centerline_radii(
            image_path=image_path,
            seg_path=seg_path,
            output_path=out_nii,
            skeleton_path=skel_path,
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
        `(data, affine, header)` with `data` dtype native and no scaling applied.

    Files written
    -------------
    - None.
    """
    img = nib.load(str(path))
    data = img.get_fdata()
    return data, img.affine, img.header


def _save_nii(data: np.ndarray, affine, header, out_path: str | Path):
    """
    Save a float32 NIfTI at the provided location (folders created as needed).

    Parameters
    ----------
    data : ndarray
        Volume to save; cast to float32.
    affine : ndarray, shape (4, 4)
        Image affine.
    header : nibabel header
        Header propagated to the output image.
    out_path : str or pathlib.Path
        Destination path.

    Returns
    -------
    None
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
        Subject token `sub-<ID>`.
    space : str
        Native space token (e.g., `TOF`, `MRV`, `hT2W`).
    klass : str
        Structure class (`arteries`, `veins`, `pvs`).

    Returns
    -------
    str
        `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-centerline_radii.tsv`
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
        Native space token (e.g., `TOF`, `MRV`, `hT2W`).
    klass : str
        Structure class.

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
        Native space token (e.g., `TOF`, `MRV`, `hT2W`).
    klass : str
        Structure class.

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
# def has_native_radii(sp, klass: str) -> bool:
#     """
#     Check if a native-space radii map exists for a given class.

#     Parameters
#     ----------
#     sp : SubjectPaths
#         Subject paths.
#     klass : str
#         Structure class.

#     Returns
#     -------
#     bool
#         True if the native-space radii map exists.
#     """
#     space = _NATIVE_SPACE.get(klass)
#     if not space:
#         return False
#     p = Path(sp.radii_dir) / deriv_name(sp.sub, space, klass, "radius", "map")
#     return p.exists()


# def has_mreg_radii(sp, klass: str) -> bool:
#     """
#     Check if an MREG-space radii map exists for a given class.

#     Parameters
#     ----------
#     sp : SubjectPaths
#         Subject paths.
#     klass : str
#         Structure class.

#     Returns
#     -------
#     bool
#         True if the MREG-space radii map exists.
#     """
#     p = Path(sp.radii_dir) / deriv_name(sp.sub, "MREG", klass, "radius", "map")
#     return p.exists()
