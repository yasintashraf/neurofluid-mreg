# SPDX-License-Identifier: MIT
"""
io.py
-----
BIDS-first I/O utilities for Neurofluid–MREG. This module provides deterministic
path resolution from a `PipelineConfig`, enforces a subject-scoped derivatives
layout, reads basic metadata (TR), and standardizes derivative filenames.

Pipeline steps
--------------
1. Resolve subject-scoped input paths (no auto-discovery; explicit filenames)
2. Validate required inputs; warn on missing optional ones
3. Create derivatives directory tree (idempotent)
4. Build standardized derivative filenames

Inputs / Outputs
----------------
Inputs  : A `PipelineConfig` object (fields consumed by `SubjectPaths`).
Outputs : `SubjectPaths` instance with resolved `Path`s; created directories.

Files written
-------------
- Directory tree under: derivatives/neurofluid-mreg/sub-<ID>/
  - `anat/`, `masks/`, `distmaps/`, `mreg/`, `bandmaps/`, `clusters/`,
    `spectra/`, `stats/`, `figures/`, `qc/`, `manifest/`, `anat/` (transforms),
    `radii/`
- Filenames constructed by `deriv_name` follow:
  `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-<DESC>_<SUFFIX>.nii.gz`

Assumptions / Preconditions
---------------------------
- Spaces: Handled by callers when naming via `deriv_name`; this module does not
  resample or inspect affines.
- Paths: All inputs are provided explicitly in `PipelineConfig`; relative paths
  are resolved under `<BIDS_ROOT>/sub-<ID>/<anat|func>/`.
- Types: Paths are `pathlib.Path` where practical.

Warnings
--------
- `load_TR_from_bold_json` expects a sidecar JSON next to the 4D MREG NIfTI
  (same basename); absence or missing `RepetitionTime` raises.

Public API
----------
- SubjectPaths
- validate_required_inputs
- load_TR_from_bold_json
- ensure_derivatives_layout
- deriv_name
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

# -------------------------------------------------------------
# Utilities (path resolution)
# -------------------------------------------------------------
def _resolve_under(base: Path, spec: Optional[str]) -> Optional[Path]:
    """
    Resolve a path specification relative to a base directory.

    Parameters
    ----------
    base : Path
        Base directory (e.g., `<sub>/anat` or `<sub>/func`).
    spec : str or None
        Path spec from config. If absolute, `expanduser()` is applied and
        returned. If relative, it is joined under `base`. If None/empty,
        returns None.

    Returns
    -------
    Path or None
        Resolved path or None.

    Files written
    -------------
    - None.

    Assumptions / Preconditions
    ---------------------------
    - `base` exists or is a valid target for resolution.
    """
    if not spec:
        return None
    p = Path(spec).expanduser()
    if p.is_absolute():
        return p
    return base / p


# -------------------------------------------------------------
# I/O helpers (BIDS naming, paths)
# -------------------------------------------------------------
@dataclass
class SubjectPaths:
    """
    Resolved subject-scoped paths using explicit filenames from PipelineConfig.

    Summary
    -------
    No auto-discovery; no guessing. All input files are resolved relative to
    the subject's BIDS folders unless absolute paths are provided.

    Required (config)
    -----------------
    - anat: `t1w`, `tof`
    - func: `mreg`  (4D MREG BOLD)

    Optional (config)
    -----------------
    - anat: `inv1`, `inv2`, `mrv`, `hT2w`
      → Attributes: `anat_inv1`, `anat_inv2`, `anat_mrv`, `anat_heavy_t2w`.

    Derived folders (under derivatives root)
    ----------------------------------------
    - `anat_out`, `masks_dir`, `distmaps_dir`, `mreg_dir`, `bandmaps_dir`,
      `clusters_dir`, `spectra_dir`, `stats_dir`, `figures_dir`, `qc_dir`,
      `manifest_dir`, `transforms_dir`, `radii_dir`.

    Files written
    -------------
    - None directly; use `ensure_derivatives_layout` to create directories.

    Assumptions / Preconditions
    ---------------------------
    - `cfg.subject`, `cfg.bids_root`, `cfg.deriv_root` are defined.
    - Relative paths are resolved under `<BIDS_ROOT>/sub-<ID>/<anat|func>/`.

    Notes
    -----
    - `transforms_dir` is colocated with `anat` derivatives.
    """
    cfg: "PipelineConfig"  # type:ignore  # forward ref for type hints

    def __post_init__(self):
        self.sub = f"sub-{self.cfg.subject}"
        self.bids_root = str(Path(self.cfg.bids_root).expanduser())
        self.deriv_root = str((Path(self.cfg.deriv_root) / self.sub).expanduser())

        # Base folders
        self.sub_root = Path(self.bids_root) / self.sub
        self.anat_dir = self.sub_root / "anat"
        self.func_dir = self.sub_root / "func"

        # Resolve anatomy files (explicit)
        a = self.cfg.anat or {}
        self.anat_t1w = _resolve_under(self.anat_dir, a.get("t1w"))
        self.anat_tof = _resolve_under(self.anat_dir, a.get("tof"))
        self.anat_inv1 = _resolve_under(self.anat_dir, a.get("inv1"))
        self.anat_inv2 = _resolve_under(self.anat_dir, a.get("inv2"))
        self.anat_mrv = _resolve_under(self.anat_dir, a.get("mrv"))
        self.anat_heavy_t2w = _resolve_under(self.anat_dir, a.get("hT2w"))

        # Resolve func file (explicit)
        f = self.cfg.func or {}
        self.func_mreg_bold = _resolve_under(self.func_dir, f.get("mreg"))

        # Derivatives layout (root + subfolders)
        dr = Path(self.deriv_root)
        self.anat_out = dr / "anat"
        self.masks_dir = dr / "masks"
        self.distmaps_dir = dr / "distmaps"
        self.mreg_dir = dr / "mreg"
        self.bandmaps_dir = dr / "bandmaps"
        self.clusters_dir = dr / "clusters"
        self.spectra_dir = dr / "spectra"
        self.stats_dir = dr / "stats"
        self.figures_dir = dr / "figures"
        self.qc_dir = dr / "qc"
        self.manifest_dir = dr / "manifest"
        self.transforms_dir = dr / "anat"  # transforms live with anat derivatives
        self.radii_dir = dr / "radii"
        self.radii_dir.mkdir(parents=True, exist_ok=True)


def validate_required_inputs(sp: SubjectPaths) -> None:
    """
    Ensure required inputs exist and warn about missing optional inputs.

    Required
    --------
    - `anat_t1w` (T1w)
    - `anat_tof` (TOF)
    - `func_mreg_bold` (MREG 4D)

    Optional (warn if configured but missing)
    -----------------------------------------
    - `anat_inv1` (INV1)
    - `anat_inv2` (INV2)
    - `anat_mrv`  (MRV)
    - `anat_heavy_t2w` (hT2w)

    Parameters
    ----------
    sp : SubjectPaths
        Resolved paths container.

    Returns
    -------
    None

    Files written
    -------------
    - None.

    Raises
    ------
    FileNotFoundError
        If any required input is missing.
    """
    missing = []
    for label, p in [
        ("T1w", sp.anat_t1w),
        ("TOF", sp.anat_tof),
        ("MREG", sp.func_mreg_bold),
    ]:
        if p is None or not Path(p).exists():
            missing.append(f"{label}: {p}")

    for label, p in [
        ("INV1", sp.anat_inv1),
        ("INV2", sp.anat_inv2),
        ("MRV", sp.anat_mrv),
        ("hT2w", sp.anat_heavy_t2w),
    ]:
        if p and not Path(p).exists():
            print(f"[WARN] Optional file not found -> {label}: {p}")

    if missing:
        msg = "[ERROR] Required inputs not found:\n" + "\n".join("  - " + m for m in missing)
        raise FileNotFoundError(msg)


def load_TR_from_bold_json(bold_path: Path) -> float:
    """
    Read TR (seconds) from the JSON sidecar next to the MREG 4D NIfTI.

    Behavior
    --------
    - Sidecar path derived via `full_path.replace('.nii.gz', '.json')`.
    - Field read: `json['RepetitionTime']`.

    Parameters
    ----------
    bold_path : Path
        Path to `<...>_task-mreg_bold.nii.gz`.

    Returns
    -------
    float
        TR value in seconds read from the JSON sidecar.

    Files written
    -------------
    - None.

    Raises
    ------
    FileNotFoundError
        If the sidecar JSON does not exist.
    KeyError
        If 'RepetitionTime' is missing in the JSON.
    """
    full_path = str(Path(bold_path))
    json_file = full_path.replace(".nii.gz", ".json")
    TR = json.load(open(json_file))["RepetitionTime"]
    return float(TR)


def ensure_derivatives_layout(sp: SubjectPaths) -> None:
    """
    Create the full derivatives directory tree for this subject (idempotent).

    Creates
    -------
    - `anat/`, `masks/`, `distmaps/`, `mreg/`, `bandmaps/`, `clusters/`,
      `spectra/`, `stats/`, `figures/`, `qc/`, `manifest/`, `anat/` (transforms)

    Parameters
    ----------
    sp : SubjectPaths
        Resolved paths container.

    Returns
    -------
    None

    Files written
    -------------
    - Directories as listed above (via `mkdir(parents=True, exist_ok=True)`).

    Notes
    -----
    - Safe to call multiple times.
    """
    for d in [
        sp.anat_out,
        sp.masks_dir,
        sp.distmaps_dir,
        sp.mreg_dir,
        sp.bandmaps_dir,
        sp.clusters_dir,
        sp.spectra_dir,
        sp.stats_dir,
        sp.figures_dir,
        sp.qc_dir,
        sp.manifest_dir,
        sp.transforms_dir,
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)


def deriv_name(sub: str, space: str, klass: str, desc: str, dtype: str) -> str:
    """
    Build a standardized BIDS-like derivative filename (no directory).

    Pattern
    -------
    `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-<DESC>_<TYPE>.nii.gz`

    Parameters
    ----------
    sub : str
        Subject token, e.g., "sub-01".
    space : str
        Space label, e.g., "MREG", "T1w".
    klass : str
        Structure/class label, e.g., "arteries", "brain".
    desc : str
        Description token, e.g., "vesselness", "mean".
    dtype : str
        File type token, e.g., "map", "mask".

    Returns
    -------
    str
        Filename only (no directory).

    Files written
    -------------
    - None.

    Assumptions / Preconditions
    ---------------------------
    - Callers ensure tokens are valid for their use-case.
    """
    return f"{sub}_space-{space}_class-{klass}_desc-{desc}_{dtype}.nii.gz"
