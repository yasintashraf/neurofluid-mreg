# SPDX-License-Identifier: MIT
"""
config.py
---------
Pipeline configuration dataclass and YAML loader for the Neurofluid–MREG
pipeline.

This module centralizes BIDS-first configuration, including dataset roots and
subject IDs, optional modality path hints, spectral band definitions (Hz),
distance-bin edges (mm), and execution flags. It is intentionally declarative:
no algorithm logic lives here. Downstream modules consume `PipelineConfig`
verbatim and perform validation/IO at use time.

Pipeline steps
--------------
1. Read a YAML file describing inputs/parameters.
2. Construct a `PipelineConfig` instance from the YAML keys.
3. Pass the instance to orchestrators/analysis modules.

Inputs / Outputs
----------------
Inputs  : YAML file path (on disk) describing keys/values.
Outputs : A `PipelineConfig` instance with attributes mapped 1:1 to YAML.

Files written
-------------
- None (this module does not produce artifacts).

Assumptions / Preconditions
---------------------------
- Spaces: Analysis is performed in native MREG space downstream; TR is read from
  the MREG JSON sidecar by consumers of this config.
- Shapes/dtypes: Not applicable (no arrays handled here).
- BIDS naming: Downstream modules write artifacts using
  `sub-<ID>_space-<SPACE>_class-<CLASS>_desc-<DESC>_<SUFFIX>.nii.gz`.

Warnings
--------
- Unknown YAML keys will raise a `TypeError` during dataclass construction.
- Paths may be absolute or relative; downstream code resolves/validates them.

Public API
----------
- PipelineConfig
- PipelineConfig.from_yaml
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml

Number = Union[int, float]


# -------------------------------------------------------------
# Utilities (logging, checks)
# -------------------------------------------------------------
@dataclass
class PipelineConfig:
    """
    Declarative configuration container for the Neurofluid–MREG pipeline.

    Attributes
    ----------
    bids_root : str
        Root directory of the BIDS dataset.
    subject : str
        Subject identifier (e.g., "sub-xh33_x107"). Downstream code commonly
        derives `sp.sub` from this value.
    deriv_root : str, default "derivatives/neurofluid-mreg"
        Root directory for BIDS-derivatives outputs.
    anat : dict[str, str | None], optional
        Modality hints/overrides for anatomical inputs (e.g., T1, TOF, MRV,
        HT2w). Values are absolute/relative paths, or None to auto-discover.
    func : dict[str, str | None], optional
        Functional/MREG input hints. Values are paths or None to auto-discover.
    bands : dict[str, list[Number]] | None
        Frequency bands in Hz, as `[fmin, fmax]`. If None, downstream uses
        module defaults (e.g., cardiac, respiratory, LF, VLF).
    distance_bins : list[Number | str] | None
        Distance bin edges in mm. The last edge may be the string "max", which
        downstream resolves to `np.nanmax(distance_map)` per run.
    make_mni : bool, default True
        Whether to additionally produce MNI-space outputs (handled downstream).
    dry_run : bool, default True
        If True, orchestrators may list planned steps without executing them.
    radii_enabled : bool, default False
        Toggle for radii estimation steps (consumed by downstream modules).
    radii_overwrite : bool, default False
        Overwrite existing radii artifacts if present (downstream behavior).

    Notes
    -----
    - Declarative only: this class does not validate file existence or schema.
    - YAML keys map 1:1 to attributes; extra keys will raise `TypeError`.
    """

    # roots / id
    bids_root: str
    subject: str
    deriv_root: str = "derivatives/neurofluid-mreg"

    # explicit files
    anat: Dict[str, Optional[str]] = None
    func: Dict[str, Optional[str]] = None

    # params
    bands: Dict[str, List[Number]] = None
    distance_bins: List[Union[Number, str]] = None

    # flags
    make_mni: bool = True
    dry_run: bool = True

    radii_enabled: bool = False
    radii_overwrite: bool = False

    # -------------------------------------------------------------
    # I/O helpers (BIDS naming, paths)
    # -------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """
        Load a YAML file and construct a `PipelineConfig`.

        Parameters
        ----------
        path : pathlib.Path
            Path to a YAML file containing keys that map 1:1 to dataclass
            attributes.

        Returns
        -------
        PipelineConfig
            Populated configuration instance.

        Files written
        -------------
        - None.

        Assumptions / Preconditions
        ---------------------------
        - YAML is UTF-8 encoded and safely parsed with `yaml.safe_load`.
        - Unknown/extra keys will surface as `TypeError` on construction.
        - Relative paths (if any) are interpreted by downstream modules.

        Warnings
        --------
        - This method does not validate the existence of referenced files.

        Raises
        ------
        FileNotFoundError
            If `path` does not exist (raised by `read_text()`).
        TypeError
            If YAML keys do not match the dataclass signature.
        ValueError
            If YAML content is malformed and cannot be parsed.

        Notes
        -----
        - Keep configuration minimal and declarative; validation occurs when the
          configuration is consumed by I/O/analysis code.
        """
        data = yaml.safe_load(path.read_text())
        return cls(**data)