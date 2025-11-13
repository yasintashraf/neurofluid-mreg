# Neurofluid–MREG Pipeline

A neuroimaging analysis pipeline for studying neurofluid dynamics using ultra-fast fMRI and high-resolution vascular images. The pipeline segments major vascular structures (arteries, veins, perivascular spaces), preprocesses MREG (Magnetic Resonance Encephalography) BOLD fMRI time series, computes spectral power measures in MREG native space, and performs distance-based analysis in MNI standard space to investigate how physiological oscillations vary with distance from blood vessels.

## Features

- **Vessel Segmentation**: Automated segmentation of arteries (TOF), veins (MRV), and perivascular spaces (hT2w)
- **MREG Preprocessing**: Motion correction, detrending, and registration
- **Spectral Analysis**: Frequency-band power mapping in MREG native space with optional export to MNI
- **Distance Analysis**: Compute distance maps in MNI space to analyze BOLD signal relationships with vessel proximity
- **MNI-Based Statistics**: Clustering and statistical analysis performed in standard MNI space for group-level compatibility
- **Radii Analysis**: Vessel radius mapping in MNI space to investigate power variations with vessel size
- **BIDS-Compatible**: All inputs and outputs follow BIDS-like format
- **Configurable**: Single YAML configuration file for full pipeline control

## Installation

### Dependencies

The pipeline requires Python 3.8+ with the following major dependencies:

- NiBabel (NIfTI I/O)
- NiPy (motion realignment)
- DIPY (image registration)
- scikit-image (filtering and skeletonization)
- NumPy/SciPy
- scikit-learn
- Nilearn (brain masking)

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

### Input Data Structure

Organize data in BIDS format:

```
bids_root/
├── sub-<ID>/
│   ├── anat/
│   │   ├── sub-<ID>_T1w.nii.gz          # Required: T1-weighted MRI
│   │   ├── sub-<ID>_T1w.json
│   │   ├── sub-<ID>_TOF.nii.gz          # Required: Time-of-Flight MRA (arteries)
│   │   ├── sub-<ID>_MRV.nii.gz          # Optional: MR venography (veins)
│   │   ├── sub-<ID>_hT2w.nii.gz         # Optional: High-res T2w (PVS)
│   │   ├── sub-<ID>_INV1.nii.gz         # Optional: MP2RAGE INV1
│   │   └── sub-<ID>_INV2.nii.gz         # Optional: MP2RAGE INV2
│   └── func/
│       ├── sub-<ID>_task-rest_mreg.nii.gz  # Required: 4D MREG fMRI
│       └── sub-<ID>_task-rest_mreg.json
```

**Note**: Convert DICOM to NIfTI using [dcm2niix](https://github.com/rordenlab/dcm2niix). Ensure JSON sidecars are generated, especially for fMRI (contains essential metadata like TR).

### Required vs Optional Data

- **Required**: T1w, TOF (arteries), MREG fMRI
- **Optional**: MRV (veins), hT2w (PVS), MP2RAGE inversions
- Pipeline automatically skips steps for missing optional data
- **Note**: For more accurate and robust vessel and PVS segmentation, it is recommended to brain mask anatomical images before segmentation. This minimizes false positives from non-brain regions and ensures cleaner results.
- **Note**: For more accurate and robust analysis, it is recommended to apply distortion correction on the MREG before frequency analysis.


## Usage

### 1. Configuration

Create a YAML configuration file (e.g., `pipeline.yaml`):

```yaml
# Basic paths
bids_root: "/path/to/BIDS/root"
subject: "xh33_x107"
deriv_root: "derivatives/neurofluid-mreg"

# Anatomical images
anat:
  t1w:  "sub-xh33_x107_T1w.nii.gz"       # Required
  tof:  "sub-xh33_x107_TOF.nii.gz"       # Required
  inv1: "sub-xh33_x107_INV1.nii.gz"      # Optional
  inv2: "sub-xh33_x107_INV2.nii.gz"      # Optional
  mrv:  "sub-xh33_x107_MRV.nii.gz"       # Optional
  ht2w: "sub-xh33_x107_hT2w.nii.gz"      # Optional

# Functional images
func:
  mreg: "sub-xh33_x107_task-rest_mreg.nii.gz"  # Required

# Frequency bands for spectral analysis (Hz)
bands:
  cardiac: [0.80, 1.20]
  respiratory: [0.20, 0.30]
  LF: [0.027, 0.073]
  VLF: [0.010, 0.027]

# Distance bins (mm) for clustering
distance_bins: [0, 2, 5, 10, "max"]

# Optional: vessel radii analysis
radii_enabled: false
radii_overwrite: false
```

### 2. Run Pipeline

First, load your YAML configuration with the below in run_pipeline.py:

```bash
from pathlib import Path
cfg = PipelineConfig.from_yaml(Path("pipeline.yaml"))
```

Then, you can run the pipeline (adjust paths as needed):

```bash
python run_pipeline.py
```

The pipeline will automatically:
1. Validate input files
2. Create output directory structure
3. Execute all processing steps in sequence
4. Generate quality control figures

### 3. Output Structure

Results are organized in BIDS-derivative format:

```
derivatives/neurofluid-mreg/sub-<ID>/
├── anat/ # Anatomical derivatives and transforms
├── masks/ # Vessel segmentation masks and skeletons
├── distmaps/ # Distance maps from vessels (MNI space)
├── mreg/ # Processed MREG fMRI data (native MREG space)
├── bandmaps/ # Band power maps (MREG space + MNI exports)
├── clusters/ # Distance-based cluster masks (MNI space)
├── spectra/ # Spectral analysis outputs (MREG space)
├── stats/ # Statistical result files (MNI space)
├── figures/ # Summary plots and figures
├── qc/ # Quality control figures
├── manifest/ # Manifest/record files
└── radii/ # Vessel radii files (MNI space)
```

### Output Files

#### Segmentation Outputs
- `*_desc-vesselness_map.nii.gz`: Vessel probability maps
- `*_desc-main_mask.nii.gz`: Binary vessel masks
- `*_desc-skeleton_mask.nii.gz`: Vessel centerlines

#### Preprocessing Outputs (MREG Space)
- `*_space-MREG_desc-detrended_bold.nii.gz`: Preprocessed fMRI in native MREG space
- `*_space-MREG_desc-mean_map.nii.gz`: Mean fMRI image in native MREG space

#### Registration Outputs
- `*_xfm-*to*.txt`: Affine transform matrices
- `*_xfm-*_warp.nii.gz`: Nonlinear deformation fields

#### Spectral Analysis Outputs

**In MREG Native Space:**
- `*_space-MREG_band-cardiac_desc-power_map.nii.gz`: Cardiac band power map
- `*_space-MREG_band-respiratory_desc-power_map.nii.gz`: Respiratory band power map
- `*_space-MREG_band-LF_desc-power_map.nii.gz`: Low-frequency band power map
- `*_space-MREG_band-VLF_desc-power_map.nii.gz`: Very-low-frequency band power map
- `*_space-MREG_desc-meanamp_map.nii.gz`: Mean amplitude across all bands
- `*_space-MREG_class-<arteries|veins|pvs>_desc-cluster_spectra.npz`: Cluster-averaged spectra

**Exported to MNI Space:**
- `*_space-MNI_band-cardiac_desc-power_map.nii.gz`: Cardiac band power in MNI space
- `*_space-MNI_band-respiratory_desc-power_map.nii.gz`: Respiratory band power in MNI space
- `*_space-MNI_band-LF_desc-power_map.nii.gz`: Low-frequency band power in MNI space
- `*_space-MNI_band-VLF_desc-power_map.nii.gz`: Very-low-frequency band power in MNI space
- `*_space-MNI_desc-meanamp_map.nii.gz`: Mean amplitude in MNI space

#### Distance & Analysis Outputs (MNI Space)
- `*_space-MNI_class-arteries_desc-distance_map.nii.gz`: Euclidean distance map from arteries in MNI space
- `*_space-MNI_class-arteries_desc-radius_map.nii.gz`: Vessel radius map for arteries in MNI space
- `*_space-MNI_class-arteries_desc-clusters_mask.nii.gz`: Distance-based cluster labels in MNI space
- `*_space-MNI_class-arteries_desc-binned_stats.csv`: Binned statistics (power by distance bin)
- `*_space-MNI_class-arteries_desc-continuous_stats.csv`: Continuous regression statistics (power vs distance)
- `*_space-MNI_class-arteries_desc-radius_vs_power.csv`: Radius-based statistics

#### Visualization Outputs
- `*_space-MREG_class-arteries_desc-cluster_spectra.png`: Cluster spectra plots (MREG space)
- `*_space-MNI_class-arteries_desc-binned_bandpower.png`: Binned power plots (MNI space)
- `*_space-MNI_class-arteries_band-cardiac_desc-continuous.png`: Continuous regression plots (MNI space)
- `*_space-MNI_class-arteries_desc-radius_vs_power.png`: Radius vs power scatter plots (MNI space)

## Pipeline Workflow

The pipeline executes the following stages:

### 1. Vessel Segmentation
Segments arteries from TOF using Frangi vesselness filtering, veins from multi-echo MRV (R2* mapping), and perivascular spaces from high-res T2w. Generates masks, vesselness maps, and skeletons for each structure in their native spaces.

### 2. MREG Preprocessing
Motion correction using NiPy's 4D realignment, polynomial detrending to remove slow drifts, and temporal mean computation. All preprocessing is performed in the native MREG functional space.

### 3. Registration
- Affine registration: MREG mean → T1w
- Affine registration: Native images (TOF, MRV, hT2w) → T1w
- Nonlinear registration: T1w → MNI space (optional)
- Apply transforms to full fMRI time series and masks

### 4. Brain Masking
Generate brain mask in MNI space and project mask to T1w and MREG spaces.

### 5. Spectral Analysis (MREG Native Space)
Compute voxel-wise FFT and extract band-specific power in the native MREG functional space. Group voxels by distance from vessels and compute cluster spectra. Store all bandpower maps in MREG space.

### 6. Export Bandpower to MNI (Optional)
Transform bandpower maps from MREG native space to MNI standard space using the computed registration transforms. This enables group-level spectral analysis while preserving the native-space computation.

### 7. Distance Mapping (MNI Space)
Compute Euclidean distance transform for each vessel class in MNI space. Generate distance maps in mm units, establishing the spatial relationship between voxels and vessel structures in standard space.

### 8. Distance-Based Clustering (MNI Space)
- Create binary clusters based on predefined distance bins from vessel structures
- Generate cluster masks labeled by distance range (e.g., 0–2 mm, 2–5 mm, etc.)
- Store all cluster masks in MNI space

### 9. Binned Statistics Analysis (MNI Space)
Perform statistical analysis of bandpower vs distance using distance-based clusters. Extract mean power values for each band within each distance bin, generate summary tables and visualization plots showing power variation with distance from vessels in MNI space.

### 10. Continuous Statistical Analysis (MNI Space)
Perform robust linear regression of bandpower (log-transformed) against continuous distance values for all voxels within the brain mask in MNI space. Generate regression coefficients, p-values, and visualization plots for each frequency band.

### 11. Radii Analysis (MNI Space)
Compute local vessel diameter at each centerline voxel using 2D Gaussian fitting. Analyze BOLD power vs vessel radius by extracting power values within radius-defined annuli around vessel structures. Generate regression analysis and visualization plots.

### 12. Visualization & Quality Control
- Generate QC plots for segmentation and registration
- Create summary plots for spectral analysis in both MREG and MNI spaces
- Output cluster spectra and distance-power relationships
- Produce publication-ready figures

## Quality Control

Quality control outputs are saved in the `qc/` subdirectory:

- Vessel segmentation visualizations
- Motion parameters from fMRI realignment
- Registration quality checks (MREG → T1w → MNI)
- Spectral analysis summary plots
- Distance map visualizations in MNI space

Review these outputs to verify each processing step completed successfully.

## Advanced Options

### Vessel Radii Analysis

Enable radius estimation along vessel centerlines in MNI space:

```yaml
radii_enabled: true
radii_overwrite: false
```

This computes local vessel diameter at each centerline voxel using 2D Gaussian fitting. Radius maps are generated in MNI space, and results are used to analyze BOLD power vs vessel size. Results are saved as both continuous radius maps and binned statistics.

### MP2RAGE Denoising

If MP2RAGE inversions (INV1, INV2) are provided, the pipeline automatically generates a robust combined T1w image with improved SNR and reduced bias field.

### Custom Frequency Bands

Define custom frequency bands for spectral analysis:

```yaml
bands:
  custom_band: [0.05, 0.15]
  another_band: [1.5, 2.5]
```

Bands should be specified in Hz and must fit within the Nyquist frequency of your MREG data.

## File Naming Convention

All outputs follow BIDS-derivative naming:

```
sub-<ID>_space-<SPACE>_class-<CLASS>_desc-<DESC>_<SUFFIX>.nii.gz
```

- `space`: Image space (TOF, MRV, T1, MREG, MNI)
  - **MREG**: Native functional space (for bandpower maps and spectra)
  - **MNI**: Standard stereotactic space (for distance maps, statistics, and analysis)
  - **T1**: Anatomical space (for intermediate registration products)
- `class`: Structure type (arteries, veins, pvs, brain)
- `desc`: Description (vesselness, main, skeleton, detrended, power_map, clusters, binned_stats, continuous_stats, radius_vs_power, etc.)
- `suffix`: File type (nii.gz for images, csv for statistics, npz for numpy arrays)

**Key Space Conventions:**
- Bandpower maps are computed and stored in `space-MREG` (native fMRI space)
- Bandpower maps are also exported to `space-MNI` for group-level analysis
- Distance maps, clustering, and statistics are all performed in `space-MNI` (standard space)
- Spectral analysis and cluster labels are computed in `space-MREG` using MNI-derived cluster boundaries transformed back to native space

## Troubleshooting

### Common Issues

**Missing TR in JSON**: Ensure fMRI JSON sidecar contains `RepetitionTime` field.

**Registration Failures**: Check that input images have correct orientation and headers. Verify T1w and functional images are from the same session.

**Memory Issues**: For very high-resolution data, consider downsampling or running on a high-memory system.

**Segmentation Quality**: Review vesselness maps and adjust thresholds if needed. Some parameters may require tuning for specific datasets.

**MNI Registration Issues**: Verify that T1w → MNI nonlinear registration completed successfully. Check QC figures for obvious alignment problems.

**Missing MNI Distance Maps**: Ensure that the T1w → MNI transformation was computed. If veins or PVS distance maps are missing in MNI space, it may indicate segmentation failed for these structures.

## Citation

If you use this pipeline in your research, please cite:

```
Hussain, Yasin Tashraf; Mattern, Hendrik (2025).
Neurofluid–MREG: BIDS-first vascular segmentation and spectral analysis pipeline.
Open-source pipeline linking ultra-fast MREG BOLD signals to vascular structures via segmentation, distance/radii mapping in MNI space, and band-limited spectral analysis in native MREG space.
Otto-von-Guericke University Magdeburg, Germany.
GitHub Repository: https://github.com/yasintashraf/neurofluid-mreg
Version 0.2.0, Released 2025-11-25.
```

## License

MIT License

Copyright (c) 2025 Yasin Tashraf Hussain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## Acknowledgments

I would like to express my sincere gratitude to **Dr. Hendrik Mattern**  
(Department of Biomedical Magnetic Resonance, Institute of Physics,  
Otto-von-Guericke University Magdeburg) for his valuable guidance, discussions,  
and for sharing reference code examples and methodological insights that  
contributed to the conceptual development of this project.

## Contact

For questions and support, please open an issue on the GitHub repository.

---

**Version**: 0.2.0  
**Last Updated**: November 2025