# Neurofluid–MREG Pipeline

A neuroimaging analysis pipeline for studying neurofluid dynamics using ultra-fast fMRI and high-resolution vascular images. The pipeline segments major vascular structures (arteries, veins, perivascular spaces), preprocesses MREG (Magnetic Resonance Encephalography) BOLD fMRI time series, and computes spectral power measures to analyze how physiological oscillations vary with distance from blood vessels.

## Features

- **Vessel Segmentation**: Automated segmentation of arteries (TOF), veins (MRV), and perivascular spaces (hT2w)
- **MREG Preprocessing**: Motion correction, detrending, and registration
- **Distance Analysis**: Compute distance maps from vessels to analyze BOLD signal relationships
- **Spectral Analysis**: Frequency-band power mapping and statistical analysis
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
├── distmaps/ # Distance maps from vessels
├── mreg/ # Processed MREG fMRI data
├── bandmaps/ # Band power maps
├── clusters/ # Distance-based cluster masks
├── spectra/ # Spectral analysis outputs
├── stats/ # Statistical result files
├── figures/ # Summary plots and figures
├── qc/ # Quality control figures
├── manifest/ # Manifest/record files
└── radii/ # Vessel radii files
```

### Output Files

#### Segmentation Outputs
- `*_desc-vesselness_map.nii.gz`: Vessel probability maps
- `*_desc-main_mask.nii.gz`: Binary vessel masks
- `*_desc-skeleton_mask.nii.gz`: Vessel centerlines

#### Preprocessing Outputs
- `*_desc-detrended_bold.nii.gz`: Preprocessed fMRI
- `*_desc-mean_map.nii.gz`: Mean fMRI image

#### Registration Outputs
- `*_xfm-*to*.txt`: Affine transform matrices
- `*_xfm-*_warp.nii.gz`: Nonlinear deformation fields

#### Analysis Outputs
- `*_band-*_desc-power_map.nii.gz`: Spectral power maps per frequency band
- `*_desc-clusters_mask.nii.gz`: Distance-based cluster labels
- `*_desc-binned_bandpower.csv`: Statistical summaries
- `*.png`: Analysis plots and figures

## Pipeline Workflow

The pipeline executes the following stages:

### 1. Vessel Segmentation
- Segments arteries from TOF using Frangi vesselness filtering
- Segments veins from multi-echo MRV (R2* mapping)
- Segments perivascular spaces from high-res T2w
- Generates masks, vesselness maps, and skeletons for each structure

### 2. MREG Preprocessing
- Motion correction using NiPy's 4D realignment
- Polynomial detrending to remove slow drifts
- Temporal mean computation

### 3. Registration
- Affine registration: MREG mean → T1w
- Affine registration: Native images (TOF, MRV, hT2w) → T1w
- Nonlinear registration: T1w → MNI space (optional)
- Apply transforms to full fMRI time series and masks

### 4. Brain Masking
- Generate brain mask in MNI space
- Project mask to T1w and MREG spaces

### 5. Distance Mapping
- Compute Euclidean distance transform for each vessel class
- Generate distance maps in MREG space (mm units)

### 6. Spectral Analysis
- Compute voxel-wise FFT and extract band-specific power
- Group voxels by distance from vessels
- Statistical analysis (ANOVA, regression) of power vs distance
- Optional: Analyze power vs vessel radius

### 7. Visualization
- Generate QC plots for segmentation and registration
- Create summary plots for spectral analysis
- Output cluster spectra and distance-power relationships

## Quality Control

Quality control outputs are saved in the `qc/` subdirectory:

- Vessel segmentation visualizations
- Motion parameters from fMRI realignment
- Registration quality checks
- Spectral analysis summary plots

Review these outputs to verify each processing step completed successfully.

## Advanced Options

### Vessel Radii Analysis

Enable radius estimation along vessel centerlines:

```yaml
radii_enabled: true
radii_overwrite: false
```

This computes local vessel diameter at each centerline voxel using 2D Gaussian fitting. Results are used to analyze BOLD power vs vessel size.

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
- `class`: Structure type (arteries, veins, pvs, brain)
- `desc`: Description (vesselness, main, skeleton, detrended, etc.)
- `suffix`: File type (map, mask, bold, etc.)

## Troubleshooting

### Common Issues

**Missing TR in JSON**: Ensure fMRI JSON sidecar contains `RepetitionTime` field.

**Registration Failures**: Check that input images have correct orientation and headers. Verify T1w and functional images are from the same session.

**Memory Issues**: For very high-resolution data, consider downsampling or running on a high-memory system.

**Segmentation Quality**: Review vesselness maps and adjust thresholds if needed. Some parameters may require tuning for specific datasets.

## Citation

If you use this pipeline in your research, please cite:

```
Hussain, Yasin Tashraf; Mattern, Hendrik (2025).
Neurofluid–MREG: BIDS-first vascular segmentation and spectral analysis pipeline.
Open-source pipeline linking ultra-fast MREG BOLD signals to vascular structures via segmentation, distance/radii mapping, and band-limited spectral analysis.
Otto-von-Guericke University Magdeburg, Germany.
GitHub Repository: https://github.com/your-username/neurofluid-mreg
Version 0.1.0, Released 2025-11-03.
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

For questions and support, please open an issue on the GitHub repository, or contact yasin.tashraf@st.ovgu.de or yasintashraf14@gmail.com

---

**Version**: 1.0.0  
**Last Updated**: November 2025
