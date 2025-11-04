# Glossary

**Bandpower (amplitude-sum):** The sum of Fourier amplitude within a specified frequency band for a time-series, indicating signal strength in that range. In this pipeline, computed by summing FFT magnitudes over each band per voxel.

**BIDS-first:** A design strategy where all file/folder paths derive strictly from a BIDS dataset structure and known subject/session IDs. Pipeline inputs must follow BIDS naming; outputs are written in a BIDS-compliant derivatives folder.

**CLAHE:** Contrast Limited Adaptive Histogram Equalization. Local contrast enhancement method that limits over-amplification of noise. Used for vessel visibility in MRV images.

**Canonical MNI grid:** The spatial grid for the standard MNI152 template (often 1×1×1 mm, canonical axes). Masks on this grid are aligned to the template for group analysis.

**Centerline radii:** Vessel radii estimated at each skeleton voxel, representing local half-diameter. Computed by Gaussian fitting to vessel cross-sections; map marks each skeleton voxel with radius (mm).

**Cubic regression spline:** Flexible ordinary least squares fit using B-spline basis (e.g., patsy.bs), often used for statistical modeling.

**Digitize clusters:** Assigning continuous values into discrete bins. Pipeline uses np.digitize on distance maps to create cluster masks grouping voxels by distance from a vessel (e.g., 0–2 mm, 2–5 mm).

**Distance map:** Volume where each voxel’s value is the Euclidean distance (mm) to the nearest vessel class (e.g., artery). Computed via Euclidean Distance Transform (EDT) on binary masks. Output: *_desc-dist_map.nii.gz.

**EDT (Euclidean Distance Transform):** Algorithm that computes each voxel’s distance to the nearest object voxel in a binary image; scaled by voxel size (mm).

**EPC (Enhanced PVS Contrast):** Method combining T1w and T2w MRI to highlight perivascular spaces (PVS) by signal cancellation or inversion, resulting in enhanced bright PVS features.

**FWHM:** Full Width at Half Maximum, a measure of peak (e.g., vessel cross-section) width. Used for converting Gaussian profile to radius (radius ≈ FWHM/2).

**HuberRegressor:** Robust linear regression model less sensitive to outliers. Used for relationships like power vs. distance.

**Hysteresis thresholding (iterative):** Two-threshold binary segmentation using iterative passes to grow structures and minimize noise, particularly useful for vessel segmentation with varying intensity.

**Image space:** Operations performed in image voxel grid (as opposed to world-coordinates). Image-space registration/resampling means interpolating on the voxel grid.

**KDTree:** k-dimensional tree data structure for efficient spatial nearest-neighbor search. Used in local vessel radius fitting.

**Label-safe warp:** Resampling labeled/binary images using nearest-neighbor interpolation to preserve label integrity (masks remain binary after warp).

**Lee skeletonization:** 3D skeletonization algorithm by T.-C. Lee et al. (1994), thinning binary objects to a 1-voxel-wide centerline while preserving topology. Used for vessel masks.

**MP2RAGE:** MRI sequence yielding a T1w image and two inversion images (INV1, INV2) that can be robustly combined for a bias-field corrected T1w. Pipeline uses INV1/INV2 if available.

**MREG:** Magnetic Resonance Encephalography—ultrafast fMRI (TR~0.1s) allowing physiological oscillation capture without aliasing. Pipeline’s fMRI data is MREG.

**R2*:** Effective transverse relaxation rate (1/T2*), mapping venous blood as hyperintense. Derived from multi-echo MRI; used for vein segmentation.

**Realign4D (NiPy):** 4D motion correction and slice timing adjustment (Roche, 2011). Aligns MREG volumes for motion-corrected output.

**rFFT amplitude (|rFFT|/N):** Magnitude of real FFT normalized by timepoints, representing oscillation strength per frequency bin.

**RobustCombination:** MP2RAGE method combining INV1/INV2 images to produce a robust T1w (UNI), reducing bias/noise.

**Sidecar JSON:** BIDS-companion JSON with per-scan metadata (e.g., TR, EchoTime). Pipeline reads BOLD JSON to obtain scan parameters. Ensure presence/formatting.

**Skeletonization (Lee):** See Lee skeletonization; process of reducing binary volume to centerline.

**Staged affine (DIPY):** Multi-step affine registration with DIPY: translation → rigid → affine, with mutual information or cross-correlation similarity.

**SyN (Symmetric Diffeomorphic Registration):** Nonlinear image registration algorithm (Avants et al.) that finds smooth, invertible deformation fields for mapping between image spaces. Pipeline uses SyN (via DIPY) for warping T1 to MNI.

**TE-weighted geometric mean:** Multi-echo MRI combination highlighting later echoes: $I_{geo} = exp(Σ w_i ln I_i / Σ w_i)$. Emphasizes features persisting at long TEs (fluid/blood), used for vein enhancement.

**Top-hat (black/white):** Morphological filtering that removes background trends; white and black variants correct uneven illumination and highlight features for segmentation.

**Vesselness (Frangi):** Frangi filter (1998) using Hessian matrix eigenvalues at multiple scales to detect vessel-like structures (tubes). Used for artery, vein, and PVS segmentation.
