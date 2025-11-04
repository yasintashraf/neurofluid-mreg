# Attribution

This repository contains original code and pipeline design by **Yasin Tashraf Hussain (2025)**.

Methods and concepts were informed by the following scientific and technical works:

## Core Methods

**Vesselness (Frangi et al., 1998):** Introduced the multiscale vessel enhancement filter for detecting tubular structures in medical images. Forms the basis of our vessel segmentation approach.

**PVS Segmentation (Ballerini et al., 2018):** Applied 3D Frangi filtering for automatic perivascular space segmentation in brain MRI with expert validation. Our PVS segmentation method is inspired by their optimal filtering technique.

**Enhanced PVS Contrast – EPC (Sepehrband et al., 2019):** Developed the EPC method combining co-registered T1w and T2w images to enhance perivascular space visibility. Our pipeline implements a similar EPC option for improved PVS contrast on T2w scans.

## Distance & Vessel Analysis

**Vessel Distance Mapping (Mietzner et al., 2025):** Demonstrated in vivo vessel distance mapping at 7 T MRI to assess arterial patterns in motor cortex. This work on analyzing cortical features relative to vessel proximity directly influenced our distance-to-vessel clustering and spectral analysis approach.

*"Assessing Arterial Patterns in the Motor Cortex With 7 Tesla Magnetic Resonance Imaging and Vessel Distance Mapping"*  
DOI: 10.1002/hbm.70311

## Image Registration & Processing

**Symmetric Normalization – SyN (Avants et al., 2008):** Proposed the SyN diffeomorphic registration algorithm for high-precision nonlinear image alignment. We use SyN (via DIPY) for warping structural images to MNI space.

**Realign4D Motion Correction (Roche, 2011):** Described simultaneous motion and slice-timing correction for fMRI. We employ Realign4D (NiPy implementation) for motion-correcting high-temporal-resolution MREG data.

## Skeletonization

**3D Skeletonization (Lee et al., 1994):** Presented the Lee algorithm for thinning binary volumes to centerlines while preserving topology. We use Lee's algorithm (scikit-image) to obtain vessel centerlines for radius estimation and distance mapping.

---

## References

- Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2008). Symmetric diffeomorphic image registration with cross-correlation: Evaluating automated labeling of elderly and neurodegenerative brain. *Medical Image Analysis*, 12(1), 26–41.

- Ballerini, L., Lovreglio, R., Valdés Hernández, M. D. C., & Wardlaw, J. M. (2018). On the automated assessment of cerebral microbleeds and perivascular spaces: A systematic review. *Journal of Neuroimaging*, 28(6), 563–575.

- Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A. (1998). Multiscale vessel enhancement filtering. In *International Conference on Medical Image Computing and Computer-Assisted Intervention* (pp. 130–137). Springer.

- Lee, T.-C., Kashyap, R. L., & Chu, C. N. (1994). Building skeleton models via 3-D medial surface axis thinning algorithms. *CVGIP: Image Understanding*, 56(3), 462–478.

- Mietzner, G., et al. (2025). Assessing Arterial Patterns in the Motor Cortex With 7 Tesla Magnetic Resonance Imaging and Vessel Distance Mapping. *Human Brain Mapping*. DOI: 10.1002/hbm.70311

- Roche, A. (2011). A four-dimensional registration algorithm with application to joint correction of nodding and shaking motion in functional MRI. *Medical Image Analysis*, 15(2), 214–220.

- Sepehrband, F., et al. (2019). Image processing approaches to enhance perivascular space visibility and quantification using MRI. *Scientific Reports*, 9, 12965. DOI: 10.1038/s41598-019-48910-x
