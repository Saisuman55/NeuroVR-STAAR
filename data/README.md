# NeuroVR / STAAR Dataset Guide

This document outlines the dataset requirements, directory structures, and procedures for obtaining the medical imaging data used by the NeuroVR / STAAR project.

---

## 1. Overview of Supported Data Modalities

| Modality | Formats | Intended Purpose | Primary Directory |
| :--- | :--- | :--- | :--- |
| **2D Brain MRI** | `.jpg`, `.jpeg`, `.png` | Multiclass Classification (Glioma, Meningioma, No Tumor, Pituitary) | `data/classification/` |
| **3D Brain NIfTI** | `.nii`, `.nii.gz` | Multi-planar volumetric visualization and future segmentation | `data/segmentation/` & `project inputs/3D/` |

---

## 2. Expected Directory Structure

```text
data/
├── README.md
├── classification/
│   ├── dataset_manifest.json       # Authoritative manifest of 2D images
│   ├── Training/                   # (Excluded from git tracking due to size)
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/                    # (Excluded from git tracking due to size)
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── demo_inputs/
│   ├── 2d/                         # Small set of verified 2D MRI demo inputs
│   └── 3d/                         # Verified 3D NIfTI reference volumes
└── segmentation/
    └── brats_manifest.json         # BraTS dataset index (when imported)
```

---

## 3. How to Obtain Datasets

### 2D Classification Dataset (Brain Tumor MRI Dataset)
The 2D classification dataset consists of approximately 7,023 brain MRI slices across 4 diagnostic categories.

To download and set up the dataset automatically:
1. Configure your Kaggle API credentials (`~/.kaggle/kaggle.json` or environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`).
2. Run the automated download script:
   ```bash
   python scripts/download_classification_dataset.py
   ```
3. Generate the verification report:
   ```bash
   python scripts/dataset_report.py
   ```

### 3D Volumetric Data (BraTS / Reference Volumes)
- Small reference volumes (`anatomical.nii` and `mni152.nii.gz`) are included directly in `project inputs/3D/` and `data/demo_inputs/3d/` for testing volumetric features, affine coordinates, and orthogonal slicing.
- Full BraTS benchmark datasets can be imported using:
  ```bash
  python scripts/import_brats_dataset.py --source /path/to/brats_extracted/
  ```

---

## 4. Licensing and Ethical Compliance

- **Dataset Ownership:** NeuroVR does not own or redistribute third-party medical imaging datasets.
- **Privacy & Anonymization:** Datasets utilized must be strictly de-identified and compliant with HIPAA/GDPR standards.
- **Permitted Use:** Data is intended exclusively for non-commercial academic research and engineering evaluation.
