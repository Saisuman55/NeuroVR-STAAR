# System Architecture — NeuroVR / STAAR

The NeuroVR system is designed as a modular, research-grounded medical imaging pipeline supporting 2D MRI classification and 3D volumetric NIfTI exploration.

---

## 1. High-Level Architecture Flow

```text
[ User Interface (Web / API) ]
              │
              ▼
    [ Flask API Boundary ] (src/web_api.py)
              │
      ┌───────┴─────────────────────────┐
      ▼                                 ▼
[ 2D MRI Pipeline ]            [ 3D NIfTI Pipeline ]
  - Pillow Validation            - NiBabel Volume Loader
  - EfficientNet-B4 Backbone     - Metadata & Affine Extraction
  - Multiclass Logits            - Multi-Planar Slicer
  - Class Probabilities          - Orthogonal Views (Axial/Cor/Sag)
      │                                 │
      └───────────────┬─────────────────┘
                      ▼
        [ Service State & Results ]
                      │
              ┌───────┴───────┐
              ▼               ▼
      [ JSON Metadata ]   [ PNG Slices ]
```

---

## 2. Core Subsystems

### A. Web API & Service Layer (`src/web_api.py`)
- **Framework:** Lightweight Flask service.
- **Upload Validation:** `_validate_upload()` verifies file integrity using Pillow (for 2D) and NiBabel (for 3D), immediately rejecting corrupted headers or unauthorized file extensions (`HTTP 400`).
- **Session State:** Managed in-memory via `AnalysisService`, tracking uploaded volumes and analysis states.

### B. 2D Classification Engine (`src/classifier_2d.py`, `src/model_registry.py`)
- **Backbone:** EfficientNet-B4 transfer learning architecture.
- **Classes:** 4 categories: `glioma`, `meningioma`, `notumor`, `pituitary`.
- **Honest Guarding:** When model checkpoint weights are not loaded (`checkpoint_path: null`), the service returns `status: "unavailable"` to prevent generating false predictions.

### C. 3D Volumetric Processing (`src/volume_loader.py`, `src/preprocessor_3d.py`)
- **NIfTI Support:** Reads uncompressed (`.nii`) and gzip-compressed (`.nii.gz`) neuroimaging files.
- **Multi-Planar Reconstruction:** Dynamic orthogonal slice generation across Axial, Coronal, and Sagittal anatomical planes with min-max normalization to standard 8-bit PNG images.
- **Metadata Extraction:** Extracts voxel dimensions, physical spacing ($mm$), and anatomical orientation (e.g., RAS, LAS) directly from the affine transform matrix.
