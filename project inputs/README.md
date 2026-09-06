# NeuroVR / STAAR Project Inputs

This directory is a clean, centralized, presentation-ready collection of real test inputs for demonstrating and validating all major workflows in the **NeuroVR — Brain Tumor MRI Analysis and Visualization System** without needing to search through large raw dataset directories.

All demonstration data is derived directly from verified local project datasets or standard public medical imaging references. No medical images, NIfTI volumes, or segmentation masks are fabricated.

---

## Directory Layout

```text
project inputs/
│
├── README.md                              # This documentation and testing guide
├── input_manifest.json                    # Complete machine-readable metadata and SHA256 manifest
│
├── 2D/                                    # Real 2D brain MRI scans (12 images, 3 per class)
│   ├── glioma/
│   │   ├── Tr-gl_157.jpg
│   │   ├── Tr-gl_578.jpg
│   │   └── Tr-gl_998.jpg
│   │
│   ├── meningioma/
│   │   ├── Tr-me_157.jpg
│   │   ├── Tr-me_578.jpg
│   │   └── Tr-me_998.jpg
│   │
│   ├── notumor/
│   │   ├── Tr-no_157.jpg
│   │   ├── Tr-no_578.jpg
│   │   └── Tr-no_998.jpg
│   │
│   └── pituitary/
│       ├── Tr-pi_157.jpg
│       ├── Tr-pi_578.jpg
│       └── Tr-pi_998.jpg
│
├── 3D/                                    # Real 3D NIfTI reference volumes
│   ├── anatomical_reference/
│   │   └── anatomical.nii                 # NiBabel T1 structural reference scan (33×41×25, 2mm isotropic)
│   │
│   └── mni152_reference/
│       └── mni152.nii.gz                  # ICBM 152 standard brain template (207×256×215, 0.7375mm isotropic)
│
├── invalid_inputs/                        # Safe test files for validation & error-handling checks
│   ├── unsupported_file.txt               # Tests file extension validation (.txt rejected with HTTP 400)
│   ├── invalid_image.txt                  # Tests image parsing error handling (corrupt image header)
│   └── invalid_nifti.txt                  # Tests NIfTI header validation (corrupt magic bytes)
│
└── test_summary/
    └── test_input_report.json             # Verification report confirming file integrity and checksums
```

---

## Input Categories and Provenance

### 1. 2D Brain MRI Classification Inputs (`2D/`)
- **Source:** Verified local 2D Brain Tumor MRI dataset (`data/classification/Training/`).
- **Total:** 12 real brain MRI scans ($512 \times 512$ JPEG format).
- **Classes:**
  - `glioma`: 3 images
  - `meningioma`: 3 images
  - `notumor`: 3 images
  - `pituitary`: 3 images
- **Purpose:** Used for uploading through the Web API or CLI, verifying format validation, metadata extraction, image preview, and classification model testing.

### 2. 3D Medical MRI Reference Volumes (`3D/`)
- **`anatomical_reference/anatomical.nii`:**
  - **Source:** Official NiBabel package reference data (`nipy/nibabel`).
  - **Dimensions:** $33 \times 41 \times 25$ voxels.
  - **Voxel Spacing:** $2.0 \times 2.0 \times 2.0\text{ mm}$ isotropic.
  - **Orientation:** L-A-S (Left-Anterior-Superior).
  - **Affine:** 4×4 spatial coordinate matrix present.
  - **Segmentation Availability:** `false` (no tumor segmentation mask).
- **`mni152_reference/mni152.nii.gz`:**
  - **Source:** ICBM 152 2009a Nonlinear Symmetric Template via NiiVue public test data.
  - **Dimensions:** $207 \times 256 \times 215$ voxels.
  - **Voxel Spacing:** $0.7375 \times 0.7375 \times 0.7375\text{ mm}$ isotropic.
  - **Orientation:** R-A-S (Right-Anterior-Superior).
  - **Affine:** 4×4 spatial coordinate matrix present.
  - **Segmentation Availability:** `false` (standard anatomical reference template).
- **Purpose:** Testing volumetric NIfTI loading, affine matrix extraction, voxel spacing calculation, and multi-planar orthogonal slicing (Axial, Coronal, Sagittal).

### 3. Invalid Inputs (`invalid_inputs/`)
- Non-malicious plain text test files designed to verify that the Web API and upload handlers safely reject invalid payloads without server crashes.

---

## Testing Instructions

### 1. 2D MRI Workflow Testing
1. Start the NeuroVR demonstration server:
   ```bash
   PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python app.py
   ```
2. Open `http://127.0.0.1:5000/` in a browser.
3. Select and upload any image from `project inputs/2D/` (e.g. `2D/glioma/Tr-gl_157.jpg`).
4. Verify:
   - File upload is accepted with HTTP 200.
   - File type is detected as `2d_image`.
   - File validation succeeds (`valid: true`).
   - MRI image preview renders in the UI.
5. If no trained classification checkpoint is loaded in `config.yaml`, the system transparently reports:
   *"Classification model not loaded"*.

### 2. 3D NIfTI Workflow Testing
1. Upload `project inputs/3D/anatomical_reference/anatomical.nii` or `project inputs/3D/mni152_reference/mni152.nii.gz` via the web interface or `/api/upload`.
2. Verify:
   - Upload is accepted with HTTP 200 and assigned a unique `volume_id`.
   - File type is recognized as `3d_volume`.
   - Volumetric dimensions, voxel spacing, and orientation are extracted and displayed.
   - Multi-planar orthogonal slice endpoints return valid PNG streams:
     - Axial slice: `GET /api/volume/<volume_id>/slice/axial/<index>`
     - Coronal slice: `GET /api/volume/<volume_id>/slice/coronal/<index>`
     - Sagittal slice: `GET /api/volume/<volume_id>/slice/sagittal/<index>`
3. The interface clearly reports:
   *"3D MRI volume loaded successfully"*, and separately confirms:
   *"3D segmentation model not loaded"*.

### 3. Invalid Input / Error Handling Testing
1. Upload `project inputs/invalid_inputs/unsupported_file.txt`.
2. Verify:
   - The API immediately rejects the upload with HTTP 400.
   - Response message: *"Unsupported file type. Use JPG, JPEG, PNG, NIfTI, or NIfTI.GZ."*
   - The server does not crash.

---

## Automated Validation

To run an automated verification of all inputs in this directory:

```bash
.venv/bin/python scripts/validate_project_inputs.py
```

This verifies folder structure, image formats, NIfTI headers, multi-planar slicing, and SHA256 checksums against `input_manifest.json`.
