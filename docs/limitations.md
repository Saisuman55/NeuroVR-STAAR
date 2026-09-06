# Scope, Known Limitations, and Academic Disclaimer

This document clearly distinguishes between verified implemented features, current technical limitations, prerequisite-dependent capabilities, and academic research prototype boundaries for the NeuroVR / STAAR project.

---

## 1. Feature Status Model

| Feature | Status | Requirement | Behavior & Description |
| :--- | :---: | :--- | :--- |
| **2D EfficientNet-B4 Inference** | ⚠️ **Requires Checkpoint** | Compatible trained model checkpoint | Supported by application architecture and model registry. When unmounted, UI displays *"Classification model not loaded. Connect a trained EfficientNet-B4 checkpoint to enable analysis."* and API returns `HTTP 503`. |
| **3D BraTS U-Net Segmentation** | ⚠️ **Not Imported** | Dataset / model resources must be imported | 3D U-Net baseline architecture implemented; BraTS benchmark data and trained segmentation checkpoints are not currently imported into this environment. |
| **3D Medical Mesh Rendering** | ⚠️ **Requires Mask** | Valid segmentation mask | Three.js viewport and controls are ready. Medical brain and tumor meshes require a verified segmentation mask before rendering. UI displays *"3D tumor mesh unavailable"*. Synthetic test geometry is labelled as demo object. |
| **Quantitative Measurements** | ⚠️ **Requires Mask** | Verified segmentation mask | Tumor volume ($mm^3$ / $cm^3$), voxel count, centroid, spatial bounding box, and Dice/IoU read *"Not available"* until a verified segmentation mask exists. |
| **Visual Evidence Overlays** | ⚠️ **Requires Analysis** | Valid analysis output | Tabs for Binary Mask, Tumor Overlay, Tumor Contour, and JET Heatmap activate only when valid model analysis outputs are available. |
| **Medical PDF Report Export** | ⚠️ **Requires Analysis** | Verified completed analysis | Guarded endpoint returning `HTTP 503` (`analysis_required`) until verified analysis results exist. |

---

## 2. Currently Implemented & Verified Capabilities

The following features are implemented, actively tested, and verified:
- **Local NeuroVR Web Interface:** Responsive dark-themed scientific interface with live environment hardware detection.
- **2D MRI Processing:** Uploading, format validation, and inspection of brain MRI images ($512 \times 512$ JPEG/PNG).
- **EfficientNet-B4 Architecture & Registry:** 4-class classification pipeline (`glioma`, `meningioma`, `notumor`, `pituitary`) with model registry loading.
- **3D NIfTI Volume Handling:** Loading uncompressed (`.nii`) and gzip-compressed (`.nii.gz`) neuroimaging volumes.
- **Multi-Planar Orthogonal Slicing:** Dynamic slice extraction across Axial, Coronal, and Sagittal planes with normalization to 8-bit PNG images.
- **Affine & Orientation Extraction:** Determining physical dimensions, voxel spacing, and standard coordinate systems (RAS, LAS).
- **Three.js 3D Viewport:** Interactive 3D scene with orbit controls, reset, fullscreen, and synthetic demo geometry controls.
- **Error Handling & Security:** Safe rejection of corrupted images, empty files, and unsupported formats without crashing the server.
- **Automated Quality Assurance:** 143 passing unit/integration tests and automated CI workflows.

---

## 3. Current Limitations & Technical Constraints

1. **In-Memory Session State:** Volume uploads are tracked in memory during active server execution; restarting the server clears uploaded volume IDs.
2. **Model Weights Separation:** Heavy binary checkpoints (`checkpoints/*.pt`) are excluded from Git to comply with repository quotas. When checkpoints are not mounted, the API honestly returns `HTTP 503` with `status: "requires_checkpoint"`.
3. **Hardware Acceleration:** Native MPS is optimized for macOS. On systems without CUDA, processing defaults gracefully to CPU.
4. **No Fabricated Data:** The application strictly enforces honesty: no synthetic tumor masks, fake measurements, or simulated diagnostic reports are ever presented as patient findings.

---

## 4. Academic Research Prototype Disclaimer

> **IMPORTANT DISCLAIMER:**  
> NeuroVR is a B.Tech Computer Science and Engineering academic research prototype.
>
> It is **NOT**:
> - A clinically validated medical device
> - A diagnostic tool
> - A substitute for radiologist or physician interpretation
>
> All workflows, architectures, models, and measurements are developed strictly for educational demonstration, engineering evaluation, and computational research evaluation.
