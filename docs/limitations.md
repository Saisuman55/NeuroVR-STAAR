# Scope, Known Limitations, and Academic Disclaimer

This document clearly distinguishes between verified implemented features, current technical limitations, and planned future work for the NeuroVR / STAAR project.

---

## 1. Verified Implemented Capabilities

The following features are implemented, actively tested, and verified:
- **2D MRI Processing:** Uploading, format validation, and inspection of brain MRI images ($512 \times 512$ JPEG/PNG).
- **EfficientNet-B4 Architecture:** 4-class classification pipeline (`glioma`, `meningioma`, `notumor`, `pituitary`) with model registry loading.
- **3D NIfTI Volume Handling:** Loading uncompressed (`.nii`) and gzip-compressed (`.nii.gz`) neuroimaging volumes.
- **Multi-Planar Orthogonal Slicing:** Dynamic slice extraction across Axial, Coronal, and Sagittal planes with normalization to 8-bit PNG images.
- **Affine & Orientation Extraction:** Determining physical dimensions, voxel spacing, and standard coordinate systems (RAS, LAS).
- **Error Handling & Security:** Safe rejection of corrupted images, empty files, and unsupported formats without crashing the server.
- **Automated Quality Assurance:** 143 passing unit/integration tests and automated CI workflows.

---

## 2. Current Limitations & Technical Constraints

1. **In-Memory Session State:** Volume uploads are tracked in memory during active server execution; restarting the server clears uploaded volume IDs.
2. **Model Weights Separation:** Heavy binary checkpoints (`checkpoints/*.pt`) are excluded from Git to comply with repository quotas. When checkpoints are not mounted, the API honestly returns `status: "unavailable"`.
3. **Hardware Acceleration:** Native MPS is optimized for macOS. On systems without CUDA, processing defaults gracefully to CPU.

---

## 3. Future Work & Planned Modules

- **3D BraTS Segmentation:** Training and integrating full 3D U-Net / V-Net segmentation weights for whole-tumor and sub-region masking.
- **Interactive 3D WebXR / VR Viewer:** Rendering volumetric meshes (`.glb`/`.obj`) directly in WebXR headsets or Three.js browser viewports.
- **Automated Clinical Reporting:** Exporting structured diagnostic summaries with quantitative volumetric statistics once segmentation models are trained.

---

## 4. Academic Disclaimer

> **IMPORTANT NOTICE:**  
> The NeuroVR / STAAR system is an academic research and engineering prototype developed for educational demonstration purposes. It is **not** a certified clinical diagnostic system, medical device, or software-as-a-medical-device (SaMD). It should not be used as a substitute for professional clinical diagnosis or patient management.
