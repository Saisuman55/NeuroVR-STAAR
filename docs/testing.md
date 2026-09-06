# Testing Guide — NeuroVR / STAAR

This document details the automated testing and validation procedures available in the repository.

---

## 1. Automated Test Suite (pytest)

The project includes a comprehensive regression test suite with 143 unit and integration tests covering:
- Configuration parsing and path resolution
- 2D classification model architectures and utilities
- 3D NIfTI volume loader, orientation detection, and affine matrix extraction
- Dataset manifests, BraTS pipeline validation, and demo inputs
- Flask REST API endpoints, file uploads, slice streaming, and error handling

### Running All Tests
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -m pytest tests/ -q
```
Expected result: `143 passed`

---

## 2. Python Bytecode Compilation Check

Ensure all modules across `src/`, `scripts/`, and `tests/` compile without syntax errors:
```bash
python -m compileall src scripts tests
```
Expected result: Clean completion (exit code 0).

---

## 3. Project Inputs & Integrity Verification

Run the dedicated input validation script:
```bash
python scripts/validate_project_inputs.py
```
This checks:
- 12 real 2D MRI scans across 4 classes (`glioma`, `meningioma`, `notumor`, `pituitary`)
- 2 real 3D NIfTI volumes (`anatomical.nii`, `mni152.nii.gz`)
- Dynamic Axial, Coronal, and Sagittal slicing
- Presence of controlled invalid and unsupported inputs
- SHA-256 hash checksums matching `TEST_INPUTS_MANIFEST.json`

---

## 4. End-to-End Live Verification Script

With the server running (`python app.py`), execute the end-to-end demonstration suite:
```bash
python scripts/run_step14_verification.py
```
This script tests:
- Server health probe
- All 4 classes of 2D MRI uploads
- 3D NIfTI volume uploads and metadata inspection
- Extraction of genuine orthogonal PNG slices to `project outputs/step_14_demo/screenshots/`
- Rejection of corrupted files, 0-byte files, and invalid formats
- 12 public API routes
- Machine-readable result logging to `project outputs/step_14_demo/test_results/`
