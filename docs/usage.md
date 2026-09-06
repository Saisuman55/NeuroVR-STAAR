# Usage Guide — NeuroVR / STAAR

This guide explains how to start the NeuroVR web application, upload 2D MRI scans, inspect 3D NIfTI volumes, and interact with the REST API.

---

## 1. Starting the Application

Activate your virtual environment and start the Flask prototype server:

```bash
# Set Apple Silicon MPS fallback if running on macOS
export PYTORCH_ENABLE_MPS_FALLBACK=1

python app.py
```

The server will initialize on:
`http://127.0.0.1:5000`

---

## 2. Using the Web Interface

Open your browser and navigate to:
`http://127.0.0.1:5000/`

### 2D Brain MRI Upload & Inspection
1. Under the **Upload** section, select a 2D MRI scan from `project inputs/2D/` (e.g., `project inputs/2D/glioma/Tr-gl_157.jpg`).
2. The interface will upload and validate the file, confirming valid image resolution ($512 \times 512$) and format.

### 3D Volumetric NIfTI Exploration
1. Select a 3D NIfTI file (e.g., `project inputs/3D/anatomical_reference/anatomical.nii` or `mni152.nii.gz`).
2. The backend extracts volumetric metadata (shape, voxel dimensions, orientation).
3. The orthogonal slice viewer displays real-time 2D slices along Axial, Coronal, and Sagittal planes.

---

## 3. Testing with cURL / REST API

### Health Check
```bash
curl -X GET http://127.0.0.1:5000/api/health
```

### Uploading a 3D NIfTI Volume
```bash
curl -X POST -F "file=@project inputs/3D/anatomical_reference/anatomical.nii" http://127.0.0.1:5000/api/upload
```
Response includes a unique `volume_id` (e.g., `abc123volumeid`).

### Fetching Orthogonal Slices
- **Axial:**
  ```bash
  curl -X GET http://127.0.0.1:5000/api/volume/<volume_id>/slice/axial/12 --output axial_slice.png
  ```
- **Coronal:**
  ```bash
  curl -X GET http://127.0.0.1:5000/api/volume/<volume_id>/slice/coronal/20 --output coronal_slice.png
  ```
- **Sagittal:**
  ```bash
  curl -X GET http://127.0.0.1:5000/api/volume/<volume_id>/slice/sagittal/16 --output sagittal_slice.png
  ```

### Testing Error Handling
Upload an invalid or unsupported file:
```bash
curl -X POST -F "file=@project inputs/invalid_inputs/unsupported_file.txt" http://127.0.0.1:5000/api/upload
```
Expected output: `HTTP 400 Bad Request` (`{"status": "error", "message": "Unsupported file type..."}`).
