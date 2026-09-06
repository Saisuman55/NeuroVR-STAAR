"""Step 14 End-to-End System Demonstration and Functional Verification Script.

Executes and logs every phase:
- Server check & Health verification
- 2D MRI Workflow testing (Glioma, Meningioma, No Tumor, Pituitary)
- 3D NIfTI Workflow testing (anatomical.nii, mni152.nii.gz)
- Orthogonal Slicing verification (Axial, Coronal, Sagittal) and saving real slice images
- Invalid input testing (unsupported files, invalid images, invalid NIfTI)
- Full API endpoints verification matrix
- Exporting structured JSON test evidence
"""

import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
import uuid

BASE_URL = "http://127.0.0.1:5000"
OUTPUT_DIR = Path("project outputs/step_14_demo")
RESULTS_DIR = OUTPUT_DIR / "test_results"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def http_get(endpoint: str) -> tuple[int, dict | bytes, str]:
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()
            if "application/json" in content_type:
                return resp.status, json.loads(data.decode("utf-8")), content_type
            return resp.status, data, content_type
    except urllib.error.HTTPError as e:
        data = e.read()
        try:
            return e.code, json.loads(data.decode("utf-8")), e.headers.get("Content-Type", "")
        except Exception:
            return e.code, data, e.headers.get("Content-Type", "")


def http_post_json(endpoint: str, payload: dict | None = None) -> tuple[int, dict, str]:
    url = f"{BASE_URL}{endpoint}"
    data = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            content_type = resp.headers.get("Content-Type", "")
            return resp.status, json.loads(resp.read().decode("utf-8")), content_type
    except urllib.error.HTTPError as e:
        data = e.read()
        try:
            return e.code, json.loads(data.decode("utf-8")), e.headers.get("Content-Type", "")
        except Exception:
            return e.code, {"error": data.decode("utf-8")}, e.headers.get("Content-Type", "")


def http_upload(endpoint: str, filepath: Path) -> tuple[int, dict, str]:
    url = f"{BASE_URL}{endpoint}"
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    
    with open(filepath, "rb") as f:
        file_bytes = f.read()
        
    filename = filepath.name
    content_disposition = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'
    footer = f"\r\n--{boundary}--\r\n"
    
    body = content_disposition.encode("utf-8") + file_bytes + footer.encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    
    try:
        with urllib.request.urlopen(req) as resp:
            content_type = resp.headers.get("Content-Type", "")
            return resp.status, json.loads(resp.read().decode("utf-8")), content_type
    except urllib.error.HTTPError as e:
        data = e.read()
        try:
            return e.code, json.loads(data.decode("utf-8")), e.headers.get("Content-Type", "")
        except Exception:
            return e.code, {"error": data.decode("utf-8")}, e.headers.get("Content-Type", "")


def run_full_verification():
    report = {
        "project": "NeuroVR / STAAR",
        "step": "Step 14 — End-to-End System Demonstration and Functional Verification",
        "server_url": BASE_URL,
        "phases": {}
    }
    
    print("\n=======================================================")
    print("PHASE 2: VERIFY SYSTEM HEALTH & APPLICATION STARTUP")
    print("=======================================================")
    code, body, _ = http_get("/api/health")
    print(f"GET /api/health -> HTTP {code}: {body}")
    assert code == 200 and body.get("status") == "ok"
    report["phases"]["phase_2_health"] = {
        "status_code": code,
        "response": body,
        "verified": True
    }
    
    print("\n=======================================================")
    print("PHASE 3: COMPLETE 2D MRI WORKFLOW TEST")
    print("=======================================================")
    two_d_classes = ["glioma", "meningioma", "notumor", "pituitary"]
    two_d_results = []
    
    for cls in two_d_classes:
        folder = Path(f"project inputs/2D/{cls}")
        if not folder.exists():
            folder = Path(f"project inputs/2D_MRI/{cls}")
        files = sorted(folder.glob("*.jpg"))
        assert len(files) > 0, f"No files found in {folder}"
        test_file = files[0]
        
        print(f"\nTesting 2D MRI upload for class '{cls}': {test_file.name}")
        status_code, upload_resp, _ = http_upload("/api/upload", test_file)
        print(f"Upload -> HTTP {status_code}: {upload_resp.get('status')}")
        
        # Test analysis endpoint
        ana_code, ana_resp, _ = http_post_json("/api/analyze")
        print(f"Analyze -> HTTP {ana_code}: status={ana_resp.get('status')}, message={ana_resp.get('message')}")
        
        # Test status endpoint
        stat_code, stat_resp, _ = http_get("/api/status")
        
        # Test results endpoint
        res_code, res_resp, _ = http_get("/api/results")
        
        item_res = {
            "class": cls,
            "filename": test_file.name,
            "filepath": str(test_file),
            "upload_status_code": status_code,
            "upload_response": upload_resp,
            "analyze_status_code": ana_code,
            "analyze_response": ana_resp,
            "app_status": stat_resp,
            "result_status": res_resp.get("status"),
            "verified": (status_code == 200 and upload_resp.get("upload", {}).get("file_type") == "2d_image")
        }
        two_d_results.append(item_res)
        
    report["phases"]["phase_3_2d_workflow"] = two_d_results
    
    print("\n=======================================================")
    print("PHASE 4: COMPLETE 3D NIFTI WORKFLOW & ORTHOGONAL SLICING")
    print("=======================================================")
    three_d_targets = [
        ("anatomical.nii", Path("project inputs/3D/anatomical_reference/anatomical.nii")),
        ("mni152.nii.gz", Path("project inputs/3D/mni152_reference/mni152.nii.gz"))
    ]
    three_d_results = []
    
    for name, path in three_d_targets:
        if not path.exists():
            # fallback path check
            alt_path = Path("project inputs/3D_MRI") / ("nifti_anatomical/anatomical.nii" if "anatomical" in name else "nifti_brain_template/mni152.nii.gz")
            if alt_path.exists():
                path = alt_path
        print(f"\nTesting 3D NIfTI upload: {name} ({path})")
        status_code, upload_resp, _ = http_upload("/api/upload", path)
        print(f"Upload -> HTTP {status_code}: {upload_resp.get('status')}")
        assert status_code == 200
        
        upload_data = upload_resp.get("upload", {})
        volume_id = upload_data.get("volume_id")
        meta = upload_data.get("metadata", {})
        print(f"Volume ID: {volume_id}")
        print(f"Shape: {meta.get('shape')}, Spacing: {meta.get('spacing')}, Orientation: {meta.get('orientation')}")
        
        # Test volume metadata endpoint
        meta_code, meta_resp, _ = http_get(f"/api/volume/{volume_id}/metadata")
        print(f"GET /api/volume/{volume_id}/metadata -> HTTP {meta_code}")
        
        # Test Orthogonal slices: Axial, Coronal, Sagittal
        slices_info = {}
        shape = meta.get("shape", [30, 30, 30])
        # Central slice indices
        mid_sagittal = shape[0] // 2
        mid_coronal = shape[1] // 2
        mid_axial = shape[2] // 2
        
        slice_targets = [
            ("axial", mid_axial),
            ("coronal", mid_coronal),
            ("sagittal", mid_sagittal)
        ]
        
        for axis, idx in slice_targets:
            slice_ep = f"/api/volume/{volume_id}/slice/{axis}/{idx}"
            s_code, s_bytes, s_type = http_get(slice_ep)
            print(f"Slice ({axis}, index {idx}) -> HTTP {s_code}, bytes={len(s_bytes)}, type={s_type}")
            assert s_code == 200
            assert "image/png" in s_type
            assert len(s_bytes) > 0
            
            # Save genuine slice image artifact
            slice_filename = f"slice_{name.replace('.', '_')}_{axis}_{idx}.png"
            slice_path = SCREENSHOTS_DIR / slice_filename
            with open(slice_path, "wb") as f_slice:
                f_slice.write(s_bytes)
                
            slices_info[axis] = {
                "index": idx,
                "status_code": s_code,
                "mimetype": s_type,
                "byte_size": len(s_bytes),
                "saved_to": str(slice_path)
            }
            
        three_d_results.append({
            "name": name,
            "volume_id": volume_id,
            "metadata": meta,
            "slices": slices_info,
            "verified": True
        })
        
    report["phases"]["phase_4_3d_workflow"] = three_d_results
    
    print("\n=======================================================")
    print("PHASE 6: INVALID INPUT AND ERROR HANDLING")
    print("=======================================================")
    invalid_files = [
        Path("project inputs/invalid_inputs/unsupported_file.txt"),
        Path("project inputs/invalid_inputs/invalid_image.txt"),
        Path("project inputs/invalid_inputs/invalid_nifti.txt"),
        Path("project inputs/invalid_inputs/invalid_images/corrupted_image.jpg"),
        Path("project inputs/invalid_inputs/invalid_images/empty_image.jpg"),
        Path("project inputs/invalid_inputs/invalid_nifti/corrupted_volume.nii"),
        Path("project inputs/invalid_inputs/invalid_nifti/empty_volume.nii")
    ]
    invalid_results = []
    
    for inv_path in invalid_files:
        if not inv_path.exists():
            continue
        print(f"\nTesting rejection of invalid input: {inv_path.name}")
        status_code, resp_body, _ = http_upload("/api/upload", inv_path)
        print(f"Upload -> HTTP {status_code}: {resp_body.get('message')}")
        assert status_code == 400
        assert resp_body.get("status") == "error"
        
        invalid_results.append({
            "filename": inv_path.name,
            "path": str(inv_path),
            "status_code": status_code,
            "error_message": resp_body.get("message"),
            "server_safe": True
        })
        
    # Verify server is still completely responsive after invalid uploads
    h_code, h_body, _ = http_get("/api/health")
    assert h_code == 200
    print("\nPost-error server health verified: 200 OK (no crash)")
    report["phases"]["phase_6_invalid_inputs"] = invalid_results
    
    print("\n=======================================================")
    print("PHASE 7: FULL API ENDPOINT VERIFICATION MATRIX")
    print("=======================================================")
    api_matrix = [
        {"endpoint": "/api/health", "method": "GET", "expected_code": 200, "description": "System health probe"},
        {"endpoint": "/api/models/status", "method": "GET", "expected_code": 200, "description": "Models registry status"},
        {"endpoint": "/api/datasets", "method": "GET", "expected_code": 200, "description": "Overall dataset status"},
        {"endpoint": "/api/datasets/classification", "method": "GET", "expected_code": 200, "description": "2D classification dataset report"},
        {"endpoint": "/api/datasets/segmentation", "method": "GET", "expected_code": 200, "description": "3D segmentation dataset report"},
        {"endpoint": "/api/system/status", "method": "GET", "expected_code": 200, "description": "Hardware and environment info"},
        {"endpoint": "/api/input/2d", "method": "GET", "expected_code": 200, "description": "Demo 2D inputs catalog"},
        {"endpoint": "/api/input/3d", "method": "GET", "expected_code": 200, "description": "Demo 3D inputs catalog"},
        {"endpoint": "/api/samples/classification?per_class=2", "method": "GET", "expected_code": 200, "description": "Classification samples list"},
        {"endpoint": "/api/status", "method": "GET", "expected_code": 200, "description": "Active session analysis status"},
        {"endpoint": "/api/results", "method": "GET", "expected_code": 200, "description": "Analysis results response"},
        {"endpoint": "/api/report", "method": "POST", "expected_code": 503, "description": "Report generator unavailable guard"}
    ]
    
    endpoint_results = []
    for item in api_matrix:
        ep = item["endpoint"]
        method = item["method"]
        exp = item["expected_code"]
        if method == "GET":
            c, b, _ = http_get(ep)
        else:
            c, b, _ = http_post_json(ep)
        passed = (c == exp)
        print(f"{method} {ep} -> HTTP {c} (Expected {exp}) [{'PASS' if passed else 'FAIL'}]")
        assert passed
        endpoint_results.append({
            "endpoint": ep,
            "method": method,
            "expected_code": exp,
            "actual_code": c,
            "description": item["description"],
            "status": "PASS" if passed else "FAIL"
        })
        
    report["phases"]["phase_7_api_matrix"] = endpoint_results
    
    # Save test results JSON
    results_json_path = RESULTS_DIR / "verification_results.json"
    with open(results_json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull verification results saved to: {results_json_path}")
    return report

if __name__ == "__main__":
    run_full_verification()
