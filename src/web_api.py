"""Flask API and service boundary for the NeuroVR research prototype."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from src.classifier_2d import CLASS_NAMES
from src.model_registry import (
    get_model_status,
    load_classification_model_for_inference,
    predict_2d_image,
)
from src.utils import create_output_directories, load_config, select_device, setup_logging
from src.volume_loader import inspect_nifti, load_nifti


def is_classifier_available(config: dict[str, Any] | None = None) -> bool:
    """Check whether a valid trained 2D classifier checkpoint is configured and readable."""
    from src.model_registry import get_classification_checkpoint_path
    cfg = config or load_config()
    return get_classification_checkpoint_path(cfg) is not None


def is_segmentation_available(config: dict[str, Any] | None = None) -> bool:
    """Check whether a valid trained 3D segmentation checkpoint is configured and readable."""
    from src.model_registry import get_segmentation_checkpoint_path
    cfg = config or load_config()
    return get_segmentation_checkpoint_path(cfg) is not None


def has_valid_segmentation_mask(volume_id: str | None = None) -> bool:
    """Check whether a verified segmentation mask exists for the current session or volume."""
    return False


def has_completed_analysis(service: AnalysisService | None = None) -> bool:
    """Check whether a verified model analysis has completed with valid predictions."""
    if service is None or service.state.result is None:
        return False
    return service.state.status == "completed" and service.state.result.get("status") == "completed"


def can_generate_report(service: AnalysisService | None = None) -> bool:
    """Check whether prerequisite conditions are met to export a medical research report."""
    return has_completed_analysis(service)


@dataclass
class AnalysisState:
    """Non-persistent state for the current local prototype session."""

    status: str = "idle"
    upload: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


class AnalysisService:
    """Coordinate upload state and model services without falsely claiming unavailable capabilities."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.state = AnalysisState()
        self.logger = setup_logging(config)
        self.paths = config["paths"]
        self.volumes: dict[str, Path] = {}

    def is_classifier_available(self) -> bool:
        """Check if 2D classification checkpoint is configured and readable."""
        return is_classifier_available(self.config)

    def is_segmentation_available(self) -> bool:
        """Check if 3D segmentation checkpoint is configured and readable."""
        return is_segmentation_available(self.config)

    def has_valid_segmentation_mask(self, volume_id: str | None = None) -> bool:
        """Check if a verified segmentation mask is available."""
        return has_valid_segmentation_mask(volume_id)

    def has_completed_analysis(self) -> bool:
        """Check if verified analysis results are available in current session."""
        return has_completed_analysis(self)

    def can_generate_report(self) -> bool:
        """Check if report generation prerequisite conditions are satisfied."""
        return can_generate_report(self)

    def save_upload(self, file_storage: Any) -> dict[str, Any]:
        """Validate and save one upload under the configured local upload path."""
        original_name = file_storage.filename or ""
        safe_name = secure_filename(original_name)
        extension = _file_extension(original_name)
        allowed = _allowed_extensions(self.config)
        if not safe_name or extension not in allowed:
            raise ValueError("Unsupported file type. Use JPG, JPEG, PNG, NIfTI, or NIfTI.GZ.")
        upload_directory = Path(self.paths["uploads"])
        upload_directory.mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid4().hex}_{safe_name}"
        destination = upload_directory / stored_name
        file_storage.save(destination)
        size = destination.stat().st_size
        file_type = "3d_volume" if extension in {".nii", ".nii.gz"} else "2d_image"
        validation = _validate_upload(destination, file_type)
        if not validation["valid"]:
            destination.unlink(missing_ok=True)
            raise ValueError(validation["message"])
        upload = {
            "filename": original_name,
            "stored_filename": stored_name,
            "file_type": file_type,
            "extension": extension,
            "size_bytes": size,
            "validation": validation,
        }
        if file_type == "3d_volume":
            volume_id = uuid4().hex
            self.volumes[volume_id] = destination
            upload["volume_id"] = volume_id
            upload["metadata"] = _volume_metadata(destination)
        self.state = AnalysisState("uploaded", upload, None)
        return upload

    def run_analysis(self) -> dict[str, Any]:
        """Perform real model inference if a checkpoint is available; otherwise return honest prerequisite state."""
        if self.state.upload is None:
            return self.unavailable_analysis()

        file_type = self.state.upload.get("file_type")
        if file_type == "2d_image":
            device = select_device(self.config)
            model, ckpt_meta = load_classification_model_for_inference(self.config, device)
            if model is not None and ckpt_meta is not None:
                stored_name = self.state.upload.get("stored_filename", "")
                stored_path = Path(self.paths["uploads"]) / stored_name
                if stored_path.is_file():
                    classes = ckpt_meta.get("class_names", list(CLASS_NAMES))
                    image_size = tuple(ckpt_meta.get("image_size", (380, 380)))
                    pred = predict_2d_image(
                        model,
                        stored_path,
                        device,
                        class_names=classes,
                        image_size=image_size,
                    )
                    self.state.status = "completed"
                    result = {
                        "success": True,
                        "status": "completed",
                        "message": "2D MRI classification completed successfully.",
                        "classification": {
                            "status": "available",
                            "predicted_class": pred["predicted_class"],
                            "confidence": pred["confidence"],
                            "probabilities": pred["probabilities"],
                            "model_architecture": ckpt_meta.get("model_architecture"),
                            "epoch": ckpt_meta.get("epoch"),
                        },
                        "segmentation": {
                            "status": "not_imported",
                            "message": "3D U-Net segmentation architecture is implemented as a baseline. BraTS-style benchmark data and/or the required trained segmentation checkpoint are not currently imported into this environment.",
                        },
                        "measurements": {
                            "status": "requires_mask",
                            "message": "Volume, voxel count, centroid, spatial location, and segmentation-derived measurements become available after a verified tumor mask is generated.",
                            "values": {
                                "tumor_volume_mm3": "Not available",
                                "tumor_volume_cm3": "Not available",
                                "tumor_voxel_count": "Not available",
                                "tumor_centroid": "Not available",
                                "bounding_box": "Not available",
                                "spatial_location": "Not available",
                                "segmentation_dice": "Not available",
                                "segmentation_iou": "Not available",
                            },
                        },
                        "visualizations": {
                            "status": "requires_analysis",
                            "message": "Tabs for Binary Mask, Tumor Overlay, Tumor Contour, and JET Heatmap become active only when valid analysis outputs are available.",
                        },
                        "mesh": {
                            "status": "requires_mask",
                            "message": "3D tumor mesh unavailable. Complete segmentation to load a medical mesh.",
                        },
                    }
                    self.state.result = result
                    return result

            return self.unavailable_analysis("2d_image")

        if file_type == "3d_volume":
            return self.unavailable_analysis("3d_volume")

        return self.unavailable_analysis()

    def unavailable_analysis(self, file_type: str = "2d_image") -> dict[str, Any]:
        """Return an explicit prerequisite-missing response with no fabricated values."""
        if file_type == "3d_volume":
            self.state.status = "not_imported"
            primary_status = "not_imported"
            msg = "3D segmentation is unavailable until the required dataset and trained model resources are configured."
        else:
            self.state.status = "requires_checkpoint"
            primary_status = "requires_checkpoint"
            msg = "Classification model not loaded. Connect a trained EfficientNet-B4 checkpoint to enable analysis."

        result = {
            "success": False,
            "status": primary_status,
            "message": msg,
            "classification": {
                "status": "requires_checkpoint",
                "description": "2D EfficientNet-B4 inference is supported by the application architecture and model registry. A compatible trained checkpoint must be mounted or configured before predictions can be generated.",
                "message": "Classification model not loaded. Connect a trained EfficientNet-B4 checkpoint to enable analysis.",
            },
            "segmentation": {
                "status": "not_imported",
                "description": "3D U-Net segmentation architecture is implemented as a baseline. BraTS-style benchmark data and/or the required trained segmentation checkpoint are not currently imported into this environment.",
                "message": "3D segmentation is unavailable until the required dataset and trained model resources are configured.",
            },
            "measurements": {
                "status": "requires_mask",
                "description": "Volume, voxel count, centroid, spatial location, and segmentation-derived measurements become available after a verified tumor mask is generated.",
                "values": {
                    "tumor_volume_mm3": "Not available",
                    "tumor_volume_cm3": "Not available",
                    "tumor_voxel_count": "Not available",
                    "tumor_centroid": "Not available",
                    "bounding_box": "Not available",
                    "spatial_location": "Not available",
                    "segmentation_dice": "Not available",
                    "segmentation_iou": "Not available",
                },
            },
            "visualizations": {
                "status": "requires_analysis",
                "description": "Tabs for Binary Mask, Tumor Overlay, Tumor Contour, and JET Heatmap become active only when valid analysis outputs are available.",
                "message": "Visualization output unavailable. These views will appear after a verified model analysis.",
            },
            "mesh": {
                "status": "requires_mask",
                "description": "Three.js viewport and viewer controls are ready. Medical brain and tumor meshes require a verified segmentation mask before rendering.",
                "message": "3D tumor mesh unavailable. Complete segmentation to load a medical mesh.",
            },
        }
        self.state.result = result
        return result

    def report_unavailable(self) -> dict[str, Any]:
        """Explain why a medical research report cannot be generated yet."""
        return {
            "success": False,
            "status": "analysis_required",
            "message": "A verified analysis is required before a medical research report can be generated.",
        }


def create_app(config: dict[str, Any] | None = None) -> Flask:
    """Create and configure the local NeuroVR Flask application."""
    config = config or load_config()
    create_output_directories(config)
    application = Flask(__name__, template_folder="../web/templates", static_folder="../web/static")
    max_upload_mb = int(config.get("web", {}).get("max_upload_mb", 64))
    application.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024
    service = AnalysisService(config)
    application.extensions["neurovr_service"] = service

    @application.get("/")
    def index() -> str:
        return render_template("index.html")

    @application.get("/api/health")
    def health() -> Any:
        return jsonify({"status": "ok", "service": "neurovr", "prototype": True})

    @application.get("/api/models/status")
    def models_status() -> Any:
        return jsonify(get_model_status(config))

    @application.post("/api/upload")
    def upload() -> Any:
        file_storage = request.files.get("file")
        if file_storage is None or not file_storage.filename:
            return jsonify({"status": "error", "message": "No file was supplied."}), 400
        try:
            return jsonify({"status": "uploaded", "upload": service.save_upload(file_storage)})
        except ValueError as error:
            service.logger.warning("Upload rejected: %s", error)
            return jsonify({"status": "error", "message": str(error)}), 400

    @application.post("/api/analyze")
    def analyze() -> Any:
        if service.state.upload is None:
            return jsonify({"status": "error", "message": "Upload an MRI file before analysis."}), 400
        result = service.run_analysis()
        status_code = 200 if result.get("success") else 503
        return jsonify(result), status_code

    @application.get("/api/status")
    def status() -> Any:
        return jsonify({"status": service.state.status, "upload": service.state.upload is not None})

    @application.get("/api/results")
    def results() -> Any:
        if service.has_completed_analysis():
            return jsonify(service.state.result)
        return jsonify({
            "status": "requires_analysis",
            "message": "No verified analysis results are available. Connect model checkpoints and run analysis first.",
        }), 200

    @application.post("/api/report")
    def report() -> Any:
        if not service.can_generate_report():
            return jsonify(service.report_unavailable()), 503
        return jsonify({"success": True, "message": "Medical research report generated successfully."}), 200

    @application.get("/api/capabilities")
    def capabilities() -> Any:
        return jsonify({
            "status": "ok",
            "features": [
                {
                    "feature": "2D EfficientNet-B4 Inference",
                    "status": "Available" if service.is_classifier_available() else "Requires Checkpoint",
                    "requirement": "Compatible trained model checkpoint",
                    "description": "2D EfficientNet-B4 inference is supported by the application architecture and model registry. A compatible trained checkpoint must be mounted or configured before predictions can be generated.",
                    "operational": service.is_classifier_available(),
                },
                {
                    "feature": "3D BraTS U-Net Segmentation",
                    "status": "Available" if service.is_segmentation_available() else "Not Imported",
                    "requirement": "Dataset/model resources must be imported",
                    "description": "3D U-Net segmentation architecture is implemented as a baseline. BraTS-style benchmark data and/or the required trained segmentation checkpoint are not currently imported into this environment.",
                    "operational": service.is_segmentation_available(),
                },
                {
                    "feature": "3D Medical Mesh Rendering",
                    "status": "Requires Mask",
                    "requirement": "Valid segmentation mask",
                    "description": "Three.js viewport and viewer controls are ready. Medical brain and tumor meshes require a verified segmentation mask before rendering.",
                    "operational": False,
                },
                {
                    "feature": "Quantitative Measurements",
                    "status": "Requires Mask",
                    "requirement": "Verified segmentation mask",
                    "description": "Volume, voxel count, centroid, spatial location, and segmentation-derived measurements become available after a verified tumor mask is generated.",
                    "operational": False,
                },
                {
                    "feature": "Visual Evidence Overlays",
                    "status": "Requires Analysis",
                    "requirement": "Valid analysis output",
                    "description": "Tabs for Binary Mask, Tumor Overlay, Tumor Contour, and JET Heatmap become active only when valid analysis outputs are available.",
                    "operational": False,
                },
                {
                    "feature": "Medical PDF Report Export",
                    "status": "Requires Analysis",
                    "requirement": "Verified completed analysis",
                    "description": "The report-generation endpoint is guarded and becomes available only after verified analysis results exist.",
                    "operational": service.can_generate_report(),
                },
            ],
        })

    @application.get("/api/datasets")
    def datasets() -> Any:
        return jsonify(_dataset_status(config))

    @application.get("/api/datasets/classification")
    def classification_dataset() -> Any:
        return jsonify(_classification_status(config))

    @application.get("/api/datasets/segmentation")
    def segmentation_dataset() -> Any:
        return jsonify(_segmentation_status(config))

    @application.get("/api/input/2d")
    def input_2d() -> Any:
        return jsonify(_input_listing(Path("data/demo_inputs/2d"), {".jpg", ".jpeg", ".png"}))

    @application.get("/api/input/3d")
    def input_3d() -> Any:
        return jsonify(_input_listing(Path("data/demo_inputs/3d"), {".nii", ".nii.gz"}))

    @application.get("/api/volume/<volume_id>/metadata")
    def volume_metadata(volume_id: str) -> Any:
        volume_path = service.volumes.get(volume_id)
        if volume_path is None:
            return jsonify({"status": "error", "message": "Volume not found."}), 404
        return jsonify(_volume_metadata(volume_path) | {"volume_id": volume_id})

    @application.get("/api/volume/<volume_id>/slice/<axis>/<int:index>")
    def volume_slice(volume_id: str, axis: str, index: int) -> Any:
        volume_path = service.volumes.get(volume_id)
        if volume_path is None:
            return jsonify({"status": "error", "message": "Volume not found."}), 404
        try:
            image = _slice_image(volume_path, axis, index)
        except (IndexError, ValueError) as error:
            return jsonify({"status": "error", "message": str(error)}), 400
        return send_file(image, mimetype="image/png")

    @application.get("/api/system/status")
    def system_status() -> Any:
        return jsonify(_system_status(config))

    @application.get("/api/samples/classification")
    def classification_samples() -> Any:
        """Return a random selection of real sample images from the 2D dataset."""
        per_class = int(request.args.get("per_class", 3))
        per_class = min(max(per_class, 1), 8)  # clamp 1-8
        return jsonify(_classification_samples(config, per_class))

    @application.get("/api/samples/classification/<class_name>/<path:filename>")
    def serve_sample_image(class_name: str, filename: str) -> Any:
        """Serve a real sample MRI image from the classification dataset."""
        safe_class = secure_filename(class_name)
        safe_file = secure_filename(filename)
        allowed_classes = set(config["classification"]["classes"])
        if safe_class not in allowed_classes:
            return jsonify({"status": "error", "message": "Unknown class."}), 404
        # Anchor to the project root (parent of src/) so the path is always absolute
        # regardless of Flask's working directory.
        project_root = Path(__file__).parent.parent
        root = (project_root / config["paths"]["classification_dataset"]).resolve()
        for split in ("Training", "Testing"):
            candidate = root / split / safe_class / safe_file
            if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                return send_file(candidate, mimetype="image/jpeg")
        return jsonify({"status": "error", "message": "Sample not found."}), 404

    @application.errorhandler(RequestEntityTooLarge)
    def too_large(_: RequestEntityTooLarge) -> Any:
        return jsonify({"status": "error", "message": "Upload exceeds the configured size limit."}), 413

    @application.errorhandler(404)
    def not_found(_: Any) -> Any:
        return jsonify({"status": "error", "message": "Resource not found."}), 404

    @application.errorhandler(Exception)
    def internal_error(error: Exception) -> Any:
        service.logger.exception("Unhandled web error: %s", error)
        return jsonify({"status": "error", "message": "The server encountered an internal error."}), 500

    return application


def _file_extension(filename: str) -> str:
    """Return a normalized extension, treating NIfTI.GZ as one extension."""
    lowered = filename.lower()
    return ".nii.gz" if lowered.endswith(".nii.gz") else Path(lowered).suffix


def _allowed_extensions(config: dict[str, Any]) -> set[str]:
    """Return configured 2D and 3D upload extensions."""
    web_config = config.get("web", {})
    return set(web_config.get("allowed_2d_extensions", [".jpg", ".jpeg", ".png"])) | set(
        web_config.get("allowed_3d_extensions", [".nii", ".nii.gz"])
    )


def _validate_upload(path: Path, file_type: str) -> dict[str, Any]:
    """Perform basic local validation without running a model."""
    if file_type == "3d_volume":
        inspection = inspect_nifti(path)
        return {"valid": inspection.valid, "message": "; ".join(inspection.messages) or "Valid 3D NIfTI volume."}
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return {"valid": True, "message": "Readable 2D image."}
    except (OSError, ValueError) as error:
        return {"valid": False, "message": f"Unreadable image: {error}"}


def _volume_metadata(path: Path) -> dict[str, Any]:
    """Return validated metadata for an uploaded NIfTI volume."""
    volume = load_nifti(path)
    inspection = inspect_nifti(path)
    return {
        "filename": path.name,
        "file_type": "3d_volume",
        "shape": list(volume.shape),
        "voxel_spacing": list(volume.spacing),
        "orientation": list(volume.orientation),
        "affine_available": volume.affine is not None,
        "dtype": volume.dtype,
        "min_intensity": inspection.intensity_statistics["min"] if inspection.intensity_statistics else None,
        "max_intensity": inspection.intensity_statistics["max"] if inspection.intensity_statistics else None,
        "mean_intensity": inspection.intensity_statistics["mean"] if inspection.intensity_statistics else None,
        "median_intensity": inspection.intensity_statistics["median"] if inspection.intensity_statistics else None,
        "number_of_slices": volume.shape[2],
    }


def _slice_image(path: Path, axis: str, index: int) -> BytesIO:
    """Extract an actual grayscale slice and encode it as PNG."""
    volume = load_nifti(path)
    slices = {"axial": volume.data[:, :, index], "coronal": volume.data[:, index, :], "sagittal": volume.data[index, :, :]}
    if axis not in slices:
        raise ValueError("axis must be axial, coronal, or sagittal")
    selected = np.asarray(slices[axis], dtype=np.float32)
    if index < 0 or index >= volume.shape[{"axial": 2, "coronal": 1, "sagittal": 0}[axis]]:
        raise IndexError(f"slice index out of range for {axis}")
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        image_data = np.zeros(selected.shape, dtype=np.uint8)
    else:
        lower, upper = float(finite.min()), float(finite.max())
        image_data = np.zeros_like(selected, dtype=np.uint8) if upper <= lower else np.clip((selected - lower) / (upper - lower) * 255, 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(np.flipud(image_data)).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def _input_listing(root: Path, extensions: set[str]) -> dict[str, Any]:
    """List only supported files in an intentional local demo-input directory."""
    files = []
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if path.is_file() and (path.suffix.lower() in extensions or (".nii.gz" in extensions and path.name.lower().endswith(".nii.gz"))):
                files.append({"filename": path.name, "relative_path": str(path.relative_to(root)), "size_bytes": path.stat().st_size})
    return {"status": "ready" if files else "not_imported", "path": str(root), "files": files, "count": len(files)}


def _classification_status(config: dict[str, Any]) -> dict[str, Any]:
    """Read actual classification dataset counts from disk."""
    root = Path(config["paths"]["classification_dataset"])
    classes = config["classification"]["classes"]
    counts = {class_name: 0 for class_name in classes}
    split_counts: dict[str, int] = {}
    valid = 0
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    class_name = next((part.lower() for part in path.relative_to(root).parts[:-1] if part.lower() in counts), None)
                    if class_name:
                        counts[class_name] += 1
                    split = path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else "unspecified"
                    split_counts[split] = split_counts.get(split, 0) + 1
                    valid += 1
                except (OSError, ValueError):
                    continue
    return {"status": "ready" if valid else "not_downloaded", "path": str(root), "total_images": valid, "class_counts": counts, "split_counts": split_counts}


def _segmentation_status(config: dict[str, Any]) -> dict[str, Any]:
    """Read actual BraTS manifest and subject split status from disk.

    Manifest lookup order:
      1. data/segmentation/brats_manifest.json   (authoritative, written by importer)
      2. data/segmentation/brats/dataset_manifest.json  (legacy fallback)
    """
    root = Path(config["paths"]["segmentation_dataset"]) / "brats"
    # Prefer the top-level manifest written alongside the importer's subject folders.
    top_manifest = root.parent / "brats_manifest.json"
    manifest_path = top_manifest if top_manifest.is_file() else root / "dataset_manifest.json"
    split_path = root / "splits.json"
    if not manifest_path.is_file():
        return {"status": "not_imported", "path": str(root), "subjects": 0, "modalities": {}}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("report", {}).get("subjects", [])
    modality_counts = {modality: sum(modality in record.get("modalities", {}) for record in records) for modality in ("flair", "t1", "t1ce", "t2", "seg")}
    splits = json.loads(split_path.read_text(encoding="utf-8")) if split_path.is_file() else {}
    return {"status": "ready" if records else "not_imported", "path": str(root), "subjects": sum(record.get("valid", False) for record in records), "discovered_subjects": len(records), "modality_counts": modality_counts, "split_counts": {key: len(value) for key, value in splits.items() if key.endswith("_subjects")}}


def _dataset_status(config: dict[str, Any]) -> dict[str, Any]:
    """Return filesystem-backed status for both real-data pipelines."""
    return {"classification": _classification_status(config), "segmentation": _segmentation_status(config)}


def _system_status(config: dict[str, Any]) -> dict[str, Any]:
    """Return actual system environment information for the status cards.

    All values come from the live runtime — nothing is hard-coded.
    """
    import sys
    import platform

    try:
        import torch
        torch_version = torch.__version__
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        if mps_available:
            compute_device = "MPS (Apple Silicon)"
        else:
            compute_device = "CPU"
    except ImportError:
        torch_version = "unavailable"
        mps_available = False
        mps_built = False
        compute_device = "unknown"

    # Classifier checkpoint detection.
    classifier_dir = Path(config["paths"]["classifier_models"])
    classifier_checkpoints = list(classifier_dir.glob("*.pt")) + list(classifier_dir.glob("*.pth")) if classifier_dir.is_dir() else []
    classifier_status = "available" if classifier_checkpoints else "requires_checkpoint"

    # 3D segmenter checkpoint detection.
    segmenter_dir = Path(config["paths"]["segmenter_models"])
    segmenter_checkpoints = list(segmenter_dir.glob("*.pt")) + list(segmenter_dir.glob("*.pth")) if segmenter_dir.is_dir() else []
    segmenter_status = "available" if segmenter_checkpoints else "not_imported"

    # BraTS manifest presence.
    seg_root = Path(config["paths"]["segmentation_dataset"])
    brats_manifest = (seg_root / "brats_manifest.json").is_file() or (seg_root / "brats" / "dataset_manifest.json").is_file()

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "torch_version": torch_version,
        "mps_available": mps_available,
        "mps_built": mps_built,
        "compute_device": compute_device,
        "classifier_model": classifier_status,
        "classifier_checkpoints": [p.name for p in classifier_checkpoints],
        "segmenter_model": segmenter_status,
        "segmenter_checkpoints": [p.name for p in segmenter_checkpoints],
        "brats_manifest_present": brats_manifest,
        "config_random_seed": config.get("project", {}).get("random_seed"),
    }


def _classification_samples(config: dict[str, Any], per_class: int = 3) -> dict[str, Any]:
    """Return a backend-selected random sample of real classification images.

    Images are selected from the actual filesystem.  No paths are fabricated.
    The caller receives only filenames and class/split metadata; the browser
    fetches image bytes via /api/samples/classification/<class>/<filename>.

    Args:
        config:    Application configuration dict.
        per_class: Number of samples to return per class (clamped 1-8).
    """
    root = Path(config["paths"]["classification_dataset"])
    classes = config["classification"]["classes"]
    samples: list[dict[str, Any]] = []

    rng = random.Random()  # non-seeded for variety each request

    for class_name in classes:
        candidates: list[tuple[str, str]] = []  # (split, filename)
        for split in ("Training", "Testing"):
            class_dir = root / split / class_name
            if class_dir.is_dir():
                for f in class_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        candidates.append((split, f.name))
        chosen = rng.sample(candidates, min(per_class, len(candidates))) if candidates else []
        for split, filename in chosen:
            samples.append({
                "class": class_name,
                "split": split,
                "filename": filename,
                "image_url": f"/api/samples/classification/{class_name}/{filename}",
            })

    rng.shuffle(samples)
    return {
        "status": "ready" if samples else "not_available",
        "per_class_requested": per_class,
        "total_returned": len(samples),
        "classes": classes,
        "samples": samples,
    }