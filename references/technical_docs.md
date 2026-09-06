# Official Technical Documentation

These sources document APIs and file formats that may be used by NeuroVR. Documentation describes software behavior and interfaces; it does not establish medical validity or NeuroVR performance.

## Structured reference table

| Reference ID | Organization/Authors | Documentation title | Year/Version | Official URL | Topic | Relevant API or standard | How it relates to NeuroVR | Use |
|---|---|---|---|---|---|---|---|---|
| D01 | PyTorch Contributors | PyTorch Documentation | 2.14 documentation | [PyTorch docs](https://docs.pytorch.org/docs/2.14/) | Tensor and model framework | Tensors, modules, optimizers, random seeding, device handling | Primary framework documentation for future training and inference code | Technical implementation |
| D02 | PyTorch Contributors | MPS backend | 2.14 documentation | [MPS backend](https://docs.pytorch.org/docs/2.14/notes/mps.html) | Apple Silicon acceleration | `torch.backends.mps.is_available()`, `torch.backends.mps.is_built()`, and the `mps` device | Supports the current MPS-first, CPU-fallback environment policy without assuming CUDA | Technical implementation |
| D03 | TorchVision Contributors | EfficientNet model documentation | Current stable documentation | [TorchVision EfficientNet](https://docs.pytorch.org/vision/stable/models/efficientnet.html) | Classification model API | EfficientNet model builders, including `efficientnet_b4` | API reference for the configured classifier architecture; pretrained weights will require a separately approved download step | Technical implementation |
| D04 | Segmentation Models PyTorch Contributors | Segmentation Models documentation | Current documentation | [Segmentation Models](https://smp.readthedocs.io/en/latest/) | Segmentation model library | U-Net, encoder choices, losses, metrics, and saving/loading | API reference for the configured U-Net and ResNet34 design; does not replace validation on NeuroVR data | Technical implementation |
| D05 | Albumentations Contributors | Albumentations Documentation | Current documentation | [Albumentations docs](https://albumentations.ai/docs/) | Image and mask augmentation | Compose pipelines, image/mask targets, classification, semantic segmentation, and reproducibility guidance | Future augmentation implementation reference, especially for applying identical spatial transforms to images and masks | Technical implementation |
| D06 | MONAI Contributors | MONAI Documentation | Current documentation | [MONAI docs](https://docs.monai.io/en/latest/) | Medical imaging workflows | Medical-imaging transforms, datasets, and 2D/3D workflows | Candidate reference for future medical-image preprocessing and volumetric experiments; not used to claim current implementation | Technical background; future implementation |
| D07 | NiBabel Developers | Working with NIfTI images | Current documentation | [NiBabel NIfTI images](https://nipy.org/nibabel/nifti_images.html) | NIfTI loading and spatial metadata | NIfTI-1/NIfTI-2 headers, affines, voxel spacing, orientation, and data scaling | Future 3D section must preserve voxel spacing and spatial transforms when reading NIfTI volumes | Technical implementation; future methodology |
| D08 | NIfTI Working Group | NIfTI-1 reference | NIfTI-1 specification | [NIfTI-1 reference](https://nifti.nimh.nih.gov/nifti-1/) | Neuroimaging file standard | NIfTI header and image format specification | Standard reference for future volumetric input and output; no NIfTI data is downloaded in this step | Technical background; future methodology |
| D09 | Three.js Contributors | Three.js Documentation | Current documentation | [Three.js docs](https://threejs.org/docs/) | Browser 3D rendering | Scenes, meshes, textures, loaders, renderers, `MarchingCubes`, and WebXR helpers | Future browser visualization reference for rendering reconstructed tumor meshes | Technical implementation; future functionality |
| D10 | Mozilla Contributors | WebXR Device API | Current documentation | [MDN WebXR Device API](https://developer.mozilla.org/en-US/docs/Web/API/WebXR_Device_API) | Web-based AR/VR | XR sessions, reference spaces, frames, input, hit testing, and security/compatibility constraints | Future AR planning reference; browser/device support must be tested before any academic claim | Technical background; future functionality |
| D11 | NiBabel Developers | Coordinate systems and affines | Current documentation | [NiBabel coordinate systems](https://nipy.org/nibabel/coordinate_systems.html) | Voxel and physical coordinates | Affine transforms, voxel spacing, RAS+ coordinates, and inverse mappings | Directly supports preserving affine metadata and separating voxel coordinates from physical coordinates in the 3D data layer | Technical methodology; implementation |

## Documentation use policy

- Pin the relevant documentation version when implementation begins.
- Verify API behavior against the installed package versions before relying on it.
- Record access dates for mutable documentation.
- Treat framework documentation as implementation guidance, not as evidence that a medical workflow is clinically valid.
