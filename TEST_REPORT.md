# Local Parity Test Report

## Scope

The copied Python inference files were compared byte-for-byte with each original Docker `Task` directory. The model bundle files were checked individually with SHA-256. We first compared the original Task source and repository source in the local Python/CUDA environment, then built both repository Dockerfiles and ran the actual containers on the local RTX 3060. No remote cluster was used.

Docker Desktop 4.77.0 / Engine 29.5.3 (`desktop-linux`) was used. The build used the locally cached PyTorch 2.6 CUDA 12.4 runtime image through its mirror tag `docker.m.daocloud.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` (image digest `sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee`); the Dockerfile runtime settings and app/model paths were unchanged. Docker reused the existing pip-install build-cache layer for the same base image and `requirements.txt`, so the dependency installation command was not force-re-executed from a clean cache. The actual image tests used the official `/input` and `/output` mounts and Dockerfile defaults. The per-Dockerfile ignore files ensure only the selected variant's model bundle is sent in each build context.

## Environment

- GPU: NVIDIA GeForce RTX 3060, 12 GiB
- Docker: Desktop `4.77.0`, Engine `29.5.3`, Linux containers
- PyTorch: `2.6.0+cu124`, CUDA available
- nnU-Net: `2.6.0`
- SimpleITK: `2.5.3`
- SciPy: `1.16.3`
- pandas: `2.3.3`
- scikit-learn: `1.9.0`
- CatBoost: `1.2.10`
- scikit-survival: `0.28.0`
- Inference settings: single-case sequential, folds `0-4`, TTA enabled, tile step `0.8`, tile batch `1`

## Source and Model Integrity

All checked source files matched their original Docker Task copies byte-for-byte: `inference.py`, `hecktor_pipeline.py`, `submission_models.py`, `train_hecktor_tn_stage_baseline.py`, and `requirements.txt` for both variants.

The 19 files in the first model bundle (2,537,436,705 bytes) and 26 files in the refinement bundle (3,772,015,773 bytes) all matched their source Docker model files by SHA-256. Manifests are under each `weights/<variant>/SHA256SUMS.txt`.

## End-to-End Results

| Variant | Case | Docker vs repository source | Docker vs original Docker Task source |
| --- | --- | --- | --- |
| `dataset804_probtab_t0_p3rfs_v1` | CHUM-001 | Geometry, voxels, T/N/RFS exact | Geometry, voxels, T/N/RFS exact |
| `dataset804_probtab_t0_p3rfs_v1` | CHUP-001 | Geometry exact; 1 voxel differs out of `568 x 512 x 512` (Docker 12,165 vs repository 12,164 positive voxels); T/N/RFS exact | Geometry and voxels exact; T/N exact; RFS differs only in the last floating-point digit (`0.7128974465954745` vs `0.7128974465954746`) |
| `dataset804_807_refine_emptytiny_t0_804p3rfs_v1` | USZ-001 | Geometry, voxels, T/N/RFS exact | Geometry, voxels, T/N/RFS exact |
| `dataset804_807_refine_emptytiny_t0_804p3rfs_v1` | MDA-001 | Geometry, voxels, T/N/RFS exact | Geometry, voxels, T/N/RFS exact |

The CHUP-001 mask difference is one boundary voxel only and occurs between the container and the repository's direct Windows-host inference; the container mask exactly matches the original Docker Task source run. T/N/RFS from the container match the repository run exactly. The original Task source's RFS differs from the container/repository value only at the final floating-point digit. Repeated tests and crossed source/weight-path controls showed tiny path/runtime-sensitive numerical variation; no source or model file mismatch was found. The exact cause is not established, so strict bitwise parity is not claimed for CHUP-001 across all execution modes. The other three cases matched exactly in both comparisons.

## Existing Submission Image Comparison

The two pre-existing local submission images were also run directly: `hecktor2026-pipeline:dataset804_probtab_t0_p3rfs_v1` on CHUP-001 and `hecktor2026-pipeline:dataset804_807_refine_emptytiny_t0_804p3rfs_v1` on USZ-001. The old 804 image matched the repository's direct Windows-host output exactly; compared with the newly built repository Docker image, it differed by one segmentation boundary voxel, while geometry and T/N/RFS outputs matched. The old 807 image matched both the new repository Docker image and direct repository inference exactly on USZ-001. This reinforces that the only observed non-bitwise case is a single boundary voxel in 804 CHUP-001; its numerical cause is still undetermined.

All parity inputs were staged from CT/PET only. The EHR JSON whitelist contains only allowed inference fields; `M-stage`, `Relapse`, `RFS`, `T-stage`, and `N-stage` were excluded. No labels were supplied to inference.
