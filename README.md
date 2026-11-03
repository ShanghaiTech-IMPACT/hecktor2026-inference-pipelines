# HECKTOR 2026 Inference Pipelines

Two HECKTOR 2026 submission pipelines are preserved here as separate variants. The Python inference sources were copied from their Docker contexts without algorithmic edits; only the Docker build context was adapted so both variants can be built from this repository root.

## Variants

| Variant | Segmentation | Tabular prediction |
| --- | --- | --- |
| `dataset804_probtab_t0_p3rfs_v1` | Dataset804 3d_lowres + 3d_cascade_fullres, five-fold TTA ensembles, `checkpoint_final` | T hard-only ExtraTrees; N hard + soft probability CatBoost; RFS P3 fusion |
| `dataset804_807_refine_emptytiny_t0_804p3rfs_v1` | Dataset804 cascade followed by Dataset807 refinement, five-fold TTA, with empty/tiny-mask fallback | T/N five-seed CatBoost; RFS reuses Dataset804 P3 fusion |

Both entrypoints accept one case in the official layout:

```text
input/
  images/ct/<case>.mha
  images/pet/<case>.mha
  ehr.json
```

NIfTI inputs are also supported by the pipeline for local testing. EHR at inference contains allowed patient metadata only; M-stage and training targets are not used. Predictions are written under `output/`, including `images/head-neck-tumor-segmentation/output.mha` and the T-stage, N-stage, and RFS JSON files.

## Build

Model weights are hosted separately on the [Hugging Face model repository](https://huggingface.co/SunnyXcf/HECKTOR2026); no checkpoints or serialized estimators are stored in GitHub. The local `weights/<variant>/model/` directories are ignored by Git and are populated from the Hub before building an image. The Hub keeps the same variant directory names used here.

Install the Hugging Face CLI if needed, then download only the model payloads into the expected local paths and verify them:

```powershell
pip install -U huggingface_hub
hf download SunnyXcf/HECKTOR2026 --repo-type model --local-dir weights --include "*/model/**"
python tools/verify_weight_hashes.py
```

Each bundle must remain paired with its matching source variant. The tracked `weights/<variant>/SHA256SUMS.txt` manifests verify the downloaded model files.

```powershell
docker build -f submissions/dataset804_probtab_t0_p3rfs_v1/Dockerfile -t hecktor2026:dataset804-probtab .
docker build -f submissions/dataset804_807_refine_emptytiny_t0_804p3rfs_v1/Dockerfile -t hecktor2026:dataset807-refine .
```

Run one input using the official `/input` and `/output` mount points:

```powershell
docker run --rm --gpus all -v C:/path/to/input:/input -v C:/path/to/output:/output hecktor2026:dataset804-probtab
```

The Dockerfiles preserve the original runtime settings: nnU-Net 2.6.0, five-fold TTA, single-case sequential execution, and the original tile step size. Each Dockerfile has a matching `Dockerfile.dockerignore`, so a build sends only that variant's source and weights instead of both large model bundles. See each variant's `PIPELINE_ORIGINAL.md` for the original description and `PIPELINE.md` for bundle mapping.

## Weight integrity and tests

`weights/README.md` documents the independent Hugging Face bundles. A SHA-256 manifest is tracked for each bundle. `tools/` contains local-only staging and parity utilities; no remote cluster is used. Test inputs and prediction outputs are excluded from Git. See [TEST_REPORT.md](TEST_REPORT.md) for the cases and comparison results.

Verify local model files against their manifests with:

```powershell
python tools/verify_weight_hashes.py
```

The original Docker image archives in the source folders are not copied here; they are large generated artifacts. The original Docker task sources are preserved here, while the model bundles are downloaded from Hugging Face before building.

The source and bundled model files are derived from existing challenge submission artifacts. No license is asserted here: obtain/confirm redistribution rights for the challenge code, trained weights, and any third-party components before publishing. Add a license only after that review.
