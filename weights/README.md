# Model Bundles

The model files are hosted in the [SunnyXcf/HECKTOR2026 Hugging Face repository](https://huggingface.co/SunnyXcf/HECKTOR2026), not in this GitHub repository. Download the model payloads into this directory with:

```powershell
hf download SunnyXcf/HECKTOR2026 --repo-type model --local-dir weights --include "*/model/**"
```

This recreates the paths expected by the Dockerfiles:

```text
weights/<variant>/model/
```

The model folders are Git-ignored. Keep each bundle paired with its matching submission variant and verify it using `python tools/verify_weight_hashes.py`. The SHA-256 manifests are tracked here; the model files themselves are not.
