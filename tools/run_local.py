from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "dataset804_probtab_t0_p3rfs_v1": "dataset804_probtab_t0_p3rfs_v1",
    "dataset804_807_refine_emptytiny_t0_804p3rfs_v1": "dataset804_807_refine_emptytiny_t0_804p3rfs_v1",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the copied official inference function outside Docker.")
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-root", type=Path, help="Defaults to this repo's matching weights/model folder.")
    parser.add_argument("--source-root", type=Path, help="Use an external source tree, e.g. the original Docker Task folder.")
    args = parser.parse_args()

    slug = VARIANTS[args.variant]
    source = args.source_root or ROOT / "submissions" / slug / "src"
    model_root = args.model_root or ROOT / "weights" / slug / "model"
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("hecktor_repo_inference", source / "inference.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import inference entrypoint from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_official_io(args.input, args.output, model_root=model_root)


if __name__ == "__main__":
    main()
