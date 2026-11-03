from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "dataset804_probtab_t0_p3rfs_v1",
    "dataset804_807_refine_emptytiny_t0_804p3rfs_v1",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    failed = False
    for variant in VARIANTS:
        bundle = ROOT / "weights" / variant
        model = bundle / "model"
        manifest = bundle / "SHA256SUMS.txt"
        if not model.is_dir() or not manifest.is_file():
            print(f"MISSING bundle or manifest: {variant}")
            failed = True
            continue

        rows = [line.strip().split("  ", 1) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = {relative: digest for digest, relative in rows}
        actual_files = {path.relative_to(model).as_posix(): path for path in model.rglob("*") if path.is_file()}
        missing = sorted(set(expected) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected))
        bad = [relative for relative, path in actual_files.items() if relative in expected and sha256(path) != expected[relative]]
        print(f"{variant}: {len(expected)} listed, {len(actual_files)} present, {len(bad)} hash mismatch")
        if missing:
            print(f"  missing: {missing}")
        if extra:
            print(f"  unlisted: {extra}")
        if bad:
            print(f"  mismatched: {bad}")
        failed = failed or bool(missing or extra or bad)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
