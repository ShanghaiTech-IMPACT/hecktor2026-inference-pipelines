from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def compare_json(left: Path, right: Path) -> bool:
    a = json.loads(left.read_text(encoding="utf-8"))
    b = json.loads(right.read_text(encoding="utf-8"))
    equal = a == b
    print(f"{left.name}: {'EXACT' if equal else 'DIFFERENT'}")
    if not equal:
        print(f"  reference: {a}")
        print(f"  repository: {b}")
    return equal


def compare_segmentation(left: Path, right: Path) -> bool:
    a_img = sitk.ReadImage(str(left))
    b_img = sitk.ReadImage(str(right))
    a = sitk.GetArrayFromImage(a_img)
    b = sitk.GetArrayFromImage(b_img)
    geometry_equal = (
        a_img.GetSize() == b_img.GetSize()
        and np.allclose(a_img.GetSpacing(), b_img.GetSpacing(), atol=1e-6)
        and np.allclose(a_img.GetOrigin(), b_img.GetOrigin(), atol=1e-6)
        and np.allclose(a_img.GetDirection(), b_img.GetDirection(), atol=1e-6)
    )
    voxel_equal = a.shape == b.shape and np.array_equal(a, b)
    print(f"output.mha: geometry={'EXACT' if geometry_equal else 'DIFFERENT'}, voxels={'EXACT' if voxel_equal else 'DIFFERENT'}")
    if a.shape == b.shape:
        print(f"  shape={a.shape}, nonzero(reference/repository)={np.count_nonzero(a)}/{np.count_nonzero(b)}")
    return geometry_equal and voxel_equal


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one Docker output tree against the repository build.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("repository", type=Path)
    args = parser.parse_args()

    segmentation_rel = Path("images/head-neck-tumor-segmentation/output.mha")
    equal = compare_segmentation(args.reference / segmentation_rel, args.repository / segmentation_rel)
    for name in ("t-stage.json", "n-stage.json", "rfs.json"):
        equal = compare_json(args.reference / name, args.repository / name) and equal
    if not equal:
        raise SystemExit(1)
    print("All compared outputs match exactly.")


if __name__ == "__main__":
    main()
