from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


ALLOWED_EHR_FIELDS = {
    "CenterID",
    "Gender",
    "Age",
    "Tobacco Consumption",
    "Alcohol Consumption",
    "Performance Status",
    "Treatment",
    "HPV Status",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage image-only local parity inputs from HECKTOR training data.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing one folder per case.")
    parser.add_argument("--ehr-csv", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True, help="Case ID; repeat for multiple cases.")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    with args.ehr_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = {row["PatientID"]: row for row in csv.DictReader(stream)}

    for case_id in args.case:
        case_dir = args.raw_root / case_id
        ct = case_dir / f"{case_id}__CT.nii.gz"
        pet = case_dir / f"{case_id}__PT.nii.gz"
        if not ct.is_file() or not pet.is_file():
            raise FileNotFoundError(f"Expected CT/PET pair not found for {case_id}: {ct}, {pet}")
        if case_id not in rows:
            raise KeyError(f"No EHR row for {case_id}")

        out = args.output_root / case_id
        ct_dir = out / "images" / "ct"
        pet_dir = out / "images" / "pet"
        ct_dir.mkdir(parents=True, exist_ok=True)
        pet_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ct, ct_dir / ct.name)
        shutil.copy2(pet, pet_dir / pet.name)

        row = rows[case_id]
        ehr = {key: row[key] for key in ALLOWED_EHR_FIELDS if key in row and row[key].strip()}
        (out / "ehr.json").write_text(json.dumps(ehr, indent=2), encoding="utf-8")
        print(f"Staged {case_id}: CT={ct.stat().st_size:,} bytes, PET={pet.stat().st_size:,} bytes; EHR={sorted(ehr)}")


if __name__ == "__main__":
    main()
