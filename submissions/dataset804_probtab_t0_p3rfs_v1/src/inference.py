from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from hecktor_pipeline import run_official_io


def main() -> None:
    input_dir = Path("/input")
    output_dir = Path("/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        debug = run_official_io(input_dir, output_dir, model_root=Path("/opt/ml/model"))
        if os.environ.get("HECKTOR_DEBUG", "").strip() == "1":
            (output_dir / "debug.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
    except Exception as exc:
        (output_dir / "error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
