from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd

from hecktor_pipeline import get_resident_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the HECKTOR Docker pipeline on multiple local cases.")
    parser.add_argument("--case-root", type=Path, required=True, help="Directory containing <case>/<case>__CT.nii.gz.")
    parser.add_argument("--clinical-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--case-ids", default="", help="Comma-separated case IDs. If omitted, scan case-root.")
    parser.add_argument("--model-root", type=Path, default=Path("/opt/ml/model"))
    parser.add_argument("--mode", choices=("sequential", "threaded"), default="threaded")
    parser.add_argument("--cpu-workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--post-workers", type=int, default=2)
    parser.add_argument("--prefetch", type=int, default=0, help="Max in-flight preprocessed cases; 0 uses 2 * cpu-workers.")
    return parser.parse_args()


def discover_cases(case_root: Path, case_ids: str, max_cases: int) -> list[str]:
    if case_ids.strip():
        cases = [item.strip() for item in case_ids.split(",") if item.strip()]
    else:
        cases = sorted(p.name for p in case_root.iterdir() if p.is_dir())
    return cases[:max_cases]


def load_clinical(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "PatientID" not in df.columns:
        return {}
    return {str(row["PatientID"]): dict(row) for _, row in df.iterrows()}


def case_paths(case_root: Path, case_id: str) -> tuple[Path, Path]:
    case_dir = case_root / case_id
    suffixes = (".mha", ".nii.gz", ".nii", ".nrrd", ".mhd")
    for suffix in suffixes:
        ct = case_dir / f"{case_id}__CT{suffix}"
        pet = case_dir / f"{case_id}__PT{suffix}"
        if ct.exists() and pet.exists():
            return ct, pet
    for suffix in suffixes:
        ct = case_root / f"{case_id}_0000{suffix}"
        pet = case_root / f"{case_id}_0001{suffix}"
        if ct.exists() and pet.exists():
            return ct, pet
    for suffix in suffixes:
        ct = case_root / f"{case_id}__CT{suffix}"
        pet = case_root / f"{case_id}__PT{suffix}"
        if ct.exists() and pet.exists():
            return ct, pet
    raise FileNotFoundError(f"Could not find CT/PET for {case_id} under {case_root}")


def make_result_row(case_id: str, started_at: float, status: str, error: str, debug: dict[str, object]) -> dict[str, object]:
    return {
        "case_id": case_id,
        "seconds": time.perf_counter() - started_at,
        "status": status,
        "error": error,
        "timing": debug.get("timing", {}),
    }


def run_cases_sequential(
    pipeline,
    case_root: Path,
    cases: list[str],
    clinical: dict[str, dict[str, str]],
    output_root: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_id in cases:
        ct, pet = case_paths(case_root, case_id)
        ehr = clinical.get(case_id, {"PatientID": case_id})
        out_dir = output_root / case_id
        started_at = time.perf_counter()
        status = "ok"
        error = ""
        debug: dict[str, object] = {}
        try:
            debug = pipeline.run_case(ct, pet, ehr, out_dir)
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        row = make_result_row(case_id, started_at, status, error, debug)
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def run_cases_threaded(
    pipeline,
    case_root: Path,
    cases: list[str],
    clinical: dict[str, dict[str, str]],
    output_root: Path,
    cpu_workers: int,
    post_workers: int,
    prefetch: int,
) -> list[dict[str, object]]:
    cpu_workers = max(1, cpu_workers)
    post_workers = max(0, post_workers)
    prefetch = max(1, prefetch)
    rows: list[dict[str, object]] = []
    case_iter = iter(cases)
    pending_prepare: dict[object, dict[str, object]] = {}
    pending_finalize: dict[object, dict[str, object]] = {}

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            case_id = next(case_iter)
        except StopIteration:
            return False
        ct, pet = case_paths(case_root, case_id)
        ehr = clinical.get(case_id, {"PatientID": case_id})
        out_dir = output_root / case_id
        meta = {
            "case_id": case_id,
            "started_at": time.perf_counter(),
        }
        future = executor.submit(pipeline.prepare_case, ct, pet, ehr, out_dir)
        pending_prepare[future] = meta
        return True

    def finalize_now(meta: dict[str, object], segmented) -> None:
        status = "ok"
        error = ""
        debug: dict[str, object] = {}
        try:
            debug = pipeline.finalize_case(segmented)
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        row = make_result_row(str(meta["case_id"]), float(meta["started_at"]), status, error, debug)
        rows.append(row)
        print(json.dumps(row), flush=True)

    with ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="hecktor-pre") as pre_executor:
        post_executor = (
            ThreadPoolExecutor(max_workers=post_workers, thread_name_prefix="hecktor-post") if post_workers > 0 else None
        )
        try:
            while len(pending_prepare) < min(prefetch, len(cases)) and submit_next(pre_executor):
                pass

            while pending_prepare or pending_finalize:
                if not pending_prepare and pending_finalize:
                    done_finalize, _ = wait(list(pending_finalize), return_when=FIRST_COMPLETED)
                    done_prepare = set()
                else:
                    done_prepare, _ = wait(list(pending_prepare), return_when=FIRST_COMPLETED)
                    done_finalize, _ = wait(list(pending_finalize), timeout=0, return_when=FIRST_COMPLETED) if pending_finalize else (set(), set())

                for future in done_finalize:
                    meta = pending_finalize.pop(future)
                    status = "ok"
                    error = ""
                    debug: dict[str, object] = {}
                    try:
                        debug = future.result()
                    except Exception as exc:
                        status = "error"
                        error = f"{type(exc).__name__}: {exc}"
                    row = make_result_row(str(meta["case_id"]), float(meta["started_at"]), status, error, debug)
                    rows.append(row)
                    print(json.dumps(row), flush=True)

                for future in done_prepare:
                    meta = pending_prepare.pop(future)
                    try:
                        prepared = future.result()
                        coarse_prob = pipeline.run_tinyunet_coarse(prepared)
                        localized = pipeline.build_local_roi(prepared, coarse_prob)
                        segmented = pipeline.run_fine_segmentation(localized)
                        if post_executor is None:
                            finalize_now(meta, segmented)
                        else:
                            pending_finalize[post_executor.submit(pipeline.finalize_case, segmented)] = meta
                    except Exception as exc:
                        row = make_result_row(
                            str(meta["case_id"]),
                            float(meta["started_at"]),
                            "error",
                            f"{type(exc).__name__}: {exc}",
                            {},
                        )
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                    while len(pending_prepare) < prefetch and submit_next(pre_executor):
                        pass
        finally:
            if post_executor is not None:
                post_executor.shutdown(wait=True, cancel_futures=False)
    return rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    clinical = load_clinical(args.clinical_csv)
    cases = discover_cases(args.case_root, args.case_ids, args.max_cases)

    t0 = time.perf_counter()
    pipeline = get_resident_pipeline(model_root=args.model_root)
    load_seconds = time.perf_counter() - t0

    run_t0 = time.perf_counter()
    if args.mode == "threaded":
        prefetch = args.prefetch if args.prefetch > 0 else max(1, args.cpu_workers * 2)
        rows = run_cases_threaded(
            pipeline=pipeline,
            case_root=args.case_root,
            cases=cases,
            clinical=clinical,
            output_root=args.output_root,
            cpu_workers=args.cpu_workers,
            post_workers=args.post_workers,
            prefetch=prefetch,
        )
    else:
        rows = run_cases_sequential(
            pipeline=pipeline,
            case_root=args.case_root,
            cases=cases,
            clinical=clinical,
            output_root=args.output_root,
        )
    run_wall_seconds = time.perf_counter() - run_t0

    ok_seconds = [row["seconds"] for row in rows if row["status"] == "ok"]
    timing_keys = sorted({key for row in rows if row["status"] == "ok" for key in row.get("timing", {})})
    stage_means = {
        key: sum(float(row["timing"].get(key, 0.0)) for row in rows if row["status"] == "ok") / len(ok_seconds)
        for key in timing_keys
    } if ok_seconds else {}
    stage_percent_of_warm_mean = {
        key: 100.0 * value / (sum(ok_seconds) / len(ok_seconds))
        for key, value in stage_means.items()
    } if ok_seconds else {}
    summary = {
        "case_count": len(rows),
        "ok_count": len(ok_seconds),
        "load_seconds": load_seconds,
        "mode": args.mode,
        "cpu_workers": args.cpu_workers if args.mode == "threaded" else 1,
        "post_workers": args.post_workers if args.mode == "threaded" else 0,
        "prefetch": (args.prefetch if args.prefetch > 0 else max(1, args.cpu_workers * 2)) if args.mode == "threaded" else 1,
        "mean_case_seconds_warm": sum(ok_seconds) / len(ok_seconds) if ok_seconds else None,
        "benchmark_wall_seconds": run_wall_seconds,
        "benchmark_wall_seconds_with_load": load_seconds + run_wall_seconds,
        "throughput_cases_per_minute": (60.0 * len(ok_seconds) / run_wall_seconds) if ok_seconds and run_wall_seconds > 0 else None,
        "total_seconds_with_load": load_seconds + run_wall_seconds,
        "mean_case_seconds_with_load_amortized": (load_seconds + run_wall_seconds) / len(ok_seconds)
        if ok_seconds
        else None,
        "stage_means_seconds": stage_means,
        "stage_percent_of_warm_mean": stage_percent_of_warm_mean,
        "cases": rows,
    }
    (args.output_root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
