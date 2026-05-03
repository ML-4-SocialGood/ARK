#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

from scripts_evaluate.evaluate import evaluate_model_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all species' P5 resolution results for a specific model."
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl:30b",
        help="Model tag used in results directories, e.g. qwen3-vl:30b",
    )
    parser.add_argument(
        "--species",
        default="",
        help="Optional species filter, e.g. Lion",
    )
    parser.add_argument(
        "--project-root",
        default="/data/yil708/ARK",
        help="Project root directory",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    results_root = project_root / "results"
    model_safe = args.model.replace(":", "_")

    if args.species:
        target_dirs = sorted(
            (results_root / args.species).glob(
                f"p5/predictions/{model_safe}/*_I2I_P5_resolution_*"
            )
        )
    else:
        target_dirs = sorted(
            results_root.glob(f"*/p5/predictions/{model_safe}/*_I2I_P5_resolution_*")
        )

    if not target_dirs:
        if args.species:
            print(
                f"No P5 resolution result directories found for species {args.species} and model {args.model} ({model_safe})."
            )
        else:
            print(
                f"No P5 resolution result directories found for model {args.model} ({model_safe})."
            )
        raise SystemExit(1)

    summary_rows = []

    print(f"Model: {args.model}")
    print(f"Species filter: {args.species or '<all>'}")
    print(f"Matched run directories: {len(target_dirs)}")
    print("=" * 105)
    print(
        f"{'Species':<20} | {'Run':<35} | {'Acc(Str)':<9} | {'Acc(Ans)':<9} | {'Acc(Exp)':<9} | {'Corr':<5} | {'Total':<5}"
    )
    print("-" * 105)

    for target_dir in target_dirs:
        species = target_dir.parts[-5]
        metrics = evaluate_model_directory(target_dir, "p5", model_name=model_safe)
        if metrics is None:
            continue

        with open(target_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        summary_rows.append(
            {
                "species": species,
                "model": metrics["model"],
                "run_name": metrics["run_name"],
                "acc_strict": metrics["acc_strict"],
                "acc_answered": metrics["acc_answered"],
                "acc_expected": metrics["acc_expected"],
                "correct": metrics["correct"],
                "total": metrics["total"],
                "answered": metrics["answered"],
                "missing_extraction": metrics["missing_extraction"],
                "target_dir": str(target_dir),
            }
        )

        print(
            f"{species:<20} | {metrics['run_name']:<35} | "
            f"{metrics['acc_strict']:<9.2%} | {metrics['acc_answered']:<9.2%} | {metrics['acc_expected']:<9.2%} | "
            f"{metrics['correct']:<5} | {metrics['total']:<5}"
        )

    print("=" * 105)

    suffix = f"{model_safe}_{args.species}" if args.species else model_safe
    summary_json = results_root / f"p5_resolution_summary_{suffix}.json"
    summary_csv = results_root / f"p5_resolution_summary_{suffix}.csv"

    with open(summary_json, "w") as f:
        json.dump(summary_rows, f, indent=4)

    fieldnames = [
        "species",
        "model",
        "run_name",
        "acc_strict",
        "acc_answered",
        "acc_expected",
        "correct",
        "total",
        "answered",
        "missing_extraction",
        "target_dir",
    ]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary JSON to {summary_json}")
    print(f"Saved summary CSV  to {summary_csv}")


if __name__ == "__main__":
    main()
