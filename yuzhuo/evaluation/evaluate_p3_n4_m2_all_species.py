#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

from scripts_evaluate.evaluate import evaluate_model_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all species' P3_N4_M2 results for a specific model."
    )
    parser.add_argument(
        "--model",
        # default="qwen3-vl:30b",
        default="gemma3:4b",
        help="Model tag used in results directories, e.g. qwen3-vl:30b",
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

    target_dirs = sorted(
        results_root.glob(f"*/p3/predictions/{model_safe}/*_MIA_P3_N4_M2")
    )

    if not target_dirs:
        print(
            f"No P3_N4_M2 result directories found for model {args.model} ({model_safe})."
        )
        raise SystemExit(1)

    summary_rows = []

    print(f"Model: {args.model}")
    print(f"Matched run directories: {len(target_dirs)}")
    print("=" * 110)
    print(
        f"{'Species':<20} | {'Run':<30} | {'Acc(Str)':<9} | {'Acc(Ans)':<9} | {'Acc(Exp)':<9} | {'Prec':<7} | {'Recall':<7} | {'F1':<7}"
    )
    print("-" * 110)

    for target_dir in target_dirs:
        species = target_dir.parts[-5]
        metrics = evaluate_model_directory(target_dir, "p3", model_name=model_safe)
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
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "correct": metrics["correct"],
                "total": metrics["total"],
                "answered": metrics["answered"],
                "missing_extraction": metrics["missing_extraction"],
                "target_dir": str(target_dir),
            }
        )

        print(
            f"{species:<20} | {metrics['run_name']:<30} | "
            f"{metrics['acc_strict']:<9.2%} | {metrics['acc_answered']:<9.2%} | {metrics['acc_expected']:<9.2%} | "
            f"{metrics['precision']:<7.2%} | {metrics['recall']:<7.2%} | {metrics['f1_score']:<7.2%}"
        )

    print("=" * 110)

    summary_json = results_root / f"p3_n4_m2_summary_{model_safe}.json"
    summary_csv = results_root / f"p3_n4_m2_summary_{model_safe}.csv"

    with open(summary_json, "w") as f:
        json.dump(summary_rows, f, indent=4)

    fieldnames = [
        "species",
        "model",
        "run_name",
        "acc_strict",
        "acc_answered",
        "acc_expected",
        "precision",
        "recall",
        "f1_score",
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
