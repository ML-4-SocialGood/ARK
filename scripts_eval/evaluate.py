"""
/home/dzha866/Projects/ARK/scripts_eval/evaluate.py
Evaluates Re-ID inference results.
Calculates accuracy and generates a detailed failure analysis report.
"""

import argparse
import json
import logging
import os
import sys
import csv
from pathlib import Path

# Ensure project root is in path so we can import scripts_eval modules
sys.path.append(os.getcwd())

from scripts_eval.utils import ensure_directories, setup_logging


def evaluate_model_directory(model_dir: Path) -> dict:
    """
    Evaluates all task JSON files in a specific model's directory.

    Returns:
        dict: Metrics containing accuracy, total counts, etc.
    """
    json_files = sorted(list(model_dir.glob("*.json")))

    if not json_files:
        logging.warning(f"No result files found in {model_dir}")
        return None

    total_count = 0
    correct_count = 0
    missing_answer_count = 0

    # Store detailed results for CSV export
    detailed_results = []

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            task_id = data.get("task_id")
            ground_truth = data.get("ground_truth")
            extracted_answer = data.get("extracted_answer")
            prediction_text = data.get("prediction_text", "")

            # Skip tasks that don't have ground truth (e.g. pure test set)
            if not ground_truth:
                continue

            total_count += 1

            # Normalize for comparison
            gt_norm = ground_truth.strip().upper()
            pred_norm = extracted_answer.strip().upper() if extracted_answer else None

            is_correct = False
            if pred_norm == gt_norm:
                correct_count += 1
                is_correct = True

            if not extracted_answer:
                missing_answer_count += 1

            detailed_results.append(
                {
                    "task_id": task_id,
                    "ground_truth": gt_norm,
                    "prediction": pred_norm if pred_norm else "N/A",
                    "is_correct": is_correct,
                    "raw_output_snippet": prediction_text[:100].replace(
                        "\n", " "
                    ),  # First 100 chars for quick check
                }
            )

        except Exception as e:
            logging.error(f"Error reading {json_file}: {e}")

    if total_count == 0:
        return None

    accuracy = correct_count / total_count

    metrics = {
        "model": model_dir.name,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total_count,
        "missing_extraction": missing_answer_count,
    }

    # Save a detailed CSV report for this model (useful for debugging failures)
    csv_path = model_dir / "evaluation_details.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "ground_truth",
                "prediction",
                "is_correct",
                "raw_output_snippet",
            ],
        )
        writer.writeheader()
        writer.writerows(detailed_results)

    logging.info(f"Saved detailed analysis to {csv_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Re-ID Inference Results")
    parser.add_argument(
        "--species", type=str, required=True, help="Species name (e.g., BelugaID)"
    )
    parser.add_argument(
        "--protocol", type=str, required=True, help="Protocol (e.g., P1)"
    )
    parser.add_argument(
        "--model", type=str, help="Specific model name to evaluate (optional)"
    )

    args = parser.parse_args()

    # 1. Setup Paths
    paths = ensure_directories(args.species, args.protocol)
    predictions_root = paths["predictions"]

    setup_logging(paths["base"] / "evaluation.log")
    logging.info(f"Starting evaluation for {args.species} / {args.protocol}")

    # 2. Identify Model Directories
    if args.model:
        # Check specific model
        model_safe_name = args.model.replace(":", "_")
        model_dirs = [predictions_root / model_safe_name]
    else:
        # Auto-discover all model directories
        model_dirs = [d for d in predictions_root.iterdir() if d.is_dir()]

    if not model_dirs:
        logging.error(f"No prediction directories found in {predictions_root}")
        return

    # 3. Evaluate Each Model
    final_report = []

    print("\n" + "=" * 60)
    print(f"{'Model':<30} | {'Accuracy':<10} | {'Correct':<8} | {'Total':<8}")
    print("-" * 60)

    for model_dir in model_dirs:
        if not model_dir.exists():
            logging.warning(f"Directory not found: {model_dir}")
            continue

        metrics = evaluate_model_directory(model_dir)

        if metrics:
            print(
                f"{metrics['model']:<30} | {metrics['accuracy']:.2%}    | {metrics['correct']:<8} | {metrics['total']:<8}"
            )
            final_report.append(metrics)

            # Save metrics.json in the model folder
            with open(model_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
        else:
            print(f"{model_dir.name:<30} | {'No Data':<10} | {'-':<8} | {'-':<8}")

    print("=" * 60 + "\n")

    # 4. Save Overall Report
    report_path = paths["base"] / "evaluation_summary.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=4)

    logging.info(f"Evaluation finished. Summary saved to {report_path}")


if __name__ == "__main__":
    main()
