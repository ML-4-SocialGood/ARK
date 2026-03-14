"""
/home/dzha866/Projects/ARK/scripts_evaluate/evaluate.py
Evaluates Re-ID inference results.
Calculates accuracy and generates a detailed failure analysis report.
"""

import argparse
import json
import logging
import os
import sys
import csv
from typing import Optional
from pathlib import Path

# Ensure project root is in path so we can import scripts_eval modules
sys.path.append(os.getcwd())

from scripts_evaluate.utils import ensure_directories, setup_logging


def evaluate_model_directory(model_dir: Path, protocol: str) -> Optional[dict]:
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
    
    total_p = 0.0
    total_r = 0.0
    total_f1 = 0.0

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
            task_p, task_r, task_f1 = 0.0, 0.0, 0.0
            
            if protocol.upper() == "P3":
                gt_set = set([x.strip() for x in gt_norm.split(",") if x.strip()])
                pred_set = set([x.strip() for x in pred_norm.split(",") if x.strip()]) if pred_norm else set()
                
                if pred_set == gt_set:
                    correct_count += 1
                    is_correct = True
                    
                tp = len(gt_set.intersection(pred_set))
                fp = len(pred_set - gt_set)
                fn = len(gt_set - pred_set)
                
                task_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                task_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                task_f1 = 2 * task_p * task_r / (task_p + task_r) if (task_p + task_r) > 0 else 0.0
                
                total_p += task_p
                total_r += task_r
                total_f1 += task_f1
            else:
                if pred_norm == gt_norm:
                    correct_count += 1
                    is_correct = True

            if not extracted_answer:
                missing_answer_count += 1

            row = {
                "task_id": task_id,
                "ground_truth": gt_norm,
                "prediction": pred_norm if pred_norm else "N/A",
                "is_correct": is_correct,
                "raw_output_snippet": prediction_text[:100].replace("\n", " "),
            }
            
            if protocol.upper() == "P3":
                row.update({
                    "precision": round(task_p, 4),
                    "recall": round(task_r, 4),
                    "f1_score": round(task_f1, 4),
                })

            detailed_results.append(row)

        except Exception as e:
            logging.error(f"Error reading {json_file}: {e}")

    if total_count == 0:
        return None

    answered_count = total_count - missing_answer_count
    acc_strict = correct_count / total_count if total_count > 0 else 0.0
    acc_answered = correct_count / answered_count if answered_count > 0 else 0.0
    # 按照要求，将 null (missing) 答案按 0.5 (一半) 的正确率折算
    acc_half_null = (correct_count + 0.5 * missing_answer_count) / total_count if total_count > 0 else 0.0

    metrics = {
        "model": model_dir.name,
        "accuracy": acc_strict,  # 保留为 acc_strict 以向下兼容
        "acc_strict": acc_strict,
        "acc_answered": acc_answered,
        "acc_half_null": acc_half_null,
        "correct": correct_count,
        "total": total_count,
        "answered": answered_count,
        "missing_extraction": missing_answer_count,
    }
    
    if protocol.upper() == "P3":
        metrics["precision"] = total_p / total_count
        metrics["recall"] = total_r / total_count
        metrics["f1_score"] = total_f1 / total_count

    # Save a detailed CSV report for this model (useful for debugging failures)
    csv_path = model_dir / "evaluation_details.csv"
    
    fieldnames = ["task_id", "ground_truth", "prediction", "is_correct", "raw_output_snippet"]
    if protocol.upper() == "P3":
        fieldnames.extend(["precision", "recall", "f1_score"])
        
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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

    setup_logging(paths["logs"] / "evaluation.log")
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

    logging.info("=" * 95)
    if args.protocol.upper() == "P3":
        logging.info(f"{'Model':<25} | {'Acc(Str)':<9} | {'Acc(Ans)':<9} | {'Acc(Half)':<9} | {'Prec':<7} | {'Recall':<7} | {'F1':<7}")
    else:
        logging.info(f"{'Model':<25} | {'Acc(Str)':<9} | {'Acc(Ans)':<9} | {'Acc(Half)':<9} | {'Corr':<5} | {'Total':<5}")
    logging.info("-" * 95)

    for model_dir in model_dirs:
        if not model_dir.exists():
            logging.warning(f"Directory not found: {model_dir}")
            continue

        metrics = evaluate_model_directory(model_dir, args.protocol)

        if metrics:
            if args.protocol.upper() == "P3":
                logging.info(
                    f"{metrics['model']:<25} | {metrics['acc_strict']:<9.2%} | {metrics['acc_answered']:<9.2%} | {metrics['acc_half_null']:<9.2%} | {metrics['precision']:<7.2%} | {metrics['recall']:<7.2%} | {metrics['f1_score']:<7.2%}"
                )
            else:
                logging.info(
                    f"{metrics['model']:<25} | {metrics['acc_strict']:<9.2%} | {metrics['acc_answered']:<9.2%} | {metrics['acc_half_null']:<9.2%} | {metrics['correct']:<5} | {metrics['total']:<5}"
                )
            final_report.append(metrics)

            # Save metrics.json in the model folder
            with open(model_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
        else:
            logging.info(f"{model_dir.name:<25} | {'No Data':<9} | {'-':<9} | {'-':<9} | {'-':<5} | {'-':<5}")

    logging.info("=" * 95)

    # 4. Save Overall Report
    report_path = paths["base"] / "evaluation_summary.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=4)

    logging.info(f"Evaluation finished. Summary saved to {report_path}")


if __name__ == "__main__":
    main()
