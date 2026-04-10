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


def evaluate_model_directory(
    target_dir: Path, protocol: str, model_name: str
) -> Optional[dict]:
    """
    Evaluates all task JSON files in a specific directory (could be a model dir or a sub-run dir).

    Returns:
        dict: Metrics containing accuracy, total counts, etc.
    """
    json_files = sorted(list(target_dir.glob("*.json")))

    if not json_files:
        logging.warning(f"No result files found in {target_dir}")
        return None

    total_count = 0
    correct_count = 0
    missing_answer_count = 0

    total_p = 0.0
    total_r = 0.0
    total_f1 = 0.0

    # Variables for P7
    correct_n = 0
    correct_c = 0
    missing_n = 0
    missing_c = 0
    correct_both = 0

    # Store detailed results for CSV export
    detailed_results = []

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            task_id = data.get("task_id")
            ground_truth = data.get("ground_truth")

            # Skip tasks that don't have ground truth (e.g. pure test set)
            if not ground_truth:
                continue

            total_count += 1

            if protocol.upper() == "P7":
                # P7 GT mapping: same -> YES, different -> NO
                gt_norm = "YES" if ground_truth.strip().lower() == "same" else "NO"

                res_n = data.get("neutral", {})
                res_c = data.get("counterfactual", {})

                pred_n = res_n.get("extracted_answer")
                pred_n_norm = pred_n.strip().upper() if pred_n else None

                pred_c = res_c.get("extracted_answer")
                pred_c_norm = pred_c.strip().upper() if pred_c else None

                is_correct_n = pred_n_norm == gt_norm
                is_correct_c = pred_c_norm == gt_norm

                if is_correct_n:
                    correct_n += 1
                if is_correct_c:
                    correct_c += 1
                if is_correct_n and is_correct_c:
                    correct_both += 1

                if not pred_n_norm:
                    missing_n += 1
                if not pred_c_norm:
                    missing_c += 1

                row = {
                    "task_id": task_id,
                    "ground_truth": gt_norm,
                    "pred_neutral": pred_n_norm if pred_n_norm else "N/A",
                    "is_correct_n": is_correct_n,
                    "pred_counterfactual": pred_c_norm if pred_c_norm else "N/A",
                    "is_correct_c": is_correct_c,
                    "raw_output_n_snippet": res_n.get("prediction_text", "")[
                        :100
                    ].replace("\n", " "),
                    "raw_output_c_snippet": res_c.get("prediction_text", "")[
                        :100
                    ].replace("\n", " "),
                }
                detailed_results.append(row)
                continue

            # P1-P6 & P3 logic
            extracted_answer = data.get("extracted_answer")
            prediction_text = data.get("prediction_text", "")

            # Normalize for comparison
            gt_norm = ground_truth.strip().upper()
            pred_norm = extracted_answer.strip().upper() if extracted_answer else None

            is_correct = False
            task_p, task_r, task_f1 = 0.0, 0.0, 0.0

            if protocol.upper() == "P3":
                gt_set = set([x.strip() for x in gt_norm.split(",") if x.strip()])
                pred_set = (
                    set([x.strip() for x in pred_norm.split(",") if x.strip()])
                    if pred_norm
                    else set()
                )

                if pred_set == gt_set:
                    correct_count += 1
                    is_correct = True

                tp = len(gt_set.intersection(pred_set))
                fp = len(pred_set - gt_set)
                fn = len(gt_set - pred_set)

                task_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                task_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                task_f1 = (
                    2 * task_p * task_r / (task_p + task_r)
                    if (task_p + task_r) > 0
                    else 0.0
                )

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
                row.update(
                    {
                        "precision": round(task_p, 4),
                        "recall": round(task_r, 4),
                        "f1_score": round(task_f1, 4),
                    }
                )

            detailed_results.append(row)

        except Exception as e:
            logging.error(f"Error reading {json_file}: {e}")

    if total_count == 0:
        return None

    # Return P7 specific metrics
    if protocol.upper() == "P7":
        acc_n = correct_n / total_count if total_count > 0 else 0.0
        acc_c = correct_c / total_count if total_count > 0 else 0.0
        delta_acc = acc_n - acc_c
        # RA (Resilience Accuracy): % of correctly answered neutral queries that remain correct under counterfactual
        ra = correct_both / correct_n if correct_n > 0 else 0.0

        metrics = {
            "model": model_name,
            "run_name": target_dir.name,
            "accuracy_neutral": acc_n,
            "accuracy_counterfactual": acc_c,
            "delta_acc": delta_acc,
            "resilience_accuracy": ra,
            "correct_neutral": correct_n,
            "correct_counterfactual": correct_c,
            "correct_both": correct_both,
            "total": total_count,
            "missing_neutral": missing_n,
            "missing_counterfactual": missing_c,
        }

        csv_path = target_dir / "evaluation_details.csv"
        fieldnames = [
            "task_id",
            "ground_truth",
            "pred_neutral",
            "is_correct_n",
            "pred_counterfactual",
            "is_correct_c",
            "raw_output_n_snippet",
            "raw_output_c_snippet",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_results)

        logging.info(f"Saved detailed analysis to {csv_path}")
        return metrics

    answered_count = total_count - missing_answer_count
    acc_strict = correct_count / total_count if total_count > 0 else 0.0
    acc_answered = correct_count / answered_count if answered_count > 0 else 0.0

    # 动态调整盲猜期望值：P6有5个选项(A-E)，盲猜期望为0.20；其他单选题有4个选项(A-D)，盲猜期望为0.25
    guessing_prob = 0.20 if protocol.upper().startswith("P6") else 0.25
    acc_expected = (
        (correct_count + guessing_prob * missing_answer_count) / total_count
        if total_count > 0
        else 0.0
    )

    metrics = {
        "model": model_name,
        "run_name": target_dir.name,
        "accuracy": acc_strict,  # 保留为 acc_strict 以向下兼容
        "acc_strict": acc_strict,
        "acc_answered": acc_answered,
        "acc_expected": acc_expected,
        "correct": correct_count,
        "total": total_count,
        "answered": answered_count,
        "missing_extraction": missing_answer_count,
    }

    # --- Protocol 6 特殊逻辑：分离闭集(Target-Present)和开集(Target-Absent)的指标 ---
    if protocol.upper().startswith("P6") and detailed_results:
        # 动态找到 "None of the above" 的选项字母（即所有 ground_truth 中字母顺序最大的那个）
        valid_gts = [
            r["ground_truth"]
            for r in detailed_results
            if r["ground_truth"] and r["ground_truth"] != "N/A"
        ]
        none_label = max(valid_gts) if valid_gts else "E"

        correct_open = 0
        total_open = 0
        correct_closed = 0
        total_closed = 0

        for r in detailed_results:
            if r["ground_truth"] == none_label:
                total_open += 1
                if r["is_correct"]:
                    correct_open += 1
            else:
                total_closed += 1
                if r["is_correct"]:
                    correct_closed += 1

        metrics["acc_open"] = correct_open / total_open if total_open > 0 else 0.0
        metrics["acc_closed"] = (
            correct_closed / total_closed if total_closed > 0 else 0.0
        )
        metrics["total_open"] = total_open
        metrics["total_closed"] = total_closed

    if protocol.upper() == "P3":
        metrics["precision"] = total_p / total_count
        metrics["recall"] = total_r / total_count
        metrics["f1_score"] = total_f1 / total_count

    # Save a detailed CSV report for this model (useful for debugging failures)
    csv_path = target_dir / "evaluation_details.csv"

    fieldnames = [
        "task_id",
        "ground_truth",
        "prediction",
        "is_correct",
        "raw_output_snippet",
    ]
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

    logging.info("=" * 115)
    if args.protocol.upper() == "P3":
        logging.info(
            f"{'Model':<20} | {'Condition / Run':<30} | {'Acc(Str)':<9} | {'Acc(Ans)':<9} | {'Acc(Exp)':<9} | {'Prec':<7} | {'Recall':<7} | {'F1':<7}"
        )
    elif args.protocol.upper() == "P7":
        logging.info(
            f"{'Model':<20} | {'Condition / Run':<30} | {'Acc(N)':<9} | {'Acc(C)':<9} | {'ΔAcc':<9} | {'RA':<9} | {'Total':<5}"
        )
    elif args.protocol.upper().startswith("P6"):
        logging.info(
            f"{'Model':<20} | {'Condition / Run':<30} | {'Acc(All)':<9} | {'Acc(Cls)':<9} | {'Acc(Opn)':<9} | {'Corr':<5} | {'Total':<5}"
        )
    else:
        logging.info(
            f"{'Model':<20} | {'Condition / Run':<30} | {'Acc(Str)':<9} | {'Acc(Ans)':<9} | {'Acc(Exp)':<9} | {'Corr':<5} | {'Total':<5}"
        )
    logging.info("-" * 115)

    for model_dir in model_dirs:
        if not model_dir.exists():
            logging.warning(f"Directory not found: {model_dir}")
            continue

        # Auto-detect if we have subdirectories (like P5 isolated runs) or flat json files
        sub_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        targets_to_evaluate = sub_dirs if sub_dirs else [model_dir]

        for target_dir in sorted(targets_to_evaluate):
            metrics = evaluate_model_directory(
                target_dir, args.protocol, model_name=model_dir.name
            )

            if metrics:
                if args.protocol.upper() == "P3":
                    logging.info(
                        f"{metrics['model']:<20} | {metrics['run_name']:<30} | {metrics['acc_strict']:<9.2%} | {metrics['acc_answered']:<9.2%} | {metrics['acc_expected']:<9.2%} | {metrics['precision']:<7.2%} | {metrics['recall']:<7.2%} | {metrics['f1_score']:<7.2%}"
                    )
                elif args.protocol.upper() == "P7":
                    logging.info(
                        f"{metrics['model']:<20} | {metrics['run_name']:<30} | {metrics['accuracy_neutral']:<9.2%} | {metrics['accuracy_counterfactual']:<9.2%} | {metrics['delta_acc']:<9.2%} | {metrics['resilience_accuracy']:<9.2%} | {metrics['total']:<5}"
                    )
                elif args.protocol.upper().startswith("P6"):
                        logging.info(
                            f"{metrics['model']:<20} | {metrics['run_name']:<30} | {metrics['acc_strict']:<9.2%} | {metrics['acc_closed']:<9.2%} | {metrics['acc_open']:<9.2%} | {metrics['correct']:<5} | {metrics['total']:<5}"
                        )
                else:
                        logging.info(
                            f"{metrics['model']:<20} | {metrics['run_name']:<30} | {metrics['acc_strict']:<9.2%} | {metrics['acc_answered']:<9.2%} | {metrics['acc_expected']:<9.2%} | {metrics['correct']:<5} | {metrics['total']:<5}"
                        )
                
                final_report.append(metrics)

                # Save metrics.json in the specific target folder
                with open(target_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=4)
            else:
                if args.protocol.upper() == "P7":
                    logging.info(
                        f"{model_dir.name:<20} | {target_dir.name:<30} | {'No Data':<9} | {'-':<9} | {'-':<9} | {'-':<9} | {'-':<5}"
                    )
                else:
                    logging.info(
                        f"{model_dir.name:<20} | {target_dir.name:<30} | {'No Data':<9} | {'-':<9} | {'-':<9} | {'-':<5} | {'-':<5}"
                    )

    logging.info("=" * 115)

    # 4. Save Overall Report
    report_path = paths["base"] / "evaluation_summary.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=4)

    logging.info(f"Evaluation finished. Summary saved to {report_path}")


if __name__ == "__main__":
    main()
