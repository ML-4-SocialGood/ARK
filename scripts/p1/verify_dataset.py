import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List


def check_image_exists(path: str, root: str) -> bool:
    full_path = os.path.join(root, path)
    return os.path.exists(full_path)


def verify_task(task: Dict[str, Any], n_val: int, data_root: str) -> List[str]:
    errors = []
    task_id = task.get("task_id", "Unknown")

    # 1. Schema Check
    if "query" not in task or "gallery" not in task or "answer" not in task:
        return [f"Task {task_id}: Missing required fields (query, gallery, answer)."]

    query = task["query"]
    gallery = task["gallery"]
    answer = task["answer"]

    # 2. Gallery Size Check
    if n_val != -1 and len(gallery) != n_val:
        errors.append(
            f"Task {task_id}: Gallery size mismatch. Expected {n_val}, got {len(gallery)}."
        )

    # 3. Ground Truth & Answer Logic
    gt_id = query.get("ground_truth_id")
    if gt_id is None:
        errors.append(f"Task {task_id}: Missing ground_truth_id in query.")
        return errors

    # Find positives in gallery
    # Note: IDs are usually strings, ensure comparison is robust
    positives = [opt for opt in gallery if str(opt.get("id")) == str(gt_id)]

    if len(positives) == 0:
        errors.append(
            f"Task {task_id}: No positive match found in gallery for GT ID {gt_id}."
        )
    elif len(positives) > 1:
        errors.append(
            f"Task {task_id}: Multiple positive matches found in gallery for GT ID {gt_id}."
        )
    else:
        # Exactly one positive
        positive_opt = positives[0]

        # Check Answer Label
        if positive_opt.get("option") != answer:
            errors.append(
                f"Task {task_id}: Answer mismatch. GT is option {positive_opt.get('option')}, but answer says {answer}."
            )

        # Check Data Leakage (Query path vs Gallery path)
        # P1 query usually has 'image_path' (singular)
        q_path = query.get("image_path")
        g_path = positive_opt.get("image_path")

        if q_path and g_path and q_path == g_path:
            errors.append(
                f"Task {task_id}: DATA LEAKAGE. Query image is identical to gallery image: {q_path}"
            )

    # 4. File Existence
    q_path = query.get("image_path")
    if q_path:
        if not check_image_exists(q_path, data_root):
            errors.append(f"Task {task_id}: Query image not found on disk: {q_path}")
    else:
        # Fallback check if schema uses image_paths (plural)
        if "image_paths" in query:
            for p in query["image_paths"]:
                if not check_image_exists(p, data_root):
                    errors.append(f"Task {task_id}: Query image not found on disk: {p}")
        else:
            errors.append(f"Task {task_id}: Query image path missing.")

    for opt in gallery:
        g_path = opt.get("image_path")
        if g_path:
            if not check_image_exists(g_path, data_root):
                errors.append(
                    f"Task {task_id}: Gallery image not found on disk: {g_path}"
                )
        else:
            errors.append(f"Task {task_id}: Gallery option missing image_path.")

    return errors


def verify_dataset(dataset_name: str, annotations_dir: str, data_root: str) -> bool:
    p1_dir = os.path.join(annotations_dir, dataset_name, "p1")
    if not os.path.exists(p1_dir):
        print(f"  [SKIP] P1 directory not found: {p1_dir}")
        return True

    print(f"Verifying P1 dataset for {dataset_name}...")

    json_files = [
        f for f in os.listdir(p1_dir) if f.endswith(".json") and "I2I_P1" in f
    ]

    if not json_files:
        print("  No valid P1 JSON files found.")
        return True

    total_errors = 0

    for f in sorted(json_files):
        file_path = os.path.join(p1_dir, f)

        # Extract N from filename
        match = re.search(r"_N(\d+)\.json$", f)
        if match:
            n_val = int(match.group(1))
        else:
            # If N is not in filename, we can infer it from the first task later, or skip size check
            n_val = -1

        print(f"  Checking {f} (Expected N={n_val if n_val != -1 else 'Auto'})...")

        try:
            with open(file_path, "r") as handle:
                data = json.load(handle)
        except Exception as e:
            print(f"  [ERROR] Failed to load {f}: {e}")
            total_errors += 1
            continue

        if not isinstance(data, list):
            print(f"  [ERROR] Root of {f} is not a list.")
            total_errors += 1
            continue

        file_errors = 0
        for task in data:
            # If n_val was not found in filename, try to infer from first task
            if n_val == -1 and "gallery" in task:
                n_val = len(task["gallery"])

            errs = verify_task(task, n_val, data_root)
            if errs:
                file_errors += 1
                total_errors += 1
                if file_errors <= 5:  # Limit output per file
                    for e in errs:
                        print(f"    - {e}")

        if file_errors > 5:
            print(f"    ... and {file_errors - 5} more errors in this file.")

        if file_errors == 0:
            print("    OK.")

    print("-" * 40)
    if total_errors == 0:
        print(f"SUCCESS: Dataset {dataset_name} passed P1 checks.")
        return True
    else:
        print(f"FAILURE: Found {total_errors} errors in {dataset_name}.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify P1 (I2I) Dataset Integrity")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the dataset (e.g., BelugaID). If not provided, verifies all.",
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="annotations",
        help="Root of annotations",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="Root for checking image paths (usually project root)",
    )
    args = parser.parse_args()

    if args.dataset_name:
        datasets = [args.dataset_name]
    else:
        if not os.path.exists(args.annotations_dir):
            print(f"Error: Annotations directory '{args.annotations_dir}' not found.")
            return
        datasets = sorted(
            [
                d
                for d in os.listdir(args.annotations_dir)
                if os.path.isdir(os.path.join(args.annotations_dir, d))
            ]
        )
        print(f"Found {len(datasets)} datasets. Verifying all...")

    failed_datasets = []
    for d in datasets:
        if not verify_dataset(d, args.annotations_dir, args.data_root):
            failed_datasets.append(d)

    print("\n" + "=" * 60)
    if failed_datasets:
        print(f"Verification FAILED for {len(failed_datasets)} datasets:")
        for d in failed_datasets:
            print(f"  - {d}")
        sys.exit(1)
    else:
        print("All datasets passed verification successfully.")


if __name__ == "__main__":
    main()
