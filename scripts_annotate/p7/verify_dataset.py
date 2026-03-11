"""
Verify Dataset for Protocol 7: Counterfactual Discernment.
Checks for file existence and ensures pairs are actually different IDs.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List


def check_image_exists(path: str, root: str) -> bool:
    full_path = os.path.join(root, path)
    return os.path.exists(full_path)


def verify_task(task: Dict[str, Any], data_root: str) -> List[str]:
    errors = []
    task_id = task.get("task_id", "Unknown")

    # 1. Schema Check
    required_fields = [
        "image_a",
        "image_b",
        "ground_truth",
        "instruction_counterfactual",
        "instruction_neutral",
    ]
    for field in required_fields:
        if field not in task:
            return [f"Task {task_id}: Missing required field '{field}'."]

    img_a = task["image_a"]
    img_b = task["image_b"]
    gt = task["ground_truth"]

    # 2. Protocol Check: Must be different IDs
    if str(img_a["id"]) == str(img_b["id"]):
        errors.append(
            f"Task {task_id}: PROTOCOL VIOLATION. IDs are identical ({img_a['id']}). P7 requires different IDs."
        )

    if gt != "different":
        errors.append(f"Task {task_id}: Ground truth is '{gt}', expected 'different'.")

    # 3. File Existence
    if not check_image_exists(img_a["image_path"], data_root):
        errors.append(f"Task {task_id}: Image A not found: {img_a['image_path']}")

    if not check_image_exists(img_b["image_path"], data_root):
        errors.append(f"Task {task_id}: Image B not found: {img_b['image_path']}")

    return errors


def verify_dataset(dataset_name: str, annotations_dir: str, data_root: str) -> bool:
    p7_dir = os.path.join(annotations_dir, dataset_name, "p7")
    if not os.path.exists(p7_dir):
        print(f"  [SKIP] P7 directory not found: {p7_dir}")
        return True

    print(f"Verifying P7 dataset for {dataset_name}...")

    json_files = [f for f in os.listdir(p7_dir) if f.endswith(".json") and "_P7" in f]

    if not json_files:
        print("  No valid P7 JSON files found.")
        return True

    total_errors = 0

    for f in sorted(json_files):
        file_path = os.path.join(p7_dir, f)
        print(f"  Checking {f}...")

        try:
            with open(file_path, "r") as handle:
                data = json.load(handle)
        except Exception as e:
            print(f"  [ERROR] Failed to load {f}: {e}")
            total_errors += 1
            continue

        file_errors = 0
        for task in data:
            errs = verify_task(task, data_root)
            if errs:
                file_errors += 1
                total_errors += 1
                if file_errors <= 5:
                    for e in errs:
                        print(f"    - {e}")

        if file_errors == 0:
            print("    OK.")
        else:
            print(f"    Found {file_errors} errors in this file.")

    return total_errors == 0


def main():
    parser = argparse.ArgumentParser(
        description="Verify P7 (Counterfactual) Dataset Integrity"
    )
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--annotations_dir", type=str, default="annotations")
    parser.add_argument("--data_root", type=str, default=".")
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

    failed = []
    for d in datasets:
        if not verify_dataset(d, args.annotations_dir, args.data_root):
            failed.append(d)

    if failed:
        print(f"\nVerification FAILED for: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll P7 datasets passed verification.")


if __name__ == "__main__":
    main()