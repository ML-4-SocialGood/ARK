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

    if "query" not in task or "gallery" not in task or "answer" not in task:
        return [f"Task {task_id}: Missing required fields."]

    query = task["query"]
    gallery = task["gallery"]

    if query is None:
        return [f"Task {task_id}: Query field is None."]
    
    if gallery is None:
        return [f"Task {task_id}: Gallery field is None."]

    # 1. Check Query Image (Must be corrupted)
    q_path = query.get("image_path")
    if not q_path:
        errors.append(f"Task {task_id}: Query image path missing.")
    else:
        # P5 specific check: Query should point to the corrupted directory
        if "corrupted" not in q_path:
            errors.append(
                f"Task {task_id}: Query image path does not look corrupted (missing 'corrupted' in path): {q_path}"
            )
        
        if not check_image_exists(q_path, data_root):
            errors.append(f"Task {task_id}: Query image missing on disk: {q_path}")

    # 2. Check Gallery Images (Must be clear/original)
    for opt in gallery:
        g_path = opt.get("image_path")
        if not g_path:
            errors.append(f"Task {task_id}: Gallery option missing image_path.")
            continue

        # P5 specific check: Gallery should NOT point to the corrupted directory
        if "corrupted" in g_path:
            errors.append(
                f"Task {task_id}: Gallery image path looks corrupted (contains 'corrupted'): {g_path}. Gallery should be clear."
            )

        if not check_image_exists(g_path, data_root):
            errors.append(f"Task {task_id}: Gallery image missing on disk: {g_path}")

    # 3. Ground Truth & Answer Logic
    gt_id = query.get("ground_truth_id")
    if not gt_id:
        errors.append(f"Task {task_id}: Missing ground_truth_id.")

    # Find positive in gallery
    positives = [opt for opt in gallery if str(opt.get("id")) == str(gt_id)]

    if len(positives) != 1:
        errors.append(
            f"Task {task_id}: Expected exactly 1 positive in gallery, found {len(positives)}."
        )
    else:
        if positives[0].get("option") != task["answer"]:
            errors.append(
                f"Task {task_id}: Answer mismatch. GT option is {positives[0].get('option')}, but answer says {task['answer']}."
            )

    return errors


def verify_dataset(dataset_name: str, annotations_dir: str, data_root: str) -> bool:
    p5_dir = os.path.join(annotations_dir, dataset_name, "p5")
    if not os.path.exists(p5_dir):
        print(f"  [SKIP] P5 directory not found: {p5_dir}")
        return True

    print(f"Verifying P5 dataset for {dataset_name}...")
    json_files = [
        f for f in os.listdir(p5_dir) if f.endswith(".json") and "_P5_" in f
    ]

    if not json_files:
        print("  No valid P5 JSON files found.")
        return True

    total_errors = 0
    for f in sorted(json_files):
        file_path = os.path.join(p5_dir, f)
        print(f"  Checking {f}...")

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
            errs = verify_task(task, data_root)
            if errs:
                file_errors += 1
                total_errors += 1
                if file_errors <= 5:
                    for e in errs:
                        print(f"    - {e}")

        if file_errors > 5:
            print(f"    ... and {file_errors - 5} more errors in this file.")

        if file_errors == 0:
            print("    OK.")
        else:
            print(f"    Found {file_errors} errors in this file.")

    return total_errors == 0


def main():
    parser = argparse.ArgumentParser(description="Verify P5 (Corrupted) Dataset Integrity")
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