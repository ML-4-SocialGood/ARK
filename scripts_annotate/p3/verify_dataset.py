"""
scripts/p3/verify_dataset.py
Verify integrity of Protocol 3 (Context-aware) datasets.
"""

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

    # 1. Basic Schema Check
    if "query" not in task or "gallery" not in task or "answer" not in task:
        return [f"Task {task_id}: Missing required fields."]

    query = task["query"]
    gallery = task["gallery"]
    answer = task["answer"]

    # 2. P3 Specific Checks (Metadata & Context)
    if "context_text" not in query:
        errors.append(f"Task {task_id}: Missing 'context_text' in query.")
    elif not isinstance(query["context_text"], dict):
        # We updated sampler to return a dict, so verify this
        errors.append(f"Task {task_id}: 'context_text' should be a dictionary.")
    elif not query["context_text"]:
        errors.append(f"Task {task_id}: 'context_text' is empty.")

    if "metadata" not in query:
        errors.append(f"Task {task_id}: Missing 'metadata' object in query.")

    # 3. Gallery Size
    if n_val != -1 and len(gallery) != n_val:
        errors.append(
            f"Task {task_id}: Gallery size mismatch. Expected {n_val}, got {len(gallery)}."
        )

    # 4. Ground Truth Logic
    gt_id = query.get("ground_truth_id")
    positives = [opt for opt in gallery if str(opt.get("id")) == str(gt_id)]

    q_path = query.get("image_path")

    if len(positives) != 1:
        errors.append(
            f"Task {task_id}: Found {len(positives)} positive matches (expected 1)."
        )
    else:
        if positives[0].get("option") != answer:
            errors.append(f"Task {task_id}: Answer mismatch.")

        # Data Leakage Check
        g_path = positives[0].get("image_path")
        if q_path and g_path and q_path == g_path:
            errors.append(f"Task {task_id}: DATA LEAKAGE (Query == Gallery Image).")

    # 5. File Existence
    if q_path and not check_image_exists(q_path, data_root):
        errors.append(f"Task {task_id}: Query image missing: {q_path}")

    for opt in gallery:
        g_path = opt.get("image_path")
        if g_path and not check_image_exists(g_path, data_root):
            errors.append(f"Task {task_id}: Gallery image missing: {g_path}")

    return errors


def verify_dataset(dataset_name: str, annotations_dir: str, data_root: str) -> bool:
    p3_dir = os.path.join(annotations_dir, dataset_name, "p3")
    if not os.path.exists(p3_dir):
        print(f"  [SKIP] P3 directory not found: {p3_dir}")
        return True

    print(f"Verifying P3 dataset for {dataset_name}...")
    json_files = [
        f for f in os.listdir(p3_dir) if f.endswith(".json") and "CIR_P3" in f
    ]

    if not json_files:
        print("  No valid P3 JSON files found.")
        return True

    total_errors = 0
    for f in sorted(json_files):
        file_path = os.path.join(p3_dir, f)

        # Infer N from filename (e.g., ..._N4.json)
        match = re.search(r"_N(\d+)\.json$", f)
        n_val = int(match.group(1)) if match else -1

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
            # Fallback if N not in filename
            if n_val == -1 and "gallery" in task:
                n_val = len(task["gallery"])

            errs = verify_task(task, n_val, data_root)
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
    parser = argparse.ArgumentParser(description="Verify P3 (CIR) Dataset Integrity")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--annotations_dir", type=str, default="annotations")
    parser.add_argument("--data_root", type=str, default=".")
    args = parser.parse_args()

    if args.dataset_name:
        datasets = [args.dataset_name]
    else:
        if not os.path.exists(args.annotations_dir):
            print("Annotations directory not found.")
            return

        # Recursively find all directories containing a 'p3' folder
        datasets = []
        for root, dirs, files in os.walk(args.annotations_dir):
            if "p3" in dirs:
                # The dataset name is the path relative to annotations_dir
                # e.g., annotations/MetaWild/Deer -> MetaWild/Deer
                rel_path = os.path.relpath(root, args.annotations_dir)
                datasets.append(rel_path)
        datasets.sort()

    failed = []
    for d in datasets:
        if not verify_dataset(d, args.annotations_dir, args.data_root):
            failed.append(d)

    if failed:
        print(f"\nVerification FAILED for: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll datasets passed verification.")


if __name__ == "__main__":
    main()
