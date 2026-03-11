import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List


def check_image_exists(path: str, root: str) -> bool:
    if path is None:  # Handle text-only options which have null image_path
        return True
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
    gt_id = query.get("ground_truth_id")

    # 2. Gallery Size Check (N images + 1 text option)
    # n_val passed here is usually the "N" from filename, which refers to number of distractors
    if n_val != -1:
        expected_len = n_val + 1
        if len(gallery) != expected_len:
            errors.append(
                f"Task {task_id}: Gallery size mismatch. Expected {expected_len} (N={n_val}+1), got {len(gallery)}."
            )

    # 3. Protocol 6 Specific Logic: Open Set / Negative Query
    
    # A. Ensure GT ID is NOT in gallery
    for opt in gallery:
        opt_id = opt.get("id")
        # Skip the text option which might have id=None
        if opt_id is not None and str(opt_id) == str(gt_id):
            errors.append(
                f"Task {task_id}: PROTOCOL VIOLATION. Ground Truth ID {gt_id} found in gallery option {opt.get('option')}."
            )

    # B. Ensure Answer is 'None of the above'
    if not gallery:
        errors.append(f"Task {task_id}: Gallery is empty.")
        return errors

    # The last option should be the text option
    last_option = gallery[-1]
    
    if last_option.get("text") != "None of the above":
        errors.append(f"Task {task_id}: Last option is not 'None of the above'. Found: {last_option.get('text')}")
    
    if answer != last_option.get("option"):
        errors.append(
            f"Task {task_id}: Answer mismatch. Expected {last_option.get('option')} (None), got {answer}."
        )

    # 4. File Existence
    q_path = query.get("image_path")
    if q_path and not check_image_exists(q_path, data_root):
        errors.append(f"Task {task_id}: Query image not found: {q_path}")

    for opt in gallery:
        g_path = opt.get("image_path")
        if g_path and not check_image_exists(g_path, data_root):
            errors.append(f"Task {task_id}: Gallery image not found: {g_path}")

    return errors


def verify_dataset(dataset_name: str, annotations_dir: str, data_root: str) -> bool:
    p6_dir = os.path.join(annotations_dir, dataset_name, "p6")
    if not os.path.exists(p6_dir):
        print(f"  [SKIP] P6 directory not found: {p6_dir}")
        return True

    print(f"Verifying P6 dataset for {dataset_name}...")
    
    # Look for {Species}_MCQ_P6_N{N}.json
    json_files = [f for f in os.listdir(p6_dir) if f.endswith(".json") and "_P6_" in f]

    if not json_files:
        print("  No valid P6 JSON files found.")
        return True

    total_errors = 0

    for f in sorted(json_files):
        file_path = os.path.join(p6_dir, f)
        
        # Extract N
        match = re.search(r"_N(\d+)\.json$", f)
        n_val = int(match.group(1)) if match else -1

        print(f"  Checking {f} (N={n_val})...")

        try:
            with open(file_path, "r") as handle:
                data = json.load(handle)
        except Exception as e:
            print(f"  [ERROR] Failed to load {f}: {e}")
            total_errors += 1
            continue

        file_errors = 0
        for task in data:
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
    parser = argparse.ArgumentParser(description="Verify P6 (Open-Set) Dataset Integrity")
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
        datasets = sorted([d for d in os.listdir(args.annotations_dir) if os.path.isdir(os.path.join(args.annotations_dir, d))])

    failed = []
    for d in datasets:
        if not verify_dataset(d, args.annotations_dir, args.data_root):
            failed.append(d)

    if failed:
        print(f"\nVerification FAILED for: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll P6 datasets passed verification.")

if __name__ == "__main__":
    main()