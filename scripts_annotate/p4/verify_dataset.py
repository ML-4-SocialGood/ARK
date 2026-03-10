import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List


def check_image_exists(path: str, root: str) -> bool:
    full_path = os.path.join(root, path)
    return os.path.exists(full_path)


def verify_task(task: Dict[str, Any], n_val: int, data_root: str) -> List[str]:
    errors = []
    task_id = task.get("task_id", "Unknown")

    if "query" not in task or "gallery" not in task or "answer" not in task:
        return [f"Task {task_id}: Missing required fields."]

    query = task["query"]
    gallery = task["gallery"]
    answer = task["answer"]

    # 1. Gallery Size Check
    if n_val != -1 and len(gallery) != n_val:
        errors.append(
            f"Task {task_id}: Gallery size mismatch. Expected {n_val}, got {len(gallery)}."
        )

    # 2. Ground Truth Logic
    gt_id = query.get("ground_truth_id")
    if not gt_id:
        errors.append(f"Task {task_id}: Missing ground_truth_id.")

    # 3. Answer Validity
    # Find option matching GT
    correct_opts = [opt for opt in gallery if str(opt.get("id")) == str(gt_id)]
    
    if len(correct_opts) != 1:
        errors.append(f"Task {task_id}: Expected exactly 1 correct option, found {len(correct_opts)}.")
    elif correct_opts[0]["option"] != answer:
        errors.append(f"Task {task_id}: Answer '{answer}' does not match correct option '{correct_opts[0]['option']}'.")

    # 4. Context Check (CIR specific)
    # Protocol 4 usually involves context_text
    if "context_text" not in query:
        # Warning or Error depending on strictness. Let's error for P4.
        errors.append(f"Task {task_id}: Missing 'context_text' in query (Required for P4 CIR).")

    # 5. Data Leakage
    q_path = query.get("image_path")
    for opt in gallery:
        if opt.get("image_path") == q_path:
             errors.append(f"Task {task_id}: DATA LEAKAGE. Query image found in gallery: {q_path}")

    # 6. File Existence
    if q_path and not check_image_exists(q_path, data_root):
        errors.append(f"Task {task_id}: Query image missing: {q_path}")

    for opt in gallery:
        if not check_image_exists(opt["image_path"], data_root):
            errors.append(f"Task {task_id}: Gallery image missing: {opt['image_path']}")

    return errors


def verify_dataset(dataset_name: str, annotations_dir: str, data_root: str) -> bool:
    p4_dir = os.path.join(annotations_dir, dataset_name, "p4")
    if not os.path.exists(p4_dir):
        print(f"  [SKIP] P4 directory not found: {p4_dir}")
        return True

    print(f"Verifying P4 dataset for {dataset_name}...")

    # Group files by N (Gallery Size)
    files_by_n = defaultdict(list)
    for f in os.listdir(p4_dir):
        if f.endswith(".json") and "_CIR_P4_" in f:
            # Expected: {Species}_CIR_P4_N{N}.json
            try:
                parts = f.split("_")
                # Find part starting with N
                n_part = next(
                    (p for p in parts if p.startswith("N") and p[1:].isdigit()), None
                )
                if n_part:
                    n = int(n_part[1:])
                    files_by_n[n].append(os.path.join(p4_dir, f))
            except Exception:
                print(f"Skipping unrecognized file: {f}")

    if not files_by_n:
        print("No valid P4 (CIR) JSON files found.")
        return True

    total_errors = 0

    for n in sorted(files_by_n.keys()):
        print(f"\n=== Verifying Gallery Size N={n} ===")
        file_paths = files_by_n[n]

        print(f"  Loading {len(file_paths)} files...")
        for fp in file_paths:
            try:
                with open(fp, "r") as f:
                    data = json.load(f)
                    
                    file_errors = 0
                    for task in data:
                        errors = verify_task(task, n, data_root)
                        if errors:
                            file_errors += 1
                            total_errors += 1
                            if file_errors <= 5: # Limit output per file
                                for err in errors:
                                    print(f"    - {err}")
                    
                    if file_errors > 0:
                        print(f"  [WARN] {file_errors} errors in {os.path.basename(fp)}")
                        
            except Exception as e:
                print(f"  [ERROR] Failed to load {fp}: {e}")
                total_errors += 1


    print("\n" + "=" * 50)
    if total_errors == 0:
        print(f"SUCCESS: Dataset {dataset_name} passed all P4 integrity checks.")
        return True
    else:
        print(f"FAILURE: Found issues in {dataset_name}.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify P4 CIR Dataset Integrity")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the dataset (e.g., BelugaID). If not provided, verifies all datasets.",
    )
    parser.add_argument(
        "--annotations_dir", type=str, default="annotations", help="Root of annotations"
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
        print(
            f"Found {len(datasets)} datasets in '{args.annotations_dir}'. Verifying all..."
        )

    failed_datasets = []
    for dataset in datasets:
        print(f"\n>>> Processing Dataset: {dataset}")
        if not verify_dataset(dataset, args.annotations_dir, args.data_root):
            failed_datasets.append(dataset)

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
