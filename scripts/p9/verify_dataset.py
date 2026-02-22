import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List


def check_image_exists(path: str, root: str) -> bool:
    full_path = os.path.join(root, path)
    return os.path.exists(full_path)


def verify_task(
    task: Dict[str, Any], n_val: int, m_val: int, data_root: str
) -> List[str]:
    errors = []
    task_id = task.get("task_id", "Unknown")

    if "query" not in task or "gallery" not in task or "answer" not in task:
        return [f"Task {task_id}: Missing required fields."]

    query = task["query"]
    gallery = task["gallery"]
    answer_str = task["answer"]

    # 1. Gallery Size Check
    if n_val != -1 and len(gallery) != n_val:
        errors.append(
            f"Task {task_id}: Gallery size mismatch. Expected {n_val}, got {len(gallery)}."
        )

    # 2. Ground Truth Logic
    gt_id = query.get("ground_truth_id")
    if not gt_id:
        errors.append(f"Task {task_id}: Missing ground_truth_id.")
        return errors

    # Find positives in gallery
    positives = [opt for opt in gallery if str(opt.get("id")) == str(gt_id)]

    # P9 Requirement: Multiple positives
    if len(positives) < 2:
        errors.append(
            f"Task {task_id}: Protocol 9 violation. Found {len(positives)} positives, expected >= 2."
        )

    if m_val != -1 and len(positives) != m_val:
        errors.append(
            f"Task {task_id}: Positive count mismatch. Expected {m_val}, found {len(positives)}."
        )

    # 3. Answer String Check
    # Answer should be like "A, C"
    answer_labels = [x.strip() for x in answer_str.split(",")]
    positive_labels = sorted([opt["option"] for opt in positives])

    if sorted(answer_labels) != positive_labels:
        errors.append(
            f"Task {task_id}: Answer mismatch. GT options are {positive_labels}, but answer says {answer_labels}."
        )

    # 4. Data Leakage
    q_path = query.get("image_path")
    for opt in gallery:
        if opt.get("image_path") == q_path:
            errors.append(
                f"Task {task_id}: DATA LEAKAGE. Query image found in gallery: {q_path}"
            )

    # 5. File Existence
    if q_path and not check_image_exists(q_path, data_root):
        errors.append(f"Task {task_id}: Query image missing: {q_path}")

    for opt in gallery:
        if not check_image_exists(opt["image_path"], data_root):
            errors.append(f"Task {task_id}: Gallery image missing: {opt['image_path']}")

    return errors


def verify_dataset(dataset_name: str, annotations_dir: str, data_root: str) -> bool:
    p9_dir = os.path.join(annotations_dir, dataset_name, "p9")
    if not os.path.exists(p9_dir):
        print(f"  [SKIP] P9 directory not found: {p9_dir}")
        return True

    print(f"Verifying P9 dataset for {dataset_name}...")
    json_files = [
        f for f in os.listdir(p9_dir) if f.endswith(".json") and "MIA_P9" in f
    ]

    if not json_files:
        print("  No valid P9 JSON files found.")
        return True

    total_errors = 0
    for f in sorted(json_files):
        file_path = os.path.join(p9_dir, f)

        # Extract N and M from filename: {Species}_MIA_P9_N{N}_M{M}.json
        n_match = re.search(r"_N(\d+)", f)
        n_val = int(n_match.group(1)) if n_match else -1

        m_match = re.search(r"_M(\d+)", f)
        m_val = int(m_match.group(1)) if m_match else -1

        print(f"  Checking {f} (Expect N={n_val}, M={m_val})...")

        try:
            with open(file_path, "r") as handle:
                data = json.load(handle)
        except Exception as e:
            print(f"  [ERROR] Failed to load {f}: {e}")
            total_errors += 1
            continue

        if not data:
            continue

        # Fallback inference if filename parsing failed
        if n_val == -1:
            n_val = len(data[0]["gallery"])

        file_errors = 0
        for task in data:
            errs = verify_task(task, n_val, m_val, data_root)
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
    parser = argparse.ArgumentParser(description="Verify P9 Dataset Integrity")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--annotations_dir", type=str, default="annotations")
    parser.add_argument("--data_root", type=str, default=".")
    args = parser.parse_args()

    if args.dataset_name:
        datasets = [args.dataset_name]
    else:
        if not os.path.exists(args.annotations_dir):
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
        print(f"Verification FAILED for: {failed}")
        sys.exit(1)
    else:
        print("All datasets passed.")


if __name__ == "__main__":
    main()
