import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List


def check_image_exists(path: str, root: str) -> bool:
    full_path = os.path.join(root, path)
    return os.path.exists(full_path)


def verify_batch(batch_id: int, tasks: Dict[int, Any], data_root: str) -> List[str]:
    errors = []
    sorted_ks = sorted(tasks.keys())

    if not sorted_ks:
        return ["No tasks found for batch."]

    # Use K=1 as the reference for Gallery and Ground Truth
    ref_k = sorted_ks[0]
    ref_task = tasks[ref_k]
    ref_gallery = ref_task["gallery"]
    ref_gt_id = ref_task["query"]["ground_truth_id"]

    # Serialize gallery for easy comparison
    ref_gallery_str = json.dumps(ref_gallery, sort_keys=True)

    # Track query images to ensure incremental growth
    prev_query_images = set()

    for k in sorted_ks:
        task = tasks[k]

        # 1. Verify Ground Truth ID Consistency
        if task["query"]["ground_truth_id"] != ref_gt_id:
            errors.append(
                f"[K={k}] Ground Truth ID mismatch. Expected {ref_gt_id}, got {task['query']['ground_truth_id']}"
            )

        # 2. Verify Fixed Gallery (Must be identical across all K)
        curr_gallery_str = json.dumps(task["gallery"], sort_keys=True)
        if curr_gallery_str != ref_gallery_str:
            errors.append(
                f"[K={k}] Gallery does not match the reference gallery (from K={ref_k}). Gallery must be fixed within a batch."
            )

        # 3. Verify Answer Validity
        # Find the option in gallery that matches GT ID
        correct_opts = [
            opt["option"] for opt in task["gallery"] if opt["id"] == ref_gt_id
        ]
        if len(correct_opts) != 1:
            errors.append(
                f"[K={k}] Gallery must contain exactly one correct option. Found {len(correct_opts)}."
            )
        elif task["answer"] != correct_opts[0]:
            errors.append(
                f"[K={k}] Answer field '{task['answer']}' does not match the correct option '{correct_opts[0]}'."
            )

        # 4. Verify Query Set Growth (Incremental)
        curr_query_images = set(task["query"]["image_paths"])
        if len(curr_query_images) != k:
            errors.append(
                f"[K={k}] Query image count mismatch. Expected {k}, got {len(curr_query_images)}."
            )

        if k > ref_k:
            # Current query set must be a superset of the previous one
            if not prev_query_images.issubset(curr_query_images):
                errors.append(
                    f"[K={k}] Query set is not a superset of K={k - 1}. Incremental growth violation."
                )

        prev_query_images = curr_query_images

        # 5. Verify Data Leakage (Positive Gallery Image vs Query Images)
        # The positive image in the gallery must NOT be in the query set
        pos_img_in_gallery = next(
            (opt["image_path"] for opt in task["gallery"] if opt["id"] == ref_gt_id),
            None,
        )
        if pos_img_in_gallery and pos_img_in_gallery in curr_query_images:
            errors.append(
                f"[K={k}] DATA LEAKAGE: Positive gallery image '{pos_img_in_gallery}' is present in the query set."
            )

        # 6. Verify File Existence (Sample check or full check)
        # Checking all files might be slow, but let's do it for correctness
        for img in curr_query_images:
            if not check_image_exists(img, data_root):
                errors.append(f"[K={k}] Query image not found on disk: {img}")

        for opt in task["gallery"]:
            if not check_image_exists(opt["image_path"], data_root):
                errors.append(
                    f"[K={k}] Gallery image not found on disk: {opt['image_path']}"
                )

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
        if f.endswith(".json") and "_P4_" in f:
            # Expected: {Species}_MCQ_P4_N{N}_K{K}.json
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
        print("No valid P4 JSON files found.")
        return True

    total_errors = 0

    for n in sorted(files_by_n.keys()):
        print(f"\n=== Verifying Gallery Size N={n} ===")
        file_paths = files_by_n[n]

        # Load and group tasks by batch_id
        tasks_by_batch = defaultdict(dict)  # batch_id -> {k -> task}

        print(f"  Loading {len(file_paths)} files...")
        for fp in file_paths:
            try:
                with open(fp, "r") as f:
                    data = json.load(f)
                    for task in data:
                        batch_id = task["meta"]["batch_id"]
                        k = task["meta"]["query_size"]
                        tasks_by_batch[batch_id][k] = task
            except Exception as e:
                print(f"  [ERROR] Failed to load {fp}: {e}")
                total_errors += 1

        print(f"  Verifying {len(tasks_by_batch)} batches...")

        batch_errors = 0
        for batch_id, tasks in tasks_by_batch.items():
            errors = verify_batch(batch_id, tasks, data_root)
            if errors:
                batch_errors += 1
                total_errors += 1
                print(f"  [Batch {batch_id} Issues]:")
                for err in errors:
                    print(f"    - {err}")
                # Limit error output to avoid flooding
                if batch_errors >= 10:
                    print("  ... Too many errors, stopping verification for this N.")
                    break

        if batch_errors == 0:
            print(f"  [OK] All batches for N={n} passed verification.")

    print("\n" + "=" * 50)
    if total_errors == 0:
        print(f"SUCCESS: Dataset {dataset_name} passed all P4 integrity checks.")
        return True
    else:
        print(f"FAILURE: Found issues in {dataset_name}.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify P4 MCQ Dataset Integrity")
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
