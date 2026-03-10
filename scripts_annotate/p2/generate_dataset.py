import argparse
import json
import os
import random
import sys
from collections import defaultdict

# Ensure imports work when running from project root
sys.path.append(os.getcwd())

from scripts_annotate.p2.sampler import MultiImageBatchSampler


def main():
    parser = argparse.ArgumentParser(
        description="Generate MCQ Dataset for Animal Re-ID (Protocol 2: Multi-Image Query)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., BelugaID)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="annotations",
        help="Root directory for output annotations",
    )
    parser.add_argument(
        "--gallery_size",
        type=int,
        default=4,
        help="Number of options in the gallery (N)",
    )
    parser.add_argument(
        "--max_query_size",
        type=int,
        default=4,
        help="Maximum number of images in the query set (K_max)",
    )
    parser.add_argument(
        "--max_queries_per_id",
        type=int,
        default=5,
        help="Max times an ID can be used as a target (generates a batch each time)",
    )
    parser.add_argument(
        "--target_batches",
        type=int,
        default=-1,
        help="Target number of batches to generate. -1 (default) to generate until exhaustion.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist.")
        return

    # Set random seed
    random.seed(args.seed)

    print(f"Initializing P2 Sampler for {args.dataset_name}...")
    print(f"  Gallery Size (N): {args.gallery_size}")
    print(f"  Max Query Size (K): {args.max_query_size}")

    try:
        sampler = MultiImageBatchSampler(
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            gallery_size=args.gallery_size,
            max_query_size=args.max_query_size,
            max_queries_per_id=args.max_queries_per_id,
        )
    except Exception as e:
        print(f"Failed to initialize sampler: {e}")
        return

    # Calculate theoretical maximum batches for information purposes
    theoretical_max_batches = 0
    for qid in sampler.valid_query_ids:
        limit = min(args.max_queries_per_id, len(sampler.image_map[qid]))
        theoretical_max_batches += limit

    print(f"Estimated maximum batches: {theoretical_max_batches}")
    print(
        f"Total expected tasks (approx): {theoretical_max_batches * args.max_query_size}"
    )

    tasks_by_k = defaultdict(list)
    batches_generated = 0

    print("Starting generation...")

    while True:
        # Check if we reached the user-specified target (if any)
        if args.target_batches != -1 and batches_generated >= args.target_batches:
            print(f"\nReached target batch count: {args.target_batches}")
            break

        batch_tasks = sampler.generate_batch()

        if batch_tasks is None:
            print("\nSampler returned None (exhausted). Stopping generation.")
            break

        # Distribute tasks into separate lists based on K
        for task in batch_tasks:
            k = task["meta"]["query_size"]
            tasks_by_k[k].append(task)

        batches_generated += 1

        if batches_generated % 10 == 0:
            total_tasks = sum(len(t) for t in tasks_by_k.values())
            print(
                f"Generated {batches_generated} batches ({total_tasks} tasks)...",
                end="\r",
            )

    print("\nGeneration complete.")
    print(f"Total Batches: {batches_generated}")

    # Construct output path
    output_subdir = os.path.join(args.output_dir, args.dataset_name, "p2")
    os.makedirs(output_subdir, exist_ok=True)

    # Save separate files for each K
    for k in sorted(tasks_by_k.keys()):
        tasks = tasks_by_k[k]
        output_filename = f"{args.dataset_name}_MCQ_P2_N{args.gallery_size}_K{k}.json"
        output_path = os.path.join(output_subdir, output_filename)

        with open(output_path, "w") as f:
            json.dump(tasks, f, indent=2)

        print(f"Dataset (K={k}) saved to {output_path} ({len(tasks)} tasks)")


if __name__ == "__main__":
    main()
