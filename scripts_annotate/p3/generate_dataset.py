"""
scripts/p3/generate_dataset.py
Generate Dataset for Protocol 3: Context-aware Interleaved Reasoning.
"""

import argparse
import json
import os
import random
import sys

# Ensure imports work when running from project root
sys.path.append(os.getcwd())

from scripts.p3.sampler import ContextAwareSampler


def main():
    parser = argparse.ArgumentParser(
        description="Generate Dataset for Protocol 3 (Context-aware Interleaved Reasoning)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., MetaWild_Deer)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to species directory (e.g., data/MetaWild/Deer)",
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
        "--max_queries_per_id", type=int, default=5, help="Max queries per ID"
    )
    parser.add_argument(
        "--target_samples",
        type=int,
        default=-1,
        help="Target number of samples. -1 for maximum possible.",
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

    # Sanitize dataset_name for filename and task_id (replace / with _)
    safe_dataset_name = args.dataset_name.replace("/", "_").replace("\\", "_")

    print(f"Initializing P3 Sampler for {safe_dataset_name} (N={args.gallery_size})...")

    try:
        sampler = ContextAwareSampler(
            data_dir=args.data_dir,
            gallery_size=args.gallery_size,
            max_queries_per_id=args.max_queries_per_id,
            dataset_name=safe_dataset_name,
        )
    except Exception as e:
        print(f"Failed to initialize sampler: {e}")
        return

    # Calculate theoretical maximum
    theoretical_max = 0
    for qid in sampler.valid_query_ids:
        limit = min(args.max_queries_per_id, len(sampler.image_map[qid]))
        theoretical_max += limit

    print(f"Theoretical maximum samples: {theoretical_max}")

    if args.target_samples == -1:
        target_count = theoretical_max
    else:
        target_count = args.target_samples

    generated_samples = []
    print(f"Starting generation of {target_count} samples...")

    for i in range(target_count):
        sample = sampler.generate_sample()

        if sample is None:
            print(f"\nSampler exhausted at iteration {i + 1}.")
            break

        generated_samples.append(sample)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{target_count}...", end="\r")

    print(f"\nTotal generated: {len(generated_samples)} samples.")

    # Output path: annotations/{dataset_name}/p3/{safe_dataset_name}_CIR_P3_N{N}.json
    # Use the raw dataset_name for directory structure to match verify_dataset.py logic
    output_subdir = os.path.join(args.output_dir, args.dataset_name, "p3")
    os.makedirs(output_subdir, exist_ok=True)

    output_filename = f"{safe_dataset_name}_CIR_P3_N{args.gallery_size}.json"
    output_path = os.path.join(output_subdir, output_filename)

    with open(output_path, "w") as f:
        json.dump(generated_samples, f, indent=2)

    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
