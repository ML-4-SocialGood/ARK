import argparse
import json
import os
import random
import sys

# Ensure imports work when running from project root
sys.path.append(os.getcwd())

from scripts_annotate.p6.sampler import OpenSetSampler


def main():
    parser = argparse.ArgumentParser(
        description="Generate MCQ Dataset for Animal Re-ID (Protocol 6: Open-set Reliability)"
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
        help="Number of distractor images in the gallery (N). Total options will be N+1.",
    )
    parser.add_argument(
        "--max_queries_per_id", type=int, default=5, help="Max queries per ID"
    )
    parser.add_argument(
        "--target_samples",
        type=int,
        default=-1,
        help="Target number of samples to generate. -1 for maximum possible.",
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

    print(f"Initializing P6 Sampler (Open-set) for {args.dataset_name}...")
    try:
        sampler = OpenSetSampler(
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            gallery_size=args.gallery_size,
            max_queries_per_id=args.max_queries_per_id,
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
    print("Starting generation...")

    for i in range(target_count):
        sample = sampler.generate_sample()

        if sample is None:
            print(f"\nSampler exhausted at iteration {i + 1}.")
            break

        generated_samples.append(sample)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{target_count} samples...", end="\r")

    print(f"\nTotal generated: {len(generated_samples)} samples.")

    # Construct output path
    output_subdir = os.path.join(args.output_dir, args.dataset_name, "p6_new")
    os.makedirs(output_subdir, exist_ok=True)

    # Filename format: {Species}_MCQ_P6_N{N}.json
    output_filename = f"{args.dataset_name}_MCQ_P6_N{args.gallery_size}.json"
    output_path = os.path.join(output_subdir, output_filename)

    with open(output_path, "w") as f:
        json.dump(generated_samples, f, indent=2)

    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
