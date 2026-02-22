import argparse
import json
import os
import random
import sys

# Ensure imports work when running from project root
sys.path.append(os.getcwd())

from scripts.p9.sampler import MultiIdentitySampler


def main():
    parser = argparse.ArgumentParser(
        description="Generate Dataset for Animal Re-ID (Protocol 9: Multi-Identity Association)"
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
        help="Total number of options in the gallery (N)",
    )
    parser.add_argument(
        "--num_positives",
        type=int,
        default=2,
        help="Number of positive images in the gallery (M). Must be >= 2.",
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
        help="Random seed",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist.")
        return

    random.seed(args.seed)

    print(f"Initializing P9 Sampler for {args.dataset_name} (N={args.gallery_size}, M={args.num_positives})...")
    try:
        sampler = MultiIdentitySampler(
            data_dir=args.data_dir,
            gallery_size=args.gallery_size,
            num_positives=args.num_positives,
            max_queries_per_id=args.max_queries_per_id,
            dataset_name=args.dataset_name,
        )
    except Exception as e:
        print(f"Failed to initialize sampler: {e}")
        return

    # Calculate theoretical max
    theoretical_max = 0
    for qid in sampler.valid_query_ids:
        limit = min(args.max_queries_per_id, len(sampler.image_map[qid]))
        theoretical_max += limit
    
    print(f"Theoretical maximum samples: {theoretical_max}")

    if args.target_samples == -1:
        target_count = theoretical_max
        print(
            f"Target samples not specified. Attempting to generate maximum possible: {target_count}"
        )
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

        if (i + 1) % 10 == 0 or (i + 1) == target_count:
            print(f"Generated {i + 1}/{target_count} samples...", end="\r")

    print(f"\nTotal generated: {len(generated_samples)} samples.")

    # Output path: annotations/{dataset_name}/p9/{dataset_name}_MIA_P9_N{N}.json
    output_subdir = os.path.join(args.output_dir, args.dataset_name, "p9")
    os.makedirs(output_subdir, exist_ok=True)

    output_filename = f"{args.dataset_name}_MIA_P9_N{args.gallery_size}_M{args.num_positives}.json"
    output_path = os.path.join(output_subdir, output_filename)

    with open(output_path, "w") as f:
        json.dump(generated_samples, f, indent=2)

    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()