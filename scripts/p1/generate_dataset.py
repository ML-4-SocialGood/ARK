import argparse
import json
import os
import random
import sys

# Ensure imports work when running from project root
sys.path.append(os.getcwd())

from scripts.p1.sampler import DynamicReIDSampler


def main():
    parser = argparse.ArgumentParser(
        description="Generate MCQ Dataset for Animal Re-ID (Protocol 1)"
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
        "--max_queries_per_id", type=int, default=5, help="Max queries per ID"
    )
    parser.add_argument(
        "--target_samples",
        type=int,
        default=-1,
        help="Target number of samples to generate. -1 for maximum possible based on constraints.",
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

    # Set random seed for reproducibility
    random.seed(args.seed)

    print(f"Initializing Sampler for {args.dataset_name} with N={args.gallery_size}...")
    try:
        sampler = DynamicReIDSampler(
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
        # Dynamic limit: min(max_queries, num_images)
        limit = min(args.max_queries_per_id, len(sampler.image_map[qid]))
        theoretical_max += limit
    print(
        f"Theoretical maximum samples: {theoretical_max} (Calculated with dynamic per-ID caps)"
    )

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
            print(
                f"\nSampler returned None at iteration {i + 1}. Stopping generation early."
            )
            print("Reason: Constraints met or data exhausted.")
            break

        generated_samples.append(sample)

        if (i + 1) % 10 == 0 or (i + 1) == target_count:
            print(f"Generated {i + 1}/{target_count} samples...", end="\r")

    print(f"\nTotal generated: {len(generated_samples)} samples.")

    # Construct output path: annotations/{dataset_name}/p1/{dataset_name}_I2I_P1.json
    output_subdir = os.path.join(args.output_dir, args.dataset_name, "p1")
    os.makedirs(output_subdir, exist_ok=True)


    output_filename = f"{args.dataset_name}_I2I_P1_N{args.gallery_size}.json"
    output_path = os.path.join(output_subdir, output_filename)

    with open(output_path, "w") as f:
        json.dump(generated_samples, f, indent=2)

    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
