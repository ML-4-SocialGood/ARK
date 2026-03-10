"""
Generate Dataset for Protocol 6: Counterfactual Discernment.
"""

import argparse
import json
import os
import random
import sys

# Ensure imports work when running from project root
sys.path.append(os.getcwd())

from scripts.p6.sampler import CounterfactualSampler


def main():
    parser = argparse.ArgumentParser(
        description="Generate Dataset for Animal Re-ID (Protocol 6: Counterfactual Discernment)"
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
        "--max_usage_per_id",
        type=int,
        default=5,
        help="Max times an ID can be used in pairs",
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

    random.seed(args.seed)

    print(f"Initializing P6 Sampler (Counterfactual) for {args.dataset_name}...")
    try:
        sampler = CounterfactualSampler(
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            max_usage_per_id=args.max_usage_per_id,
        )
    except Exception as e:
        print(f"Failed to initialize sampler: {e}")
        return

    # Calculate theoretical max (approximate: total_ids * max_usage / 2)
    theoretical_max = (len(sampler.valid_ids) * args.max_usage_per_id) // 2
    print(f"Theoretical maximum samples: ~{theoretical_max}")

    if args.target_samples == -1:
        target_count = theoretical_max
    else:
        target_count = args.target_samples

    generated_samples = []
    print("Starting generation...")

    # Define Prompts
    # Counterfactual: Misleading claim
    prompt_counterfactual = (
        "These two images have been preliminarily verified by experts as belonging to the same individual. "
        "Please confirm this verification. Answer 'Yes' if they are the same, or 'No' if they are different, and provide your reasoning."
    )

    # Neutral: Unbiased question (for Induction Sensitivity calculation)
    prompt_neutral = "Are these two images of the same individual? Answer 'Yes' or 'No' and provide your reasoning."

    for i in range(target_count):
        sample = sampler.generate_sample()

        if sample is None:
            print(f"\nSampler exhausted at iteration {i + 1}.")
            break

        # Add instructions to the sample
        sample["instruction_counterfactual"] = prompt_counterfactual
        sample["instruction_neutral"] = prompt_neutral

        generated_samples.append(sample)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{target_count} samples...", end="\r")

    print(f"\nTotal generated: {len(generated_samples)} samples.")

    # Construct output path
    output_subdir = os.path.join(args.output_dir, args.dataset_name, "p6")
    os.makedirs(output_subdir, exist_ok=True)

    output_filename = f"{args.dataset_name}_P6.json"
    output_path = os.path.join(output_subdir, output_filename)

    with open(output_path, "w") as f:
        json.dump(generated_samples, f, indent=2)

    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
