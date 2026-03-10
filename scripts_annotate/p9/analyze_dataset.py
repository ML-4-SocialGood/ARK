import argparse
import json
import os
from collections import Counter
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Analyze generated P9 (Multi-Identity Association) dataset statistics."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to the generated JSON file",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default=None,
        help="Path to save the distribution plot",
    )

    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: File {args.json_file} not found.")
        return

    if args.output_plot is None:
        base_name = os.path.splitext(os.path.basename(args.json_file))[0]
        output_dir = os.path.dirname(args.json_file)
        args.output_plot = os.path.join(output_dir, f"{base_name}_distribution.png")

    print(f"Loading dataset from {args.json_file}...")
    try:
        with open(args.json_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    total_samples = len(data)
    print(f"Total tasks in file: {total_samples}")

    if total_samples == 0:
        print("Dataset is empty.")
        return

    # 1. Count samples per ID
    id_counts = Counter()
    for item in data:
        if "query" in item and "ground_truth_id" in item["query"]:
            id_counts[item["query"]["ground_truth_id"]] += 1
    
    unique_ids = len(id_counts)
    print(f"Unique IDs sampled: {unique_ids}")

    # 2. Analyze distribution
    counts = list(id_counts.values())
    if not counts:
        print("No data found.")
        return

    min_samples = min(counts)
    max_samples = max(counts)
    avg_samples = sum(counts) / len(counts)

    print("\n--- Sample Distribution Statistics ---")
    print(f"Min samples per ID: {min_samples}")
    print(f"Max samples per ID: {max_samples}")
    print(f"Avg samples per ID: {avg_samples:.2f}")

    # 3. Plot Histogram
    plt.figure(figsize=(12, 6))

    # Use a range of bins that covers all counts
    bins = range(min_samples, max_samples + 2)

    plt.hist(
        counts, bins=bins, color="skyblue", edgecolor="black", align="left", rwidth=0.8
    )

    plt.title(f"Distribution of Samples per ID (P9 MIA, Total: {total_samples})")
    plt.xlabel("Number of Samples Generated for an ID")
    plt.ylabel("Count of IDs")

    if max_samples - min_samples < 40:
        plt.xticks(range(min_samples, max_samples + 1))

    plt.grid(axis="y", alpha=0.5, linestyle="--")
    plt.tight_layout()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_plot)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(args.output_plot)
    print(f"\nHistogram saved to {args.output_plot}")


if __name__ == "__main__":
    main()