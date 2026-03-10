"""
Analyze P6 Dataset Statistics.
"""

import argparse
import json
import os
from collections import Counter
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Analyze generated P6 (Counterfactual) dataset statistics."
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

    # 1. Count usage of IDs
    # In P6, each task involves two IDs: image_a.id and image_b.id
    all_ids = []
    for item in data:
        if "image_a" in item and "id" in item["image_a"]:
            all_ids.append(item["image_a"]["id"])
        if "image_b" in item and "id" in item["image_b"]:
            all_ids.append(item["image_b"]["id"])

    id_counts = Counter(all_ids)
    unique_ids = len(id_counts)
    print(f"Unique IDs involved: {unique_ids}")

    # 2. Analyze distribution
    counts = list(id_counts.values())
    if not counts:
        print("No ID data found.")
        return

    min_usage = min(counts)
    max_usage = max(counts)
    avg_usage = sum(counts) / len(counts)

    print("\n--- ID Usage Statistics ---")
    print(f"Min usage per ID: {min_usage}")
    print(f"Max usage per ID: {max_usage}")
    print(f"Avg usage per ID: {avg_usage:.2f}")

    # 3. Plot Histogram
    plt.figure(figsize=(10, 6))
    
    # Use a range of bins that covers all counts
    bins = range(min_usage, max_usage + 2)

    plt.hist(counts, bins=bins, color="mediumpurple", edgecolor="black", align="left", rwidth=0.8)
    plt.title(f"Distribution of ID Usage in P6 (Total Pairs: {total_samples})")
    plt.xlabel("Number of Times an ID was Used")
    plt.ylabel("Count of IDs")
    
    if max_usage - min_usage < 40:
        plt.xticks(range(min_usage, max_usage + 1))
        
    plt.grid(axis="y", alpha=0.5, linestyle="--")
    plt.tight_layout()

    output_dir = os.path.dirname(args.output_plot)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(args.output_plot)
    print(f"\nHistogram saved to {args.output_plot}")


if __name__ == "__main__":
    main()
