"""
scripts/p3/analyze_dataset.py
Analyze Protocol 3 Datasets.
Focuses on:
1. ID Analysis: Number of Samples Generated for an ID vs Count of IDs.
2. Metadata Distribution: Distribution of context attributes.
"""

import argparse
import json
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt

# Ensure imports work when running from project root
sys.path.append(os.getcwd())


def analyze_file(json_path, output_dir):
    print(f"Analyzing {json_path}...")

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return

    if not data:
        print("Dataset is empty.")
        return

    # Prepare output directory
    dataset_filename = os.path.splitext(os.path.basename(json_path))[0]
    plot_dir = os.path.join(output_dir, dataset_filename)
    os.makedirs(plot_dir, exist_ok=True)

    # --- 1. ID Analysis: Number of Samples Generated for an ID vs Count of IDs ---
    # Count how many samples each ID has
    query_ids = [str(task["query"]["ground_truth_id"]) for task in data]
    id_counts = Counter(query_ids)

    # Get the list of counts (e.g., [5, 5, 5, 2, 2, 1, ...])
    samples_per_id = list(id_counts.values())

    plt.figure(figsize=(10, 6))
    if samples_per_id:
        max_count = max(samples_per_id)
        min_count = min(samples_per_id)
        # Create bins for each integer value
        bins = range(min_count, max_count + 2)

        plt.hist(
            samples_per_id,
            bins=bins,
            align="left",
            rwidth=0.8,
            color="skyblue",
            edgecolor="black",
        )
        plt.title(
            f"Number of Samples Generated for an ID vs Count of IDs\n(Total Unique IDs: {len(id_counts)})"
        )
        plt.xlabel("Number of Samples Generated")
        plt.ylabel("Count of IDs")
        # Set x-ticks to be integers only
        if max_count - min_count < 20:
            plt.xticks(range(min_count, max_count + 1))
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "id_sample_distribution.png"))
        plt.close()

    # --- 2. Metadata Distribution ---
    # Infer keys from the first item's context_text
    first_task = data[0]
    metadata_keys = []
    if "query" in first_task and "context_text" in first_task["query"]:
        metadata_keys = list(first_task["query"]["context_text"].keys())

    for key in metadata_keys:
        # Extract values for this key across all tasks
        values = [t.get("query", {}).get("context_text", {}).get(key) for t in data]
        # Filter Nones and convert to string for consistent counting/plotting
        clean_values = [str(v) for v in values if v is not None]

        if not clean_values:
            continue

        counts = Counter(clean_values)

        # Plot
        plt.figure(figsize=(10, 6))

        # Sort by count descending
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        # Limit to top 20 for readability if many unique values
        if len(sorted_items) > 20:
            sorted_items = sorted_items[:20]
            plt.title(f"Distribution of {key} (Top 20)")
        else:
            plt.title(f"Distribution of {key}")

        labels, values = zip(*sorted_items)

        plt.bar(labels, values, color="salmon", edgecolor="black", alpha=0.7)
        plt.xlabel(key)
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"meta_{key}_distribution.png"))
        plt.close()

    print(f"Analysis saved to: {plot_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze P3 Dataset")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of dataset (e.g. MetaWild/Deer) or path to .json",
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="annotations",
        help="Root annotations directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory",
    )

    args = parser.parse_args()

    # Case 1: Direct file
    if args.dataset_name.endswith(".json") and os.path.exists(args.dataset_name):
        if args.output_dir is None:
            args.output_dir = os.path.join(os.path.dirname(args.dataset_name), "analysis_results")
        analyze_file(args.dataset_name, args.output_dir)
        return

    # Case 2: Logical name (scan directory)
    p3_dir = os.path.join(args.annotations_dir, args.dataset_name, "p3")
    if not os.path.exists(p3_dir):
        print(f"Directory not found: {p3_dir}")
        return

    if args.output_dir is None:
        args.output_dir = os.path.join(p3_dir, "analysis_results")

    json_files = [
        f for f in os.listdir(p3_dir) if f.endswith(".json") and "CIR_P3" in f
    ]
    json_files.sort()

    if not json_files:
        print(f"No P3 JSON files found in {p3_dir}")
        return

    print(f"Found {len(json_files)} files to analyze.")
    for f in json_files:
        analyze_file(os.path.join(p3_dir, f), args.output_dir)


if __name__ == "__main__":
    main()
