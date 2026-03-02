"""
scripts/p3/generate_stats.py
Generate statistical report for Protocol 3 Datasets (Context-aware).
Outputs a JSON file with numerical statistics.
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

# Ensure imports work when running from project root
sys.path.append(os.getcwd())


def generate_stats_for_file(json_path, output_dir):
    print(f"Generating statistics for {json_path}...")

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return

    if not data:
        print("Dataset is empty.")
        return

    stats = {}

    # 1. Basic Counts
    stats["total_samples"] = len(data)

    # 2. ID Stats
    query_ids = [str(task["query"]["ground_truth_id"]) for task in data]
    unique_ids = set(query_ids)
    stats["unique_query_ids"] = len(unique_ids)

    id_counts = Counter(query_ids)
    counts = list(id_counts.values())
    if counts:
        stats["samples_per_id"] = {
            "mean": float(np.mean(counts)),
            "std": float(np.std(counts)),
            "min": int(min(counts)),
            "max": int(max(counts)),
        }
    else:
        stats["samples_per_id"] = {}

    # 3. Answer Distribution
    answers = [task["answer"] for task in data]
    answer_counts = Counter(answers)
    # Sort by key (A, B, C, D)
    stats["answer_distribution"] = dict(sorted(answer_counts.items()))

    # 4. Metadata Stats (P3 Specific)
    # Infer keys from the first sample's context_text
    if "query" in data[0] and "context_text" in data[0]["query"]:
        meta_keys = list(data[0]["query"]["context_text"].keys())
        stats["metadata_distribution"] = {}

        for key in meta_keys:
            # Get all values for this key
            values = [str(t["query"]["context_text"].get(key)) for t in data]
            # Count and store
            stats["metadata_distribution"][key] = dict(Counter(values).most_common())

    # Output
    dataset_filename = os.path.splitext(os.path.basename(json_path))[0]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{dataset_filename}_stats.json")

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate P3 Dataset Statistics")
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
            # Default to 'analysis_results' in the same folder as the json file
            args.output_dir = os.path.join(
                os.path.dirname(args.dataset_name), "analysis_results"
            )
        generate_stats_for_file(args.dataset_name, args.output_dir)
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

    print(f"Found {len(json_files)} files to process.")
    for f in json_files:
        generate_stats_for_file(os.path.join(p3_dir, f), args.output_dir)


if __name__ == "__main__":
    main()
