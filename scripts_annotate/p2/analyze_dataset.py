import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Analyze generated MCQ P2 dataset statistics."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to the generated JSON file (e.g., annotations/BelugaID/p2/BelugaID_MCQ_P4_N4_K1.json)",
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
        # Auto-generate output filename from input JSON filename
        # e.g. BelugaID_MCQ_P4_N4_K1.json -> BelugaID_MCQ_P4_N4_K1_distribution.png
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

    # Check metadata for P4 specifics (Optional info display)
    first_task = data[0]
    if "meta" in first_task:
        meta = first_task["meta"]
        print("Protocol 4 Metadata detected:")
        if "batch_id" in meta:
            print(
                f"  Batch ID range: {data[0]['meta']['batch_id']} - {data[-1]['meta']['batch_id']}"
            )
        if "query_size" in meta:
            print(f"  Query Size (K): {meta['query_size']}")

    # 1. Count samples per ID
    # We use 'ground_truth_id' from the query object to identify the ID
    id_counts = Counter(item["query"]["ground_truth_id"] for item in data)
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

    n, bins, patches = plt.hist(
        counts, bins=bins, color="skyblue", edgecolor="black", align="left", rwidth=0.8
    )

    plt.title(f"Distribution of Samples per ID (Total: {total_samples})")
    plt.xlabel("Number of Samples Generated for an ID")
    plt.ylabel("Count of IDs")

    # Only set specific xticks if the range is reasonable to avoid overcrowding
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

    # 4. Print detailed frequency table
    print("\n--- Frequency Table ---")
    print(f"{'Samples/ID':<12} | {'Count of IDs':<12}")
    print("-" * 27)
    freq_of_counts = Counter(counts)
    for k in sorted(freq_of_counts.keys()):
        print(f"{k:<12} | {freq_of_counts[k]:<12}")


if __name__ == "__main__":
    main()
