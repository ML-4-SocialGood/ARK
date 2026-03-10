"""
scripts/p3/generate_stats.py
Generate a summary Excel report for all Protocol 3 datasets found in the annotations directory.
Output: p3_dataset_stats.xlsx
"""

import argparse
import json
import os
import re
from collections import Counter

import pandas as pd


def analyze_file(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    if not data:
        return None

    filename = os.path.basename(file_path)

    # Extract N from filename
    # Expected format: ..._N{N}.json
    match = re.search(r"_N(\d+)\.json$", filename)
    if match:
        n_val = int(match.group(1))
    else:
        n_val = 4

    # Calculate stats
    total_tasks = len(data)

    # Extract ground truth IDs
    query_ids = []
    for task in data:
        if "query" in task and "ground_truth_id" in task["query"]:
            query_ids.append(task["query"]["ground_truth_id"])

    if not query_ids:
        return None

    id_counts = Counter(query_ids)
    unique_ids = len(id_counts)
    counts = list(id_counts.values())

    if counts:
        min_samples = min(counts)
        max_samples = max(counts)
        avg_samples = sum(counts) / len(counts)
    else:
        min_samples = 0
        max_samples = 0
        avg_samples = 0

    # Infer dataset name from directory structure
    # annotations/{DatasetName}/p3/filename
    parent_dir = os.path.dirname(os.path.dirname(file_path))
    dataset_name = os.path.basename(parent_dir)

    return {
        "Dataset": dataset_name,
        "Protocol": "P3 (CIR)",
        "Gallery Size (N)": n_val,
        "Total Tasks": total_tasks,
        "Unique IDs": unique_ids,
        "Min Samples/ID": min_samples,
        "Max Samples/ID": max_samples,
        "Avg Samples/ID": round(avg_samples, 2),
        "Filename": filename,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate Excel Statistics for P3 (CIR) Annotations"
    )

    # Determine default output path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, "p3_dataset_stats.xlsx")

    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="annotations",
        help="Root directory containing annotations (e.g., annotations/)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=default_output,
        help="Path to output Excel file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.annotations_dir):
        print(f"Error: Directory '{args.annotations_dir}' does not exist.")
        return

    all_stats = []
    print(f"Scanning '{args.annotations_dir}' for P3 datasets...")

    for root, dirs, files in os.walk(args.annotations_dir):
        # We are looking for files inside a 'p3' subdirectory
        if os.path.basename(root) == "p3":
            for file in files:
                if file.endswith(".json") and "CIR_P3" in file:
                    file_path = os.path.join(root, file)
                    print(f"  Processing: {file}", end="\r")

                    stats = analyze_file(file_path)
                    if stats:
                        all_stats.append(stats)

    print("\n" + "-" * 50)

    if not all_stats:
        print("No valid P3 annotation files found.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_stats)

    # Sort by Dataset and N
    if not df.empty:
        df = df.sort_values(by=["Dataset", "Gallery Size (N)"])

    # Determine output format based on extension
    ext = os.path.splitext(args.output_file)[1].lower()

    if ext == ".csv":
        print(f"Saving statistics to {args.output_file}...")
        df.to_csv(args.output_file, index=False)
        print("Done.")
    else:
        # Default to Excel for .xlsx or other extensions
        try:
            print(f"Saving statistics to {args.output_file}...")
            df.to_excel(args.output_file, index=False)
            print("Done.")
        except ImportError:
            print("Error: 'openpyxl' library is required to save Excel files.")
            print("Please install it using: pip install openpyxl")
            # Fallback to CSV
            csv_file = os.path.splitext(args.output_file)[0] + ".csv"
            print(f"Attempting to save as CSV instead: {csv_file}")
            df.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"Error saving file: {e}")


if __name__ == "__main__":
    main()
