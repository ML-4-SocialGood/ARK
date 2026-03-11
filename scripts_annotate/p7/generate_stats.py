import argparse
import json
import os
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

    # Calculate stats
    total_tasks = len(data)

    # Extract IDs to calculate distribution
    # In P7, we have image_a and image_b, both contribute to ID usage
    all_ids = []
    for task in data:
        if "image_a" in task and "id" in task["image_a"]:
            all_ids.append(task["image_a"]["id"])
        if "image_b" in task and "id" in task["image_b"]:
            all_ids.append(task["image_b"]["id"])

    if not all_ids:
        return None

    id_counts = Counter(all_ids)
    unique_ids = len(id_counts)
    counts = list(id_counts.values())

    if counts:
        min_usage = min(counts)
        max_usage = max(counts)
        avg_usage = sum(counts) / len(counts)
    else:
        min_usage = 0
        max_usage = 0
        avg_usage = 0

    # Infer dataset name from directory structure
    # annotations/{DatasetName}/p7/filename
    parent_dir = os.path.dirname(os.path.dirname(file_path))
    dataset_name = os.path.basename(parent_dir)

    return {
        "Dataset": dataset_name,
        "Protocol": "P7 (Counterfactual)",
        "Total Pairs": total_tasks,
        "Unique IDs": unique_ids,
        "Min Usage/ID": min_usage,
        "Max Usage/ID": max_usage,
        "Avg Usage/ID": round(avg_usage, 2),
        "Filename": filename,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate Excel Statistics for P7 (Counterfactual) Annotations"
    )

    # Determine default output path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, "p7_dataset_stats.xlsx")

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
        help="Path for the output Excel file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.annotations_dir):
        print(f"Error: Directory '{args.annotations_dir}' does not exist.")
        return

    all_stats = []
    print(f"Scanning '{args.annotations_dir}' for P7 JSON files...")

    for root, dirs, files in os.walk(args.annotations_dir):
        # We are looking for files inside a 'p7' subdirectory
        if os.path.basename(root) == "p7":
            for file in files:
                if file.endswith(".json") and "_P7" in file:
                    file_path = os.path.join(root, file)
                    print(f"  Processing: {file}", end="\r")

                    stats = analyze_file(file_path)
                    if stats:
                        all_stats.append(stats)

    print("\n" + "-" * 50)

    if not all_stats:
        print("No valid P7 annotation files found.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_stats)

    # Sort
    if not df.empty:
        df = df.sort_values(by=["Dataset"])

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