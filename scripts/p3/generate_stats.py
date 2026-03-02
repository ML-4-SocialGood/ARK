"""
scripts/p3/generate_stats.py
Generate a summary Excel report for all Protocol 3 datasets found in the annotations directory.
Output: p3_dataset_stats.xlsx
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

import pandas as pd

# Ensure imports work when running from project root
sys.path.append(os.getcwd())


def get_n_from_filename(filename):
    # Expecting format like ..._N4.json
    match = re.search(r"_N(\d+)\.json$", filename)
    if match:
        return int(match.group(1))
    return None


def process_file(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

    if not data:
        return None

    filename = os.path.basename(json_path)

    # Extract N (Gallery Size)
    n_val = get_n_from_filename(filename)

    # Extract Dataset Name (everything before _CIR_P3_)
    if "_CIR_P3_" in filename:
        dataset_name = filename.split("_CIR_P3_")[0]
    else:
        dataset_name = filename.replace(".json", "")

    total_samples = len(data)

    # ID stats
    query_ids = [t["query"]["ground_truth_id"] for t in data if "query" in t]
    unique_ids = len(set(query_ids))

    # Answer stats
    answers = [t["answer"] for t in data if "answer" in t]
    answer_counts = Counter(answers)

    row = {
        "Dataset": dataset_name,
        "Protocol": "P3",
        "N": n_val if n_val else "Unknown",
        "Num Samples": total_samples,
        "Unique Query IDs": unique_ids,
        "Avg Samples/ID": round(total_samples / unique_ids, 2) if unique_ids else 0,
    }

    # Add answer counts (A, B, C, D...)
    for ans in sorted(answer_counts.keys()):
        row[f"Ans {ans}"] = answer_counts[ans]

    return row


def main():
    parser = argparse.ArgumentParser(description="Generate P3 Dataset Statistics Excel")
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="annotations",
        help="Root directory containing annotations",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="p3_dataset_stats.xlsx",
        help="Path to output Excel file",
    )

    args = parser.parse_args()

    print(f"Scanning '{args.annotations_dir}' for P3 datasets...")

    # Recursive search for P3 JSON files
    # Pattern matches: annotations/**/p3/*_CIR_P3_*.json
    pattern = os.path.join(args.annotations_dir, "**", "p3", "*_CIR_P3_*.json")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print("No P3 dataset files found.")
        return

    print(f"Found {len(files)} files. Processing...")

    rows = []
    for f in sorted(files):
        print(f"  Processing {os.path.basename(f)}...")
        row = process_file(f)
        if row:
            rows.append(row)

    if not rows:
        print("No valid data extracted.")
        return

    df = pd.DataFrame(rows)

    # Reorder columns to put Ans columns at the end
    cols = [c for c in df.columns if not c.startswith("Ans ")]
    ans_cols = sorted([c for c in df.columns if c.startswith("Ans ")])
    df = df[cols + ans_cols]

    # Sort by Dataset and N
    if "Dataset" in df.columns and "N" in df.columns:
        df = df.sort_values(by=["Dataset", "N"])

    print(f"Saving statistics to {args.output_file}...")
    try:
        df.to_excel(args.output_file, index=False)
        print("Done.")
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        print("Make sure you have 'openpyxl' installed (pip install openpyxl).")


if __name__ == "__main__":
    main()
