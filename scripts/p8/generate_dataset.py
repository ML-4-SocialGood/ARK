import argparse
import json
import os
import random
import sys

from tqdm import tqdm

# Ensure imports work when running from project root
sys.path.append(os.getcwd())

from scripts.p8.corruption_utils import apply_corruption


def main():
    parser = argparse.ArgumentParser(
        description="Generate P8 (Corrupted Consistency) Dataset from P1"
    )
    parser.add_argument(
        "--p1_json",
        type=str,
        required=True,
        help="Path to the source P1 JSON file",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="Root directory of the project (to resolve relative image paths)",
    )
    parser.add_argument(
        "--corruption_type",
        type=str,
        choices=["occlusion", "resolution", "grayscale"],
        required=True,
        help="Type of corruption to apply",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Severity of corruption (1=Low, 2=Medium, 3=High). Ignored for grayscale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (e.g. occlusion position)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.p1_json):
        print(f"Error: P1 JSON file not found: {args.p1_json}")
        return

    # Load P1 Data
    with open(args.p1_json, "r") as f:
        p1_data = json.load(f)

    print(f"Loaded {len(p1_data)} tasks from {args.p1_json}")

    # Determine Output Paths
    # Input: annotations/{Species}/p1/{Species}_I2I_P1_N{N}.json
    # Output: annotations/{Species}/p8/{Species}_I2I_P8_{Type}_S{Sev}_N{N}.json

    p1_dir = os.path.dirname(args.p1_json)
    species_dir = os.path.dirname(p1_dir)
    species_name = os.path.basename(species_dir)

    p8_dir = os.path.join(species_dir, "p8")
    os.makedirs(p8_dir, exist_ok=True)

    # Extract N from filename or data
    filename = os.path.basename(args.p1_json)
    n_part = ""
    if "_N" in filename:
        import re

        match = re.search(r"_N(\d+)", filename)
        if match:
            n_part = f"_N{match.group(1)}"

    output_filename = (
        f"{species_name}_I2I_P8_{args.corruption_type}_S{args.severity}{n_part}.json"
    )
    output_json_path = os.path.join(p8_dir, output_filename)

    # Directory for corrupted images
    # data/{Species}/corrupted/{corruption_type}_s{severity}/
    corrupted_img_root = os.path.join(
        "data", species_name, "corrupted", f"{args.corruption_type}_s{args.severity}"
    )
    full_corrupted_root = os.path.join(args.data_root, corrupted_img_root)
    os.makedirs(full_corrupted_root, exist_ok=True)

    random.seed(args.seed)

    p8_data = []

    print(
        f"Generating P8 dataset ({args.corruption_type}, Severity {args.severity})..."
    )
    print(f"Corrupted images will be saved to: {full_corrupted_root}")

    for task in tqdm(p1_data):
        # Deep copy task to avoid modifying original if we were keeping it in memory
        new_task = json.loads(json.dumps(task))

        # Update Task ID
        # P1 ID: {Species}_MCQ_000001 -> P8 ID: {Species}_MCQ_P8_{Type}_000001
        old_id = new_task.get("task_id", "Unknown")
        new_task["task_id"] = old_id.replace("_P1_", "_P8_").replace(
            "_MCQ_", f"_MCQ_P8_{args.corruption_type}_"
        )

        # Process Query Image
        query_path = new_task["query"]["image_path"]
        full_query_path = os.path.join(args.data_root, query_path)

        if not os.path.exists(full_query_path):
            print(f"Warning: Source image not found: {full_query_path}. Skipping task.")
            continue

        # Generate Corrupted Filename
        # Structure: corrupted_root/{ID}/{filename}
        path_parts = query_path.split(os.sep)
        if len(path_parts) >= 2:
            id_name = path_parts[-2]  # Assuming data/Species/IDs/{ID}/{Img}
            img_name = path_parts[-1]
        else:
            id_name = "unknown"
            img_name = os.path.basename(query_path)

        save_dir = os.path.join(full_corrupted_root, id_name)
        os.makedirs(save_dir, exist_ok=True)

        full_save_path = os.path.join(save_dir, img_name)
        rel_save_path = os.path.join(corrupted_img_root, id_name, img_name)

        # Check if already exists to avoid re-processing (caching)
        if not os.path.exists(full_save_path):
            corrupted_img = apply_corruption(
                full_query_path, args.corruption_type, args.severity
            )
            if corrupted_img:
                corrupted_img.save(full_save_path)
            else:
                print(f"Failed to corrupt {full_query_path}")
                continue

        # Update Task
        new_task["query"]["image_path"] = rel_save_path

        # Add Meta info
        if "meta" not in new_task:
            new_task["meta"] = {}
        new_task["meta"]["protocol"] = "P8"
        new_task["meta"]["corruption"] = args.corruption_type
        new_task["meta"]["severity"] = args.severity
        new_task["meta"]["original_query_path"] = query_path

        p8_data.append(new_task)

    # Save JSON
    with open(output_json_path, "w") as f:
        json.dump(p8_data, f, indent=2)

    print(f"Saved P8 dataset to {output_json_path}")


if __name__ == "__main__":
    main()
