import argparse
import json
import os
import sys

# Ensure imports work when running from project root
sys.path.append(os.getcwd())

from scripts_annotate.p2.sampler import MultiImageBatchSampler


def main():
    parser = argparse.ArgumentParser(description="Test P2 MultiImageBatchSampler")
    parser.add_argument(
        "--data_dir", type=str, default="data/BelugaID/IDs", help="Path to dataset IDs"
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        print("Please run this script from the project root or specify --data_dir.")
        return

    print(f"Testing MultiImageBatchSampler with data_dir: {args.data_dir}")

    # Configuration
    gallery_size = 4
    max_query_size = 4
    max_queries_per_id = 5

    try:
        sampler = MultiImageBatchSampler(
            dataset_name="Beluga",
            data_dir=args.data_dir,
            gallery_size=gallery_size,
            max_query_size=max_query_size,
            max_queries_per_id=max_queries_per_id,
        )
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    print("\n--- Generating a Batch ---")
    batch = sampler.generate_batch()

    if batch is None:
        print(
            "Sampler returned None. Check if data is sufficient (need IDs with >= max_query_size + 1 images)."
        )
        return

    print(f"Generated batch with {len(batch)} tasks (Expected: {max_query_size}).")

    # Validation
    prev_query_images = []
    first_gallery = None

    for i, task in enumerate(batch):
        print(f"\nTask {i + 1} (K={task['meta']['query_size']}):")
        print(f"  ID: {task['task_id']}")

        # Check Query Growth
        current_query_images = task["query"]["image_paths"]
        print(f"  Query Images: {len(current_query_images)} -> {current_query_images}")

        if i == 0:
            if len(current_query_images) != 1:
                print("  [ERROR] First task should have 1 query image.")
        else:
            # Check if current is prev + 1 new image
            if current_query_images[:-1] != prev_query_images:
                print("  [ERROR] Query images are not incremental subsets!")
            elif len(current_query_images) != len(prev_query_images) + 1:
                print("  [ERROR] Query size did not increase by 1.")
            else:
                print("  [OK] Query growth correct.")

        prev_query_images = current_query_images

        # Check Fixed Gallery
        current_gallery_json = json.dumps(task["gallery"], sort_keys=True)
        if first_gallery is None:
            first_gallery = current_gallery_json
        else:
            if current_gallery_json != first_gallery:
                print("  [ERROR] Gallery is not fixed across the batch!")
            else:
                print("  [OK] Gallery matches previous task.")

    print("\n--- JSON Structure Check (Last Task) ---")
    print(json.dumps(batch[-1], indent=2))


if __name__ == "__main__":
    main()
