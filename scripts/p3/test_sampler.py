"""
scripts/p3/test_sampler.py
Unit test for Protocol 3 Sampler.
"""

import json
import os
import sys

# Ensure imports work
sys.path.append(os.getcwd())

from scripts.p3.sampler import ContextAwareSampler


def test_sampler():
    # Configuration for test
    # Assuming you have data in data/MetaWild/Deer (or similar)
    # You might need to adjust this path to a real directory on your machine for testing
    test_data_dir = "data/MetaWild/Deer"

    if not os.path.exists(test_data_dir):
        # Try to find first available species in MetaWild
        metawild_root = "data/MetaWild"
        if os.path.exists(metawild_root):
            subdirs = [
                d
                for d in os.listdir(metawild_root)
                if os.path.isdir(os.path.join(metawild_root, d))
            ]
            if subdirs:
                test_data_dir = os.path.join(metawild_root, subdirs[0])
            else:
                print(f"No species directories found in {metawild_root}")
                return
        else:
            print(f"Data root {metawild_root} not found. Skipping test.")
            return

    print(f"Testing P3 Sampler with data: {test_data_dir}")

    sampler = ContextAwareSampler(
        data_dir=test_data_dir,
        gallery_size=4,
        max_queries_per_id=2,
        dataset_name="TestSpecies"
    )

    print("\n--- Generating 3 Samples ---")
    for i in range(3):
        sample = sampler.generate_sample()
        if sample:
            print(f"\nSample {i + 1}:")
            print(f"Task ID: {sample['task_id']}")
            print(f"Query Image: {sample['query']['image_path']}")
            print(f"Context Text: {sample['query']['context_text']}")
            print(f"Answer: {sample['answer']}")
            print(json.dumps(sample, indent=2))  # Uncomment to see full JSON
        else:
            print("Sampler returned None (exhausted or error).")
            break


if __name__ == "__main__":
    test_sampler()
