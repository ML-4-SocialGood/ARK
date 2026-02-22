import json
import os
import sys

# Ensure imports work when running from project root
sys.path.append(os.getcwd())

from scripts.p9.sampler import MultiIdentitySampler

if __name__ == "__main__":
    # Test configuration
    # NOTE: Ensure you run this script from the project root directory
    # We try to find a valid data directory automatically
    test_data_dir = "data/BelugaID/IDs"

    if not os.path.exists(test_data_dir):
        # Fallback: try to find any valid species directory
        if os.path.exists("data"):
            potential_species = [
                d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))
            ]
            for species in potential_species:
                ids_path = os.path.join("data", species, "IDs")
                if os.path.exists(ids_path):
                    test_data_dir = ids_path
                    break

    print(f"Testing MultiIdentitySampler (P9) with data_dir: {test_data_dir}")

    if not os.path.exists(test_data_dir):
        print(
            f"Error: Data directory {test_data_dir} not found. Please run from project root or adjust path."
        )
        sys.exit(1)

    # Test parameters for Protocol 9
    gallery_size = 4
    num_positives = 2  # P9 requires at least 2 positives
    max_queries = 5
    dataset_name = "TestDataset"

    print("\n--- Configuration ---")
    print(f"Gallery Size (N): {gallery_size}")
    print(f"Num Positives (M): {num_positives}")

    try:
        sampler = MultiIdentitySampler(
            data_dir=test_data_dir,
            gallery_size=gallery_size,
            num_positives=num_positives,
            max_queries_per_id=max_queries,
            dataset_name=dataset_name,
        )

        print("\n--- Generating a sample ---")
        sample = sampler.generate_sample()

        if sample:
            print("Sample generated successfully:")
            print(json.dumps(sample, indent=2))

            # Basic Validation Logic
            print("\n--- Basic Validation ---")
            gallery = sample["gallery"]
            answer_str = sample["answer"]
            query_id = sample["query"]["ground_truth_id"]
            query_img = sample["query"]["image_path"]

            # Check 1: Gallery Size
            print(
                f"1. Gallery size: {len(gallery)} (Expected: {gallery_size}) -> {'PASS' if len(gallery) == gallery_size else 'FAIL'}"
            )

            # Check 2: Number of Positives in Gallery
            positives = [item for item in gallery if str(item["id"]) == str(query_id)]
            print(
                f"2. Positives in gallery: {len(positives)} (Expected: {num_positives}) -> {'PASS' if len(positives) == num_positives else 'FAIL'}"
            )

            # Check 3: Answer String Format
            correct_options = sorted([item["option"] for item in positives])
            expected_answer = ", ".join(correct_options)
            print(
                f"3. Answer string: '{answer_str}' (Expected: '{expected_answer}') -> {'PASS' if answer_str == expected_answer else 'FAIL'}"
            )

            # Check 4: Data Leakage
            leakage = any(item["image_path"] == query_img for item in gallery)
            print(
                f"4. Data Leakage Check (Query not in Gallery): -> {'PASS' if not leakage else 'FAIL'}"
            )

        else:
            print(
                "Failed to generate a sample. Constraints might be too tight or no eligible queries left."
            )

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
