import json
import os
import sys

# Ensure imports work when running from project root
sys.path.append(os.getcwd())

from scripts.p6.sampler import CounterfactualSampler

if __name__ == "__main__":
    # Test configuration
    # NOTE: Ensure you run this script from the project root directory
    # You might need to change this path depending on where your data is located
    test_data_dir = "data/BelugaID/IDs"

    print(f"Testing CounterfactualSampler (Protocol 6) with data_dir: {test_data_dir}")

    if not os.path.exists(test_data_dir):
        print(f"Error: Data directory {test_data_dir} does not exist.")
        print(
            "Please adjust 'test_data_dir' in the script to point to a valid dataset."
        )
        sys.exit(1)

    try:
        sampler = CounterfactualSampler(
            dataset_name="TestDataset",
            data_dir=test_data_dir,
            max_usage_per_id=5,
        )

        # Generate and print one sample to verify the structure
        print("\n--- Generating a sample ---")
        sample = sampler.generate_sample()

        if sample:
            print("Sample generated successfully:")
            print(json.dumps(sample, indent=2))

            # Basic validation logic
            img_a = sample["image_a"]
            img_b = sample["image_b"]
            gt = sample["ground_truth"]

            # Check 1: IDs must be different (Core requirement of P6)
            if str(img_a["id"]) == str(img_b["id"]):
                print(
                    f"[FAIL] Protocol Violation: IDs are identical ({img_a['id']}). P6 requires Negative Pairs."
                )
            else:
                print(
                    f"[PASS] IDs are different ({img_a['id']} vs {img_b['id']})."
                )

            # Check 2: Ground truth must be 'different'
            if gt == "different":
                print(f"[PASS] Ground truth is '{gt}'.")
            else:
                print(f"[FAIL] Ground truth is '{gt}', expected 'different'.")

            # Check 3: Image paths should be strings
            if isinstance(img_a["image_path"], str) and isinstance(
                img_b["image_path"], str
            ):
                print("[PASS] Image paths are strings.")
            else:
                print("[FAIL] Image paths are not strings.")

            # Check 4: Task ID format
            if "P6" in sample["task_id"]:
                print(f"[PASS] Task ID contains protocol tag: {sample['task_id']}")
            else:
                print(f"[FAIL] Task ID missing protocol tag: {sample['task_id']}")

        else:
            print("Failed to generate a sample. Not enough IDs or constraints too tight.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()