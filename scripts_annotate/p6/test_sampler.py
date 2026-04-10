import json
import os
import sys

# Ensure imports work when running from project root
sys.path.append(os.getcwd())

from scripts_annotate.p6.sampler import OpenSetSampler

if __name__ == "__main__":
    # Test configuration
    # NOTE: Ensure you run this script from the project root directory
    # You might need to change this path depending on where your data is located
    test_data_dir = "data/BelugaID/IDs"

    print(f"Testing OpenSetSampler (Protocol 6) with data_dir: {test_data_dir}")

    if not os.path.exists(test_data_dir):
        print(f"Error: Data directory {test_data_dir} does not exist.")
        print(
            "Please adjust 'test_data_dir' in the script to point to a valid dataset."
        )
        sys.exit(1)

    # Test different gallery sizes
    # Note: In P6, gallery_size N refers to the number of distractors.
    # The total options presented to the model will be N + 1 (None of the above).
    test_gallery_sizes = [4, 8]

    for N in test_gallery_sizes:
        print(f"\n--- Testing Gallery Size N={N} (Total Options={N + 1}) ---")
        try:
            sampler = OpenSetSampler(
                dataset_name="TestDataset",
                data_dir=test_data_dir,
                gallery_size=N,
                max_queries_per_id=5,
                max_jaccard_sim=None,  # Enable adaptive threshold
            )

            # Generate and print one sample to verify the structure
            print("\n--- Generating a sample ---")
            sample = sampler.generate_sample()
            if sample:
                print("Sample generated successfully:")
                print(json.dumps(sample, indent=2))

                # Basic validation logic
                gallery = sample["gallery"]
                answer = sample["answer"]

                # Check 1: Gallery size should be N + 1 (images + text option)
                if len(gallery) != N + 1:
                    print(
                        f"[FAIL] Gallery size mismatch. Expected {N + 1}, got {len(gallery)}"
                    )
                else:
                    print(
                        f"[PASS] Gallery size is {len(gallery)} (N={N} images + 1 text option)"
                    )

                # Check 2: Last option should be "None of the above"
                last_opt = gallery[-1]
                if last_opt.get("text") == "None of the above":
                    print("[PASS] Last option is 'None of the above'")
                else:
                    print(f"[FAIL] Last option is NOT 'None of the above': {last_opt}")

                # Check 3 & 4: Query ID presence and Answer correctness
                query_id = sample["query"]["ground_truth_id"]
                gallery_ids = [item["id"] for item in gallery if item["id"] is not None]

                is_target_present = query_id in gallery_ids
                if is_target_present:
                    print("[INFO] Target IS present in gallery (Closed-set query).")
                    
                    # Find expected answer option
                    expected_ans = None
                    for opt in gallery:
                        if opt.get("id") == query_id:
                            expected_ans = opt["option"]
                            break
                    
                    if answer == expected_ans:
                        print(f"[PASS] Answer correctly points to the Target image option ({answer}).")
                    else:
                        print(f"[FAIL] Answer mismatch. Target is at {expected_ans}, but Answer is {answer}.")
                else:
                    print("[INFO] Target is NOT in gallery (Open-set query).")
                    if answer == last_opt["option"]:
                        print(f"[PASS] Answer correctly points to the 'None of the above' option ({answer}).")
                    else:
                        print(f"[FAIL] Answer mismatch. Answer: {answer}, expected: {last_opt['option']}")

            else:
                print(
                    "Failed to generate a sample. Constraints might be too tight or no eligible queries left."
                )
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback

            traceback.print_exc()
