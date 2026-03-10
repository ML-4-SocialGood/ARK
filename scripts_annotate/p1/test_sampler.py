import sys
import os
import json

# Ensure imports work when running from project root
sys.path.append(os.getcwd())

from scripts.p1.sampler import DynamicReIDSampler

if __name__ == "__main__":
    # Test configuration
    # NOTE: Ensure you run this script from the project root directory
    test_data_dir = "data/BelugaID/IDs"

    print(f"Testing DynamicReIDSampler with data_dir: {test_data_dir}")

    # Test different gallery sizes to verify adaptive threshold logic
    # test_gallery_sizes = [4, 8, 16, 32]
    test_gallery_sizes = [4]
    
    for N in test_gallery_sizes:
        print(f"\n--- Testing Gallery Size N={N} ---")
        try:
            sampler = DynamicReIDSampler(
                data_dir=test_data_dir,
                gallery_size=N,
                max_queries_per_id=5,  # Arbitrary for testing
                max_jaccard_sim=None,  # Enable adaptive threshold
            )

            # Generate and print one sample to verify the structure
            print("\n--- Generating a sample ---")
            sample = sampler.generate_sample()
            if sample:
                print("Sample generated successfully:")
                print(json.dumps(sample, indent=2))
            else:
                print("Failed to generate a sample. Constraints might be too tight or no eligible queries left.")
        except Exception as e:
            print(f"An error occurred: {e}")