import os
import subprocess
import sys


def main():
    # Define paths relative to the project root
    project_root = os.getcwd()
    data_root = os.path.join(project_root, "data")

    # Path to the scripts we want to run
    generate_script = os.path.join(project_root, "scripts", "p4", "generate_dataset.py")

    # Check if data directory exists
    if not os.path.exists(data_root):
        print(f"Error: Data root directory '{data_root}' not found.")
        return

    # 1. Identify Species
    # Scan data/ for folders that contain an 'IDs' subdirectory
    species_list = []
    for item in os.listdir(data_root):
        species_path = os.path.join(data_root, item)
        ids_path = os.path.join(species_path, "IDs")
        if os.path.isdir(species_path) and os.path.isdir(ids_path):
            species_list.append(item)

    species_list.sort()

    if not species_list:
        print(
            "No valid species datasets found in 'data/'. Expected structure: data/{SpeciesName}/IDs/"
        )
        return

    print(f"Found {len(species_list)} species: {', '.join(species_list)}")
    print("=" * 60)

    # 2. Process each species
    for i, species in enumerate(species_list):
        print(f"\n[{i + 1}/{len(species_list)}] Processing Species: {species}")
        data_dir = os.path.join("data", species, "IDs")

        # Define the gallery sizes to generate
        # Protocol 4 focuses on multi-image query, but we still vary gallery size
        gallery_sizes = [4, 8, 16, 32]
        max_query_size = 4

        for N in gallery_sizes:
            print(f"  > Processing Gallery Size N={N}...")

            # --- Step 1: Generate Dataset ---
            # We use max_queries_per_id=10000 to generate the maximum possible samples
            # (capped dynamically by the number of images per ID).
            # target_batches=-1 ensures we generate until exhaustion.
            gen_cmd = [
                sys.executable,
                generate_script,
                "--dataset_name",
                species,
                "--data_dir",
                data_dir,
                "--gallery_size",
                str(N),
                "--max_query_size",
                str(max_query_size),
                "--max_queries_per_id",
                "10000",
                "--target_batches",
                "-1",
                "--seed",
                "42",
            ]

            print(f"    Generating P4 (MCQ) dataset (N={N}, K_max={max_query_size})...")
            try:
                subprocess.run(gen_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"    Error generating dataset for {species} (N={N}): {e}")
                continue

            # Note: Analysis step is skipped here because P4 generates multiple output files
            # (one for each K value: _K1.json, _K2.json, etc.), making it complex to
            # map directly to the single-file analyzer used in P1.

    print("\n" + "=" * 60)
    print("All species processed successfully.")


if __name__ == "__main__":
    main()
