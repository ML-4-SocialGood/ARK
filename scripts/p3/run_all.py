"""
scripts/p3/run_all.py
Batch generation script for Protocol 3 (MetaWild data).
"""

import os
import subprocess
import sys


def main():
    # Define paths relative to the project root
    project_root = os.getcwd()
    # P3 data is specifically in data/MetaWild
    data_root = os.path.join(project_root, "data", "MetaWild")

    # Path to the generation script
    generate_script = os.path.join(project_root, "scripts", "p3", "generate_dataset.py")
    analyze_script = os.path.join(project_root, "scripts", "p3", "analyze_dataset.py")
    stats_script = os.path.join(project_root, "scripts", "p3", "generate_stats.py")

    # Check if data directory exists
    if not os.path.exists(data_root):
        print(f"Error: Data root directory '{data_root}' not found.")
        return

    # 1. Identify Species in MetaWild
    # Structure: data/MetaWild/{Species}/IDs/
    species_list = []
    for item in os.listdir(data_root):
        species_path = os.path.join(data_root, item)
        ids_path = os.path.join(species_path, "IDs")
        # Check for IDs folder to confirm it's a valid species dir
        if os.path.isdir(species_path) and os.path.isdir(ids_path):
            species_list.append(item)

    species_list.sort()

    if not species_list:
        print("No valid species datasets found in 'data/MetaWild/'.")
        return

    print(f"Found {len(species_list)} species in MetaWild: {', '.join(species_list)}")
    print("=" * 60)

    # 2. Process each species
    for i, species in enumerate(species_list):
        print(f"\n[{i + 1}/{len(species_list)}] Processing Species: {species}")

        # For P3, the data_dir passed to sampler is the species root (containing IDs/ and json)
        species_data_dir = os.path.join("data", "MetaWild", species)

        # We use "MetaWild/{Species}" as the dataset name to maintain directory structure
        # generate_dataset.py will handle sanitizing this for the filename
        dataset_name = f"MetaWild/{species}"

        # Define the gallery sizes to generate
        gallery_sizes = [4, 8, 16, 32]

        for N in gallery_sizes:
            print(f"  > Processing Gallery Size N={N}...")

            # --- Step 1: Generate Dataset ---
            gen_cmd = [
                sys.executable,
                generate_script,
                "--dataset_name",
                dataset_name,
                "--data_dir",
                species_data_dir,
                "--gallery_size",
                str(N),
                "--max_queries_per_id",
                "10000",  # Maximize samples
                "--seed",
                "42",
            ]

            try:
                subprocess.run(gen_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"    Error generating dataset for {species} (N={N}): {e}")
                continue
        
        # --- Step 2: Analyze Dataset ---
        print(f"  > Analyzing dataset for {species}...")
        analyze_cmd = [
            sys.executable,
            analyze_script,
            "--dataset_name",
            dataset_name,
        ]
        try:
            subprocess.run(analyze_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"    Error analyzing dataset for {species}: {e}")

    print("\n" + "=" * 60)
    print("All species processed successfully.")

    # --- Final Step: Generate Global Statistics Excel ---
    print("Generating global statistics report (Excel)...")
    stats_cmd = [
        sys.executable,
        stats_script,
        "--annotations_dir", "annotations",
        "--output_file", "p3_dataset_stats.xlsx"
    ]
    subprocess.run(stats_cmd, check=True)


if __name__ == "__main__":
    main()
