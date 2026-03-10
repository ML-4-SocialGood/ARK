import os
import subprocess
import sys


def main():
    # Define paths relative to the project root
    project_root = os.getcwd()
    data_root = os.path.join(project_root, "data")

    # Path to the scripts we want to run
    generate_script = os.path.join(project_root, "scripts_annotate", "p2", "generate_dataset.py")
    analyze_script = os.path.join(project_root, "scripts_annotate", "p2", "analyze_dataset.py")

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
        # Protocol 2 focuses on multi-image query, but we still vary gallery size
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

            print(f"    Generating P2 (MCQ) dataset (N={N}, K_max={max_query_size})...")
            try:
                subprocess.run(gen_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"    Error generating dataset for {species} (N={N}): {e}")
                continue

            # --- Step 2: Analyze Dataset (using K=1 as representative) ---
            # Since P4 generates batches, the distribution of IDs is the same for all K.
            # We analyze K=1 to get the distribution of unique batches per ID.
            
            base_name_k1 = f"{species}_MCQ_P2_N{N}_K1"
            json_file_k1 = os.path.join(project_root, "annotations", species, "p2", f"{base_name_k1}.json")
            plot_file = os.path.join(project_root, "annotations", species, "p2", f"{base_name_k1}_distribution.png")

            if os.path.exists(json_file_k1):
                print(f"    Analyzing dataset statistics (N={N}, K=1)...")
                analyze_cmd = [
                    sys.executable,
                    analyze_script,
                    "--json_file",
                    json_file_k1,
                    "--output_plot",
                    plot_file,
                ]
                try:
                    subprocess.run(analyze_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"    Error analyzing dataset for {species} (N={N}): {e}")
            else:
                 print(f"    Warning: Expected JSON file not found for analysis: {json_file_k1}")

    print("\n" + "=" * 60)
    print("All species processed successfully.")


if __name__ == "__main__":
    main()
